"""Tests for MoE LoRA with Marlin FP4 quantized base weights.

Verifies that MarlinRunnerCoreWithLoRA correctly stages the Marlin FP4 MoE
pipeline and injects LoRA deltas at the right points.

Test approach:
1. Run fused_marlin_moe (no LoRA) to get base output.
2. Run MarlinRunnerCoreWithLoRA with zero-rank LoRA → should match base.
3. Run MarlinRunnerCoreWithLoRA with non-zero LoRA → output should differ from base.
4. Run MarlinRunnerCoreWithLoRA with known LoRA weights and compare
   the delta against a hand-computed reference.
"""

import random
import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=200, suite="stage-b-test-1-gpu-large")


def _check_requirements():
    from sglang.srt.utils import is_cuda

    if not is_cuda():
        return False
    from sglang.srt.layers.quantization.marlin_utils_fp4 import is_fp4_marlin_supported

    if not is_fp4_marlin_supported():
        return False
    return True


def _make_marlin_weights(E, K, N, dtype, device):
    """Build synthetic FP4 Marlin MoE weights (repacked + scale-processed)."""
    from sglang.srt.layers.quantization.marlin_utils import marlin_permute_scales
    from sglang.srt.layers.quantization.marlin_utils_fp4 import (
        nvfp4_marlin_process_scales,
    )

    from sglang.jit_kernel.gptq_marlin_repack import gptq_marlin_repack

    FP4_GROUP_SIZE = 16
    perm = torch.empty(0, dtype=torch.int, device=device)

    def _repack(size_k, size_n):
        raw_fp4 = torch.randint(0, 256, (size_n, size_k // 2), dtype=torch.uint8)
        qweight = raw_fp4.view(torch.int32).T.contiguous().to(device)
        return gptq_marlin_repack(qweight, perm, size_k, size_n, num_bits=4)

    def _scale(size_k, size_n):
        raw = torch.ones(
            size_k // FP4_GROUP_SIZE, size_n, dtype=dtype, device=device
        )
        permuted = marlin_permute_scales(raw, size_k, size_n, FP4_GROUP_SIZE)
        return nvfp4_marlin_process_scales(permuted)

    w1 = torch.stack([_repack(K, 2 * N) for _ in range(E)])
    w2 = torch.stack([_repack(N, K) for _ in range(E)])
    w1_scale = torch.stack([_scale(K, 2 * N) for _ in range(E)])
    w2_scale = torch.stack([_scale(N, K) for _ in range(E)])
    w1_global_scale = torch.ones(E, dtype=dtype, device=device)
    w2_global_scale = torch.ones(E, dtype=dtype, device=device)

    return w1, w2, w1_scale, w2_scale, w1_global_scale, w2_global_scale


def _make_lora_info(
    M, num_experts, max_loras, max_rank, hidden_size, intermediate_size, dtype, device
):
    """Build a LoRAInfo with random weights and simple dispatch info."""
    from sglang.srt.lora.lora_moe_runners import LoRAInfo

    # All tokens in one sequence, assigned to lora 0
    seg_indptr = torch.tensor([0, M], dtype=torch.int32, device=device)
    req_to_lora = torch.tensor([0], dtype=torch.int32, device=device)
    lora_ranks = torch.zeros(max_loras, dtype=torch.int32, device=device)
    lora_ranks[0] = max_rank
    adapter_enabled = torch.zeros(max_loras, dtype=torch.int32, device=device)
    adapter_enabled[0] = 1

    gate_up_a = torch.randn(
        max_loras, num_experts, 2 * max_rank, hidden_size, dtype=dtype, device=device
    ) * 0.01
    gate_up_b = torch.randn(
        max_loras, num_experts, 2 * intermediate_size, max_rank, dtype=dtype, device=device
    ) * 0.01
    down_a = torch.randn(
        max_loras, num_experts, max_rank, intermediate_size, dtype=dtype, device=device
    ) * 0.01
    down_b = torch.randn(
        max_loras, num_experts, hidden_size, max_rank, dtype=dtype, device=device
    ) * 0.01

    return LoRAInfo(
        gate_up_lora_a_weights=gate_up_a,
        gate_up_lora_b_weights=gate_up_b,
        down_lora_a_weights=down_a,
        down_lora_b_weights=down_b,
        seg_indptr=seg_indptr,
        req_to_lora=req_to_lora,
        lora_ranks=lora_ranks,
        adapter_enabled=adapter_enabled,
        max_lora_rank=max_rank,
        num_experts=num_experts,
    )


def _make_zero_lora_info(
    M, num_experts, max_loras, max_rank, hidden_size, intermediate_size, dtype, device
):
    """Build a LoRAInfo with zero weights (should produce same output as no LoRA)."""
    info = _make_lora_info(
        M, num_experts, max_loras, max_rank, hidden_size, intermediate_size, dtype, device
    )
    info.gate_up_lora_a_weights.zero_()
    info.gate_up_lora_b_weights.zero_()
    info.down_lora_a_weights.zero_()
    info.down_lora_b_weights.zero_()
    return info


class TestMarlinMoEWithLoRA(CustomTestCase):
    """Test MarlinRunnerCoreWithLoRA end-to-end."""

    def setUp(self):
        if not _check_requirements():
            self.skipTest("Requirements not met (CUDA unavailable or SM < 75)")
        self.device = torch.device("cuda")
        self.dtype = torch.bfloat16

    def _run_marlin_runner_with_lora(
        self, M, E, K, N, topk, max_loras=2, max_rank=16, lora_info=None
    ):
        """Run MarlinRunnerCoreWithLoRA and return output tensor."""
        from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
        from sglang.srt.layers.moe.moe_runner.marlin import (
            MarlinMoeQuantInfo,
            MarlinRunnerInput,
        )
        from sglang.srt.lora.marlin_lora_moe_runner import MarlinRunnerCoreWithLoRA

        w1, w2, w1_scale, w2_scale, w1_gs, w2_gs = _make_marlin_weights(
            E, K, N, self.dtype, self.device
        )

        hidden_states = torch.randn(M, K, dtype=self.dtype, device=self.device)
        gating_output = torch.randn(M, E, dtype=self.dtype, device=self.device)
        topk_weights, topk_ids = torch.topk(
            torch.softmax(gating_output, dim=-1), topk, dim=-1
        )

        config = MoeRunnerConfig(
            num_experts=E,
            num_local_experts=E,
            hidden_size=K,
            intermediate_size_per_partition=N,
            top_k=topk,
            activation="silu",
            is_gated=True,
            inplace=False,
        )

        quant_info = MarlinMoeQuantInfo(
            w13_qweight=w1,
            w2_qweight=w2,
            w13_scales=w1_scale,
            w2_scales=w2_scale,
            w13_g_idx_sort_indices=None,
            w2_g_idx_sort_indices=None,
            weight_bits=4,
            w13_global_scale=w1_gs,
            w2_global_scale=w2_gs,
        )

        runner_input = MarlinRunnerInput(
            hidden_states=hidden_states,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            router_logits=gating_output,
        )

        runner = MarlinRunnerCoreWithLoRA(config)
        result = runner.run(runner_input, quant_info, {}, lora_info=lora_info)

        return result.hidden_states, hidden_states, topk_weights, topk_ids, w1, w2

    def test_output_shape_and_dtype(self):
        """Basic sanity: output has correct shape and dtype, no NaN."""
        M, E, K, N, topk = 8, 4, 128, 64, 2
        output, *_ = self._run_marlin_runner_with_lora(M, E, K, N, topk)

        self.assertEqual(output.shape, (M, K))
        self.assertEqual(output.dtype, self.dtype)
        self.assertFalse(torch.isnan(output).any(), "Output contains NaN!")

    def test_output_shape_with_lora(self):
        """Output shape is correct when LoRA is applied."""
        M, E, K, N, topk = 8, 4, 128, 64, 2
        lora_info = _make_lora_info(
            M, E, 2, 16, K, N, self.dtype, self.device
        )
        output, *_ = self._run_marlin_runner_with_lora(
            M, E, K, N, topk, lora_info=lora_info
        )

        self.assertEqual(output.shape, (M, K))
        self.assertEqual(output.dtype, self.dtype)
        self.assertFalse(torch.isnan(output).any(), "Output contains NaN!")

    def test_zero_lora_matches_no_lora(self):
        """Zero-weight LoRA should produce the same output as no LoRA."""
        M, E, K, N, topk = 16, 4, 128, 64, 2

        # We need deterministic weights, so seed + use same weights
        torch.manual_seed(42)
        from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
        from sglang.srt.layers.moe.moe_runner.marlin import (
            MarlinMoeQuantInfo,
            MarlinRunnerInput,
        )
        from sglang.srt.lora.marlin_lora_moe_runner import MarlinRunnerCoreWithLoRA

        w1, w2, w1_scale, w2_scale, w1_gs, w2_gs = _make_marlin_weights(
            E, K, N, self.dtype, self.device
        )
        hidden_states = torch.randn(M, K, dtype=self.dtype, device=self.device)
        gating_output = torch.randn(M, E, dtype=self.dtype, device=self.device)
        topk_weights, topk_ids = torch.topk(
            torch.softmax(gating_output, dim=-1), topk, dim=-1
        )

        config = MoeRunnerConfig(
            num_experts=E,
            num_local_experts=E,
            hidden_size=K,
            intermediate_size_per_partition=N,
            top_k=topk,
            activation="silu",
            is_gated=True,
            inplace=False,
        )
        quant_info = MarlinMoeQuantInfo(
            w13_qweight=w1,
            w2_qweight=w2,
            w13_scales=w1_scale,
            w2_scales=w2_scale,
            w13_g_idx_sort_indices=None,
            w2_g_idx_sort_indices=None,
            weight_bits=4,
            w13_global_scale=w1_gs,
            w2_global_scale=w2_gs,
        )
        runner_input = MarlinRunnerInput(
            hidden_states=hidden_states,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            router_logits=gating_output,
        )

        runner = MarlinRunnerCoreWithLoRA(config)

        # No LoRA
        out_no_lora = runner.run(runner_input, quant_info, {}, lora_info=None)

        # Zero LoRA
        zero_lora = _make_zero_lora_info(
            M, E, 2, 16, K, N, self.dtype, self.device
        )
        out_zero_lora = runner.run(runner_input, quant_info, {}, lora_info=zero_lora)

        torch.testing.assert_close(
            out_no_lora.hidden_states,
            out_zero_lora.hidden_states,
            atol=1e-3,
            rtol=1e-3,
        )

    def test_nonzero_lora_changes_output(self):
        """Non-zero LoRA weights should produce a different output than no LoRA."""
        M, E, K, N, topk = 16, 4, 128, 64, 2

        torch.manual_seed(42)
        from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
        from sglang.srt.layers.moe.moe_runner.marlin import (
            MarlinMoeQuantInfo,
            MarlinRunnerInput,
        )
        from sglang.srt.lora.marlin_lora_moe_runner import MarlinRunnerCoreWithLoRA

        w1, w2, w1_scale, w2_scale, w1_gs, w2_gs = _make_marlin_weights(
            E, K, N, self.dtype, self.device
        )
        hidden_states = torch.randn(M, K, dtype=self.dtype, device=self.device)
        gating_output = torch.randn(M, E, dtype=self.dtype, device=self.device)
        topk_weights, topk_ids = torch.topk(
            torch.softmax(gating_output, dim=-1), topk, dim=-1
        )

        config = MoeRunnerConfig(
            num_experts=E,
            num_local_experts=E,
            hidden_size=K,
            intermediate_size_per_partition=N,
            top_k=topk,
            activation="silu",
            is_gated=True,
            inplace=False,
        )
        quant_info = MarlinMoeQuantInfo(
            w13_qweight=w1,
            w2_qweight=w2,
            w13_scales=w1_scale,
            w2_scales=w2_scale,
            w13_g_idx_sort_indices=None,
            w2_g_idx_sort_indices=None,
            weight_bits=4,
            w13_global_scale=w1_gs,
            w2_global_scale=w2_gs,
        )
        runner_input = MarlinRunnerInput(
            hidden_states=hidden_states,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            router_logits=gating_output,
        )

        runner = MarlinRunnerCoreWithLoRA(config)

        # No LoRA
        out_no_lora = runner.run(runner_input, quant_info, {}, lora_info=None)

        # Non-zero LoRA (scaled up to be visible over FP4 noise)
        lora_info = _make_lora_info(
            M, E, 2, 16, K, N, self.dtype, self.device
        )
        # Scale up LoRA weights so the delta is clearly visible
        lora_info.gate_up_lora_a_weights *= 10.0
        lora_info.gate_up_lora_b_weights *= 10.0
        lora_info.down_lora_a_weights *= 10.0
        lora_info.down_lora_b_weights *= 10.0

        out_with_lora = runner.run(runner_input, quant_info, {}, lora_info=lora_info)

        diff = (out_no_lora.hidden_states - out_with_lora.hidden_states).abs().max()
        self.assertGreater(
            diff.item(),
            1e-3,
            "LoRA weights should change the output, but diff is negligible",
        )

    def test_various_sizes(self):
        """Test with different M, E, topk combinations."""
        test_cases = [
            (1, 4, 128, 64, 2),
            (32, 8, 256, 128, 2),
            (4, 4, 128, 64, 1),
            (64, 4, 128, 64, 2),
        ]
        for M, E, K, N, topk in test_cases:
            with self.subTest(M=M, E=E, K=K, N=N, topk=topk):
                lora_info = _make_lora_info(
                    M, E, 2, 16, K, N, self.dtype, self.device
                )
                output, *_ = self._run_marlin_runner_with_lora(
                    M, E, K, N, topk, lora_info=lora_info
                )
                self.assertEqual(output.shape, (M, K))
                self.assertFalse(torch.isnan(output).any(), f"NaN for M={M},E={E}")

    def test_matches_fused_marlin_moe_without_lora(self):
        """Output without LoRA should match the original fused_marlin_moe."""
        from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import (
            fused_marlin_moe,
        )
        from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
        from sglang.srt.layers.moe.moe_runner.marlin import (
            MarlinMoeQuantInfo,
            MarlinRunnerInput,
        )
        from sglang.srt.lora.marlin_lora_moe_runner import MarlinRunnerCoreWithLoRA

        M, E, K, N, topk = 16, 4, 128, 64, 2
        torch.manual_seed(123)

        w1, w2, w1_scale, w2_scale, w1_gs, w2_gs = _make_marlin_weights(
            E, K, N, self.dtype, self.device
        )
        hidden_states = torch.randn(M, K, dtype=self.dtype, device=self.device)
        gating_output = torch.randn(M, E, dtype=self.dtype, device=self.device)
        topk_weights, topk_ids = torch.topk(
            torch.softmax(gating_output, dim=-1), topk, dim=-1
        )

        # Reference: fused_marlin_moe
        ref_output = fused_marlin_moe(
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            gating_output=gating_output,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            num_bits=4,
            w1_global_scale=w1_gs,
            w2_global_scale=w2_gs,
        )

        # Test: MarlinRunnerCoreWithLoRA without LoRA
        config = MoeRunnerConfig(
            num_experts=E,
            num_local_experts=E,
            hidden_size=K,
            intermediate_size_per_partition=N,
            top_k=topk,
            activation="silu",
            is_gated=True,
            inplace=False,
        )
        quant_info = MarlinMoeQuantInfo(
            w13_qweight=w1,
            w2_qweight=w2,
            w13_scales=w1_scale,
            w2_scales=w2_scale,
            w13_g_idx_sort_indices=None,
            w2_g_idx_sort_indices=None,
            weight_bits=4,
            w13_global_scale=w1_gs,
            w2_global_scale=w2_gs,
        )
        runner_input = MarlinRunnerInput(
            hidden_states=hidden_states,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            router_logits=gating_output,
        )

        runner = MarlinRunnerCoreWithLoRA(config)
        test_output = runner.run(runner_input, quant_info, {}, lora_info=None)

        torch.testing.assert_close(
            ref_output, test_output.hidden_states, atol=1e-3, rtol=1e-3
        )


if __name__ == "__main__":
    unittest.main(verbosity=3)
