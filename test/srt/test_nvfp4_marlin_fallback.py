"""
Test script for NVFP4 Marlin fallback on non-FP4 GPUs (SM75-SM89).

This validates the Marlin fallback path that allows NVFP4-quantized models
to run on GPUs without native FP4 hardware (RTX 3090, A100, etc.).

Usage:
    python test_nvfp4_marlin_fallback.py

Requirements:
    - SM75+ GPU (Turing or newer)
    - SGLang installed with Marlin kernel support
"""

import logging
import sys
import unittest

import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _check_requirements():
    from sglang.srt.utils import is_cuda

    if not is_cuda():
        print("SKIP: CUDA not available")
        return False
    from sglang.srt.layers.quantization.marlin_utils_fp4 import is_fp4_marlin_supported

    if not is_fp4_marlin_supported():
        print("SKIP: GPU is SM < 75, Marlin FP4 not supported")
        return False
    return True


class TestNvfp4MarlinLinear(unittest.TestCase):
    """Test the FP4 Marlin linear layer fallback (non-MoE)."""

    def setUp(self):
        if not _check_requirements():
            self.skipTest("Requirements not met")
        self.device = torch.device("cuda")
        self.dtype = torch.bfloat16

    def _make_fake_fp4_layer(self, N, K):
        """Build a fake layer with NVFP4 weight attributes."""

        class FakeLayer(torch.nn.Module):
            pass

        layer = FakeLayer()
        layer.params_dtype = self.dtype
        layer.input_size_per_partition = K
        layer.output_size_per_partition = N

        # weight: [N, K//2] uint8 (2 FP4 values packed per byte)
        layer.weight = torch.nn.Parameter(
            torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=self.device),
            requires_grad=False,
        )

        # weight_scale: [N, K//16] float8_e4m3fn (per-group FP8 scales)
        layer.weight_scale = torch.nn.Parameter(
            torch.ones(N, K // 16, dtype=torch.float8_e4m3fn, device=self.device),
            requires_grad=False,
        )

        # weight_scale_2_marlin: scalar float32 global scale
        layer.weight_scale_2_marlin = torch.nn.Parameter(
            torch.tensor(1.0, dtype=torch.float32, device=self.device),
            requires_grad=False,
        )

        return layer

    def test_prepare_and_apply_fp4_marlin_linear(self):
        """Test prepare_fp4_layer_for_marlin + apply_fp4_marlin_linear."""
        from sglang.srt.layers.quantization.marlin_utils_fp4 import (
            apply_fp4_marlin_linear,
            prepare_fp4_layer_for_marlin,
        )

        N, K = 256, 128
        layer = self._make_fake_fp4_layer(N, K)

        # Prepare: repack weights to Marlin format
        prepare_fp4_layer_for_marlin(
            layer,
            weight_attr="weight",
            weight_scale_attr="weight_scale",
            weight_global_scale_attr="weight_scale_2_marlin",
        )

        # Verify Marlin workspace was created
        self.assertTrue(hasattr(layer, "marlin_workspace"))

        # Run inference
        M = 16
        x = torch.randn(M, K, dtype=self.dtype, device=self.device)
        output = apply_fp4_marlin_linear(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            weight_global_scale=layer.weight_scale_2_marlin,
            workspace=layer.marlin_workspace,
            size_n=N,
            size_k=K,
        )

        self.assertEqual(output.shape, (M, N))
        self.assertEqual(output.dtype, self.dtype)
        logger.debug(
            f"Linear Marlin FP4: input [{M}, {K}] -> output [{M}, {N}] dtype={self.dtype}"
        )

    def test_nvfp4_marlin_process_scales(self):
        """Test that scale conversion functions produce non-NaN outputs."""
        from sglang.srt.layers.quantization.marlin_utils_fp4 import (
            nvfp4_marlin_process_global_scale,
            nvfp4_marlin_process_scales,
        )

        N, K_div_group = 64, 16  # K=256, group_size=16

        # Per-group FP8 scale (values from 0.5 to 1.5)
        raw_scale = torch.ones(
            N, K_div_group, dtype=torch.float8_e4m3fn, device=self.device
        )
        raw_scale = raw_scale.to(self.dtype)  # convert to half for processing
        processed = nvfp4_marlin_process_scales(raw_scale)

        self.assertFalse(torch.isnan(processed.to(self.dtype)).any())
        self.assertEqual(processed.dtype, torch.float8_e4m3fn)
        logger.debug(
            f"nvfp4_marlin_process_scales: input {raw_scale.shape} -> output {processed.shape}"
        )

        # Global scale
        global_scale = torch.tensor(1.0, dtype=self.dtype, device=self.device)
        processed_global = nvfp4_marlin_process_global_scale(global_scale)
        self.assertFalse(torch.isnan(processed_global).any())
        self.assertEqual(processed_global.dtype, self.dtype)
        logger.debug(
            f"nvfp4_marlin_process_global_scale: {global_scale.item():.4f} -> {processed_global.item():.4f}"
        )


class TestNvfp4MarlinMoe(unittest.TestCase):
    """Test the FP4 Marlin MoE fallback."""

    def setUp(self):
        if not _check_requirements():
            self.skipTest("Requirements not met")
        self.device = torch.device("cuda")
        self.dtype = torch.bfloat16

    def test_fused_marlin_moe_fp4(self):
        """Test fused_marlin_moe with FP4 global scales."""
        from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import (
            fused_marlin_moe,
        )
        from sglang.srt.layers.quantization.marlin_utils_fp4 import (
            nvfp4_marlin_process_global_scale,
            nvfp4_marlin_process_scales,
        )
        from sglang.srt.layers.quantization.marlin_utils import marlin_permute_scales

        try:
            from sglang.jit_kernel.gptq_marlin_repack import gptq_marlin_repack
        except ImportError:
            self.skipTest("gptq_marlin_repack not available")

        E = 4  # num experts
        K = 128  # hidden size
        N = 64  # intermediate size
        topk = 2
        M = 8  # batch size

        FP4_MARLIN_GROUP_SIZE = 16
        perm = torch.empty(0, dtype=torch.int, device=self.device)

        def _make_marlin_weight(size_k, size_n):
            """Create a Marlin-repacked FP4 weight for one expert."""
            raw_fp4 = torch.randint(0, 256, (size_n, size_k // 2), dtype=torch.uint8)
            qweight = raw_fp4.view(torch.int32).T.contiguous().to(self.device)
            return gptq_marlin_repack(qweight, perm, size_k, size_n, num_bits=4)

        def _make_marlin_scale(size_k, size_n):
            """Create processed Marlin FP8-S0E5M3 scale."""
            raw = torch.ones(
                size_k // FP4_MARLIN_GROUP_SIZE,
                size_n,
                dtype=self.dtype,
                device=self.device,
            )
            permuted = marlin_permute_scales(raw, size_k, size_n, FP4_MARLIN_GROUP_SIZE)
            return nvfp4_marlin_process_scales(permuted)

        def _make_global_scale():
            gs = torch.tensor(1.0, dtype=self.dtype, device=self.device)
            return nvfp4_marlin_process_global_scale(gs)

        # Build w1 (gate+up proj): [E, K//16, 4*N]
        w1_list = [_make_marlin_weight(K, 2 * N) for _ in range(E)]
        w1 = torch.stack(w1_list, dim=0)  # [E, K//16, 4*N]

        # Build w2 (down proj): [E, N//16, 2*K]
        w2_list = [_make_marlin_weight(N, K) for _ in range(E)]
        w2 = torch.stack(w2_list, dim=0)  # [E, N//16, 2*K]

        # Build scales
        w1_scale_list = [_make_marlin_scale(K, 2 * N) for _ in range(E)]
        w1_scale = torch.stack(w1_scale_list, dim=0)

        w2_scale_list = [_make_marlin_scale(N, K) for _ in range(E)]
        w2_scale = torch.stack(w2_scale_list, dim=0)

        # Global scales: one per expert
        w1_global_scale = torch.stack([_make_global_scale() for _ in range(E)])
        w2_global_scale = torch.stack([_make_global_scale() for _ in range(E)])

        # Inputs
        hidden_states = torch.randn(M, K, dtype=self.dtype, device=self.device)
        gating_output = torch.randn(M, E, dtype=self.dtype, device=self.device)

        # Compute topk
        topk_weights, topk_ids = torch.topk(
            torch.softmax(gating_output, dim=-1), topk, dim=-1
        )

        output = fused_marlin_moe(
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            gating_output=gating_output,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            num_bits=4,
            w1_global_scale=w1_global_scale,
            w2_global_scale=w2_global_scale,
        )

        self.assertEqual(output.shape, (M, K))
        self.assertEqual(output.dtype, self.dtype)
        self.assertFalse(torch.isnan(output).any(), "Output contains NaN!")
        logger.debug(
            f"fused_marlin_moe FP4: [{M}, {K}] E={E} topk={topk} -> [{M}, {K}] ✓"
        )

    def test_prepare_moe_fp4_layer_for_marlin(self):
        """Test that prepare_moe_fp4_layer_for_marlin correctly repacks weights."""
        from sglang.srt.layers.moe.utils import MoeRunnerBackend
        from sglang.srt.layers.quantization.marlin_utils_fp4 import (
            prepare_moe_fp4_layer_for_marlin,
        )

        try:
            from sglang.jit_kernel.gptq_marlin_repack import gptq_marlin_repack  # noqa
        except ImportError:
            self.skipTest("gptq_marlin_repack not available")

        E = 4
        K = 128  # hidden size
        N = 64  # intermediate size

        class FakeMoeRunnerConfig:
            is_gated = True

        class FakeLayer(torch.nn.Module):
            pass

        layer = FakeLayer()
        layer.num_local_experts = E
        layer.intermediate_size_per_partition = N
        layer.params_dtype = self.dtype
        layer.moe_runner_config = FakeMoeRunnerConfig()

        # w13_weight: [E, 2*N, K//2] uint8 (gate+up proj, packed FP4)
        layer.w13_weight = torch.nn.Parameter(
            torch.randint(0, 256, (E, 2 * N, K // 2), dtype=torch.uint8, device=self.device),
            requires_grad=False,
        )
        # w2_weight: [E, K, N//2] uint8 (down proj, packed FP4)
        layer.w2_weight = torch.nn.Parameter(
            torch.randint(0, 256, (E, K, N // 2), dtype=torch.uint8, device=self.device),
            requires_grad=False,
        )

        # w13_weight_scale: [E, 2*N, K//16] float8_e4m3fn
        layer.w13_weight_scale = torch.nn.Parameter(
            torch.ones(E, 2 * N, K // 16, dtype=torch.float8_e4m3fn, device=self.device),
            requires_grad=False,
        )
        # w2_weight_scale: [E, K, N//16] float8_e4m3fn
        layer.w2_weight_scale = torch.nn.Parameter(
            torch.ones(E, K, N // 16, dtype=torch.float8_e4m3fn, device=self.device),
            requires_grad=False,
        )

        # Global scales (actual scales, not inverted)
        layer.w13_weight_scale_2 = torch.nn.Parameter(
            torch.ones(E, 2, dtype=torch.float32, device=self.device),
            requires_grad=False,
        )
        layer.w2_weight_scale_2 = torch.nn.Parameter(
            torch.ones(E, dtype=torch.float32, device=self.device),
            requires_grad=False,
        )

        prepare_moe_fp4_layer_for_marlin(layer)

        # Verify workspace was created
        self.assertTrue(hasattr(layer, "marlin_workspace"))
        # Verify weights were repacked (shape changes)
        self.assertEqual(layer.w13_weight.shape[0], E)
        self.assertEqual(layer.w2_weight.shape[0], E)
        # Verify global scales were processed
        self.assertEqual(layer.w13_weight_scale_2.shape, (E,))
        self.assertEqual(layer.w2_weight_scale_2.shape, (E,))
        logger.debug(
            f"prepare_moe_fp4_layer_for_marlin: E={E}, K={K}, N={N} ✓"
        )


class TestFp4MarlinSupport(unittest.TestCase):
    """Test the capability detection functions."""

    def test_is_fp4_marlin_supported(self):
        from sglang.srt.layers.quantization.marlin_utils_fp4 import (
            is_fp4_marlin_supported,
        )

        result = is_fp4_marlin_supported()
        cap = torch.cuda.get_device_capability()
        sm = cap[0] * 10 + cap[1]
        expected = sm >= 75
        self.assertEqual(result, expected)
        logger.debug(f"SM{sm}: is_fp4_marlin_supported() = {result}")

    def test_min_capability_changed(self):
        """Verify get_min_capability() returns 75 (not 100)."""
        from sglang.srt.layers.quantization.modelopt_quant import ModelOptFp4Config

        cap = ModelOptFp4Config.get_min_capability()
        self.assertEqual(cap, 75, f"Expected 75, got {cap}")
        logger.debug(f"ModelOptFp4Config.get_min_capability() = {cap} ✓")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("SKIP: CUDA not available")
        sys.exit(0)

    cap = torch.cuda.get_device_capability()
    sm = cap[0] * 10 + cap[1]
    print(f"GPU: {torch.cuda.get_device_name(0)} (SM{sm})")

    if sm < 75:
        print(f"SKIP: SM{sm} < SM75, Marlin FP4 not supported")
        sys.exit(0)

    if sm >= 100:
        print(f"Note: SM{sm} has native FP4 support. Testing Marlin fallback paths anyway.")
    else:
        print(f"SM{sm}: Testing Marlin FP4 fallback (non-Blackwell GPU).")

    unittest.main(verbosity=2)
