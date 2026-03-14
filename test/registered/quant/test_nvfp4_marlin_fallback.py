"""Tests for NVFP4 Marlin fallback on non-Blackwell GPUs (SM75+)."""

import logging
import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=1200, suite="stage-b-test-large-1-gpu")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _check_requirements():
    from sglang.srt.utils import is_cuda

    if not is_cuda():
        return False
    from sglang.srt.layers.quantization.marlin_utils_fp4 import is_fp4_marlin_supported

    if not is_fp4_marlin_supported():
        return False
    return True


class TestNvfp4MarlinLinear(CustomTestCase):
    """Test the FP4 Marlin linear layer fallback (non-MoE)."""

    def setUp(self):
        if not _check_requirements():
            self.skipTest("Requirements not met (CUDA unavailable or SM < 75)")
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

        layer.weight = torch.nn.Parameter(
            torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=self.device),
            requires_grad=False,
        )

        layer.weight_scale = torch.nn.Parameter(
            torch.ones(N, K // 16, dtype=torch.float8_e4m3fn, device=self.device),
            requires_grad=False,
        )

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

        prepare_fp4_layer_for_marlin(
            layer,
            weight_attr="weight",
            weight_scale_attr="weight_scale",
            weight_global_scale_attr="weight_scale_2_marlin",
        )

        self.assertTrue(hasattr(layer, "marlin_workspace"))

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

    def test_nvfp4_marlin_process_scales(self):
        """Test that scale conversion functions produce non-NaN outputs."""
        from sglang.srt.layers.quantization.marlin_utils_fp4 import (
            nvfp4_marlin_process_global_scale,
            nvfp4_marlin_process_scales,
        )

        N, K_div_group = 64, 16

        raw_scale = torch.ones(
            N, K_div_group, dtype=torch.float8_e4m3fn, device=self.device
        )
        raw_scale = raw_scale.to(self.dtype)
        processed = nvfp4_marlin_process_scales(raw_scale)

        self.assertFalse(torch.isnan(processed.to(self.dtype)).any())
        self.assertEqual(processed.dtype, torch.float8_e4m3fn)

        # Large scales (e.g. 448 = FP8 E4M3 max) are valid. The int16 wrapping
        # from (scale*128) << 1 preserves bit patterns correctly for the kernel.
        large_scale = torch.full(
            (N, K_div_group), 448.0, dtype=self.dtype, device=self.device
        )
        proc_large = nvfp4_marlin_process_scales(large_scale)
        self.assertFalse(torch.isnan(proc_large.to(self.dtype)).any())

        global_scale = torch.tensor(1.0, dtype=self.dtype, device=self.device)
        processed_global = nvfp4_marlin_process_global_scale(global_scale)
        self.assertFalse(torch.isnan(processed_global).any())
        self.assertEqual(processed_global.dtype, self.dtype)


class TestNvfp4MarlinMoe(CustomTestCase):
    """Test the FP4 Marlin MoE fallback."""

    def setUp(self):
        if not _check_requirements():
            self.skipTest("Requirements not met (CUDA unavailable or SM < 75)")
        self.device = torch.device("cuda")
        self.dtype = torch.bfloat16

    def test_fused_marlin_moe_fp4(self):
        """Test fused_marlin_moe with FP4 global scales."""
        from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import (
            fused_marlin_moe,
        )
        from sglang.srt.layers.quantization.marlin_utils import marlin_permute_scales
        from sglang.srt.layers.quantization.marlin_utils_fp4 import (
            nvfp4_marlin_process_scales,
        )

        try:
            from sglang.jit_kernel.gptq_marlin_repack import gptq_marlin_repack
        except ImportError:
            self.skipTest("gptq_marlin_repack not available")

        E = 4
        K = 128
        N = 64
        topk = 2
        M = 8

        FP4_MARLIN_GROUP_SIZE = 16
        perm = torch.empty(0, dtype=torch.int, device=self.device)

        def _make_marlin_weight(size_k, size_n):
            raw_fp4 = torch.randint(0, 256, (size_n, size_k // 2), dtype=torch.uint8)
            qweight = raw_fp4.view(torch.int32).T.contiguous().to(self.device)
            return gptq_marlin_repack(qweight, perm, size_k, size_n, num_bits=4)

        def _make_marlin_scale(size_k, size_n):
            raw = torch.ones(
                size_k // FP4_MARLIN_GROUP_SIZE,
                size_n,
                dtype=self.dtype,
                device=self.device,
            )
            permuted = marlin_permute_scales(raw, size_k, size_n, FP4_MARLIN_GROUP_SIZE)
            processed = nvfp4_marlin_process_scales(permuted)
            return processed

        def _make_global_scale():
            return torch.tensor(1.0, dtype=self.dtype, device=self.device)

        w1_list = [_make_marlin_weight(K, 2 * N) for _ in range(E)]
        w1 = torch.stack(w1_list, dim=0)

        w2_list = [_make_marlin_weight(N, K) for _ in range(E)]
        w2 = torch.stack(w2_list, dim=0)

        w1_scale_list = [_make_marlin_scale(K, 2 * N) for _ in range(E)]
        w1_scale = torch.stack(w1_scale_list, dim=0)

        w2_scale_list = [_make_marlin_scale(N, K) for _ in range(E)]
        w2_scale = torch.stack(w2_scale_list, dim=0)

        w1_global_scale = torch.stack([_make_global_scale() for _ in range(E)])
        w2_global_scale = torch.stack([_make_global_scale() for _ in range(E)])

        hidden_states = torch.randn(M, K, dtype=self.dtype, device=self.device)
        gating_output = torch.randn(M, E, dtype=self.dtype, device=self.device)

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

    def test_prepare_moe_fp4_layer_for_marlin(self):
        """Test that prepare_moe_fp4_layer_for_marlin correctly repacks weights."""
        from sglang.srt.layers.quantization.marlin_utils_fp4 import (
            prepare_moe_fp4_layer_for_marlin,
        )

        try:
            from sglang.jit_kernel.gptq_marlin_repack import gptq_marlin_repack  # noqa
        except ImportError:
            self.skipTest("gptq_marlin_repack not available")

        E = 4
        K = 128
        N = 64

        class FakeMoeRunnerConfig:
            is_gated = True

        class FakeLayer(torch.nn.Module):
            pass

        layer = FakeLayer()
        layer.num_local_experts = E
        layer.intermediate_size_per_partition = N
        layer.params_dtype = self.dtype
        layer.moe_runner_config = FakeMoeRunnerConfig()

        layer.w13_weight = torch.nn.Parameter(
            torch.randint(
                0, 256, (E, 2 * N, K // 2), dtype=torch.uint8, device=self.device
            ),
            requires_grad=False,
        )
        layer.w2_weight = torch.nn.Parameter(
            torch.randint(
                0, 256, (E, K, N // 2), dtype=torch.uint8, device=self.device
            ),
            requires_grad=False,
        )

        layer.w13_weight_scale = torch.nn.Parameter(
            torch.ones(
                E, 2 * N, K // 16, dtype=torch.float8_e4m3fn, device=self.device
            ),
            requires_grad=False,
        )
        layer.w2_weight_scale = torch.nn.Parameter(
            torch.ones(E, K, N // 16, dtype=torch.float8_e4m3fn, device=self.device),
            requires_grad=False,
        )

        layer.w13_weight_scale_2 = torch.nn.Parameter(
            torch.ones(E, 2, dtype=torch.float32, device=self.device),
            requires_grad=False,
        )
        layer.w2_weight_scale_2 = torch.nn.Parameter(
            torch.ones(E, dtype=torch.float32, device=self.device),
            requires_grad=False,
        )

        prepare_moe_fp4_layer_for_marlin(layer)

        self.assertEqual(layer.w13_weight.shape[0], E)
        self.assertEqual(layer.w2_weight.shape[0], E)
        self.assertEqual(layer.w13_weight_scale_2.shape, (E,))
        self.assertEqual(layer.w2_weight_scale_2.shape, (E,))


class TestNvfp4MarlinCorrectness(CustomTestCase):
    """End-to-end correctness test: Marlin kernel output vs Python reference."""

    def setUp(self):
        if not _check_requirements():
            self.skipTest("Requirements not met (CUDA unavailable or SM < 75)")
        self.device = torch.device("cuda")
        self.dtype = torch.bfloat16

    @staticmethod
    def _reference_dequant_fp4(packed_uint8: torch.Tensor, size_n: int, size_k: int):
        """Dequantize FP4 E2M1 packed uint8 weights to float (vLLM reference)."""
        fp4 = packed_uint8  # (N, K//2) uint8

        # High nibble (bits [7:4]) → odd K indices
        part_hi = (fp4 & 0b10000000) | ((fp4 & 0b01110000) >> 2)
        part_hi = part_hi.view(torch.float8_e4m3fn).to(torch.float32) * (2**6)

        # Low nibble (bits [3:0] shifted to [7:4]) → even K indices
        fp4_lo = fp4 << 4
        part_lo = (fp4_lo & 0b10000000) | ((fp4_lo & 0b01110000) >> 2)
        part_lo = part_lo.view(torch.float8_e4m3fn).to(torch.float32) * (2**6)

        # Interleave: [even, odd] = [low_nibble, high_nibble]
        weight_ref = torch.cat(
            [part_lo.unsqueeze(2), part_hi.unsqueeze(2)], dim=2
        ).view(size_n, size_k)
        return weight_ref

    def test_marlin_gemm_matches_reference(self):
        """Compare Marlin kernel output against Python reference dequantization."""
        from sglang.srt.layers.quantization.marlin_utils import marlin_permute_scales
        from sglang.srt.layers.quantization.marlin_utils_fp4 import (
            FP4_MARLIN_GROUP_SIZE,
            apply_fp4_marlin_linear,
            marlin_make_workspace,
            nvfp4_marlin_process_global_scale,
            nvfp4_marlin_process_scales,
        )

        try:
            from sglang.jit_kernel.gptq_marlin_repack import gptq_marlin_repack
        except ImportError:
            self.skipTest("gptq_marlin_repack not available")

        N, K = 256, 256
        M = 16
        GROUP_SIZE = FP4_MARLIN_GROUP_SIZE  # 16

        torch.manual_seed(42)

        # --- Create synthetic quantized weights ---
        fp4_packed = torch.randint(
            0, 256, (N, K // 2), dtype=torch.uint8, device=self.device
        )

        # Reference dequant
        weight_float = self._reference_dequant_fp4(fp4_packed, N, K)  # (N, K) float32

        # Per-group scales: choose values from valid FP8 E4M3 range
        num_groups = K // GROUP_SIZE
        scales_float = (
            torch.rand(N, num_groups, device=self.device) * 10 + 0.5
        )  # [0.5, 10.5]
        scales_fp8 = scales_float.to(torch.float8_e4m3fn)
        scales_float_actual = scales_fp8.to(torch.float32)  # after FP8 rounding

        # Global scale: small positive value (typical for NVFP4 models)
        global_scale_val = 0.001
        global_scale = torch.tensor(
            global_scale_val, dtype=self.dtype, device=self.device
        )

        # --- Reference output ---
        # weight_deq = FP4_decoded * per_group_scale * global_scale
        scales_expanded = scales_float_actual.repeat_interleave(
            GROUP_SIZE, dim=1
        )  # (N, K)
        weight_deq = weight_float * scales_expanded * global_scale_val  # (N, K) float32
        weight_deq = weight_deq.to(self.dtype)

        x = torch.randn(M, K, dtype=self.dtype, device=self.device)
        ref_output = x @ weight_deq.T  # (M, N)

        # --- Marlin kernel output ---
        perm = torch.empty(0, dtype=torch.int, device=self.device)
        qweight = fp4_packed.view(torch.int32).T.contiguous()
        marlin_qweight = gptq_marlin_repack(
            b_q_weight=qweight, perm=perm, size_k=K, size_n=N, num_bits=4
        )

        # Process scales through Marlin pipeline
        weight_scale = scales_fp8.T.contiguous().to(self.dtype)  # (num_groups, N)
        weight_scale = marlin_permute_scales(
            s=weight_scale, size_k=K, size_n=N, group_size=GROUP_SIZE
        )
        weight_scale = nvfp4_marlin_process_scales(weight_scale)

        global_scale_processed = nvfp4_marlin_process_global_scale(global_scale)

        workspace = marlin_make_workspace(self.device)

        marlin_output = apply_fp4_marlin_linear(
            input=x,
            weight=marlin_qweight,
            weight_scale=weight_scale,
            weight_global_scale=global_scale_processed,
            workspace=workspace,
            size_n=N,
            size_k=K,
        )

        # --- Compare ---
        ref_f32 = ref_output.float()
        mar_f32 = marlin_output.float()

        # Log statistics for debugging
        abs_diff = (ref_f32 - mar_f32).abs()
        rel_diff = abs_diff / (ref_f32.abs() + 1e-6)

        logger.info(
            "Correctness test: ref range [%.4f, %.4f], marlin range [%.4f, %.4f]",
            ref_f32.min().item(),
            ref_f32.max().item(),
            mar_f32.min().item(),
            mar_f32.max().item(),
        )
        logger.info(
            "Abs diff: mean=%.6f, max=%.6f. Rel diff: mean=%.6f, max=%.6f",
            abs_diff.mean().item(),
            abs_diff.max().item(),
            rel_diff.mean().item(),
            rel_diff.max().item(),
        )

        # BF16 has ~7 bits of mantissa → relative tolerance ~1%.
        # FP4 quantization adds additional error, so allow ~5% relative error.
        cos_sim = torch.nn.functional.cosine_similarity(
            ref_f32.flatten().unsqueeze(0), mar_f32.flatten().unsqueeze(0)
        ).item()
        logger.info("Cosine similarity: %.6f", cos_sim)

        self.assertGreater(cos_sim, 0.99, f"Cosine similarity too low: {cos_sim:.6f}")

        # Also check max absolute error is bounded
        max_abs = abs_diff.max().item()
        ref_range = ref_f32.abs().max().item()
        self.assertLess(
            max_abs / (ref_range + 1e-8),
            0.05,
            f"Max relative error too large: {max_abs / (ref_range + 1e-8):.4f}",
        )

    def test_marlin_gemm_bf16_vs_fp16(self):
        """Verify that BF16 and FP16 models produce finite non-zero Marlin output."""
        from sglang.srt.layers.quantization.marlin_utils import marlin_permute_scales
        from sglang.srt.layers.quantization.marlin_utils_fp4 import (
            FP4_MARLIN_GROUP_SIZE,
            apply_fp4_marlin_linear,
            marlin_make_workspace,
            nvfp4_marlin_process_global_scale,
            nvfp4_marlin_process_scales,
        )

        try:
            from sglang.jit_kernel.gptq_marlin_repack import gptq_marlin_repack
        except ImportError:
            self.skipTest("gptq_marlin_repack not available")

        N, K = 256, 256
        M = 16
        GROUP_SIZE = FP4_MARLIN_GROUP_SIZE

        torch.manual_seed(123)

        fp4_packed = torch.randint(
            0, 256, (N, K // 2), dtype=torch.uint8, device=self.device
        )
        scales_fp8 = (torch.rand(N, K // GROUP_SIZE, device=self.device) * 5 + 0.5).to(
            torch.float8_e4m3fn
        )
        global_scale_val = 0.0005

        perm = torch.empty(0, dtype=torch.int, device=self.device)
        qweight = fp4_packed.view(torch.int32).T.contiguous()
        marlin_qweight = gptq_marlin_repack(
            b_q_weight=qweight, perm=perm, size_k=K, size_n=N, num_bits=4
        )
        workspace = marlin_make_workspace(self.device)

        outputs = {}
        for dtype in [torch.float16, torch.bfloat16]:
            x = torch.randn(M, K, dtype=dtype, device=self.device)
            gs = torch.tensor(global_scale_val, dtype=dtype, device=self.device)

            ws = scales_fp8.T.contiguous().to(dtype)
            ws = marlin_permute_scales(s=ws, size_k=K, size_n=N, group_size=GROUP_SIZE)
            ws = nvfp4_marlin_process_scales(ws)
            gs_proc = nvfp4_marlin_process_global_scale(gs)

            out = apply_fp4_marlin_linear(
                input=x,
                weight=marlin_qweight,
                weight_scale=ws,
                weight_global_scale=gs_proc,
                workspace=workspace,
                size_n=N,
                size_k=K,
            )
            outputs[dtype] = out.float()

        # Both should produce finite values
        for dtype, out in outputs.items():
            self.assertFalse(torch.isnan(out).any(), f"{dtype} output contains NaN")
            self.assertFalse(torch.isinf(out).any(), f"{dtype} output contains Inf")
            self.assertFalse((out == 0).all(), f"{dtype} output is all zeros")

        # The outputs won't be identical (different dtypes, different inputs),
        # but both should have similar magnitude distributions
        for dtype, out in outputs.items():
            logger.info(
                "%s output: min=%.4f, max=%.4f, mean=%.4f, std=%.4f",
                dtype,
                out.min().item(),
                out.max().item(),
                out.mean().item(),
                out.std().item(),
            )


class TestFp4MarlinSupport(CustomTestCase):
    """Test the capability detection functions."""

    def test_is_fp4_marlin_supported(self):
        from sglang.srt.layers.quantization.marlin_utils_fp4 import (
            is_fp4_marlin_supported,
        )

        result = is_fp4_marlin_supported()
        if torch.cuda.is_available() and torch.version.hip is None:
            cap = torch.cuda.get_device_capability()
            sm = cap[0] * 10 + cap[1]
            expected = sm >= 75
            self.assertEqual(result, expected)
        elif torch.version.hip is not None:
            self.assertFalse(result, "FP4 Marlin should not be supported on ROCm/HIP")

    def test_min_capability_changed(self):
        """Verify get_min_capability() returns 75 (not 100)."""
        from sglang.srt.layers.quantization.modelopt_quant import ModelOptFp4Config

        cap = ModelOptFp4Config.get_min_capability()
        self.assertEqual(cap, 75, f"Expected 75, got {cap}")


if __name__ == "__main__":
    unittest.main(verbosity=3)
