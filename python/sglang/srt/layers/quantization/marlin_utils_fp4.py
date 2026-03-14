# SPDX-License-Identifier: Apache-2.0
# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/utils/marlin_utils_fp4.py

"""
NVFP4 Marlin fallback utilities for non-FP4 GPUs.

On GPUs without native FP4 hardware (i.e., pre-Blackwell / SM < 100),
NVFP4-quantized weights can still be used efficiently via the Marlin kernel.
The weights remain in compressed FP4 format in GPU memory (no VRAM explosion),
and the Marlin kernel handles FP4 dequantization in-flight using fast bitwise
operations during tensor core matmul.

Key differences from naive BF16 dequantization:
  - Weights stay compressed (FP4 packed in uint8, 2 values per byte)
  - Only the tiling layout is changed (gptq_marlin_repack)
  - Scales are converted to a special FP8-S0E5M3 format for faster dequant
  - Global scale is pre-adjusted with exponent bias
  - Zero additional VRAM overhead from decompression
"""

import logging
from typing import Optional

import torch

from sglang.srt.layers.quantization.marlin_utils import (
    USE_FP32_REDUCE_DEFAULT,
    marlin_make_workspace,
    marlin_permute_bias,
    marlin_permute_scales,
    should_use_atomic_add_reduce,
)
from sglang.srt.layers.quantization.utils import get_scalar_types
from sglang.srt.utils import get_bool_env_var, get_device_capability, is_cuda

_NVFP4_MARLIN_DEBUG = get_bool_env_var("SGLANG_NVFP4_MARLIN_DEBUG")
_nvfp4_marlin_prepare_count = 0
_nvfp4_marlin_apply_count = 0

_is_cuda = is_cuda()
if _is_cuda:
    from sglang.jit_kernel.gptq_marlin import gptq_marlin_gemm
    from sglang.jit_kernel.gptq_marlin_repack import gptq_marlin_repack

ScalarType, scalar_types = get_scalar_types()

logger = logging.getLogger(__name__)

# NVFP4 always uses group_size=16
FP4_MARLIN_GROUP_SIZE = 16


def is_fp4_marlin_supported() -> bool:
    """Check if the current GPU supports FP4 Marlin fallback (requires SM >= 75).

    Marlin kernel is CUDA-only; ROCm/HIP is not supported.
    """
    if not _is_cuda:
        return False
    if torch.version.hip is not None:
        return False
    major, minor = get_device_capability()
    if major is None or minor is None:
        return False
    return (major * 10 + minor) >= 75


def nvfp4_marlin_process_scales(marlin_scales: torch.Tensor) -> torch.Tensor:
    """Convert NVFP4 scales from FP8-S1E4M3 to special FP8-S0E5M3 format.

    This transformation allows the Marlin kernel to perform faster dequantization
    by bringing the exponent bias closer to zero (NVFP4 guarantees non-negative scales).

    The intermediate (scale * 128).view(int16) << 1 may wrap around for large scales
    (e.g. 448 * 128 = 57344, which as int16 after <<1 wraps to negative), but the BIT
    PATTERN is preserved correctly. The kernel reads the raw bytes and does its own bit
    manipulation, so the int16 sign interpretation is irrelevant.
    """
    if _NVFP4_MARLIN_DEBUG and _nvfp4_marlin_prepare_count <= 3:
        _m = marlin_scales.to(torch.float32) if marlin_scales.numel() > 0 else None
        logger.info(
            "[NVFP4_MARLIN_DEBUG] nvfp4_marlin_process_scales IN (layer#%d): shape=%s dtype=%s "
            "min=%s max=%s mean=%s",
            _nvfp4_marlin_prepare_count,
            tuple(marlin_scales.shape),
            marlin_scales.dtype,
            _m.min().item() if _m is not None else "N/A",
            _m.max().item() if _m is not None else "N/A",
            _m.mean().item() if _m is not None else "N/A",
        )
    marlin_scales = marlin_scales.to(torch.half)

    if not (marlin_scales >= 0).all():
        logger.warning_once(
            "NVFP4 Marlin assumes scales >= 0, but encountered negative scales. "
            "Accuracy may be degraded. The scales are converted from FP8-S1E4M3 "
            "to a special FP8-S0E5M3 format to speed up dequantization."
        )

    marlin_scales = marlin_scales.view(-1, 4)[:, [0, 2, 1, 3]].view(
        marlin_scales.size(0), -1
    )

    marlin_scales = (marlin_scales * (2**7)).view(torch.int16) << 1
    marlin_scales = marlin_scales.view(torch.float8_e4m3fn)
    marlin_scales = marlin_scales[:, 1::2].contiguous()

    if _NVFP4_MARLIN_DEBUG and _nvfp4_marlin_prepare_count <= 3:
        _m = marlin_scales.to(torch.float32) if marlin_scales.numel() > 0 else None
        logger.info(
            "[NVFP4_MARLIN_DEBUG] nvfp4_marlin_process_scales OUT (layer#%d): shape=%s dtype=%s "
            "min=%s max=%s",
            _nvfp4_marlin_prepare_count,
            tuple(marlin_scales.shape),
            marlin_scales.dtype,
            _m.min().item() if _m is not None else "N/A",
            _m.max().item() if _m is not None else "N/A",
        )
    return marlin_scales


def nvfp4_marlin_process_global_scale(global_scale: torch.Tensor) -> torch.Tensor:
    """Pre-adjust NVFP4 global scale with exponent bias for the Marlin kernel.

    FP4 (E2M1) and FP16/BF16 have different exponent ranges. Pre-multiplying
    the global scale avoids repeated exponent bias computation during inference.

    The exponent bias depends on the model dtype:
      - FP16 (5 exp bits): bias = 14, multiplier = 2^7
      - BF16 (8 exp bits): bias = 126, multiplier = 2^119
    BF16 has sufficient range (max ~3.4e38) to hold the result without overflow.
    """
    assert global_scale.dtype in [
        torch.half,
        torch.bfloat16,
    ], f"global_scale dtype must be half or bfloat16, got {global_scale.dtype}"
    if _NVFP4_MARLIN_DEBUG and _nvfp4_marlin_prepare_count <= 3:
        _val = global_scale.float().item() if global_scale.numel() == 1 else global_scale.float().tolist()
        logger.info(
            "[NVFP4_MARLIN_DEBUG] nvfp4_marlin_process_global_scale IN (layer#%d): shape=%s "
            "dtype=%s value=%s",
            _nvfp4_marlin_prepare_count,
            tuple(global_scale.shape),
            global_scale.dtype,
            _val,
        )
    fp4_exponent = 2
    if global_scale.dtype == torch.half:
        target_exponent = 5
    elif global_scale.dtype == torch.bfloat16:
        target_exponent = 8
    exponent_bias = 2 ** (target_exponent - 1) - 2 ** (fp4_exponent - 1)
    result = global_scale * (2.0 ** (exponent_bias - 7))
    if _NVFP4_MARLIN_DEBUG and _nvfp4_marlin_prepare_count <= 3:
        _val = result.float().item() if result.numel() == 1 else result.float().tolist()
        logger.info(
            "[NVFP4_MARLIN_DEBUG] nvfp4_marlin_process_global_scale OUT (layer#%d): value=%s",
            _nvfp4_marlin_prepare_count,
            _val,
        )
    return result


def apply_fp4_marlin_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_global_scale: Optional[torch.Tensor],
    workspace: torch.Tensor,
    size_n: int,
    size_k: int,
    bias: Optional[torch.Tensor] = None,
    use_fp32_reduce: bool = USE_FP32_REDUCE_DEFAULT,
) -> torch.Tensor:
    """Apply FP4-quantized linear using the Marlin kernel (non-FP4 GPU fallback).

    Weights stay compressed in FP4 format. The Marlin kernel handles dequantization
    in-flight via bitwise operations during tensor core operations.

    Args:
        input: Activation tensor [M, K] in FP16/BF16.
        weight: Marlin-repacked FP4 weight tensor.
        weight_scale: Processed per-group FP8 scale tensor.
        weight_global_scale: Pre-adjusted global FP32 scale tensor.
        workspace: Marlin workspace tensor.
        size_n: Output dimension (N).
        size_k: Input dimension (K).
        bias: Optional bias tensor.
        use_fp32_reduce: Whether to use FP32 reduction.

    Returns:
        Output tensor [M, N] in same dtype as input.
    """
    reshaped_x = input.reshape(-1, input.shape[-1])
    out_shape = input.shape[:-1] + (size_n,)

    global _nvfp4_marlin_apply_count
    _nvfp4_marlin_apply_count += 1
    if _NVFP4_MARLIN_DEBUG and _nvfp4_marlin_apply_count <= 5:
        logger.info(
            "[NVFP4_MARLIN_DEBUG] apply_fp4_marlin_linear call#%d: input shape=%s "
            "weight shape=%s weight_scale shape=%s global_scale=%s "
            "size_n=%d size_k=%d",
            _nvfp4_marlin_apply_count,
            tuple(reshaped_x.shape),
            tuple(weight.shape),
            tuple(weight_scale.shape),
            weight_global_scale.float().item() if weight_global_scale is not None and weight_global_scale.numel() == 1 else (weight_global_scale.float().tolist() if weight_global_scale is not None else None),
            size_n,
            size_k,
        )

    use_atomic_add = should_use_atomic_add_reduce(
        m=reshaped_x.size(0),
        n=size_n,
        k=size_k,
        device=input.device,
        dtype=input.dtype,
    )

    output = gptq_marlin_gemm(
        a=reshaped_x,
        c=None,
        b_q_weight=weight,
        b_scales=weight_scale,
        global_scale=weight_global_scale.reshape(-1),
        b_zeros=None,
        g_idx=None,
        perm=None,
        workspace=workspace,
        b_q_type=scalar_types.float4_e2m1f,
        size_m=reshaped_x.size(0),
        size_n=size_n,
        size_k=size_k,
        use_atomic_add=use_atomic_add,
        use_fp32_reduce=use_fp32_reduce,
    )

    if bias is not None:
        output.add_(bias)

    if _NVFP4_MARLIN_DEBUG and _nvfp4_marlin_apply_count <= 5:
        _o = output.reshape(out_shape)
        logger.info(
            "[NVFP4_MARLIN_DEBUG] apply_fp4_marlin_linear call#%d OUT: shape=%s "
            "min=%s max=%s mean=%s has_nan=%s has_inf=%s",
            _nvfp4_marlin_apply_count,
            tuple(_o.shape),
            _o.float().min().item(),
            _o.float().max().item(),
            _o.float().mean().item(),
            torch.isnan(_o.float()).any().item(),
            torch.isinf(_o.float()).any().item(),
        )
    return output.reshape(out_shape)


def prepare_fp4_layer_for_marlin(
    layer: torch.nn.Module,
    weight_attr: str = "weight",
    weight_scale_attr: str = "weight_scale",
    weight_global_scale_attr: str = "weight_global_scale",
) -> None:
    """Repack NVFP4 linear layer weights into Marlin format for non-FP4 GPU.

    The FP4 weights remain compressed (no VRAM explosion). Only the tile layout
    changes to match Marlin's access pattern. Scales are converted to the special
    FP8-S0E5M3 format required by the Marlin dequantization kernel.

    Args:
        layer: The linear layer to prepare in-place.
        weight_attr: Attribute name of the packed FP4 weight tensor (N, K//2) uint8.
        weight_scale_attr: Attribute name of the per-group FP8 scale (N, K//16) fp8.
        weight_global_scale_attr: Attribute name of the global scale scalar fp16/bf16.
    """
    logger.warning_once(
        "Your GPU does not have native support for FP4 computation but "
        "FP4 quantization is being used. Weight-only FP4 compression will "
        "be used leveraging the Marlin kernel. This may degrade "
        "performance for compute-heavy workloads."
    )

    global _nvfp4_marlin_prepare_count
    _nvfp4_marlin_prepare_count += 1
    _log_this_layer = _NVFP4_MARLIN_DEBUG and _nvfp4_marlin_prepare_count <= 3

    part_size_n = layer.output_size_per_partition
    part_size_k = layer.input_size_per_partition
    param_dtype = layer.params_dtype

    weight = getattr(layer, weight_attr)
    if _log_this_layer:
        _ws = getattr(layer, weight_scale_attr, None)
        _wgs = getattr(layer, weight_global_scale_attr, None)
        logger.info(
            "[NVFP4_MARLIN_DEBUG] prepare_fp4_layer_for_marlin layer#%d: "
            "weight shape=%s dtype=%s, weight_scale shape=%s dtype=%s, "
            "weight_global_scale shape=%s dtype=%s value=%s",
            _nvfp4_marlin_prepare_count,
            tuple(weight.shape),
            weight.dtype,
            tuple(_ws.shape) if _ws is not None else None,
            _ws.dtype if _ws is not None else None,
            tuple(_wgs.shape) if _wgs is not None else None,
            _wgs.dtype if _wgs is not None else None,
            _wgs.float().item() if _wgs is not None and _wgs.numel() == 1 else "N/A",
        )
    assert weight.shape == (part_size_n, part_size_k // 2), (
        f"Expected {weight_attr} shape ({part_size_n}, {part_size_k // 2}), "
        f"got {weight.shape}"
    )

    device = weight.device

    # WORKSPACE
    layer.marlin_workspace = marlin_make_workspace(device)

    # WEIGHT: Repack from NVFP4 native layout to Marlin tile layout.
    # Native: (N, K//2) packed uint8 -> Marlin: (K//16, N*16//pack_factor)
    # Weights stay in FP4 packed format; only tile organization changes.
    perm = torch.empty(0, dtype=torch.int, device=device)
    qweight = weight.data.view(torch.int32).T.contiguous()
    del weight
    marlin_qweight = gptq_marlin_repack(
        b_q_weight=qweight,
        perm=perm,
        size_k=part_size_k,
        size_n=part_size_n,
        num_bits=4,
    )
    del qweight
    setattr(layer, weight_attr, torch.nn.Parameter(marlin_qweight, requires_grad=False))
    if _log_this_layer:
        logger.info(
            "[NVFP4_MARLIN_DEBUG] prepare_fp4_layer_for_marlin layer#%d: "
            "marlin_qweight shape=%s dtype=%s",
            _nvfp4_marlin_prepare_count,
            tuple(marlin_qweight.shape),
            marlin_qweight.dtype,
        )

    # WEIGHT SCALES: Transpose, permute, convert to FP8-S0E5M3 format.
    # Match vLLM: convert to param_dtype before permute (nvfp4_marlin_process_scales
    # converts to half first for the comparison, so FP8->param_dtype here is fine).
    weight_scale = getattr(layer, weight_scale_attr)
    weight_scale = weight_scale.data.T.contiguous().to(param_dtype)
    weight_scale = marlin_permute_scales(
        s=weight_scale,
        size_k=part_size_k,
        size_n=part_size_n,
        group_size=FP4_MARLIN_GROUP_SIZE,
    )
    weight_scale = nvfp4_marlin_process_scales(weight_scale)
    setattr(
        layer, weight_scale_attr, torch.nn.Parameter(weight_scale, requires_grad=False)
    )
    if _log_this_layer:
        logger.info(
            "[NVFP4_MARLIN_DEBUG] prepare_fp4_layer_for_marlin layer#%d: "
            "weight_scale after process shape=%s",
            _nvfp4_marlin_prepare_count,
            tuple(weight_scale.shape),
        )

    # GLOBAL SCALE: Pre-adjust exponent bias for Marlin kernel.
    weight_global_scale = getattr(layer, weight_global_scale_attr)
    weight_global_scale = weight_global_scale.to(param_dtype)
    weight_global_scale = nvfp4_marlin_process_global_scale(weight_global_scale)
    setattr(
        layer,
        weight_global_scale_attr,
        torch.nn.Parameter(weight_global_scale, requires_grad=False),
    )
    if _log_this_layer:
        logger.info(
            "[NVFP4_MARLIN_DEBUG] prepare_fp4_layer_for_marlin layer#%d DONE: "
            "weight_global_scale shape=%s value=%s",
            _nvfp4_marlin_prepare_count,
            tuple(weight_global_scale.shape),
            weight_global_scale.float().item() if weight_global_scale.numel() == 1 else weight_global_scale.float().tolist(),
        )

    # BIAS (if present): Permute for Marlin's fast access pattern
    if hasattr(layer, "bias") and layer.bias is not None:
        assert layer.bias.shape == (part_size_n,)
        bias = marlin_permute_bias(layer.bias)
        layer.bias = torch.nn.Parameter(bias, requires_grad=False)


def prepare_moe_fp4_layer_for_marlin(layer: torch.nn.Module) -> None:
    """Repack NVFP4 MoE weights into Marlin format for non-FP4 GPU fallback.

    Each expert's weights are repacked individually. FP4 compression is preserved
    in memory; only the tile layout changes for the Marlin kernel's access pattern.

    Expects layer attributes:
        w13_weight: [E, 2*intermediate, hidden//2] uint8 (gate+up proj, packed FP4)
        w2_weight:  [E, hidden, intermediate//2] uint8 (down proj, packed FP4)
        w13_weight_scale: [E, 2*intermediate, hidden//16] float8_e4m3fn
        w2_weight_scale:  [E, hidden, intermediate//16] float8_e4m3fn
        w13_weight_scale_2: [E] or [E, 2] float32 (per-expert global scale)
        w2_weight_scale_2:  [E] float32 (per-expert global scale)
        layer.intermediate_size_per_partition: int
        layer.params_dtype: torch.dtype
        layer.moe_runner_config.is_gated: bool
    """
    logger.warning_once(
        "Your GPU does not have native support for FP4 computation but "
        "FP4 quantization is being used. Weight-only FP4 compression will "
        "be used leveraging the Marlin kernel for MoE layers. This may "
        "degrade performance for compute-heavy workloads."
    )

    e = layer.num_local_experts
    # hidden_size = last dim of w13_weight * 2 (packed: K//2 per uint8)
    k = layer.w13_weight.shape[2] * 2
    n = layer.intermediate_size_per_partition
    param_dtype = layer.params_dtype
    is_gated = layer.moe_runner_config.is_gated
    num_shards = 2 if is_gated else 1

    device = layer.w13_weight.device
    perm = torch.empty(0, dtype=torch.int, device=device)

    # --- WEIGHT REPACKING ---
    for name in ["w13_weight", "w2_weight"]:
        weight = getattr(layer, name)
        if "w13" in name:
            size_n = n * num_shards
            size_k = k
        else:
            size_n = k
            size_k = n

        assert weight.shape == (e, size_n, size_k // 2), (
            f"Expected {name} shape ({e}, {size_n}, {size_k // 2}), "
            f"got {weight.shape}"
        )

        tensor_list = []
        for i in range(e):
            # (size_n, size_k//2) uint8 -> view as int32 -> transpose -> repack
            qweight = weight.data[i].view(torch.int32).T.contiguous()
            marlin_qweight = gptq_marlin_repack(
                b_q_weight=qweight,
                perm=perm,
                size_k=size_k,
                size_n=size_n,
                num_bits=4,
            )
            tensor_list.append(marlin_qweight)

        del weight
        packed = torch.cat([x.unsqueeze(0) for x in tensor_list], 0)
        del tensor_list
        setattr(layer, name, torch.nn.Parameter(packed, requires_grad=False))

    # --- WEIGHT SCALE PROCESSING ---
    for name in ["w13", "w2"]:
        scales = getattr(layer, name + "_weight_scale")
        global_scale = getattr(layer, name + "_weight_scale_2")

        scales = scales.to(param_dtype)
        global_scale = global_scale.to(param_dtype)

        if "w13" in name:
            size_n = n * num_shards
            size_k = k
        else:
            size_n = k
            size_k = n

        tensor_list = []
        for i in range(e):
            scale = scales.data[i].T
            marlin_scales = marlin_permute_scales(
                s=scale,
                size_k=size_k,
                size_n=size_n,
                group_size=FP4_MARLIN_GROUP_SIZE,
            )
            marlin_scales = nvfp4_marlin_process_scales(marlin_scales)
            tensor_list.append(marlin_scales)

        del scales
        processed_scales = torch.cat([x.unsqueeze(0) for x in tensor_list], 0)
        del tensor_list
        setattr(
            layer,
            name + "_weight_scale",
            torch.nn.Parameter(processed_scales, requires_grad=False),
        )

        if global_scale.dim() > 1:
            global_scale = global_scale.max(dim=-1).values

        global_scale = nvfp4_marlin_process_global_scale(global_scale)
        setattr(
            layer,
            name + "_weight_scale_2",
            torch.nn.Parameter(global_scale, requires_grad=False),
        )
