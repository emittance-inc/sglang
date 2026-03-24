"""Core NVFP4 (E2M1) quantization primitives.

Shared by:
  - tools/convert_hf_to_nvfp4.py  (offline checkpoint conversion)
  - quantizer_nfvp4.py            (Megatron weight-sync quantization)

The quantization follows the NVFP4 reference in TransformerEngine with 1D
block scaling (group_size=16).  Weights belonging to the same SGLang fused
layer (Q/K/V  or  gate/up) must share a single global amax so that the
global decode scale (weight_scale_2) is identical across all partitions.
"""

from __future__ import annotations

import torch

FP4_E2M1_MAX = 6.0
FP8_E4M3_MAX = 448.0
NVFP4_GROUP_SIZE = 16

GATED_PAIR_SUFFIXES: dict[str, str] = {
    ".gate_proj.weight": "gate",
    ".up_proj.weight": "up",
    ".w1.weight": "gate",
    ".w3.weight": "up",
}

FUSED_QKV_WEIGHT_SUFFIXES: dict[str, str] = {
    ".q_proj.weight": "q",
    ".k_proj.weight": "k",
    ".v_proj.weight": "v",
}


def split_gated_pair_name(name: str) -> tuple[str | None, str | None]:
    """Return (base, role) if *name* belongs to a gate/up fused pair."""
    for suffix, role in GATED_PAIR_SUFFIXES.items():
        if name.endswith(suffix):
            return name[: -len(suffix)], role
    return None, None


def split_qkv_name(name: str) -> tuple[str | None, str | None]:
    """Return (base, role) if *name* belongs to a Q/K/V fused group."""
    for suffix, role in FUSED_QKV_WEIGHT_SUFFIXES.items():
        if name.endswith(suffix):
            return name[: -len(suffix)], role
    return None, None


def lookup_shared_amax(
    name: str,
    shared_global_amax: dict[str, torch.Tensor],
) -> torch.Tensor | None:
    """Look up the shared global amax for *name*, trying gate/up then QKV."""
    base, _ = split_gated_pair_name(name)
    if base is not None:
        amax = shared_global_amax.get(base)
        if amax is not None:
            return amax
    qkv_base, _ = split_qkv_name(name)
    if qkv_base is not None:
        return shared_global_amax.get(qkv_base)
    return None


# ── FP4 packing ──────────────────────────────────────────────────────────


def cast_to_fp4x2(x: torch.Tensor) -> torch.Tensor:
    """Quantize a tensor to FP4 E2M1 and pack two values per byte."""
    result = torch.zeros_like(x, dtype=torch.uint8)
    result[(x >= 0.0) & (x <= 0.25)] = 0
    result[(x > 0.25) & (x < 0.75)] = 1
    result[(x >= 0.75) & (x <= 1.25)] = 2
    result[(x > 1.25) & (x < 1.75)] = 3
    result[(x >= 1.75) & (x <= 2.5)] = 4
    result[(x > 2.5) & (x < 3.5)] = 5
    result[(x >= 3.5) & (x <= 5.0)] = 6
    result[x > 5.0] = 7

    result[(x >= -0.25) & (x < -0.0)] = 8
    result[(x < -0.25) & (x > -0.75)] = 9
    result[(x <= -0.75) & (x >= -1.25)] = 10
    result[(x < -1.25) & (x > -1.75)] = 11
    result[(x <= -1.75) & (x >= -2.5)] = 12
    result[(x < -2.5) & (x > -3.5)] = 13
    result[(x <= -3.5) & (x >= -5.0)] = 14
    result[x < -5.0] = 15

    return result[:, ::2] + result[:, 1::2] * 16


# ── Quantization ─────────────────────────────────────────────────────────


def _quantize_nvfp4_1d(
    weight: torch.Tensor,
    global_amax: torch.Tensor | None = None,
    group_size: int = NVFP4_GROUP_SIZE,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """NVFP4 1D quantization (tile shape = 1×16).

    Returns
    -------
    qweight      : uint8 packed fp4, shape (M, K // 2)
    block_scale  : float8_e4m3fn,     shape (M, K // group_size)
    global_scale : float32 scalar     (decode scale = 1 / encode_scale)
    """
    weight = weight.contiguous()
    m, n = weight.shape
    if n % group_size != 0:
        raise ValueError(f"NVFP4 requires K divisible by {group_size}, got {n}.")

    weight_f = weight.to(torch.float32)
    if global_amax is None:
        global_amax = torch.max(torch.abs(weight_f))
    else:
        global_amax = global_amax.to(device=weight.device, dtype=torch.float32)

    if global_amax.item() == 0.0:
        return (
            torch.zeros((m, n // 2), dtype=torch.uint8, device=weight.device),
            torch.zeros((m, n // group_size), dtype=torch.float8_e4m3fn, device=weight.device),
            torch.tensor(1.0, device=weight.device, dtype=torch.float32),
        )

    fp4_max = torch.tensor(FP4_E2M1_MAX, device=weight.device, dtype=torch.float32)
    fp8_max = torch.tensor(FP8_E4M3_MAX, device=weight.device, dtype=torch.float32)

    global_encode_scale = torch.div(fp8_max * fp4_max, global_amax)
    global_encode_scale = torch.min(
        global_encode_scale,
        torch.tensor(torch.finfo(torch.float32).max, device=weight.device, dtype=torch.float32),
    )
    if global_encode_scale.item() == 0.0:
        global_encode_scale = torch.tensor(1.0, device=weight.device, dtype=torch.float32)
    global_decode_scale = torch.div(1.0, global_encode_scale)

    weight_blocks = weight_f.view(m, n // group_size, group_size)
    vec_max = torch.amax(torch.abs(weight_blocks), dim=-1, keepdim=True)
    decode_scale = torch.div(vec_max, fp4_max) * global_encode_scale
    decode_scale = torch.clamp(decode_scale, min=-fp8_max, max=fp8_max).to(torch.float8_e4m3fn)

    encode_scale = torch.div(1.0, decode_scale.to(torch.float32) * global_decode_scale)
    scaled = weight_blocks * encode_scale
    clipped = torch.clamp(scaled, -fp4_max, fp4_max).reshape(m, n)

    qweight = cast_to_fp4x2(clipped)
    block_scale = decode_scale.squeeze(-1)
    return qweight, block_scale, global_decode_scale


def quantize_nvfp4(
    weight: torch.Tensor,
    global_amax: torch.Tensor | None = None,
    group_size: int = NVFP4_GROUP_SIZE,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize a 2-D or 3-D weight tensor to NVFP4.

    For 3-D tensors (MoE experts stacked on dim-0), each expert is quantized
    independently.  ``global_amax`` override is only supported for 2-D.
    """
    if weight.dim() == 2:
        return _quantize_nvfp4_1d(weight, global_amax=global_amax, group_size=group_size)
    if weight.dim() == 3:
        if global_amax is not None:
            raise ValueError("global_amax override is only supported for 2D weights.")
        qweights, block_scales, global_scales = [], [], []
        for idx in range(weight.shape[0]):
            qw, bs, gs = _quantize_nvfp4_1d(weight[idx], group_size=group_size)
            qweights.append(qw)
            block_scales.append(bs)
            global_scales.append(gs)
        return (
            torch.stack(qweights, dim=0),
            torch.stack(block_scales, dim=0),
            torch.stack(global_scales, dim=0),
        )
    raise ValueError(f"Unsupported weight rank {weight.dim()} for NVFP4 quantization.")
