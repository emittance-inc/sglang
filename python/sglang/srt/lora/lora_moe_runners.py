# Copyright 2023-2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""LoRA-aware MoE runners that integrate LoRA deltas into the MoE computation.

The key insight is that LoRA deltas must be added at specific points:
1. After gate_up projection, BEFORE activation (halfway through)
2. After down projection, BEFORE final reduction (at the end)

This differs from computing LoRA independently and adding at the very end.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import torch
import triton.language as tl

from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig, MoeRunnerCore
from sglang.srt.layers.moe.moe_runner.triton import (
    TritonMoeQuantInfo,
    TritonRunnerCore,
    TritonRunnerInput,
    TritonRunnerOutput,
)
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.utils import cpu_has_amx_support, is_cpu, is_cuda, is_hip, is_xpu

_is_hip = is_hip()
_is_cuda = is_cuda()
_is_cpu_amx_available = cpu_has_amx_support()
_is_cpu = is_cpu()
_use_aiter = bool(int(os.getenv("SGLANG_USE_AITER", "0")))
_is_xpu = is_xpu()
_MOE_PADDING_SIZE = 128 if bool(int(os.getenv("SGLANG_MOE_PADDING", "0"))) else 0


if _is_cuda or _is_hip:
    from sgl_kernel import gelu_and_mul, moe_sum_reduce, silu_and_mul

    if _is_hip:
        from vllm import _custom_ops as vllm_ops  # moe_sum
elif _is_cpu and _is_cpu_amx_available:
    pass
elif _is_xpu:
    from sgl_kernel import silu_and_mul


if _is_cuda or _is_hip or _is_xpu:
    from sgl_kernel import (  # noqa: F401
        moe_align_block_size as sgl_moe_align_block_size,
    )

    from sglang.jit_kernel.moe_lora_align import moe_lora_align_block_size


@dataclass
class LoRAInfo:
    """LoRA weights and dispatch info for MoE computation."""

    # LoRA weights: [num_loras, num_experts, dim1, dim2]
    gate_up_lora_a_weights: (
        torch.Tensor
    )  # [num_loras, num_experts, max_rank, hidden_dim]
    gate_up_lora_b_weights: (
        torch.Tensor
    )  # [num_loras, num_experts, gate_up_dim, max_rank]
    down_lora_a_weights: (
        torch.Tensor
    )  # [num_loras, num_experts, max_rank, intermediate_dim]
    down_lora_b_weights: torch.Tensor  # [num_loras, num_experts, hidden_dim, max_rank]

    # Indice pointers of each segment in shape (num_segments + 1, )
    seg_indptr: torch.Tensor

    # The index of lora adapter used by each segment, in shape (num_segments,)
    req_to_lora: torch.Tensor

    # LoRA config per adapter
    lora_ranks: torch.Tensor  # [num_loras]
    adapter_enabled: torch.Tensor  # [num_loras] - which adapters are enabled
    max_lora_rank: int  # Maximum LoRA rank across all adapters

    num_experts: int

    fully_sharded: bool = False
    tp_size: int = 1
    tp_rank: int = 0
    hidden_size: int = 0


class MoeLoRADeltaMixin:
    """Shared LoRA delta methods for MoE runner cores.

    Provides the LoRA alignment and delta computation logic that is
    independent of the base GEMM backend (Triton, Marlin, etc.).
    """

    def _run_lora_align(
        self,
        topk_ids: torch.Tensor,
        lora_info: LoRAInfo,
    ):
        """Run moe_lora_align_block_size and return alignment tensors."""
        block_size_m = 64
        max_loras = len(lora_info.lora_ranks)

        max_num_tokens_padded = topk_ids.numel() + lora_info.num_experts * (
            block_size_m - 1
        )
        max_num_tokens_padded = (
            (max_num_tokens_padded + block_size_m - 1) // block_size_m
        ) * block_size_m
        max_num_m_blocks = (max_num_tokens_padded + block_size_m - 1) // block_size_m

        device = topk_ids.device
        sorted_token_ids_lora = torch.empty(
            (max_loras * max_num_tokens_padded,),
            dtype=torch.int32,
            device=device,
        )
        expert_ids_lora = torch.empty(
            (max_loras * max_num_m_blocks,),
            dtype=torch.int32,
            device=device,
        )
        num_tokens_post_padded_lora = torch.empty(
            (max_loras,), dtype=torch.int32, device=device
        )

        lora_ids = torch.arange(max_loras, dtype=torch.int32, device=device)

        moe_lora_align_block_size(
            topk_ids,
            lora_info.seg_indptr,
            lora_info.req_to_lora,
            int(lora_info.num_experts),
            int(block_size_m),
            int(max_loras),
            int(max_num_tokens_padded),
            int(max_num_m_blocks),
            sorted_token_ids_lora,
            expert_ids_lora,
            num_tokens_post_padded_lora,
            lora_info.adapter_enabled,
            lora_ids,
            None,  # expert_map
        )

        sorted_token_ids_reshaped = sorted_token_ids_lora.view(max_loras, -1)
        expert_ids_reshaped = expert_ids_lora.view(max_loras, -1)

        return (
            sorted_token_ids_reshaped,
            expert_ids_reshaped,
            num_tokens_post_padded_lora,
            lora_ids,
        )

    def _add_lora_gate_up_delta(
        self,
        hidden_states: torch.Tensor,  # [M, hidden_dim]
        intermediate_cache: torch.Tensor,  # [M, top_k, gate_up_dim]
        topk_weights: torch.Tensor,  # [M, top_k]
        lora_info: LoRAInfo,
        sorted_token_ids_reshaped: torch.Tensor,
        expert_ids_reshaped: torch.Tensor,
        num_tokens_post_padded_lora: torch.Tensor,
        lora_ids: torch.Tensor,
    ) -> None:
        """
        Add LoRA gate_up delta to intermediate_cache in-place.

        For each (token, expert) pair, computes:
            delta = scaling * B @ (A @ hidden_states[token])
        and adds it to intermediate_cache[token, k] where k is the top_k index.
        """
        from sglang.srt.lora.triton_ops import fused_moe_lora

        M, top_k, gate_up_dim = intermediate_cache.shape

        # Skip LoRA computation if no LoRA adapters have non-zero rank
        if lora_info.max_lora_rank == 0:
            return

        r = lora_info.max_lora_rank
        gate_up_a = lora_info.gate_up_lora_a_weights
        gate_up_b = lora_info.gate_up_lora_b_weights
        inter_size = gate_up_b.shape[2] // 2

        # Split packed gate_up weights into separate gate and up slices.
        # gate_up_lora_a has shape [max_loras, num_experts, 2*r, hidden_dim]
        # where the first r rows are gate_lora_a and the next r are up_lora_a.
        # gate_up_lora_b has shape [max_loras, num_experts, 2*inter_size, r]
        # where the first inter_size rows are gate_lora_b and the rest up_lora_b.
        # Using num_slices=2 lets the kernel handle gate and up independently,
        # keeping the rank dimension at r so shrink and expand both match.
        lora_a_stacked = [gate_up_a[:, :, :r, :], gate_up_a[:, :, r : 2 * r, :]]
        lora_b_stacked = [
            gate_up_b[:, :, :inter_size, :],
            gate_up_b[:, :, inter_size:, :],
        ]

        fused_moe_lora(
            output=intermediate_cache,
            qcurr_hidden_states=hidden_states,
            lora_a_stacked=lora_a_stacked,
            lora_b_stacked=lora_b_stacked,
            topk_weights=topk_weights,
            sorted_token_ids=sorted_token_ids_reshaped,
            expert_ids=expert_ids_reshaped,
            num_tokens_post_padded=num_tokens_post_padded_lora,
            max_lora_rank=r,
            top_k_num=top_k,
            lora_ids=lora_ids,
            adapter_enabled=lora_info.adapter_enabled,
            # TODO: Replace hardcoded block sizes with autotuned configs
            shrink_block_size_m=64,
            shrink_block_size_n=64,
            shrink_block_size_k=64,
            shrink_group_size_m=8,
            shrink_num_warps=4,
            shrink_num_stages=2,
            shrink_split_k=1,
            expand_block_size_m=64,
            expand_block_size_n=64,
            expand_block_size_k=64,
            expand_group_size_m=8,
            expand_num_warps=4,
            expand_num_stages=2,
            expand_split_k=1,
            fully_sharded=lora_info.fully_sharded,
        )

    def _add_lora_down_delta(
        self,
        intermediate_input: torch.Tensor,  # [M * top_k, intermediate_dim]
        intermediate_cache: torch.Tensor,  # [M, top_k, hidden_dim]
        topk_weights: torch.Tensor,  # [M, top_k]
        lora_info: LoRAInfo,
        sorted_token_ids_reshaped: torch.Tensor,
        expert_ids_reshaped: torch.Tensor,
        num_tokens_post_padded_lora: torch.Tensor,
        lora_ids: torch.Tensor,
    ) -> None:
        """
        Add LoRA down delta to intermediate_cache in-place.

        For each (token, expert) pair, computes:
            delta = scaling * B @ (A @ intermediate_input[dispatched_idx])
        and adds it to intermediate_cache[token, k].
        """
        from sglang.srt.lora.triton_ops import fused_moe_lora

        M, top_k, hidden_dim = intermediate_cache.shape

        # Skip LoRA computation if no LoRA adapters have non-zero rank
        if lora_info.max_lora_rank == 0:
            return

        lora_a_stacked = [lora_info.down_lora_a_weights]
        lora_b_stacked = [lora_info.down_lora_b_weights]

        if lora_info.fully_sharded and lora_info.tp_size > 1:
            shard_size = lora_info.hidden_size // lora_info.tp_size
            offset = shard_size * lora_info.tp_rank
        else:
            offset = 0

        fused_moe_lora(
            output=intermediate_cache,
            qcurr_hidden_states=intermediate_input,
            lora_a_stacked=lora_a_stacked,
            lora_b_stacked=lora_b_stacked,
            topk_weights=topk_weights,
            sorted_token_ids=sorted_token_ids_reshaped,
            expert_ids=expert_ids_reshaped,
            num_tokens_post_padded=num_tokens_post_padded_lora,
            max_lora_rank=lora_info.max_lora_rank,
            top_k_num=top_k,
            lora_ids=lora_ids,
            adapter_enabled=lora_info.adapter_enabled,
            # TODO: Replace hardcoded block sizes with autotuned configs
            shrink_block_size_m=64,
            shrink_block_size_n=64,
            shrink_block_size_k=64,
            shrink_group_size_m=8,
            shrink_num_warps=4,
            shrink_num_stages=2,
            shrink_split_k=1,
            expand_block_size_m=64,
            expand_block_size_n=64,
            expand_block_size_k=64,
            expand_group_size_m=8,
            expand_num_warps=4,
            expand_num_stages=2,
            expand_split_k=1,
            mul_routed_weight=True,
            fully_sharded=lora_info.fully_sharded,
            offset=offset,
        )


class TritonRunnerCoreWithLoRA(MoeLoRADeltaMixin, TritonRunnerCore):
    """LoRA-aware wrapper around TritonRunnerCore.

    Integrates LoRA deltas at the correct points in the MoE forward pass:
    1. Base gate_up projection + LoRA gate_up delta -> activation
    2. Base down projection + LoRA down delta -> final reduction
    """

    def __init__(self, config: MoeRunnerConfig):
        super().__init__(config)

    def run(
        self,
        runner_input: TritonRunnerInput,
        quant_info: TritonMoeQuantInfo,
        running_state: dict,
        lora_info: Optional[LoRAInfo] = None,
    ) -> TritonRunnerOutput:
        hidden_states = runner_input.hidden_states
        topk_weights = runner_input.topk_weights
        topk_ids = runner_input.topk_ids
        sorted_token_ids = runner_input.sorted_token_ids
        expert_ids = runner_input.expert_ids
        num_tokens_post_padded = runner_input.num_tokens_post_padded

        w13 = quant_info.w13_weight
        w2 = quant_info.w2_weight
        b13 = quant_info.b13
        b2 = quant_info.b2
        a13_scale = quant_info.a13_scale
        a2_scale = quant_info.a2_scale
        w13_scale = quant_info.w13_scale
        w2_scale = quant_info.w2_scale
        w13_zp = quant_info.w13_zp
        w2_zp = quant_info.w2_zp
        block_shape = quant_info.block_shape
        per_channel_quant = quant_info.per_channel_quant
        use_fp8_w8a8 = quant_info.use_fp8_w8a8
        use_int8_w8a8 = quant_info.use_int8_w8a8
        use_int8_w8a16 = quant_info.use_int8_w8a16
        use_int4_w4a16 = quant_info.use_int4_w4a16

        activation = self.config.activation
        no_combine = self.config.no_combine
        inplace = self.config.inplace
        gemm1_alpha = self.config.gemm1_alpha
        gemm1_limit = self.config.gemm1_clamp_limit
        routed_scaling_factor = self.config.routed_scaling_factor
        apply_router_weight_on_input = self.config.apply_router_weight_on_input

        assert self.config.is_gated, "Only gated MoEs are supported for Triton runner"

        M = hidden_states.shape[0]
        E, N, _ = w13.shape
        compute_type = (
            tl.bfloat16 if hidden_states.dtype == torch.bfloat16 else tl.float16
        )

        from sglang.srt.layers.moe.fused_moe_triton.fused_moe import (
            _swiglu_gpt_oss_sigmoid_alpha,
            _swiglu_silu_clamp_mul,
            invoke_fused_moe_kernel,
            moe_sum_reduce_torch_compile,
            moe_sum_reduce_triton,
        )

        # Stage 1: Gate/Up projection (base)
        intermediate_cache1 = torch.empty(
            (M, topk_ids.shape[1], N),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

        invoke_fused_moe_kernel(
            hidden_states, w13, b13, intermediate_cache1,
            a13_scale, w13_scale, w13_zp,
            topk_weights, topk_ids,
            sorted_token_ids, expert_ids, num_tokens_post_padded,
            apply_router_weight_on_input,
            topk_ids.shape[1],
            running_state["config"],
            compute_type=compute_type,
            use_fp8_w8a8=use_fp8_w8a8,
            use_int8_w8a8=use_int8_w8a8,
            use_int8_w8a16=use_int8_w8a16,
            use_int4_w4a16=use_int4_w4a16,
            per_channel_quant=per_channel_quant,
            block_shape=block_shape,
        )

        # LoRA alignment
        (
            sorted_token_ids_reshaped,
            expert_ids_reshaped,
            num_tokens_post_padded_lora,
            lora_ids,
        ) = self._run_lora_align(topk_ids, lora_info)

        # Stage 1.5: Add LoRA gate_up delta BEFORE activation
        self._add_lora_gate_up_delta(
            hidden_states=hidden_states,
            intermediate_cache=intermediate_cache1,
            topk_weights=topk_weights,
            lora_info=lora_info,
            sorted_token_ids_reshaped=sorted_token_ids_reshaped,
            expert_ids_reshaped=expert_ids_reshaped,
            num_tokens_post_padded_lora=num_tokens_post_padded_lora,
            lora_ids=lora_ids,
        )

        # Stage 2: Activation (SiLU or GELU)
        intermediate_cache2 = torch.empty(
            (M * topk_ids.shape[1], N // 2),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        if activation == "silu":
            if gemm1_alpha is not None:
                assert gemm1_limit is not None
                intermediate_cache2 = _swiglu_gpt_oss_sigmoid_alpha(
                    intermediate_cache1.view(-1, N), gemm1_alpha, gemm1_limit
                )
            elif gemm1_limit is not None:
                intermediate_cache2 = _swiglu_silu_clamp_mul(
                    intermediate_cache1.view(-1, N), gemm1_limit
                )
            elif _is_cuda or _is_hip or _is_xpu:
                silu_and_mul(intermediate_cache1.view(-1, N), intermediate_cache2)
            else:
                vllm_ops.silu_and_mul(
                    intermediate_cache2, intermediate_cache1.view(-1, N)
                )
        elif activation == "gelu":
            assert gemm1_alpha is None, "gemm1_alpha is not supported for gelu"
            assert gemm1_limit is None, "gemm1_limit is not supported for gelu"
            if _is_cuda or _is_hip:
                gelu_and_mul(intermediate_cache1.view(-1, N), intermediate_cache2)
            else:
                vllm_ops.gelu_and_mul(
                    intermediate_cache2, intermediate_cache1.view(-1, N)
                )
        else:
            raise ValueError(f"Unsupported activation: {activation=}")

        # Stage 3: Down projection (base)
        intermediate_cache3 = torch.empty(
            (M, topk_ids.shape[1], w2.shape[1]),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

        if no_combine:
            assert not inplace
            out_hidden_states = torch.empty(
                (M, topk_ids.shape[1], w2.shape[1]),
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
        elif inplace:
            out_hidden_states = hidden_states
        else:
            out_hidden_states = torch.empty_like(hidden_states)

        invoke_fused_moe_kernel(
            intermediate_cache2, w2, b2, intermediate_cache3,
            a2_scale, w2_scale, w2_zp,
            topk_weights, topk_ids,
            sorted_token_ids, expert_ids, num_tokens_post_padded,
            not apply_router_weight_on_input,
            1,
            running_state["config"],
            compute_type=compute_type,
            use_fp8_w8a8=use_fp8_w8a8,
            use_int8_w8a8=use_int8_w8a8,
            use_int8_w8a16=use_int8_w8a16,
            use_int4_w4a16=use_int4_w4a16,
            per_channel_quant=per_channel_quant,
            block_shape=block_shape,
        )

        # Stage 3.5: Add LoRA down delta BEFORE final reduction
        self._add_lora_down_delta(
            intermediate_input=intermediate_cache2,
            intermediate_cache=intermediate_cache3,
            topk_weights=topk_weights,
            lora_info=lora_info,
            sorted_token_ids_reshaped=sorted_token_ids_reshaped,
            expert_ids_reshaped=expert_ids_reshaped,
            num_tokens_post_padded_lora=num_tokens_post_padded_lora,
            lora_ids=lora_ids,
        )

        # Stage 4: Final reduction (sum across top_k)
        if routed_scaling_factor is None:
            routed_scaling_factor = 1.0

        if no_combine:
            pass
        elif _is_cuda:
            if topk_ids.shape[1] == 1 and routed_scaling_factor == 1.0:
                out_hidden_states[:] = intermediate_cache3.squeeze(1)
            elif topk_ids.shape[1] == 2 and routed_scaling_factor == 1.0:
                torch.add(
                    intermediate_cache3[:, 0],
                    intermediate_cache3[:, 1],
                    out=out_hidden_states,
                ).squeeze(dim=1)
            else:
                if M <= 32:
                    moe_sum_reduce_torch_compile(
                        intermediate_cache3.view(*intermediate_cache3.shape),
                        out_hidden_states,
                        routed_scaling_factor,
                    )
                else:
                    moe_sum_reduce_triton(
                        intermediate_cache3.view(*intermediate_cache3.shape),
                        out_hidden_states,
                        routed_scaling_factor,
                    )
        elif _is_hip:
            from vllm import _custom_ops as vllm_ops

            vllm_ops.moe_sum(
                intermediate_cache3.view(*intermediate_cache3.shape),
                out_hidden_states,
            )
        else:
            from vllm import _custom_ops as vllm_ops

            vllm_ops.moe_sum(
                intermediate_cache3.view(*intermediate_cache3.shape),
                out_hidden_states,
            )

        return TritonRunnerOutput(hidden_states=out_hidden_states)


class MarlinRunnerCoreWithLoRA(MoeLoRADeltaMixin, MoeRunnerCore):
    """LoRA-aware MoE runner for the Marlin (WNA16) backend.

    Decomposes fused_marlin_moe into stages and injects LoRA deltas
    at the same points as TritonRunnerCoreWithLoRA:
    1. After gate_up Marlin GEMM, before activation
    2. After down Marlin GEMM, before final reduction
    """

    def __init__(self, config: MoeRunnerConfig):
        super().__init__(config)

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.MARLIN

    def run(
        self,
        runner_input,
        quant_info,
        running_state: dict,
        lora_info: Optional[LoRAInfo] = None,
    ):
        from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size
        from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import (
            get_scalar_type,
        )
        from sglang.srt.layers.moe.moe_runner.marlin import (
            MarlinMoeQuantInfo,
            MarlinRunnerOutput,
        )
        from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace

        from sglang.jit_kernel.moe_wna16_marlin import moe_wna16_marlin_gemm

        hidden_states = runner_input.hidden_states
        topk_weights = runner_input.topk_weights
        topk_ids = runner_input.topk_ids

        assert isinstance(quant_info, MarlinMoeQuantInfo)

        w1 = quant_info.w13_qweight
        w2 = quant_info.w2_qweight
        w1_scale = quant_info.w13_scales
        w2_scale = quant_info.w2_scales
        g_idx1 = quant_info.w13_g_idx
        g_idx2 = quant_info.w2_g_idx
        sort_indices1 = quant_info.w13_g_idx_sort_indices
        sort_indices2 = quant_info.w2_g_idx_sort_indices
        w1_zeros = quant_info.w13_qzeros
        w2_zeros = quant_info.w2_qzeros
        num_bits = quant_info.weight_bits
        is_k_full = quant_info.is_k_full
        expert_map = quant_info.expert_map

        inplace = self.config.inplace
        routed_scaling_factor = self.config.routed_scaling_factor

        assert self.config.activation == "silu", (
            "Only SiLU activation is supported for Marlin MoE LoRA."
        )

        M, K = hidden_states.shape
        E = w1.shape[0]
        N = w2.shape[1] * 16
        topk = topk_ids.shape[1]

        # Block size M selection (same heuristic as fused_marlin_moe)
        for block_size_m in [8, 16, 32, 48, 64]:
            if M * topk / E / block_size_m < 0.9:
                break

        global_num_experts = E
        sorted_token_ids, base_expert_ids, num_tokens_post_padded = (
            moe_align_block_size(topk_ids, block_size_m, global_num_experts)
        )

        workspace = marlin_make_workspace(hidden_states.device, max_blocks_per_sm=4)

        scalar_type1 = get_scalar_type(num_bits, w1_zeros is not None)
        scalar_type2 = get_scalar_type(num_bits, w2_zeros is not None)

        use_atomic_add = (
            hidden_states.dtype == torch.half
            or torch.cuda.get_device_capability(hidden_states.device)[0] >= 9
        )

        # Stage 1: Gate/Up projection (Marlin GEMM)
        intermediate_cache1 = torch.empty(
            (M * topk, 2 * N),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

        intermediate_cache1 = moe_wna16_marlin_gemm(
            hidden_states,
            intermediate_cache1,
            w1,
            None,  # b_bias
            w1_scale,
            getattr(quant_info, "w13_global_scale", None),
            w1_zeros,
            g_idx1,
            sort_indices1,
            workspace,
            sorted_token_ids,
            base_expert_ids,
            num_tokens_post_padded,
            topk_weights,
            moe_block_size=block_size_m,
            top_k=topk,
            mul_topk_weights=False,
            is_ep=expert_map is not None,
            b_q_type=scalar_type1,
            size_m=M,
            size_n=2 * N,
            size_k=K,
            is_k_full=is_k_full,
            use_atomic_add=use_atomic_add,
            use_fp32_reduce=True,
            is_zp_float=False,
        )

        # Reshape for LoRA: [M, topk, 2*N]
        intermediate_cache1_3d = intermediate_cache1.view(M, topk, 2 * N)

        # LoRA alignment (shared with Triton path)
        (
            sorted_token_ids_reshaped,
            expert_ids_reshaped,
            num_tokens_post_padded_lora,
            lora_ids,
        ) = self._run_lora_align(topk_ids, lora_info)

        # Stage 1.5: Add LoRA gate_up delta BEFORE activation
        self._add_lora_gate_up_delta(
            hidden_states=hidden_states,
            intermediate_cache=intermediate_cache1_3d,
            topk_weights=topk_weights,
            lora_info=lora_info,
            sorted_token_ids_reshaped=sorted_token_ids_reshaped,
            expert_ids_reshaped=expert_ids_reshaped,
            num_tokens_post_padded_lora=num_tokens_post_padded_lora,
            lora_ids=lora_ids,
        )

        # Stage 2: Activation (SiLU)
        intermediate_cache2 = torch.empty(
            (M * topk, N),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        silu_and_mul(intermediate_cache1_3d.view(-1, 2 * N), intermediate_cache2)

        # Stage 3: Down projection (Marlin GEMM)
        intermediate_cache3 = torch.empty(
            (M * topk, K),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

        if expert_map is not None:
            intermediate_cache3.zero_()

        intermediate_cache3 = moe_wna16_marlin_gemm(
            intermediate_cache2,
            intermediate_cache3,
            w2,
            None,  # b_bias
            w2_scale,
            getattr(quant_info, "w2_global_scale", None),
            w2_zeros,
            g_idx2,
            sort_indices2,
            workspace,
            sorted_token_ids,
            base_expert_ids,
            num_tokens_post_padded,
            topk_weights,
            moe_block_size=block_size_m,
            top_k=1,
            mul_topk_weights=True,
            is_ep=expert_map is not None,
            b_q_type=scalar_type2,
            size_m=M * topk,
            size_n=K,
            size_k=N,
            is_k_full=is_k_full,
            use_atomic_add=use_atomic_add,
            use_fp32_reduce=True,
            is_zp_float=False,
        )

        # Reshape for LoRA: [M, topk, K]
        intermediate_cache3_3d = intermediate_cache3.view(M, topk, K)

        # Stage 3.5: Add LoRA down delta BEFORE final reduction
        self._add_lora_down_delta(
            intermediate_input=intermediate_cache2,
            intermediate_cache=intermediate_cache3_3d,
            topk_weights=topk_weights,
            lora_info=lora_info,
            sorted_token_ids_reshaped=sorted_token_ids_reshaped,
            expert_ids_reshaped=expert_ids_reshaped,
            num_tokens_post_padded_lora=num_tokens_post_padded_lora,
            lora_ids=lora_ids,
        )

        # Stage 4: Final reduction
        output = hidden_states if inplace else torch.empty_like(hidden_states)
        if routed_scaling_factor is None:
            routed_scaling_factor = 1.0

        moe_sum_reduce(intermediate_cache3_3d, output, routed_scaling_factor)

        return MarlinRunnerOutput(hidden_states=output)
