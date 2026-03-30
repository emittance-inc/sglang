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

"""LoRA-aware Marlin MoE runner.

Breaks the monolithic fused_marlin_moe into stages and injects LoRA delta
computation (via the fused_moe_lora Triton kernel) between stages:

    Stage 1:   Marlin gate_up GEMM
    Stage 1.5: LoRA gate_up delta  (BEFORE activation)
    Stage 2:   SiLU activation
    Stage 3:   Marlin down GEMM
    Stage 3.5: LoRA down delta     (BEFORE reduction)
    Stage 4:   Sum reduction across top-k experts
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig, MoeRunnerCore
from sglang.srt.layers.moe.moe_runner.marlin import (
    MarlinMoeQuantInfo,
    MarlinRunnerInput,
    MarlinRunnerOutput,
)
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.lora.lora_moe_runners import LoRADeltaMixin, LoRAInfo
from sglang.srt.utils import is_cuda

_is_cuda = is_cuda()

if _is_cuda:
    from sgl_kernel import moe_sum_reduce, silu_and_mul

    from sglang.jit_kernel.moe_wna16_marlin import moe_wna16_marlin_gemm

if TYPE_CHECKING:
    pass

MARLIN_MOE_WORKSPACE: Optional[torch.Tensor] = None


class MarlinRunnerCoreWithLoRA(LoRADeltaMixin, MoeRunnerCore):
    """LoRA-aware Marlin MoE runner.

    Stages the Marlin FP4 MoE pipeline and injects LoRA deltas at the correct
    points, reusing the quantization-agnostic fused_moe_lora Triton kernel.
    """

    def __init__(self, config: MoeRunnerConfig):
        super().__init__(config)

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.MARLIN

    def run(
        self,
        runner_input: MarlinRunnerInput,
        quant_info: MarlinMoeQuantInfo,
        running_state: dict,
        lora_info: Optional[LoRAInfo] = None,
    ) -> MarlinRunnerOutput:
        global MARLIN_MOE_WORKSPACE
        from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size
        from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import (
            _get_fp4_scalar_type,
            get_scalar_type,
        )
        from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace

        # ------------------------------------------------------------------
        # Setup: extract dimensions and prepare alignment / workspace
        # ------------------------------------------------------------------
        hidden_states = runner_input.hidden_states
        topk_weights = runner_input.topk_weights
        topk_ids = runner_input.topk_ids

        w1 = quant_info.w13_qweight
        w2 = quant_info.w2_qweight
        w1_scale = quant_info.w13_scales
        w2_scale = quant_info.w2_scales
        w1_global_scale = quant_info.w13_global_scale
        w2_global_scale = quant_info.w2_global_scale
        num_bits = quant_info.weight_bits

        _is_fp4_marlin = w1_global_scale is not None

        M, K = hidden_states.shape
        E = w1.shape[0]
        N = w2.shape[1] * 16
        topk = topk_ids.shape[1]

        # M block size selection (same heuristic as fused_marlin_moe)
        for block_size_m in [8, 16, 32, 48, 64]:
            if M * topk / E / block_size_m < 0.9:
                break

        global_num_experts = quant_info.global_num_experts
        if global_num_experts == -1:
            global_num_experts = E
        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            topk_ids, block_size_m, global_num_experts
        )

        # Workspace
        if (
            MARLIN_MOE_WORKSPACE is None
            or MARLIN_MOE_WORKSPACE.device != hidden_states.device
        ):
            MARLIN_MOE_WORKSPACE = marlin_make_workspace(
                hidden_states.device, max_blocks_per_sm=4
            )
        workspace = MARLIN_MOE_WORKSPACE

        # Scalar types
        if _is_fp4_marlin:
            scalar_type1 = _get_fp4_scalar_type()
            scalar_type2 = _get_fp4_scalar_type()
        else:
            has_zp1 = quant_info.w13_qzeros is not None
            has_zp2 = quant_info.w2_qzeros is not None
            scalar_type1 = get_scalar_type(num_bits, has_zp1)
            scalar_type2 = get_scalar_type(num_bits, has_zp2)

        use_atomic_add = (
            hidden_states.dtype == torch.half
            or torch.cuda.get_device_capability(hidden_states.device)[0] >= 9
        )

        # ------------------------------------------------------------------
        # Allocate intermediate caches
        # Aliased buffer: cache1 and cache3 share underlying memory (never
        # used simultaneously), matching the fused_marlin_moe optimization.
        # ------------------------------------------------------------------
        intermediate_cache13 = torch.empty(
            (M * topk * max(2 * N, K),),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        intermediate_cache1 = intermediate_cache13[: M * topk * 2 * N].view(-1, 2 * N)
        intermediate_cache3 = intermediate_cache13[: M * topk * K].view(-1, K)

        intermediate_cache2 = torch.empty(
            (M * topk, N),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

        is_ep = quant_info.expert_map is not None

        # ------------------------------------------------------------------
        # Stage 1: Marlin gate_up GEMM (w1)
        # ------------------------------------------------------------------
        intermediate_cache1 = moe_wna16_marlin_gemm(
            hidden_states,
            intermediate_cache1,
            w1,
            None,  # b_bias_or_none
            w1_scale,
            w1_global_scale,
            quant_info.w13_qzeros,
            quant_info.w13_g_idx,
            quant_info.w13_g_idx_sort_indices,
            workspace,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            topk_weights,
            moe_block_size=block_size_m,
            top_k=topk,
            mul_topk_weights=False,
            is_ep=is_ep,
            b_q_type=scalar_type1,
            size_m=M,
            size_n=2 * N,
            size_k=K,
            is_k_full=quant_info.is_k_full,
            use_atomic_add=use_atomic_add,
            use_fp32_reduce=True,
            is_zp_float=False,
        )

        # ------------------------------------------------------------------
        # Stage 1.5: LoRA gate_up delta (BEFORE activation)
        # Reshape 2D → 3D for LoRA kernel, then back (zero-copy views)
        # ------------------------------------------------------------------
        if lora_info is not None:
            alignment = self._prepare_lora_alignment(topk_ids, lora_info)
            intermediate_cache1_3d = intermediate_cache1.view(M, topk, 2 * N)
            self._add_lora_gate_up_delta(
                hidden_states=hidden_states,
                intermediate_cache=intermediate_cache1_3d,
                topk_weights=topk_weights,
                lora_info=lora_info,
                alignment=alignment,
            )

        # ------------------------------------------------------------------
        # Stage 2: Activation (SiLU)
        # ------------------------------------------------------------------
        silu_and_mul(intermediate_cache1.view(-1, 2 * N), intermediate_cache2)

        # ------------------------------------------------------------------
        # Stage 3: Marlin down GEMM (w2)
        # ------------------------------------------------------------------
        if is_ep:
            intermediate_cache3.zero_()

        intermediate_cache3 = moe_wna16_marlin_gemm(
            intermediate_cache2,
            intermediate_cache3,
            w2,
            None,  # b_bias_or_none
            w2_scale,
            w2_global_scale,
            quant_info.w2_qzeros,
            quant_info.w2_g_idx,
            quant_info.w2_g_idx_sort_indices,
            workspace,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            topk_weights,
            moe_block_size=block_size_m,
            top_k=1,
            mul_topk_weights=True,
            is_ep=is_ep,
            b_q_type=scalar_type2,
            size_m=M * topk,
            size_n=K,
            size_k=N,
            is_k_full=quant_info.is_k_full,
            use_atomic_add=use_atomic_add,
            use_fp32_reduce=True,
            is_zp_float=False,
        )

        # ------------------------------------------------------------------
        # Stage 3.5: LoRA down delta (BEFORE reduction)
        # ------------------------------------------------------------------
        intermediate_cache3_3d = intermediate_cache3.view(M, topk, K)
        if lora_info is not None:
            self._add_lora_down_delta(
                intermediate_input=intermediate_cache2,
                intermediate_cache=intermediate_cache3_3d,
                topk_weights=topk_weights,
                lora_info=lora_info,
                alignment=alignment,
            )

        # ------------------------------------------------------------------
        # Stage 4: Reduction (sum across top-k)
        # ------------------------------------------------------------------
        routed_scaling_factor = self.config.routed_scaling_factor
        if routed_scaling_factor is None:
            routed_scaling_factor = 1.0

        inplace = self.config.inplace
        output = hidden_states if inplace else torch.empty_like(hidden_states)

        moe_sum_reduce(
            intermediate_cache3_3d,
            output,
            routed_scaling_factor,
        )

        return MarlinRunnerOutput(hidden_states=output)
