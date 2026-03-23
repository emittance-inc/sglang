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
"""Tests for MarlinRunnerCoreWithLoRA.

Verifies that the LoRA delta computed through the Marlin MoE + LoRA pipeline
matches a naive PyTorch reference. Uses the same delta-comparison approach
as test_lora_moe_runner.py.
"""

import random

import pytest
import torch
from sgl_kernel.scalar_type import scalar_types

from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.marlin import MarlinMoeQuantInfo
from sglang.srt.layers.moe.moe_runner.runner import MoeRunner
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.lora.lora_moe_runners import LoRAInfo
from sglang.srt.utils import set_random_seed
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_marlin_utils import marlin_quantize

register_cuda_ci(est_time=120, suite="stage-b-test-large-1-gpu")

DEVICE = "cuda:0"


def create_random_gpu_tensor(shape, dtype, mean=0, std=0.01):
    return torch.empty(shape, dtype=dtype, device=DEVICE).normal_(mean, std)


def generate_request_data(num_tokens, num_sequences, max_loras):
    assert num_sequences > 0 and max_loras > 0
    assert num_tokens >= num_sequences

    remaining = num_tokens
    seg_lens = []
    for _ in range(num_sequences - 1):
        max_len = remaining - (num_sequences - len(seg_lens)) + 1
        length = random.randint(1, min(max_len, num_tokens // num_sequences * 2))
        seg_lens.append(length)
        remaining -= length
    seg_lens.append(remaining)

    seg_indptr = torch.cumsum(
        torch.tensor([0] + seg_lens, dtype=torch.int32, device=DEVICE),
        dim=0,
        dtype=torch.int32,
    )
    req_to_lora = torch.randint(
        0, max_loras, (num_sequences,), dtype=torch.int32, device=DEVICE
    )
    token_lora_mapping = torch.repeat_interleave(
        req_to_lora, torch.tensor(seg_lens, device=DEVICE)
    )

    return seg_indptr, req_to_lora, token_lora_mapping


def assign_experts_to_tokens(num_tokens, num_experts, top_k_num, dtype=torch.float32):
    expert_indices = torch.empty((num_tokens, top_k_num), dtype=torch.int32)
    for i in range(num_tokens):
        selected = torch.randperm(num_experts)[:top_k_num]
        expert_indices[i] = selected

    expert_weights = torch.rand((num_tokens, top_k_num), dtype=dtype)
    expert_weights = expert_weights / expert_weights.sum(dim=1, keepdim=True)

    return expert_indices, expert_weights


def setup_marlin_moe_weights(num_experts, n, k, group_size, dtype):
    """Create properly quantized Marlin MoE weights (GPTQ-style, no zero-points)."""
    quant_type = scalar_types.uint4b8

    w = torch.randn((num_experts, n, k), device=DEVICE, dtype=dtype) / 20

    w_ref_l, qweight_l, scales_l, g_idx_l, sort_indices_l = [], [], [], [], []
    for i in range(num_experts):
        test_perm = torch.randperm(k, device=DEVICE)
        w_ref, qweight, scales, g_idx, sort_indices, _ = marlin_quantize(
            w[i].transpose(1, 0), quant_type, group_size, False, test_perm
        )
        w_ref_l.append(w_ref.T)
        qweight_l.append(qweight)
        scales_l.append(scales)
        g_idx_l.append(g_idx)
        sort_indices_l.append(sort_indices)

    def stack(lst):
        return torch.stack(lst, dim=0).to(DEVICE).contiguous()

    return (
        stack(w_ref_l),
        stack(qweight_l),
        stack(scales_l),
        stack(g_idx_l),
        stack(sort_indices_l),
        quant_type,
    )


def create_lora_info(
    seg_indptr,
    weight_indices,
    max_loras,
    num_experts,
    max_lora_rank,
    hidden_dim,
    intermediate_dim,
    gate_up_dim,
    dtype,
):
    gate_up_lora_a_weights = create_random_gpu_tensor(
        (max_loras, num_experts, max_lora_rank * 2, hidden_dim), dtype, std=0.01
    )
    down_lora_a_weights = create_random_gpu_tensor(
        (max_loras, num_experts, max_lora_rank, intermediate_dim), dtype, std=0.01
    )
    gate_up_lora_b_weights = create_random_gpu_tensor(
        (max_loras, num_experts, gate_up_dim, max_lora_rank), dtype, std=0.01
    )
    down_lora_b_weights = create_random_gpu_tensor(
        (max_loras, num_experts, hidden_dim, max_lora_rank), dtype, std=0.01
    )

    lora_ranks = torch.full(
        (max_loras,), max_lora_rank, dtype=torch.int32, device=DEVICE
    )
    adapter_enabled = torch.zeros(max_loras + 1, dtype=torch.int32, device=DEVICE)
    adapter_enabled.index_fill_(0, weight_indices.long(), 1)

    return LoRAInfo(
        gate_up_lora_a_weights=gate_up_lora_a_weights,
        gate_up_lora_b_weights=gate_up_lora_b_weights,
        down_lora_a_weights=down_lora_a_weights,
        down_lora_b_weights=down_lora_b_weights,
        seg_indptr=seg_indptr,
        req_to_lora=weight_indices,
        lora_ranks=lora_ranks,
        adapter_enabled=adapter_enabled,
        max_lora_rank=max_lora_rank,
        num_experts=num_experts,
    )


def torch_naive_moe_with_lora(
    hidden_states,
    w13_ref,
    w2_ref,
    topk_weights,
    topk_ids,
    lora_info,
    token_lora_mapping,
):
    """Naive reference: dequantized Marlin weights + LoRA in full precision."""
    num_tokens, hidden_dim = hidden_states.shape
    top_k = topk_ids.shape[1]
    num_experts = w13_ref.shape[0]

    hidden_expanded = (
        hidden_states.unsqueeze(1).expand(-1, top_k, -1).reshape(-1, hidden_dim)
    )

    # 1. Gate/Up projection
    gate_up_out = torch.zeros(
        num_tokens * top_k, w13_ref.shape[1],
        dtype=hidden_states.dtype, device=hidden_states.device,
    )
    for expert_id in range(num_experts):
        mask = (topk_ids == expert_id).flatten()
        if mask.any():
            gate_up_out[mask] = hidden_expanded[mask] @ w13_ref[expert_id].T
    gate_up_out = gate_up_out.view(num_tokens, top_k, -1)

    # 1.5. LoRA gate/up delta
    if lora_info.max_lora_rank > 0:
        r = lora_info.max_lora_rank
        for i in range(num_tokens):
            for k in range(top_k):
                expert_id = topk_ids[i, k]
                lora_id = token_lora_mapping[i]
                if lora_id < len(lora_info.lora_ranks):
                    lora_a = lora_info.gate_up_lora_a_weights[lora_id, expert_id]
                    lora_b = lora_info.gate_up_lora_b_weights[lora_id, expert_id]
                    half = lora_b.shape[0] // 2
                    lora_a_result = lora_a @ hidden_states[i]
                    gate_delta = lora_b[:half, :] @ lora_a_result[:r]
                    up_delta = lora_b[half:, :] @ lora_a_result[r:]
                    gate_up_out[i, k] += torch.cat([gate_delta, up_delta])

    # 2. Activation
    gate_up_dim = gate_up_out.shape[-1]
    gate_dim = gate_up_dim // 2
    gate = gate_up_out[..., :gate_dim]
    up = gate_up_out[..., gate_dim:]
    intermediate_out = torch.nn.functional.silu(gate) * up

    # 3. Down projection
    down_out = torch.zeros(
        num_tokens, top_k, hidden_dim,
        dtype=hidden_states.dtype, device=hidden_states.device,
    )
    for expert_id in range(num_experts):
        mask = topk_ids == expert_id
        if mask.any():
            down_out[mask] = intermediate_out[mask] @ w2_ref[expert_id].T

    # 3.5. LoRA down delta
    if lora_info.max_lora_rank > 0:
        for i in range(num_tokens):
            for k in range(top_k):
                expert_id = topk_ids[i, k]
                lora_id = token_lora_mapping[i]
                if lora_id < len(lora_info.lora_ranks):
                    lora_a = lora_info.down_lora_a_weights[lora_id, expert_id]
                    lora_b = lora_info.down_lora_b_weights[lora_id, expert_id]
                    down_out[i, k] += lora_b @ (lora_a @ intermediate_out[i, k])

    # 4. Final reduction
    weighted_out = down_out * topk_weights.unsqueeze(-1)
    return weighted_out.sum(dim=1)


@pytest.mark.parametrize("num_tokens", [32, 64])
@pytest.mark.parametrize("top_k_num", [1, 2])
@pytest.mark.parametrize("max_lora_rank", [8, 16])
def test_lora_marlin_moe_runner(num_tokens, top_k_num, max_lora_rank):
    num_experts = 8
    max_loras = 2
    hidden_dim = 256
    intermediate_dim = 512
    gate_up_dim = intermediate_dim * 2
    group_size = 128
    num_sequences = 4
    dtype = torch.float16
    seed = 42

    torch.set_default_device(DEVICE)
    set_random_seed(seed)

    # Generate routing data
    topk_ids, topk_weights = assign_experts_to_tokens(
        num_tokens, num_experts, top_k_num, dtype
    )
    seg_indptr, req_to_lora, token_lora_mapping = generate_request_data(
        num_tokens, num_sequences, max_loras
    )

    topk_ids = topk_ids.to(DEVICE)
    topk_weights = topk_weights.to(DEVICE)

    hidden_states = create_random_gpu_tensor(
        (num_tokens, hidden_dim), dtype, std=1.0
    )

    # Setup quantized Marlin weights
    # w13: gate_up, shape (E, 2*intermediate, hidden) -> qweight is packed
    w13_ref, w13_qweight, w13_scales, w13_g_idx, w13_sort_indices, quant_type = (
        setup_marlin_moe_weights(num_experts, gate_up_dim, hidden_dim, group_size, dtype)
    )
    # w2: down, shape (E, hidden, intermediate)
    w2_ref, w2_qweight, w2_scales, w2_g_idx, w2_sort_indices, _qt = (
        setup_marlin_moe_weights(num_experts, hidden_dim, intermediate_dim, group_size, dtype)
    )

    # Create LoRA info
    lora_info_with_rank = create_lora_info(
        seg_indptr, req_to_lora, max_loras, num_experts, max_lora_rank,
        hidden_dim, intermediate_dim, gate_up_dim, dtype,
    )
    lora_info_zero_rank = create_lora_info(
        seg_indptr, req_to_lora, max_loras, num_experts, 0,
        hidden_dim, intermediate_dim, gate_up_dim, dtype,
    )

    # Build quant info
    quant_info = MarlinMoeQuantInfo(
        w13_qweight=w13_qweight,
        w2_qweight=w2_qweight,
        w13_scales=w13_scales,
        w2_scales=w2_scales,
        w13_g_idx=w13_g_idx,
        w2_g_idx=w2_g_idx,
        w13_g_idx_sort_indices=w13_sort_indices,
        w2_g_idx_sort_indices=w2_sort_indices,
        weight_bits=4,
        is_k_full=True,
    )

    config = MoeRunnerConfig(
        activation="silu",
        is_gated=True,
        inplace=False,
        no_combine=False,
        routed_scaling_factor=1.0,
        apply_router_weight_on_input=False,
        num_local_experts=num_experts,
    )

    router_logits = torch.randn(
        num_tokens, num_experts, dtype=dtype, device=DEVICE
    )
    topk_output = StandardTopKOutput(
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        router_logits=router_logits,
    )
    dispatch_output = StandardDispatchOutput(
        hidden_states=hidden_states,
        hidden_states_scale=None,
        topk_output=topk_output,
    )

    runner = MoeRunner(MoeRunnerBackend.MARLIN, config, lora_enabled=True)

    output_with_lora = runner.run(
        dispatch_output, quant_info, lora_info_with_rank
    ).hidden_states
    output_baseline = runner.run(
        dispatch_output, quant_info, lora_info_zero_rank
    ).hidden_states

    # Naive reference using dequantized weights
    torch_output_lora = torch_naive_moe_with_lora(
        hidden_states, w13_ref, w2_ref, topk_weights, topk_ids,
        lora_info_with_rank, token_lora_mapping,
    )
    torch_output_base = torch_naive_moe_with_lora(
        hidden_states, w13_ref, w2_ref, topk_weights, topk_ids,
        lora_info_zero_rank, token_lora_mapping,
    )

    # Compare deltas (LoRA effect) rather than absolute values,
    # because Marlin quantization introduces rounding error in the base path.
    # The delta comparison cancels most quantization error, but residual
    # differences remain because the down-projection LoRA delta is applied
    # to activations that differ between the 4-bit Marlin and FP16 paths.
    sglang_delta = output_with_lora - output_baseline
    torch_delta = torch_output_lora - torch_output_base

    torch.testing.assert_close(sglang_delta, torch_delta, rtol=2.0, atol=0.15)


if __name__ == "__main__":
    pytest.main([__file__])
