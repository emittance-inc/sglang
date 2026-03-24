# NVFP4 Marlin Fallback + MoE LoRA Integration

Branch: `add-moe-lora-support` (merged with `nvfp4-marlin-fallback` from PR #19652)

## What This Enables

1. **LoRA on MoE layers** — `gate_up_proj` and `down_proj` LoRA adapters are applied *inside* the MoE computation (after each GEMM, before activation/reduction), matching HF/vLLM semantics.
2. **NVFP4 models on non-Blackwell GPUs** — FP4 checkpoints (e.g. `nvidia/Qwen3-30B-A3B-FP4`) run on SM75+ GPUs via the Marlin kernel, with weights staying compressed in FP4.
3. **NVFP4 + LoRA together** — Combines both: load an NVFP4 MoE model, apply LoRA adapters, run on any modern NVIDIA GPU.

---

## How to Use

### 1. Basic NVFP4 MoE (no LoRA)

Force Marlin fallback on Blackwell (automatic on SM75-SM89):

```bash
SGLANG_FORCE_NVFP4_MARLIN=1 python -m sglang.launch_server \
  --model-path nvidia/Qwen3-30B-A3B-FP4 \
  --trust-remote-code
```

No special flags needed on non-Blackwell GPUs — `should_use_fp4_marlin_fallback()` triggers automatically.

### 2. MoE LoRA (Triton backend — default for FP16/BF16/AWQ/GPTQ MoE models)

```bash
python -m sglang.launch_server \
  --model-path <moe-model> \
  --enable-lora \
  --lora-paths adapter1=/path/to/lora \
  --trust-remote-code
```

LoRA target modules for MoE must be named `gate_up_proj_moe` and/or `down_proj_moe` in the adapter config.

### 3. NVFP4 MoE + LoRA (Marlin backend)

```bash
SGLANG_FORCE_NVFP4_MARLIN=1 python -m sglang.launch_server \
  --model-path nvidia/Qwen3-30B-A3B-FP4 \
  --enable-lora \
  --lora-paths adapter1=/path/to/lora \
  --trust-remote-code
```

### 4. AWQ/GPTQ MoE + LoRA (Marlin backend)

```bash
python -m sglang.launch_server \
  --model-path <awq-or-gptq-moe-model> \
  --enable-lora \
  --lora-paths adapter1=/path/to/lora \
  --trust-remote-code
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `SGLANG_FORCE_NVFP4_MARLIN` | `false` | Force Marlin FP4 fallback even on Blackwell GPUs (SM100+). Automatic on SM75-SM89. |

---

## New Files

| File | Description |
|---|---|
| `python/sglang/srt/lora/lora_moe_runners.py` | LoRA-aware MoE runner cores for Triton and Marlin backends |
| `python/sglang/srt/lora/triton_ops/fused_moe_lora_kernel.py` | Triton kernel for batched LoRA GEMM inside MoE (per-expert, per-adapter) |
| `python/sglang/srt/layers/quantization/marlin_utils_fp4.py` | NVFP4→Marlin weight repacking, scale conversion, and linear/MoE helpers (from PR #19652) |
| `test/registered/lora/test_lora_marlin_moe_runner.py` | Unit tests for Marlin LoRA MoE runner |
| `test/registered/quant/test_nvfp4_marlin_fallback.py` | Unit tests for NVFP4 Marlin fallback (from PR #19652) |

---

## Key Classes and APIs

### `FusedMoEWithLoRA` (`sglang.srt.lora.layers`)

Wraps `FusedMoE` and injects LoRA deltas at the correct points in the MoE pipeline.

```python
class FusedMoEWithLoRA(BaseLayerWithLoRA):
    def set_lora_info(
        self,
        gate_up_lora_a_weights: torch.Tensor,  # [num_loras, num_experts, max_rank, hidden_dim]
        gate_up_lora_b_weights: torch.Tensor,  # [num_loras, num_experts, gate_up_dim, max_rank]
        down_lora_a_weights: torch.Tensor,      # [num_loras, num_experts, max_rank, intermediate_dim]
        down_lora_b_weights: torch.Tensor,      # [num_loras, num_experts, hidden_dim, max_rank]
    )

    def forward(self, hidden_states, topk_output, **kwargs) -> torch.Tensor
```

**LoRA target module names:**
- `gate_up_proj_moe` — fused gate+up projection LoRA (A shards on `down_proj_moe` rules, B shards on gate_up)
- `down_proj_moe` — down projection LoRA

**TP sharding** (handled automatically by `slice_moe_lora_a_weights` / `slice_moe_lora_b_weights`):
- `gate_up_proj_moe.A`: no shard (input is full `hidden_states`)
- `gate_up_proj_moe.B`: sharded along intermediate dim per TP rank
- `down_proj_moe.A`: sharded along intermediate dim per TP rank
- `down_proj_moe.B`: no shard (output is all-reduced)

### `LoRAInfo` (`sglang.srt.lora.lora_moe_runners`)

Dataclass bundling LoRA weights and dispatch metadata for a batch:

```python
@dataclass
class LoRAInfo:
    gate_up_lora_a_weights: torch.Tensor  # [num_loras, num_experts, max_rank, hidden_dim]
    gate_up_lora_b_weights: torch.Tensor  # [num_loras, num_experts, gate_up_dim, max_rank]
    down_lora_a_weights: torch.Tensor     # [num_loras, num_experts, max_rank, intermediate_dim]
    down_lora_b_weights: torch.Tensor     # [num_loras, num_experts, hidden_dim, max_rank]
    seg_indptr: torch.Tensor              # [num_segments + 1]
    req_to_lora: torch.Tensor             # [num_segments]
    lora_ranks: torch.Tensor              # [num_loras]
    adapter_enabled: torch.Tensor         # [num_loras]
    max_lora_rank: int
    num_experts: int
    tp_size: int = 1
    tp_rank: int = 0
    hidden_size: int = 0
```

### `MoeLoRADeltaMixin` (`sglang.srt.lora.lora_moe_runners`)

Shared mixin providing backend-agnostic LoRA delta computation:

```python
class MoeLoRADeltaMixin:
    def _run_lora_align(self, topk_ids, lora_info) -> Tuple[...]
    def _add_lora_gate_up_delta(self, hidden_states, intermediate_cache, topk_weights, lora_info, ...) -> None
    def _add_lora_down_delta(self, intermediate_input, intermediate_cache, topk_weights, lora_info, ...) -> None
```

### `TritonRunnerCoreWithLoRA` (`sglang.srt.lora.lora_moe_runners`)

LoRA-aware runner for the Triton (FP16/BF16/FP8) MoE backend. Decomposes the fused Triton MoE kernel into gate_up → LoRA delta → activation → down → LoRA delta → reduce.

### `MarlinRunnerCoreWithLoRA` (`sglang.srt.lora.lora_moe_runners`)

LoRA-aware runner for the Marlin (WNA16 / NVFP4) MoE backend. Decomposes `fused_marlin_moe` into stages and injects LoRA deltas at the same points. Supports:
- GPTQ INT4 (`uint4b8`)
- AWQ INT4 (`uint4`)
- **NVFP4** (`float4_e2m1f`) with per-expert `global_scale`

### `MarlinMoeQuantInfo` (`sglang.srt.layers.moe.moe_runner.marlin`)

Extended with FP4 and EP fields:

```python
@dataclass
class MarlinMoeQuantInfo(MoeQuantInfo):
    w13_qweight: torch.Tensor
    w2_qweight: torch.Tensor
    w13_scales: torch.Tensor
    w2_scales: torch.Tensor
    w13_g_idx_sort_indices: Optional[torch.Tensor]
    w2_g_idx_sort_indices: Optional[torch.Tensor]
    weight_bits: int

    # GPTQ specific
    w13_g_idx: Optional[torch.Tensor] = None
    w2_g_idx: Optional[torch.Tensor] = None
    is_k_full: bool = True

    # AWQ specific
    w13_qzeros: Optional[torch.Tensor] = None
    w2_qzeros: Optional[torch.Tensor] = None

    # FP4 Marlin specific (NEW)
    w13_global_scale: Optional[torch.Tensor] = None
    w2_global_scale: Optional[torch.Tensor] = None

    # EP support (NEW)
    expert_map: Optional[torch.Tensor] = None
    global_num_experts: int = -1
```

### `MoeRunner` (`sglang.srt.layers.moe.moe_runner.runner`)

Extended constructor:

```python
class MoeRunner:
    def __init__(self, runner_backend, config, lora_enabled=False):
        ...

    def run(self, dispatch_output, quant_info, lora_info=None) -> CombineInput:
        ...
```

When `lora_enabled=True`:
- Skips the fused path (LoRA requires decomposed stages)
- Selects `TritonRunnerCoreWithLoRA` or `MarlinRunnerCoreWithLoRA` based on backend
- Passes `lora_info` to the runner core

---

## NVFP4 Marlin Fallback Integration Fixes (our additions on top of PR #19652)

Three bugs in PR #19652's integration with the `FusedMoE` pipeline, all caused by `get_moe_runner_backend()` still returning `flashinfer_trtllm` when the Marlin fallback is active:

### Fix 1: Layer class selection (`ep_moe/layer.py`)

`get_fused_moe_cls()` was selecting `FlashInferFP4MoE` (which requires native FP4 attributes like `w13_input_scale_quant`). Now returns `FusedMoE` when `should_use_fp4_marlin_fallback()` is True.

### Fix 2: TopK routing format (`topk.py`)

TopK was using `BYPASSED` format (raw `router_logits`, no `topk_weights`/`topk_ids`). Marlin needs `STANDARD` format. Now checks `should_use_fp4_marlin_fallback()`.

### Fix 3: Weight loading order (`fused_moe_triton/layer.py`)

**This was the critical correctness bug.** When `use_flashinfer_trtllm_moe=True`, the weight loader swaps gate/up projections to `[up, gate]` format (FlashInfer TRTLLM convention). Marlin's `silu_and_mul` expects `[gate, up]`. This produced completely garbled model output. Fixed by setting `use_flashinfer_trtllm_moe=False` when Marlin fallback is active.

### Detection function

```python
from sglang.srt.layers.quantization.marlin_utils_fp4 import should_use_fp4_marlin_fallback

should_use_fp4_marlin_fallback()  # True when SGLANG_FORCE_NVFP4_MARLIN=1 or GPU < SM100
```

---

## LoRA Delta Injection Points

The key design insight: LoRA deltas must be injected *inside* the MoE computation, not added to the final output.

```
hidden_states
    │
    ▼
┌─────────────────────┐
│  Gate/Up GEMM       │  base: w13 @ hidden_states
│  (Triton or Marlin) │
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  + LoRA gate_up Δ   │  delta: B_gate_up @ (A_gate_up @ hidden_states)
│  (per-expert,       │  added BEFORE activation
│   per-adapter)      │
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  Activation         │  silu_and_mul / gelu_and_mul
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  Down GEMM          │  base: w2 @ intermediate
│  (Triton or Marlin) │
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  + LoRA down Δ      │  delta: B_down @ (A_down @ intermediate)
│  (per-expert,       │  added BEFORE moe_sum_reduce
│   per-adapter)      │
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  moe_sum_reduce     │  weighted sum across top-k experts
└─────────────────────┘
    │
    ▼
  output
```

---

## Weight Memory Layout

All NVFP4 weights stay in FP4 (4-bit compressed). The Marlin fallback only **repacks** the tile layout:

| Step | What happens | Memory |
|---|---|---|
| Checkpoint load | FP4 uint8 packed, FP8 block scales, FP32 global scale | ~same as checkpoint |
| `prepare_moe_fp4_layer_for_marlin` | `gptq_marlin_repack` retiles FP4 data for Marlin; scales converted to FP8-S0E5M3; global scale adjusted with exponent bias | ~same footprint |
| Runtime | Marlin kernel dequantizes FP4→BF16/FP16 on-the-fly during GEMM | No extra memory |

LoRA weights are stored in FP16/BF16 at full precision (they're small: `rank × dim` per expert per adapter).
