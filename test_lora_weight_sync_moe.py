"""Standalone test for LoRA weight sync on MoE models.

Tests that the SGLang LoRA adapter loading API correctly handles MoE expert
LoRA weights (per-expert lora_A / lora_B) alongside standard attention LoRA.

Usage (on a machine with the SGLang fork installed):
    # Attention-only LoRA on MoE model
    python tests/test_lora_weight_sync_moe.py --model Qwen/Qwen3-30B-A3B --mode attn

    # Attention + MoE expert LoRA
    python tests/test_lora_weight_sync_moe.py --model Qwen/Qwen3-30B-A3B --mode full

    # Quick smoke test with a smaller model slice
    python tests/test_lora_weight_sync_moe.py --model Qwen/Qwen3-30B-A3B --mode full --max-layers 2

Requires: the SGLang fork from ../sglang with MoE LoRA support.
"""

import argparse
import gc
import json
import sys
import time

import requests
import torch


def get_model_config(base_url: str) -> dict:
    resp = requests.get(f"{base_url}/get_model_info")
    resp.raise_for_status()
    return resp.json()


def generate(base_url: str, prompt: str, max_tokens: int = 32, lora_path: str | None = None) -> str:
    payload = {
        "text": prompt,
        "sampling_params": {"temperature": 0, "max_new_tokens": max_tokens},
    }
    if lora_path is not None:
        payload["lora_path"] = lora_path
    resp = requests.post(f"{base_url}/generate", json=payload)
    resp.raise_for_status()
    return resp.json()["text"]


def build_lora_weights_attn_only(
    num_layers: int,
    hidden_size: int,
    num_heads: int,
    num_kv_heads: int,
    lora_rank: int,
    head_dim: int | None = None,
    max_layers: int | None = None,
) -> dict[str, torch.Tensor]:
    """Build random LoRA weights for attention layers only."""
    if head_dim is None:
        head_dim = hidden_size // num_heads

    q_size = num_heads * head_dim
    kv_size = num_kv_heads * head_dim

    weights = {}
    layers = range(min(num_layers, max_layers)) if max_layers else range(num_layers)
    for layer_idx in layers:
        prefix = f"base_model.model.model.layers.{layer_idx}.self_attn"
        # lora_A shape: [rank, input_dim], lora_B shape: [output_dim, rank]
        # o_proj input is the attention output (num_heads * head_dim), not hidden_size
        for proj_name, in_size, out_size in [
            ("q_proj", hidden_size, q_size),
            ("k_proj", hidden_size, kv_size),
            ("v_proj", hidden_size, kv_size),
            ("o_proj", q_size, hidden_size),
        ]:
            weights[f"{prefix}.{proj_name}.lora_A.weight"] = torch.randn(lora_rank, in_size) * 0.01
            weights[f"{prefix}.{proj_name}.lora_B.weight"] = torch.zeros(out_size, lora_rank)
    return weights


def build_lora_weights_moe(
    num_layers: int,
    hidden_size: int,
    num_heads: int,
    num_kv_heads: int,
    num_experts: int,
    moe_intermediate_size: int,
    lora_rank: int,
    head_dim: int | None = None,
    max_layers: int | None = None,
    include_shared_expert: bool = False,
    shared_expert_intermediate_size: int | None = None,
) -> dict[str, torch.Tensor]:
    """Build random LoRA weights for attention + MoE expert layers."""
    weights = build_lora_weights_attn_only(
        num_layers, hidden_size, num_heads, num_kv_heads, lora_rank, head_dim, max_layers
    )

    layers = range(min(num_layers, max_layers)) if max_layers else range(num_layers)
    for layer_idx in layers:
        for expert_idx in range(num_experts):
            prefix = f"base_model.model.model.layers.{layer_idx}.mlp.experts.{expert_idx}"
            for proj_name, a_in, b_out in [
                ("gate_proj", hidden_size, moe_intermediate_size),
                ("up_proj", hidden_size, moe_intermediate_size),
                ("down_proj", moe_intermediate_size, hidden_size),
            ]:
                weights[f"{prefix}.{proj_name}.lora_A.weight"] = torch.randn(lora_rank, a_in) * 0.01
                weights[f"{prefix}.{proj_name}.lora_B.weight"] = torch.zeros(b_out, lora_rank)

        if include_shared_expert and shared_expert_intermediate_size:
            prefix = f"base_model.model.model.layers.{layer_idx}.mlp.shared_expert"
            for proj_name, a_in, b_out in [
                ("gate_proj", hidden_size, shared_expert_intermediate_size),
                ("up_proj", hidden_size, shared_expert_intermediate_size),
                ("down_proj", shared_expert_intermediate_size, hidden_size),
            ]:
                weights[f"{prefix}.{proj_name}.lora_A.weight"] = torch.randn(lora_rank, a_in) * 0.01
                weights[f"{prefix}.{proj_name}.lora_B.weight"] = torch.zeros(b_out, lora_rank)

    return weights


def load_lora_via_api(
    base_url: str,
    lora_name: str,
    lora_weights: dict[str, torch.Tensor],
    lora_config: dict,
) -> dict:
    """Load LoRA adapter via SGLang's load_lora_adapter_from_tensors API."""
    from sglang.srt.utils import MultiprocessingSerializer
    from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket

    named_tensors = [(name, tensor.cuda()) for name, tensor in lora_weights.items()]
    bucket = FlattenedTensorBucket(named_tensors=named_tensors)
    bucket_data = {
        "flattened_tensor": bucket.get_flattened_tensor(),
        "metadata": bucket.get_metadata(),
    }
    serialized = MultiprocessingSerializer.serialize(bucket_data, output_str=True)

    payload = {
        "lora_name": lora_name,
        "serialized_tensors": serialized,
        "config_dict": lora_config,
        "load_format": "flattened_bucket",
    }
    resp = requests.post(f"{base_url}/load_lora_adapter_from_tensors", json=payload)
    resp.raise_for_status()
    return resp.json()


def unload_lora_via_api(base_url: str, lora_name: str) -> dict:
    resp = requests.post(f"{base_url}/unload_lora_adapter", json={"lora_name": lora_name})
    resp.raise_for_status()
    return resp.json()


def run_test(args):
    from transformers import AutoConfig

    print(f"Loading model config from {args.model}...")
    hf_config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)

    hidden_size = hf_config.hidden_size
    num_heads = hf_config.num_attention_heads
    num_kv_heads = getattr(hf_config, "num_key_value_heads", num_heads)
    num_layers = hf_config.num_hidden_layers
    num_experts = getattr(hf_config, "num_experts", 1)
    moe_intermediate_size = getattr(hf_config, "moe_intermediate_size", getattr(hf_config, "intermediate_size", 0))
    shared_expert_intermediate_size = getattr(hf_config, "shared_expert_intermediate_size", 0)
    head_dim = getattr(hf_config, "head_dim", hidden_size // num_heads)

    is_moe = num_experts > 1
    if not is_moe:
        print(f"WARNING: {args.model} has num_experts={num_experts}, not an MoE model.")

    print(f"Model config: layers={num_layers}, hidden={hidden_size}, heads={num_heads}, "
          f"kv_heads={num_kv_heads}, experts={num_experts}, moe_inter={moe_intermediate_size}, "
          f"shared_expert_inter={shared_expert_intermediate_size}")

    lora_rank = args.lora_rank

    if args.mode == "attn":
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        print(f"\n=== Building attention-only LoRA weights (rank={lora_rank}) ===")
        lora_weights = build_lora_weights_attn_only(
            num_layers, hidden_size, num_heads, num_kv_heads, lora_rank, head_dim, args.max_layers
        )
    else:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        print(f"\n=== Building attention + MoE expert LoRA weights (rank={lora_rank}) ===")
        lora_weights = build_lora_weights_moe(
            num_layers, hidden_size, num_heads, num_kv_heads, num_experts,
            moe_intermediate_size, lora_rank, head_dim, args.max_layers,
            include_shared_expert=shared_expert_intermediate_size > 0,
            shared_expert_intermediate_size=shared_expert_intermediate_size,
        )

    print(f"Generated {len(lora_weights)} LoRA weight tensors")
    total_params = sum(t.numel() for t in lora_weights.values())
    print(f"Total LoRA parameters: {total_params:,} ({total_params * 2 / 1e6:.1f} MB in bf16)")

    lora_config = {
        "peft_type": "LORA",
        "r": lora_rank,
        "lora_alpha": lora_rank,
        "target_modules": target_modules,
        "lora_dropout": 0.0,
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }

    lora_name = "test_moe_lora"
    base_url = args.base_url

    if not args.skip_server:
        print(f"\n=== Starting SGLang server for {args.model} ===")
        _start_server(args)

    print(f"\n=== Waiting for server at {base_url} ===")
    _wait_healthy(base_url, timeout=args.timeout)

    print("\n=== Step 1: Generate baseline output (no LoRA) ===")
    prompt = "The capital of France is"
    baseline_output = generate(base_url, prompt)
    print(f"  Baseline: {baseline_output[:100]}...")

    print(f"\n=== Step 2: Loading LoRA adapter '{lora_name}' via tensors API ===")
    t0 = time.perf_counter()
    result = load_lora_via_api(base_url, lora_name, lora_weights, lora_config)
    load_time = time.perf_counter() - t0
    print(f"  Load result: {result}")
    print(f"  Load time: {load_time:.2f}s")

    success = result.get("success", False)
    if not success:
        print(f"\n  FAILED: LoRA loading failed: {result}")
        return False

    print(f"\n=== Step 3: Generate output with LoRA adapter ===")
    lora_output = generate(base_url, prompt, lora_path=lora_name)
    print(f"  LoRA output: {lora_output[:100]}...")

    print(f"\n=== Step 4: Unload LoRA adapter ===")
    unload_result = unload_lora_via_api(base_url, lora_name)
    print(f"  Unload result: {unload_result}")

    print(f"\n=== Step 5: Generate output after unload (should match baseline) ===")
    post_unload_output = generate(base_url, prompt)
    print(f"  Post-unload: {post_unload_output[:100]}...")

    print("\n=== Step 6: Reload with non-zero lora_B to verify LoRA takes effect ===")
    for key in lora_weights:
        if "lora_B" in key:
            lora_weights[key] = torch.randn_like(lora_weights[key]) * 0.1
    result2 = load_lora_via_api(base_url, lora_name, lora_weights, lora_config)
    print(f"  Reload result: {result2}")
    if not result2.get("success", False):
        print(f"  FAILED: LoRA reload failed: {result2}")
        return False

    lora_output_nonzero = generate(base_url, prompt, lora_path=lora_name)
    print(f"  Non-zero LoRA output: {lora_output_nonzero[:100]}...")

    outputs_differ = lora_output_nonzero != baseline_output
    print(f"\n=== Results ===")
    print(f"  Baseline matches post-unload: {baseline_output == post_unload_output}")
    print(f"  Non-zero LoRA output differs from baseline: {outputs_differ}")

    if not outputs_differ:
        print("  WARNING: LoRA output matches baseline - adapter may not be taking effect")

    unload_lora_via_api(base_url, lora_name)

    print("\n  PASSED: LoRA weight sync test completed successfully")
    return True


def _start_server(args):
    """Start SGLang server as a subprocess."""
    import subprocess

    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    if args.mode == "full":
        target_modules += ["gate_proj", "up_proj", "down_proj"]

    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", args.model,
        "--port", str(args.port),
        "--tp-size", str(args.tp_size),
        "--enable-lora",
        "--max-loras-per-batch", "1",
        "--max-lora-rank", str(args.lora_rank),
        "--lora-target-modules", *target_modules,
        "--mem-fraction-static", str(args.mem_fraction),
    ]
    if args.trust_remote_code:
        cmd.append("--trust-remote-code")

    print(f"  CMD: {' '.join(cmd)}")
    subprocess.Popen(cmd)


def _wait_healthy(base_url: str, timeout: int = 300):
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(f"{base_url}/health_generate", timeout=5)
            if resp.status_code == 200:
                print(f"  Server healthy after {time.time() - start:.1f}s")
                return
        except requests.RequestException:
            pass
        time.sleep(2)
    raise TimeoutError(f"Server not healthy after {timeout}s")


def main():
    parser = argparse.ArgumentParser(description="Test LoRA weight sync on MoE models")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-30B-A3B",
                        help="HF model path or name")
    parser.add_argument("--mode", choices=["attn", "full"], default="full",
                        help="attn: attention-only LoRA; full: attention + MoE expert LoRA")
    parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--max-layers", type=int, default=None,
                        help="Limit LoRA to first N layers (for faster testing)")
    parser.add_argument("--base-url", type=str, default="http://127.0.0.1:30000",
                        help="SGLang server base URL")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--mem-fraction", type=float, default=0.7)
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--timeout", type=int, default=300,
                        help="Server startup timeout in seconds")
    parser.add_argument("--skip-server", action="store_true",
                        help="Skip server launch (assume already running)")
    args = parser.parse_args()
    args.base_url = f"http://127.0.0.1:{args.port}"

    success = run_test(args)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
