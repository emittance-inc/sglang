"""Test each LoRA layer type in isolation on an MoE model.

For each layer type (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj),
builds a LoRA adapter where ONLY that type has non-zero lora_B weights. Verifies
that inference output changes, proving each path is actually wired up.

Usage:
    python tests/test_lora_layer_isolation.py \
        --model /root/models/Qwen3-30B-A3B-NVFP4 \
        --skip-server --max-layers 1
"""

import argparse
import sys
import time

import requests
import torch


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


def load_lora_via_api(base_url, lora_name, lora_weights, lora_config):
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


def unload_lora_via_api(base_url, lora_name):
    resp = requests.post(f"{base_url}/unload_lora_adapter", json={"lora_name": lora_name})
    resp.raise_for_status()
    return resp.json()


def flush_cache(base_url):
    resp = requests.post(f"{base_url}/flush_cache")
    resp.raise_for_status()


ATTN_PROJS = ["q_proj", "k_proj", "v_proj", "o_proj"]
MOE_PROJS = ["gate_proj", "up_proj", "down_proj"]


def build_full_zero_weights(
    num_layers, hidden_size, num_heads, num_kv_heads, head_dim,
    num_experts, moe_intermediate_size, lora_rank, max_layers,
):
    """Build a full LoRA weight dict (attn + MoE) with all zeros."""
    q_size = num_heads * head_dim
    kv_size = num_kv_heads * head_dim

    proj_dims = {
        "q_proj": (hidden_size, q_size),
        "k_proj": (hidden_size, kv_size),
        "v_proj": (hidden_size, kv_size),
        "o_proj": (q_size, hidden_size),
    }
    moe_dims = {
        "gate_proj": (hidden_size, moe_intermediate_size),
        "up_proj": (hidden_size, moe_intermediate_size),
        "down_proj": (moe_intermediate_size, hidden_size),
    }

    weights = {}
    layers = range(min(num_layers, max_layers)) if max_layers else range(num_layers)
    for layer_idx in layers:
        for proj, (in_sz, out_sz) in proj_dims.items():
            prefix = f"base_model.model.model.layers.{layer_idx}.self_attn.{proj}"
            weights[f"{prefix}.lora_A.weight"] = torch.zeros(lora_rank, in_sz)
            weights[f"{prefix}.lora_B.weight"] = torch.zeros(out_sz, lora_rank)

        for expert_idx in range(num_experts):
            for proj, (in_sz, out_sz) in moe_dims.items():
                prefix = f"base_model.model.model.layers.{layer_idx}.mlp.experts.{expert_idx}.{proj}"
                weights[f"{prefix}.lora_A.weight"] = torch.zeros(lora_rank, in_sz)
                weights[f"{prefix}.lora_B.weight"] = torch.zeros(out_sz, lora_rank)

    return weights


def activate_layer_type(weights, layer_type):
    """Set lora_A and lora_B to large random values for a specific layer type.

    layer_type is one of: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
    """
    activated = {k: v.clone() for k, v in weights.items()}
    count = 0
    for key in activated:
        # key format: ...{proj_name}.lora_A.weight or ...{proj_name}.lora_B.weight
        parts = key.split(".")
        proj_name = parts[-3]  # e.g. "q_proj", "gate_proj"
        if proj_name != layer_type:
            continue
        activated[key] = torch.randn_like(activated[key]) * 100.0
        count += 1
    return activated, count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--max-layers", type=int, default=1)
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--skip-server", action="store_true")
    args = parser.parse_args()
    base_url = f"http://127.0.0.1:{args.port}"

    from transformers import AutoConfig
    hf_config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)

    hidden_size = hf_config.hidden_size
    num_heads = hf_config.num_attention_heads
    num_kv_heads = getattr(hf_config, "num_key_value_heads", num_heads)
    num_layers = hf_config.num_hidden_layers
    num_experts = getattr(hf_config, "num_experts", 1)
    moe_intermediate_size = getattr(hf_config, "moe_intermediate_size",
                                     getattr(hf_config, "intermediate_size", 0))
    head_dim = getattr(hf_config, "head_dim", hidden_size // num_heads)

    print(f"Model: layers={num_layers}, hidden={hidden_size}, heads={num_heads}, "
          f"kv_heads={num_kv_heads}, experts={num_experts}, moe_inter={moe_intermediate_size}")

    lora_config = {
        "peft_type": "LORA",
        "r": args.lora_rank,
        "lora_alpha": args.lora_rank,
        "target_modules": ATTN_PROJS + MOE_PROJS,
        "lora_dropout": 0.0,
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }

    lora_name = "isolation_test"
    prompt = "The capital of France is"

    print(f"\nBuilding all-zero LoRA weights (rank={args.lora_rank}, layers=0..{(args.max_layers or num_layers) - 1})...")
    zero_weights = build_full_zero_weights(
        num_layers, hidden_size, num_heads, num_kv_heads, head_dim,
        num_experts, moe_intermediate_size, args.lora_rank, args.max_layers,
    )
    print(f"  Total tensors: {len(zero_weights)}")

    print("\nWaiting for server...")
    for _ in range(60):
        try:
            requests.get(f"{base_url}/health_generate", timeout=3).raise_for_status()
            break
        except Exception:
            time.sleep(2)
    else:
        print("Server not ready"); sys.exit(1)

    flush_cache(base_url)
    print("\n--- Baseline (no LoRA) ---")
    baseline = generate(base_url, prompt)
    print(f"  Output: {baseline[:120]}")

    flush_cache(base_url)
    print("\n--- All-zero LoRA (should match baseline) ---")
    load_lora_via_api(base_url, lora_name, zero_weights, lora_config)
    zero_output = generate(base_url, prompt, lora_path=lora_name)
    print(f"  Output: {zero_output[:120]}")
    print(f"  Matches baseline: {zero_output == baseline}")
    unload_lora_via_api(base_url, lora_name)

    all_types = ATTN_PROJS + MOE_PROJS
    results = {}

    for layer_type in all_types:
        flush_cache(base_url)
        print(f"\n--- Testing {layer_type} in isolation ---")
        activated, count = activate_layer_type(zero_weights, layer_type)
        print(f"  Activated {count} tensors for {layer_type}")

        load_lora_via_api(base_url, lora_name, activated, lora_config)
        try:
            output = generate(base_url, prompt, lora_path=lora_name)
            differs = output != baseline
            results[layer_type] = {"status": "PASS" if differs else "FAIL (no effect)", "output": output[:100]}
            print(f"  Output: {output[:100]}")
            print(f"  Differs from baseline: {differs}")
        except Exception as e:
            results[layer_type] = {"status": f"CRASH: {e}", "output": None}
            print(f"  CRASHED: {e}")
        unload_lora_via_api(base_url, lora_name)

    print("\n" + "=" * 60)
    print("ISOLATION TEST RESULTS")
    print("=" * 60)
    all_pass = True
    for layer_type, res in results.items():
        tag = "ATTN" if layer_type in ATTN_PROJS else "MoE "
        status = res["status"]
        icon = "OK" if status == "PASS" else "!!"
        print(f"  [{icon}] {tag} {layer_type:10s} → {status}")
        if status != "PASS":
            all_pass = False

    if all_pass:
        print("\nAll layer types verified working.")
    else:
        print("\nSome layer types failed — check output above.")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
