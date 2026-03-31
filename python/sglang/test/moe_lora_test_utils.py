"""Utilities for creating synthetic MoE LoRA adapters and merged checkpoints on-the-fly.

Used by end-to-end tests that validate MoE LoRA correctness against a
merged-weight baseline.
"""

import json
import os
import shutil
import tempfile
from typing import Optional

import torch
from safetensors.torch import save_file


def create_synthetic_moe_lora_adapter(
    base_model_path: str,
    output_dir: Optional[str] = None,
    rank: int = 8,
    lora_alpha: int = 16,
    seed: int = 42,
    num_layers: Optional[int] = None,
    num_experts: Optional[int] = None,
    hidden_size: Optional[int] = None,
    moe_intermediate_size: Optional[int] = None,
) -> str:
    """Create a synthetic PEFT-compatible MoE LoRA adapter with random weights.

    If architecture parameters (num_layers, num_experts, etc.) are not provided,
    they are loaded from the base model's config.

    Args:
        base_model_path: HuggingFace model path or local path for config.
        output_dir: Where to save the adapter. If None, creates a temp directory.
        rank: LoRA rank.
        lora_alpha: LoRA alpha scaling factor.
        seed: Random seed for reproducible weights.
        num_layers: Override number of transformer layers.
        num_experts: Override number of MoE experts.
        hidden_size: Override hidden size.
        moe_intermediate_size: Override MoE intermediate size per expert.

    Returns:
        Path to the saved adapter directory.
    """
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)

    num_layers = num_layers or config.num_hidden_layers
    num_experts = num_experts or config.num_experts
    hidden_size = hidden_size or config.hidden_size
    moe_intermediate_size = moe_intermediate_size or config.moe_intermediate_size

    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="synthetic_moe_lora_")

    os.makedirs(output_dir, exist_ok=True)

    # Create adapter_config.json (PEFT format)
    adapter_config = {
        "peft_type": "LORA",
        "auto_mapping": None,
        "base_model_name_or_path": base_model_path,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "layers_to_transform": None,
        "layers_pattern": None,
        "lora_alpha": lora_alpha,
        "lora_dropout": 0.0,
        "modules_to_save": None,
        "r": rank,
        "revision": None,
        "target_modules": ["gate_proj", "up_proj", "down_proj"],
        "task_type": "CAUSAL_LM",
    }

    with open(os.path.join(output_dir, "adapter_config.json"), "w") as f:
        json.dump(adapter_config, f, indent=2)

    # Generate weight tensors with seeded RNG
    rng = torch.Generator()
    rng.manual_seed(seed)

    tensors = {}
    for layer_id in range(num_layers):
        for expert_id in range(num_experts):
            prefix = f"base_model.model.model.layers.{layer_id}.mlp.experts.{expert_id}"

            # gate_proj: maps hidden_size -> moe_intermediate_size
            tensors[f"{prefix}.gate_proj.lora_A.weight"] = torch.randn(
                rank, hidden_size, generator=rng, dtype=torch.bfloat16
            ) * 0.1
            tensors[f"{prefix}.gate_proj.lora_B.weight"] = torch.randn(
                moe_intermediate_size, rank, generator=rng, dtype=torch.bfloat16
            ) * 0.1

            # up_proj: maps hidden_size -> moe_intermediate_size
            tensors[f"{prefix}.up_proj.lora_A.weight"] = torch.randn(
                rank, hidden_size, generator=rng, dtype=torch.bfloat16
            ) * 0.1
            tensors[f"{prefix}.up_proj.lora_B.weight"] = torch.randn(
                moe_intermediate_size, rank, generator=rng, dtype=torch.bfloat16
            ) * 0.1

            # down_proj: maps moe_intermediate_size -> hidden_size
            tensors[f"{prefix}.down_proj.lora_A.weight"] = torch.randn(
                rank, moe_intermediate_size, generator=rng, dtype=torch.bfloat16
            ) * 0.1
            tensors[f"{prefix}.down_proj.lora_B.weight"] = torch.randn(
                hidden_size, rank, generator=rng, dtype=torch.bfloat16
            ) * 0.1

    save_file(tensors, os.path.join(output_dir, "adapter_model.safetensors"))
    return output_dir


def create_merged_checkpoint(
    base_model_path: str,
    adapter_path: str,
    output_dir: Optional[str] = None,
) -> str:
    """Merge a LoRA adapter into the base model and save as a new checkpoint.

    This creates the "ground truth" — a model with LoRA weights baked in,
    whose outputs should match the base model + runtime LoRA exactly.

    Operates directly on safetensors files (no PEFT required) by computing
    delta = scaling * B @ A for each expert and adding it to the base weight.

    Args:
        base_model_path: HuggingFace model path or local path.
        adapter_path: Path to PEFT LoRA adapter directory.
        output_dir: Where to save merged checkpoint. If None, creates temp dir.

    Returns:
        Path to the saved merged checkpoint directory.
    """
    import re

    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="merged_moe_checkpoint_")

    os.makedirs(output_dir, exist_ok=True)

    # Load adapter config
    with open(os.path.join(adapter_path, "adapter_config.json")) as f:
        adapter_config = json.load(f)
    rank = adapter_config["r"]
    lora_alpha = adapter_config["lora_alpha"]
    scaling = lora_alpha / rank

    # Load adapter weights
    lora_weights = load_file(
        os.path.join(adapter_path, "adapter_model.safetensors")
    )

    # Index LoRA weights by (layer, expert, proj_type, A/B)
    lora_index = {}
    for key, tensor in lora_weights.items():
        m = re.match(
            r"base_model\.model\.model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
            r"(gate_proj|up_proj|down_proj)\.lora_(A|B)\.weight",
            key,
        )
        if m:
            layer_id, expert_id, proj, ab = (
                int(m.group(1)),
                int(m.group(2)),
                m.group(3),
                m.group(4),
            )
            lora_index[(layer_id, expert_id, proj, ab)] = tensor

    # Download / locate base model
    if os.path.isdir(base_model_path):
        model_dir = base_model_path
    else:
        model_dir = snapshot_download(base_model_path)

    # Copy all files from base model to output dir, modifying safetensors
    import glob

    for src_file in glob.glob(os.path.join(model_dir, "*")):
        fname = os.path.basename(src_file)
        dst_file = os.path.join(output_dir, fname)

        if fname.endswith(".safetensors"):
            # Load, merge LoRA deltas, save
            base_tensors = load_file(src_file)
            for key in list(base_tensors.keys()):
                m = re.match(
                    r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
                    r"(gate_proj|up_proj|down_proj)\.weight",
                    key,
                )
                if m:
                    layer_id = int(m.group(1))
                    expert_id = int(m.group(2))
                    proj = m.group(3)
                    lora_a = lora_index.get((layer_id, expert_id, proj, "A"))
                    lora_b = lora_index.get((layer_id, expert_id, proj, "B"))
                    if lora_a is not None and lora_b is not None:
                        # delta = scaling * B @ A
                        delta = scaling * (
                            lora_b.to(base_tensors[key].dtype)
                            @ lora_a.to(base_tensors[key].dtype)
                        )
                        base_tensors[key] = base_tensors[key] + delta
            save_file(base_tensors, dst_file)
        else:
            # Copy non-safetensors files as-is (config, tokenizer, etc.)
            shutil.copy2(src_file, dst_file)

    return output_dir


def cleanup_temp_dirs(*dirs: str):
    """Remove temporary directories."""
    for d in dirs:
        if d and os.path.isdir(d):
            shutil.rmtree(d, ignore_errors=True)
