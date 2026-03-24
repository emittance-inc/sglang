"""
python tools/convert_hf_to_nvfp4.py [-h] [--model-dir MODEL_DIR] [--save-dir SAVE_DIR]
                                   [--device DEVICE] [--keep-last-n KEEP_LAST_N] [--keep-first-n KEEP_FIRST_N]

Convert a BF16/FP16/FP32 HF safetensors checkpoint to NVFP4 (E2M1).

Quantises both MoE expert and dense linear layers.  Weights that belong to the
same SGLang fused layer (Q/K/V or gate/up) are quantised with a shared global
amax so their weight_scale_2 values are identical after fusion.

Uses 1D block scaling (NVTE_NVFP4_1D_SCALING, group size = 16).
"""

import argparse
import gc
import json
import os
import shutil
import sys

import safetensors
import safetensors.torch
import torch
from tqdm import tqdm

# Allow importing from the slime package when running as a standalone tool.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from nvfp4_core import (
    NVFP4_GROUP_SIZE,
    quantize_nvfp4,
    split_gated_pair_name,
    split_qkv_name,
)

DEFAULT_KV_CACHE_SCHEME = {"dynamic": False, "num_bits": 8, "type": "float"}
DEFAULT_KV_CACHE_QUANT_ALGO = "FP8"

EXPERT_WEIGHT_SUFFIXES = (
    ".w1.weight",
    ".w2.weight",
    ".w3.weight",
    ".gate_proj.weight",
    ".up_proj.weight",
    ".down_proj.weight",
    ".gate_up_proj.weight",
)

EXPERT_NAME_MARKERS = (
    ".experts.",
    ".shared_experts.",
    "block_sparse_moe.experts.",
    ".moe.experts.",
)

FUSED_QKV_SUFFIXES = (".q_proj", ".k_proj", ".v_proj")

DENSE_LINEAR_SUFFIXES = (
    ".q_proj.weight",
    ".k_proj.weight",
    ".v_proj.weight",
    ".o_proj.weight",
    ".gate_proj.weight",
    ".up_proj.weight",
    ".down_proj.weight",
    ".gate_up_proj.weight",
    ".qkv_proj.weight",
)


def _is_moe_expert_weight_name(name: str) -> bool:
    if not name.endswith(".weight"):
        return False
    if not any(marker in name for marker in EXPERT_NAME_MARKERS):
        return False
    return any(name.endswith(suffix) for suffix in EXPERT_WEIGHT_SUFFIXES)


def _is_dense_linear_weight_name(name: str) -> bool:
    return any(name.endswith(suffix) for suffix in DENSE_LINEAR_SUFFIXES)


def _extract_layer_id(name: str) -> int | None:
    parts = name.split(".")
    for idx, part in enumerate(parts):
        if part == "layers" and idx + 1 < len(parts):
            layer_id = parts[idx + 1]
            if layer_id.isdigit():
                return int(layer_id)
    return None


def _get_num_hidden_layers(model_dir: str) -> int:
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        raise ValueError("config.json is required to use --keep-first-n or --keep-last-n.")
    cfg = json.load(open(config_path))
    num_layers = cfg.get("num_hidden_layers")
    if isinstance(cfg.get("text_config"), dict):
        num_layers = num_layers or cfg["text_config"].get("num_hidden_layers")
    if num_layers is None:
        raise ValueError("num_hidden_layers not found in config.json.")
    return int(num_layers)


def _get_last_n_layer_ids(num_layers: int, keep_last_n: int) -> set[int]:
    if keep_last_n <= 0:
        return set()
    start = max(0, num_layers - keep_last_n)
    return set(range(start, num_layers))


def _get_first_n_layer_ids(num_layers: int, keep_first_n: int) -> set[int]:
    if keep_first_n <= 0:
        return set()
    end = min(num_layers, keep_first_n)
    return set(range(0, end))


def _build_keep_layer_ignore_list(num_layers: int, keep_last_n: int, keep_first_n: int) -> list[str]:
    """Build an ignore list for layers that should stay unquantized."""
    ignore_list = []
    if keep_last_n > 0:
        start = max(0, num_layers - keep_last_n)
        for layer_id in range(start, num_layers):
            prefix = f"model.layers.{layer_id}"
            ignore_list.extend([
                f"{prefix}.self_attn.qkv_proj",
                f"{prefix}.self_attn.o_proj",
                f"{prefix}.mlp",
                f"{prefix}.mlp.experts",
            ])
    if keep_first_n > 0:
        end = min(num_layers, keep_first_n)
        for layer_id in range(0, end):
            prefix = f"model.layers.{layer_id}"
            ignore_list.extend([
                f"{prefix}.self_attn.qkv_proj",
                f"{prefix}.self_attn.o_proj",
                f"{prefix}.mlp",
                f"{prefix}.mlp.experts",
            ])
    return ignore_list


def should_quantize(
    name: str,
    weight: torch.Tensor,
    skip_layers: set[int] | None = None,
) -> bool:
    if skip_layers:
        layer_id = _extract_layer_id(name)
        if layer_id is not None and layer_id in skip_layers:
            return False
    if not (_is_moe_expert_weight_name(name) or _is_dense_linear_weight_name(name)):
        return False
    if weight.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    if weight.dim() < 2:
        return False
    if weight.shape[-1] % NVFP4_GROUP_SIZE != 0:
        raise ValueError(
            f"Last dim {weight.shape[-1]} must be divisible by {NVFP4_GROUP_SIZE} "
            f"for NVFP4 quantization ({name})."
        )
    return True


class ConversionResult:
    def __init__(self) -> None:
        self.weight_map: dict[str, str] = {}
        self.total_size: int = 0
        self.modules_to_not_convert: list[str] = []

    def add_result(self, filename: str, q_weights: dict[str, torch.Tensor], module_names: list[str]) -> None:
        for key, tensor in q_weights.items():
            self.weight_map[key] = filename
            self.total_size += tensor.numel() * tensor.element_size()
        self.modules_to_not_convert.extend(module_names)


def _update_quantization_config(cfg: dict, ignore_list: list[str]) -> None:
    quant_cfg = cfg.get("quantization_config")
    if not isinstance(quant_cfg, dict):
        quant_cfg = {}

    quant_cfg["quant_algo"] = "NVFP4"
    quant_cfg["quant_method"] = "modelopt_fp4"
    quant_cfg["group_size"] = NVFP4_GROUP_SIZE
    quant_cfg["ignore"] = ignore_list
    quant_cfg.setdefault("kv_cache_scheme", DEFAULT_KV_CACHE_SCHEME)

    config_groups = quant_cfg.get("config_groups")
    if isinstance(config_groups, dict):
        for group in config_groups.values():
            if not isinstance(group, dict):
                continue
            group.setdefault("targets", ["Linear"])
            for key in ("input_activations", "weights"):
                section = group.get(key)
                if not isinstance(section, dict):
                    continue
                section.setdefault("dynamic", False)
                section.setdefault("num_bits", 4)
                section.setdefault("type", "float")
                section["group_size"] = NVFP4_GROUP_SIZE

    cfg["quantization_config"] = quant_cfg


def _write_hf_quant_config(output_path: str, ignore_list: list[str], input_path: str) -> None:
    hf_quant_path = os.path.join(input_path, "hf_quant_config.json")
    if os.path.exists(hf_quant_path):
        with open(hf_quant_path) as f:
            hf_quant_cfg = json.load(f)
    else:
        hf_quant_cfg = {"producer": {"name": "modelopt"}}

    quant_section = hf_quant_cfg.get("quantization")
    if not isinstance(quant_section, dict):
        quant_section = {}

    quant_section["quant_algo"] = "NVFP4"
    quant_section["kv_cache_quant_algo"] = DEFAULT_KV_CACHE_QUANT_ALGO
    quant_section["group_size"] = NVFP4_GROUP_SIZE
    quant_section["exclude_modules"] = ignore_list
    hf_quant_cfg["quantization"] = quant_section

    with open(os.path.join(output_path, "hf_quant_config.json"), "w") as f:
        json.dump(hf_quant_cfg, f, indent=2)


def _augment_ignore_list(ignore_list: list[str]) -> list[str]:
    """Expand individual Q/K/V ignore entries to also cover fused qkv_proj."""
    ignore_set = set(ignore_list)
    extra = set()
    for name in ignore_list:
        if name.endswith(FUSED_QKV_SUFFIXES):
            for suffix in FUSED_QKV_SUFFIXES:
                if name.endswith(suffix):
                    extra.add(name[: -len(suffix)] + ".qkv_proj")
                    break
    ignore_set.update(extra)
    return sorted(ignore_set)


def _collect_shared_global_amax(
    *,
    input_path: str,
    safetensors_files: list[str],
    device: str,
    skip_layers: set[int],
) -> dict[str, torch.Tensor]:
    """Collect shared amax for fused weight groups (gate/up pairs and Q/K/V)."""
    gate_amax: dict[str, torch.Tensor] = {}
    up_amax: dict[str, torch.Tensor] = {}
    qkv_amax: dict[str, dict[str, torch.Tensor]] = {}
    for filename in safetensors_files:
        with safetensors.safe_open(os.path.join(input_path, filename), framework="pt", device=device) as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                if not should_quantize(key, tensor, skip_layers):
                    continue
                amax = tensor.abs().max().to(torch.float32)
                base, role = split_gated_pair_name(key)
                if base is not None and role is not None:
                    if role == "gate":
                        prev = gate_amax.get(base)
                        gate_amax[base] = amax if prev is None else torch.max(prev, amax)
                    elif role == "up":
                        prev = up_amax.get(base)
                        up_amax[base] = amax if prev is None else torch.max(prev, amax)
                    continue
                qkv_base, qkv_role = split_qkv_name(key)
                if qkv_base is not None and qkv_role is not None:
                    if qkv_base not in qkv_amax:
                        qkv_amax[qkv_base] = {}
                    prev = qkv_amax[qkv_base].get(qkv_role)
                    qkv_amax[qkv_base][qkv_role] = amax if prev is None else torch.max(prev, amax)

    shared_global_amax: dict[str, torch.Tensor] = {}
    for base in gate_amax.keys() & up_amax.keys():
        shared_global_amax[base] = torch.max(gate_amax[base], up_amax[base])
    for base, roles in qkv_amax.items():
        if len(roles) >= 2:
            shared_global_amax[base] = torch.stack(list(roles.values())).max()
    return shared_global_amax


def process_file(
    input_path: str,
    output_path: str,
    filename: str,
    result_collector: ConversionResult,
    device: str,
    skip_layers: set[int],
    shared_global_amax: dict[str, torch.Tensor],
) -> None:
    if not filename.endswith(".safetensors"):
        return

    modules_to_not_convert: list[str] = []
    q_weights: dict[str, torch.Tensor] = {}

    with safetensors.safe_open(os.path.join(input_path, filename), framework="pt", device=device) as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            if should_quantize(key, tensor, skip_layers):
                global_amax = None
                base, _ = split_gated_pair_name(key)
                if base is not None:
                    global_amax = shared_global_amax.get(base)
                if global_amax is None:
                    qkv_base, _ = split_qkv_name(key)
                    if qkv_base is not None:
                        global_amax = shared_global_amax.get(qkv_base)
                qweight, block_scale, weight_scale_2 = quantize_nvfp4(tensor, global_amax=global_amax)
                q_weights[key] = qweight
                q_weights[key.replace(".weight", ".weight_scale")] = block_scale
                q_weights[key.replace(".weight", ".weight_scale_2")] = weight_scale_2
                q_weights[key.replace(".weight", ".input_scale")] = torch.ones_like(
                    weight_scale_2, dtype=torch.float32
                )
            else:
                if key.endswith(".weight"):
                    modules_to_not_convert.append(key.replace(".weight", ""))
                q_weights[key] = tensor

    safetensors.torch.save_file(q_weights, os.path.join(output_path, filename), metadata={"format": "pt"})
    result_collector.add_result(filename, q_weights, modules_to_not_convert)


def convert_nvfp4(model_dir: str, save_dir: str, device: str, keep_last_n: int, keep_first_n: int) -> None:
    input_path = os.path.abspath(model_dir)
    output_path = os.path.abspath(save_dir)
    os.makedirs(output_path, exist_ok=True)

    for filename in os.listdir(input_path):
        if not filename.endswith(".safetensors") and not os.path.isdir(os.path.join(input_path, filename)):
            shutil.copyfile(os.path.join(input_path, filename), os.path.join(output_path, filename))

    safetensors_files = [f for f in os.listdir(input_path) if f.endswith(".safetensors")]

    num_layers = _get_num_hidden_layers(input_path) if (keep_last_n > 0 or keep_first_n > 0) else 0
    skip_layers = _get_last_n_layer_ids(num_layers, keep_last_n) | _get_first_n_layer_ids(num_layers, keep_first_n)
    keep_ignore = _build_keep_layer_ignore_list(num_layers, keep_last_n, keep_first_n)

    shared_global_amax = _collect_shared_global_amax(
        input_path=input_path,
        safetensors_files=safetensors_files,
        device=device,
        skip_layers=skip_layers,
    )
    result_collector = ConversionResult()
    for filename in tqdm(safetensors_files, desc="Processing files"):
        process_file(
            input_path,
            output_path,
            filename,
            result_collector,
            device,
            skip_layers,
            shared_global_amax,
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    ignore_list = _augment_ignore_list(result_collector.modules_to_not_convert + keep_ignore)

    config_path = os.path.join(input_path, "config.json")
    if os.path.exists(config_path):
        cfg = json.load(open(config_path))
        _update_quantization_config(cfg, ignore_list)
        json.dump(cfg, open(os.path.join(output_path, "config.json"), "w"), indent=2)

    _write_hf_quant_config(output_path, ignore_list, input_path)

    index_dict = {
        "weight_map": result_collector.weight_map,
        "metadata": {"total_size": result_collector.total_size},
    }
    json.dump(index_dict, open(os.path.join(output_path, "model.safetensors.index.json"), "w"), indent=2)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, required=True, help="Path to HF safetensors model.")
    parser.add_argument("--save-dir", type=str, required=True, help="Path to save converted model.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device to run quantization on (default: cuda).",
    )
    parser.add_argument(
        "--keep-last-n",
        type=int,
        default=0,
        help="Keep the last N transformer layers unquantized (BF16/FP16).",
    )
    parser.add_argument(
        "--keep-first-n",
        type=int,
        default=0,
        help="Keep the first N transformer layers unquantized (BF16/FP16).",
    )
    args = parser.parse_args()

    if isinstance(args.device, str) and args.device.isdigit():
        device = torch.device(f"cuda:{args.device}")
    else:
        device = torch.device(args.device)

    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available, cannot run NVFP4 quantization.")
        if device.index is None:
            device = torch.device("cuda:0")
        torch.cuda.set_device(device)

    if not os.path.exists(args.save_dir):
        print(f"Creating directory {args.save_dir}")
        os.makedirs(args.save_dir)
    elif not os.path.isdir(args.save_dir):
        raise ValueError("The save_dir should be a directory.")

    convert_nvfp4(args.model_dir, args.save_dir, str(device), args.keep_last_n, args.keep_first_n)


if __name__ == "__main__":
    main()
