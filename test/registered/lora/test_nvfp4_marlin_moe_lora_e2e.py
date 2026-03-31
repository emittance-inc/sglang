"""End-to-end correctness tests for MoE LoRA on the NVFP4 Marlin FP4 path.

Validates that MarlinRunnerCoreWithLoRA injects LoRA deltas correctly by
comparing the FP4 Marlin path against the BF16 Triton path (already validated)
using a real trained LoRA adapter.

Test strategy:
1. BF16 + LoRA (Triton path) produces coherent outputs → trusted baseline.
2. FP4 + LoRA (Marlin path) should produce similar coherent outputs.
3. FP4 without LoRA should produce different outputs than FP4 with LoRA.
"""

import multiprocessing as mp
import os
import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=600, suite="stage-b-test-1-gpu-large")

# Model paths
QWEN3_MOE_BASE = "Qwen/Qwen3-30B-A3B"
QWEN3_MOE_NVFP4 = "nvidia/Qwen3-30B-A3B-NVFP4"
# Real trained LoRA targeting MoE experts (gate_proj, up_proj, down_proj)
# and attention layers (q/k/v/o_proj), rank=8, alpha=16
QWEN3_MOE_LORA = "emil-nearai/qwen3-30b-a3b-forc-sft-lora"

TEST_PROMPTS = [
    "The capital of France is",
    "Write a haiku about the ocean:",
    "What is 2 + 2?",
    "Reverse the word 'hello':",
    "The speed of light is approximately",
]

MAX_NEW_TOKENS = 10
LORA_RANK = 32


def _check_requirements():
    from sglang.srt.utils import is_cuda

    if not is_cuda():
        return False
    from sglang.srt.layers.quantization.marlin_utils_fp4 import is_fp4_marlin_supported

    if not is_fp4_marlin_supported():
        return False
    return True


def _run_sglang(
    model_path,
    prompts,
    lora_adapter_path=None,
    max_new_tokens=MAX_NEW_TOKENS,
    env_overrides=None,
):
    """Run SGLang inference and return output strings + logprobs."""
    from sglang.test.runners import SRTRunner

    old_env = {}
    if env_overrides is None:
        env_overrides = {}
    # Disable torch.compile — Marlin FP4 JIT kernels aren't traceable
    env_overrides.setdefault("TORCHDYNAMO_DISABLE", "1")
    for k, v in env_overrides.items():
        old_env[k] = os.environ.get(k)
        os.environ[k] = v

    try:
        runner_kwargs = dict(
            model_path=model_path,
            torch_dtype=torch.bfloat16,
            model_type="generation",
            tp_size=1,
            trust_remote_code=True,
            disable_radix_cache=True,
            disable_cuda_graph=True,
            attention_backend="flashinfer",
            mem_fraction_static=0.80,
        )

        if lora_adapter_path:
            runner_kwargs["lora_paths"] = [lora_adapter_path]
            runner_kwargs["max_loras_per_batch"] = 1
            runner_kwargs["max_lora_rank"] = LORA_RANK

        with SRTRunner(**runner_kwargs) as runner:
            lora_paths_per_prompt = (
                [lora_adapter_path] * len(prompts) if lora_adapter_path else None
            )
            outputs = runner.forward(
                prompts,
                max_new_tokens=max_new_tokens,
                lora_paths=lora_paths_per_prompt,
            )

        return {
            "output_strs": outputs.output_strs,
            "top_output_logprobs": outputs.top_output_logprobs,
        }
    finally:
        for k, v in old_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class TestNvfp4MarlinMoELoRAE2E(CustomTestCase):
    """End-to-end correctness tests for MoE LoRA on NVFP4 Marlin path.

    Uses a real trained LoRA adapter to validate that the Marlin FP4
    staged pipeline produces coherent, correct outputs.
    """

    @classmethod
    def setUpClass(cls):
        if not _check_requirements():
            raise unittest.SkipTest(
                "Requirements not met (CUDA unavailable or SM < 75)"
            )

    def test_fp4_lora_changes_output(self):
        """LoRA must actually change the NVFP4 model's output."""
        no_lora_results = _run_sglang(
            model_path=QWEN3_MOE_NVFP4,
            prompts=TEST_PROMPTS,
            env_overrides={"SGLANG_FORCE_NVFP4_MARLIN": "true"},
        )

        lora_results = _run_sglang(
            model_path=QWEN3_MOE_NVFP4,
            prompts=TEST_PROMPTS,
            lora_adapter_path=QWEN3_MOE_LORA,
            env_overrides={"SGLANG_FORCE_NVFP4_MARLIN": "true"},
        )

        num_different = 0
        for i in range(len(TEST_PROMPTS)):
            no_lora_str = no_lora_results["output_strs"][i].strip()
            lora_str = lora_results["output_strs"][i].strip()
            if no_lora_str != lora_str:
                num_different += 1
            print(
                f"Prompt {i}: same={no_lora_str == lora_str} "
                f"no_lora='{no_lora_str}' lora='{lora_str}'"
            )

        self.assertGreater(
            num_different,
            0,
            "LoRA should change at least some outputs, but all outputs are identical. "
            "This suggests LoRA deltas are not being applied.",
        )

    def test_fp4_marlin_lora_vs_bf16_lora(self):
        """FP4 Marlin + LoRA should produce similar results to BF16 + LoRA.

        The BF16 Triton path is already validated by existing MoE LoRA tests.
        This test checks that the Marlin FP4 staged pipeline produces
        comparable outputs with a real trained LoRA adapter.
        """
        # Run A: BF16 base + LoRA (trusted baseline)
        bf16_results = _run_sglang(
            model_path=QWEN3_MOE_BASE,
            prompts=TEST_PROMPTS,
            lora_adapter_path=QWEN3_MOE_LORA,
        )

        # Run B: NVFP4 base + LoRA (Marlin path, code under test)
        fp4_results = _run_sglang(
            model_path=QWEN3_MOE_NVFP4,
            prompts=TEST_PROMPTS,
            lora_adapter_path=QWEN3_MOE_LORA,
            env_overrides={"SGLANG_FORCE_NVFP4_MARLIN": "true"},
        )

        from sglang.test.test_utils import calculate_rouge_l

        bf16_strs = [s.strip() for s in bf16_results["output_strs"]]
        fp4_strs = [s.strip() for s in fp4_results["output_strs"]]

        rouge_l_scores = calculate_rouge_l(bf16_strs, fp4_strs)
        for i, prompt in enumerate(TEST_PROMPTS):
            score = rouge_l_scores[i]
            print(
                f"Prompt {i}: ROUGE-L={score:.3f} "
                f"bf16='{bf16_strs[i]}' fp4='{fp4_strs[i]}'"
            )

        mean_rouge = sum(rouge_l_scores) / len(rouge_l_scores)
        print(f"Mean ROUGE-L: {mean_rouge:.3f}")

        # With a real LoRA, both paths should produce coherent text.
        # FP4 quantization introduces error but outputs should be similar.
        self.assertGreaterEqual(
            mean_rouge,
            0.3,
            f"Mean ROUGE-L too low ({mean_rouge:.3f}). "
            "FP4 Marlin path may not be applying LoRA correctly.",
        )

        # Log logprob diffs for debugging
        for i in range(len(TEST_PROMPTS)):
            bf16_lps = bf16_results["top_output_logprobs"][i]
            fp4_lps = fp4_results["top_output_logprobs"][i]

            bf16_lps_flat = [
                float(t[0]) if isinstance(t, list) else float(t)
                for t in bf16_lps
            ]
            fp4_lps_flat = [
                float(t[0]) if isinstance(t, list) else float(t)
                for t in fp4_lps
            ]

            min_len = min(len(bf16_lps_flat), len(fp4_lps_flat))
            if min_len > 0:
                diffs = [
                    abs(bf16_lps_flat[t] - fp4_lps_flat[t]) for t in range(min_len)
                ]
                mean_diff = sum(diffs) / len(diffs)
                print(
                    f"Prompt {i}: mean logprob diff={mean_diff:.6f}, "
                    f"max logprob diff={max(diffs):.6f}"
                )


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    try:
        unittest.main(warnings="ignore", verbosity=2)
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
