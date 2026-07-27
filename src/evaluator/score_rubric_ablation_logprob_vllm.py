#!/usr/bin/env python3
"""Score the rubric-ablation judge dataset with vLLM prompt logprobs.

vLLM variant of score_rubric_ablation_logprob.py. Works for any local
model vLLM can load (bf16, AWQ, etc.), including models too large for the
HF transformers single-GPU path. Scoring protocol is identical: for each
row, compute the summed logprob of the forced-choice completions
(" PASS\\n" vs " FAIL\\n") after the verdict prefix, via teacher-forced
prompt_logprobs on context+completion sequences.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

from src.evaluator.score_generative_self_eval_logprob import (
    metrics_at_threshold,
    select_thresholds,
    sigmoid,
)
from src.evaluator.score_rubric_ablation_logprob import (
    paired_bootstrap_auc_diff,
    probability_metrics,
    read_jsonl,
    write_json,
    write_jsonl,
)


def teacher_forced_logprob(
    llm: Any,
    tokenizer: Any,
    contexts: list[str],
    completions: list[str],
    batch_size: int,
) -> list[float]:
    """Sum logprobs of completion tokens conditioned on context, via prompt_logprobs."""
    from vllm import SamplingParams

    sums: list[float] = []
    params = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=1)
    for start in range(0, len(contexts), batch_size):
        batch_contexts = contexts[start : start + batch_size]
        batch_completions = completions[start : start + batch_size]
        prompts = [c + t for c, t in zip(batch_contexts, batch_completions)]
        outputs = llm.generate(prompts, params)
        for context, completion, output in zip(batch_contexts, batch_completions, outputs):
            context_len = len(tokenizer(context, add_special_tokens=False).input_ids)
            total_len = len(tokenizer(context + completion, add_special_tokens=False).input_ids)
            prompt_logprobs = output.prompt_logprobs
            if prompt_logprobs is None or len(prompt_logprobs) != total_len:
                raise ValueError(f"prompt_logprobs length mismatch: {None if prompt_logprobs is None else len(prompt_logprobs)} vs {total_len}")
            total = 0.0
            for position in range(context_len, total_len):
                entry = prompt_logprobs[position]
                token_id = output.prompt_token_ids[position]
                if token_id not in entry:
                    raise ValueError(f"token {token_id} missing from prompt_logprobs at position {position}")
                total += float(entry[token_id].logprob)
            sums.append(total)
    return sums


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--scores-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--verdict-prefix", default="Verdict:")
    parser.add_argument("--pass-completion", default=" PASS\n")
    parser.add_argument("--fail-completion", default=" FAIL\n")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-overacceptance", type=float, default=0.25)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--quantization", default=None)
    parser.add_argument("--enforce-eager", action="store_true")
    args = parser.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM

    rows = read_jsonl(args.input)
    if not rows:
        raise SystemExit("no judge rows found")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    llm_kwargs: dict[str, Any] = {
        "model": args.model,
        "trust_remote_code": True,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enforce_eager": args.enforce_eager,
    }
    if args.quantization:
        llm_kwargs["quantization"] = args.quantization
    llm = LLM(**llm_kwargs)

    contexts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": row["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        + args.verdict_prefix
        for row in rows
    ]
    pass_sums = teacher_forced_logprob(llm, tokenizer, contexts, [args.pass_completion] * len(rows), args.batch_size)
    fail_sums = teacher_forced_logprob(llm, tokenizer, contexts, [args.fail_completion] * len(rows), args.batch_size)

    predictions: list[dict[str, Any]] = []
    for row, pass_sum, fail_sum in zip(rows, pass_sums, fail_sums):
        margin = float(pass_sum) - float(fail_sum)
        predictions.append(
            {
                "id": row.get("id"),
                "response_id": row.get("response_id"),
                "arm": row.get("arm"),
                "split": row.get("split"),
                "task_type": row.get("task_type"),
                "gold_passed": bool(row.get("gold_passed")),
                "gold_failure_type": row.get("gold_failure_type"),
                "pass_logprob_margin": margin,
                "pass_probability": sigmoid(margin),
                "predicted_pass_at_zero": margin >= 0.0,
            }
        )
    write_jsonl(args.scores_output, predictions)

    arms = sorted({str(row["arm"]) for row in predictions})
    arm_reports: dict[str, Any] = {}
    for arm in arms:
        arm_rows = [row for row in predictions if row["arm"] == arm]
        validation_rows = [row for row in arm_rows if row["split"] == "validation"]
        test_rows = [row for row in arm_rows if row["split"] == "test"]
        sweep = select_thresholds(validation_rows, args.max_overacceptance)
        safe = sweep["best_with_overacceptance_le_max"]
        selected = safe or sweep["best_balanced_accuracy"]
        threshold = float(selected["threshold"])
        arm_reports[arm] = {
            "rows": len(arm_rows),
            "gold_counts": dict(Counter("pass" if row["gold_passed"] else "fail" for row in arm_rows)),
            "probability_metrics_all": probability_metrics(arm_rows),
            "zero_threshold_all": metrics_at_threshold(arm_rows, 0.0),
            "validation_probability_metrics": probability_metrics(validation_rows),
            "test_probability_metrics": probability_metrics(test_rows),
            "selected_threshold_policy": "validation best balanced accuracy among thresholds satisfying overacceptance <= max; fallback to best balanced accuracy",
            "selected_threshold": threshold,
            "validation_selected_threshold": metrics_at_threshold(validation_rows, threshold),
            "test_selected_threshold": metrics_at_threshold(test_rows, threshold),
        }

    comparisons = []
    for left, right in (
        ("auto_rubric", "no_rubric"),
        ("auto_rubric", "random_rubric"),
        ("no_rubric", "random_rubric"),
        ("short_rubric", "no_rubric"),
        ("short_rubric", "auto_rubric"),
    ):
        if left not in arms or right not in arms:
            continue
        left_rows = [row for row in predictions if row["arm"] == left]
        right_rows = [row for row in predictions if row["arm"] == right]
        comparisons.append(
            {
                "a": left,
                "b": right,
                "auc_a": arm_reports[left]["probability_metrics_all"]["auc"],
                "auc_b": arm_reports[right]["probability_metrics_all"]["auc"],
                "paired_bootstrap": paired_bootstrap_auc_diff(left_rows, right_rows, args.bootstrap_iterations, args.seed),
            }
        )

    summary = {
        "input": str(args.input),
        "scores_file": str(args.scores_output),
        "model": args.model,
        "quantization": args.quantization,
        "judge": "base model via chat template; forced-choice verdict logprob margin (vLLM prompt_logprobs)",
        "rows_scored": len(predictions),
        "arms": arm_reports,
        "arm_comparisons": comparisons,
    }
    write_json(args.summary_output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
