#!/usr/bin/env python3
"""Score the rubric-ablation judge dataset with forced-choice verdict logprobs.

Adapted from score_generative_self_eval_logprob.py with three differences:

- the judge is the base instruct model (no adapter), addressed through its
  chat template;
- rows carry an ``arm`` field (no_rubric / auto_rubric / random_rubric) and
  metrics are reported per arm;
- arm differences use a paired bootstrap over response ids, since every
  response is judged under every arm.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from sklearn.metrics import accuracy_score, cohen_kappa_score, roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.evaluator.score_generative_self_eval_logprob import (
    EncodedChoice,
    batches,
    encode_choice,
    metrics_at_threshold,
    select_thresholds,
    sigmoid,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
    return rows


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def probability_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    y_true = np.array([1 if row["gold_passed"] else 0 for row in rows], dtype=int)
    scores = np.array([float(row["pass_logprob_margin"]) for row in rows], dtype=float)
    probs = np.array([float(row["pass_probability"]) for row in rows], dtype=float)
    auc = float(roc_auc_score(y_true, scores)) if len(set(y_true.tolist())) > 1 else 0.0
    eps = 1e-12
    brier = float(np.mean((probs - y_true) ** 2))
    log_loss = float(-np.mean(y_true * np.log(np.clip(probs, eps, 1.0)) + (1 - y_true) * np.log(np.clip(1 - probs, eps, 1.0))))
    return {"auc": auc, "brier": brier, "log_loss": log_loss}


def paired_bootstrap_auc_diff(
    rows_a: list[dict[str, Any]],
    rows_b: list[dict[str, Any]],
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    """Bootstrap CI for AUC(a) - AUC(b) resampling response ids jointly."""
    by_id_a = {str(row["response_id"]): row for row in rows_a}
    by_id_b = {str(row["response_id"]): row for row in rows_b}
    common = sorted(set(by_id_a) & set(by_id_b))
    rng = random.Random(seed)
    diffs: list[float] = []
    for _ in range(iterations):
        sample = [common[rng.randrange(len(common))] for _ in common]
        y_a = [1 if by_id_a[i]["gold_passed"] else 0 for i in sample]
        s_a = [float(by_id_a[i]["pass_logprob_margin"]) for i in sample]
        y_b = [1 if by_id_b[i]["gold_passed"] else 0 for i in sample]
        s_b = [float(by_id_b[i]["pass_logprob_margin"]) for i in sample]
        if len(set(y_a)) <= 1 or len(set(y_b)) <= 1:
            continue
        diffs.append(float(roc_auc_score(y_a, s_a)) - float(roc_auc_score(y_b, s_b)))
    diffs.sort()
    if not diffs:
        return {
            "common_responses": len(common),
            "bootstrap_iterations": 0,
            "auc_diff_ci95": [float("nan"), float("nan")],
            "prob_a_not_better_than_b": float("nan"),
            "note": "all resamples were single-class; sample too small for bootstrap",
        }
    lo = diffs[int(0.025 * len(diffs))]
    hi = diffs[min(len(diffs) - 1, int(0.975 * len(diffs)))]
    frac_le_zero = sum(1 for d in diffs if d <= 0.0) / len(diffs)
    return {
        "common_responses": len(common),
        "bootstrap_iterations": len(diffs),
        "auc_diff_ci95": [lo, hi],
        "prob_a_not_better_than_b": round(frac_le_zero, 4),
    }


@torch.inference_mode()
def score_choices(
    model: Any,
    tokenizer: Any,
    encoded: list[EncodedChoice],
    batch_size: int,
) -> dict[tuple[int, str], dict[str, float | int]]:
    pad_id = tokenizer.pad_token_id
    device = next(model.parameters()).device
    scores: dict[tuple[int, str], dict[str, float | int]] = {}
    model.eval()
    for batch in batches(encoded, batch_size):
        max_len = max(len(item.input_ids) for item in batch)
        input_ids = []
        attention_mask = []
        target_mask = []
        for item in batch:
            pad_len = max_len - len(item.input_ids)
            input_ids.append(item.input_ids + [pad_id] * pad_len)
            attention_mask.append([1] * len(item.input_ids) + [0] * pad_len)
            target_mask.append(item.target_mask + [0] * pad_len)
        input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device)
        attention_tensor = torch.tensor(attention_mask, dtype=torch.long, device=device)
        target_tensor = torch.tensor(target_mask, dtype=torch.bool, device=device)
        output = model(input_ids=input_tensor, attention_mask=attention_tensor)
        logits = output.logits[:, :-1, :]
        labels = input_tensor[:, 1:]
        shifted_target = target_tensor[:, 1:]
        token_logprobs = torch.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        token_logprobs = token_logprobs * shifted_target
        sums = token_logprobs.sum(dim=1).detach().cpu().tolist()
        counts = shifted_target.sum(dim=1).detach().cpu().tolist()
        for item, logprob_sum, token_count in zip(batch, sums, counts):
            count = int(token_count)
            scores[(item.row_index, item.choice)] = {
                "logprob_sum": float(logprob_sum),
                "token_count": count,
                "logprob_avg": float(logprob_sum / count) if count else float("-inf"),
            }
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--scores-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--verdict-prefix", default="Verdict:")
    parser.add_argument("--pass-completion", default=" PASS\n")
    parser.add_argument("--fail-completion", default=" FAIL\n")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-overacceptance", type=float, default=0.25)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float16", "float32"), default="bfloat16")
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    if not rows:
        raise SystemExit("no judge rows found")

    dtype_map = {
        "auto": "auto",
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype_map[args.dtype],
        trust_remote_code=True,
    )
    if torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()

    encoded: list[EncodedChoice] = []
    for index, row in enumerate(rows):
        context = (
            tokenizer.apply_chat_template(
                [{"role": "user", "content": row["prompt"]}],
                tokenize=False,
                add_generation_prompt=True,
            )
            + args.verdict_prefix
        )
        encoded.append(encode_choice(tokenizer, index, "pass", context, args.pass_completion, args.max_model_len))
        encoded.append(encode_choice(tokenizer, index, "fail", context, args.fail_completion, args.max_model_len))

    choice_scores = score_choices(model, tokenizer, encoded, args.batch_size)
    predictions: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        pass_score = choice_scores[(index, "pass")]
        fail_score = choice_scores[(index, "fail")]
        margin = float(pass_score["logprob_sum"]) - float(fail_score["logprob_sum"])
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
        "judge": "base model via chat template; forced-choice verdict logprob margin",
        "verdict_prefix": args.verdict_prefix,
        "pass_completion": args.pass_completion,
        "fail_completion": args.fail_completion,
        "rows_scored": len(predictions),
        "arms": arm_reports,
        "arm_comparisons": comparisons,
    }
    write_json(args.summary_output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
