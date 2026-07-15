#!/usr/bin/env python3
"""Score Method 1 generative self-evaluation with forced-choice verdict logprobs."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from peft import PeftModel
from sklearn.metrics import accuracy_score, cohen_kappa_score, roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer


IGNORE_INDEX = -100


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


def format_prompt(prompt: Any, prompt_format: str) -> str:
    text = str(prompt)
    if prompt_format == "as-is":
        return text
    if prompt_format == "raw":
        return text.strip() + "\n\n"
    raise ValueError(f"unsupported prompt format: {prompt_format}")


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def gold_passed(row: dict[str, Any]) -> bool:
    metadata = row.get("metadata") or {}
    if "passed" in metadata:
        return bool(metadata["passed"])
    completion = str(row.get("completion") or "")
    if "Verdict: PASS" in completion:
        return True
    if "Verdict: FAIL" in completion:
        return False
    raise ValueError(f"cannot infer gold label for row {row.get('id')}")


def pct(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def confusion_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    auc = float(roc_auc_score(y_true, y_pred)) if len(set(y_true.tolist())) > 1 else 0.0
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": (pct(tp, tp + fn) + pct(tn, tn + fp)) / 2,
        "kappa": float(cohen_kappa_score(y_true, y_pred)) if len(set(y_pred.tolist())) > 1 else 0.0,
        "auc_binary_predictions": auc,
        "predicted_pass_rate": pct(tp + fp, len(y_true)),
        "true_pass_rate": pct(int(y_true.sum()), len(y_true)),
        "overacceptance_rate": pct(fp, fp + tn),
        "false_rejection_rate": pct(fn, fn + tp),
        "precision_pass": pct(tp, tp + fp),
        "recall_pass": pct(tp, tp + fn),
        "specificity_fail": pct(tn, tn + fp),
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def probability_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    y_true = np.array([1 if row["gold_passed"] else 0 for row in rows], dtype=int)
    scores = np.array([float(row["pass_logprob_margin"]) for row in rows], dtype=float)
    probs = np.array([float(row["pass_probability"]) for row in rows], dtype=float)
    if len(set(y_true.tolist())) <= 1:
        auc = 0.0
    else:
        auc = float(roc_auc_score(y_true, scores))
    eps = 1e-12
    brier = float(np.mean((probs - y_true) ** 2))
    log_loss = float(-np.mean(y_true * np.log(np.clip(probs, eps, 1.0)) + (1 - y_true) * np.log(np.clip(1 - probs, eps, 1.0))))
    return {"auc": auc, "brier": brier, "log_loss": log_loss}


def metrics_at_threshold(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    y_true = np.array([1 if row["gold_passed"] else 0 for row in rows], dtype=int)
    y_pred = np.array([1 if float(row["pass_logprob_margin"]) >= threshold else 0 for row in rows], dtype=int)
    result = confusion_metrics(y_true, y_pred)
    result["threshold"] = float(threshold)
    return result


def threshold_candidates(rows: list[dict[str, Any]]) -> list[float]:
    scores = sorted({float(row["pass_logprob_margin"]) for row in rows})
    if not scores:
        return [0.0]
    candidates = [scores[0] - 1e-6, scores[-1] + 1e-6, 0.0]
    candidates.extend(scores)
    candidates.extend((left + right) / 2 for left, right in zip(scores, scores[1:]))
    return sorted(set(candidates))


def select_thresholds(rows: list[dict[str, Any]], max_overacceptance: float) -> dict[str, Any]:
    candidates = [metrics_at_threshold(rows, threshold) for threshold in threshold_candidates(rows)]
    best_balanced = max(candidates, key=lambda item: (item["balanced_accuracy"], item["accuracy"], -abs(item["threshold"])))
    best_accuracy = max(candidates, key=lambda item: (item["accuracy"], item["balanced_accuracy"], -abs(item["threshold"])))
    safe = [item for item in candidates if item["overacceptance_rate"] <= max_overacceptance]
    best_safe = max(safe, key=lambda item: (item["balanced_accuracy"], item["accuracy"], item["recall_pass"])) if safe else None
    selected_thresholds = []
    for threshold in [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]:
        selected_thresholds.append(metrics_at_threshold(rows, threshold))
    return {
        "best_balanced_accuracy": best_balanced,
        "best_accuracy": best_accuracy,
        "best_with_overacceptance_le_max": best_safe,
        "selected_thresholds": selected_thresholds,
    }


@dataclass
class EncodedChoice:
    row_index: int
    choice: str
    input_ids: list[int]
    target_mask: list[int]


def encode_choice(
    tokenizer: Any,
    row_index: int,
    choice: str,
    context: str,
    completion: str,
    max_model_len: int,
) -> EncodedChoice:
    context_ids = tokenizer(context, add_special_tokens=False).input_ids
    completion_ids = tokenizer(completion, add_special_tokens=False).input_ids
    input_ids = context_ids + completion_ids
    if len(input_ids) > max_model_len:
        raise ValueError(f"encoded {choice} sequence exceeds max_model_len={max_model_len}: {len(input_ids)}")
    target_mask = [0] * len(context_ids) + [1] * len(completion_ids)
    return EncodedChoice(row_index=row_index, choice=choice, input_ids=input_ids, target_mask=target_mask)


def batches(values: list[Any], batch_size: int) -> Iterable[list[Any]]:
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


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


def render_report(summary: dict[str, Any]) -> str:
    selected = summary["test_selected_threshold"]
    oracle = summary["test_oracle_threshold_sweep"]["best_with_overacceptance_le_max"]
    lines = [
        "# Generative Self-Evaluator Logprob Gate",
        "",
        "## Summary",
        "",
        f"- Rows scored: `{summary['rows_scored']}`",
        f"- Adapter: `{summary['adapter']}`",
        f"- Verdict prefix: `{summary['verdict_prefix']}`",
        f"- Pass completion: `{summary['pass_completion']}`",
        f"- Fail completion: `{summary['fail_completion']}`",
        f"- Validation selected threshold: `{summary['selected_threshold']:.6f}`",
        f"- Test selected balanced accuracy: `{selected['balanced_accuracy']:.4f}`",
        f"- Test selected overacceptance: `{selected['overacceptance_rate']:.4f}`",
        f"- Test selected false rejection: `{selected['false_rejection_rate']:.4f}`",
        f"- Test AUC: `{summary['test_probability_metrics']['auc']:.4f}`",
        f"- Canary passed: `{summary['canary_passed']}`",
        "",
        "## Gates",
        "",
    ]
    for name, passed in summary["gates"].items():
        lines.append(f"- [{'x' if passed else ' '}] {name}")
    lines.extend(
        [
            "",
            "## Oracle Diagnostic",
            "",
            f"- Test oracle safe threshold balanced accuracy: `{oracle['balanced_accuracy']:.4f}`" if oracle else "- No test threshold satisfies overacceptance constraint.",
            f"- Test oracle safe overacceptance: `{oracle['overacceptance_rate']:.4f}`" if oracle else "",
            "",
            "## Full Summary",
            "",
            "```json",
            json.dumps(summary, ensure_ascii=False, indent=2),
            "```",
            "",
        ]
    )
    return "\n".join(line for line in lines if line != "")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--scores-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["validation", "test"])
    parser.add_argument("--task-type", default="judge_single")
    parser.add_argument("--prompt-format", choices=("as-is", "raw"), default="raw")
    parser.add_argument("--verdict-prefix", default="Verdict:")
    parser.add_argument("--pass-completion", default=" PASS\n")
    parser.add_argument("--fail-completion", default=" FAIL\n")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-overacceptance", type=float, default=0.25)
    parser.add_argument("--min-balanced-accuracy", type=float, default=0.70)
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float16", "float32"), default="bfloat16")
    args = parser.parse_args()

    if not (args.adapter / "adapter_model.safetensors").is_file():
        raise FileNotFoundError(f"completed adapter not found: {args.adapter}")

    rows = [
        row
        for row in read_jsonl(args.input)
        if str(row.get("split")) in set(args.splits) and str(row.get("task_type")) == args.task_type
    ]
    if not rows:
        raise SystemExit("no rows selected for logprob scoring")

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
    model = PeftModel.from_pretrained(model, args.adapter)
    if torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()

    encoded: list[EncodedChoice] = []
    for index, row in enumerate(rows):
        context = format_prompt(row["prompt"], args.prompt_format) + args.verdict_prefix
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
                "split": row.get("split"),
                "task_type": row.get("task_type"),
                "gold_passed": gold_passed(row),
                "pass_logprob_sum": pass_score["logprob_sum"],
                "fail_logprob_sum": fail_score["logprob_sum"],
                "pass_logprob_avg": pass_score["logprob_avg"],
                "fail_logprob_avg": fail_score["logprob_avg"],
                "pass_token_count": pass_score["token_count"],
                "fail_token_count": fail_score["token_count"],
                "pass_logprob_margin": margin,
                "pass_probability": sigmoid(margin),
                "predicted_pass_at_zero": margin >= 0.0,
                "metadata": row.get("metadata"),
            }
        )

    by_split = {split: [row for row in predictions if str(row.get("split")) == split] for split in args.splits}
    validation_rows = by_split.get("validation") or []
    test_rows = by_split.get("test") or []
    if not validation_rows or not test_rows:
        raise SystemExit("logprob gate requires both validation and test splits")

    validation_sweep = select_thresholds(validation_rows, args.max_overacceptance)
    safe_validation = validation_sweep["best_with_overacceptance_le_max"]
    selected = safe_validation or validation_sweep["best_balanced_accuracy"]
    selected_threshold = float(selected["threshold"])
    summary = {
        "rows_scored": len(predictions),
        "model": args.model,
        "adapter": str(args.adapter),
        "input": str(args.input),
        "scores_file": str(args.scores_output),
        "splits": args.splits,
        "prompt_format": args.prompt_format,
        "verdict_prefix": args.verdict_prefix,
        "pass_completion": args.pass_completion,
        "fail_completion": args.fail_completion,
        "split_counts": dict(Counter(str(row.get("split")) for row in predictions)),
        "gold_counts": dict(Counter("pass" if row["gold_passed"] else "fail" for row in predictions)),
        "token_counts": {
            "pass": dict(Counter(int(row["pass_token_count"]) for row in predictions)),
            "fail": dict(Counter(int(row["fail_token_count"]) for row in predictions)),
        },
        "validation_probability_metrics": probability_metrics(validation_rows),
        "test_probability_metrics": probability_metrics(test_rows),
        "validation_zero_threshold": metrics_at_threshold(validation_rows, 0.0),
        "test_zero_threshold": metrics_at_threshold(test_rows, 0.0),
        "validation_threshold_sweep": validation_sweep,
        "selected_threshold_policy": "validation best balanced accuracy among thresholds satisfying overacceptance <= max; fallback to validation best balanced accuracy",
        "selected_threshold": selected_threshold,
        "validation_selected_threshold": metrics_at_threshold(validation_rows, selected_threshold),
        "test_selected_threshold": metrics_at_threshold(test_rows, selected_threshold),
        "test_oracle_threshold_sweep": select_thresholds(test_rows, args.max_overacceptance),
    }
    test_selected = summary["test_selected_threshold"]
    summary["gates"] = {
        "test_balanced_accuracy_ge_min": test_selected["balanced_accuracy"] >= args.min_balanced_accuracy,
        "test_overacceptance_le_max": test_selected["overacceptance_rate"] <= args.max_overacceptance,
        "validation_safe_threshold_exists": safe_validation is not None,
    }
    summary["canary_passed"] = all(summary["gates"].values())
    summary["policy"] = "forced-choice verdict logprob scoring; threshold selected only on validation"

    write_jsonl(args.scores_output, predictions)
    write_json(args.summary_output, summary)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(render_report(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
