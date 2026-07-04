#!/usr/bin/env python3
"""Evaluate GSM8K rubric discriminability against exact-answer verifier labels."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, roc_auc_score


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def safe_auc(labels: list[int], scores: list[float]) -> float | None:
    if len(set(labels)) < 2:
        return None
    return float(roc_auc_score(labels, scores))


def static_dimension_scores(row: dict, rubric: dict) -> dict[str, int]:
    text = row.get("generated_answer") or ""
    predicted = row.get("predicted_answer")
    extraction = row.get("answer_extraction_method")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    number_count = len(re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text))
    operator_count = sum(text.count(op) for op in ["+", "-", "*", "/", "="])
    has_equation = operator_count >= 2 or bool(re.search(r"\d\s*(?:\\times|\*)\s*\d", text))
    has_reasoning = len(lines) >= 2 and number_count >= 2

    scores = {}
    for dim in rubric.get("dimensions", []):
        dim_id = dim["id"]
        if dim_id == "final_answer_format":
            if predicted is None:
                scores[dim_id] = 1
            elif extraction == "hash_final" and "####" in text:
                scores[dim_id] = 5
            elif extraction in {"phrase_final", "last_number"}:
                scores[dim_id] = 3
            else:
                scores[dim_id] = 2
        elif dim_id == "stepwise_reasoning_completeness":
            if has_reasoning and len(lines) >= 3:
                scores[dim_id] = 5
            elif has_reasoning:
                scores[dim_id] = 4
            elif predicted is not None:
                scores[dim_id] = 2
            else:
                scores[dim_id] = 1
        elif dim_id == "calculation_accuracy":
            if has_equation and number_count >= 4:
                scores[dim_id] = 4
            elif number_count >= 2:
                scores[dim_id] = 3
            else:
                scores[dim_id] = 2
        elif dim_id == "problem_modeling":
            question_numbers = set(re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", row.get("question") or ""))
            used_numbers = set(re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text))
            overlap = len(question_numbers & used_numbers)
            if question_numbers and overlap >= max(1, len(question_numbers) // 2) and has_reasoning:
                scores[dim_id] = 4
            elif has_reasoning:
                scores[dim_id] = 3
            else:
                scores[dim_id] = 2
        else:
            scores[dim_id] = 4 if predicted is not None and has_reasoning else 2
    return scores


def upper_bound_dimension_scores(row: dict, pattern: str | None, rubric: dict) -> dict[str, int]:
    if row.get("passed"):
        return {dim["id"]: 5 for dim in rubric.get("dimensions", [])}

    scores = {dim["id"]: 4 for dim in rubric.get("dimensions", [])}
    for dim in rubric.get("dimensions", []):
        dim_id = dim["id"]
        linked = set(dim.get("linked_patterns") or [])
        if pattern in linked:
            if pattern in {"missing_final_answer", "reasoning_truncation"}:
                scores[dim_id] = 1
            elif pattern in {"final_format_violation", "ambiguous_final_answer"}:
                scores[dim_id] = 2
            else:
                scores[dim_id] = 2
    return scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labeled", type=Path, required=True)
    parser.add_argument("--failures", type=Path, required=True)
    parser.add_argument("--rubric", type=Path, required=True)
    parser.add_argument("--scores-output", type=Path, default=Path("data/rubrics/gsm8k_auto_rubric_scores.jsonl"))
    parser.add_argument("--metrics-output", type=Path, default=Path("data/rubrics/gsm8k_auto_rubric_metrics.json"))
    args = parser.parse_args()

    rows = list(read_jsonl(args.labeled))
    failures = {row["id"]: row.get("error_pattern") for row in read_jsonl(args.failures)}
    rubric = json.loads(args.rubric.read_text(encoding="utf-8"))
    labels = [1 if row.get("passed") else 0 for row in rows]
    static_totals = []
    static_predictions = []
    upper_totals = []
    upper_predictions = []
    records = []
    linked_patterns = {p for dim in rubric.get("dimensions", []) for p in dim.get("linked_patterns") or []}
    covered = 0
    failure_count = 0

    for row in rows:
        pattern = failures.get(row["id"])
        if not row.get("passed"):
            failure_count += 1
            if pattern in linked_patterns:
                covered += 1
        static_scores = static_dimension_scores(row, rubric)
        upper_scores = upper_bound_dimension_scores(row, pattern, rubric)
        static_total = float(np.mean(list(static_scores.values()))) if static_scores else 0.0
        upper_total = float(np.mean(list(upper_scores.values()))) if upper_scores else 0.0
        static_totals.append(static_total)
        upper_totals.append(upper_total)
        static_predictions.append(1 if static_total >= 4.0 else 0)
        upper_predictions.append(1 if upper_total >= 4.0 else 0)
        records.append({
            "id": row["id"],
            "passed": row.get("passed"),
            "error_pattern": pattern,
            "static_dimension_scores": static_scores,
            "static_total_score": static_total,
            "static_predicted_pass": static_predictions[-1],
            "upper_bound_dimension_scores": upper_scores,
            "upper_bound_total_score": upper_total,
            "upper_bound_predicted_pass": upper_predictions[-1],
        })

    metrics = {
        "labeled": str(args.labeled),
        "rubric": str(args.rubric),
        "total": len(rows),
        "pass_count": int(sum(labels)),
        "fail_count": int(len(labels) - sum(labels)),
        "failure_pattern_coverage": covered / failure_count if failure_count else None,
        "static_auc": safe_auc(labels, static_totals),
        "static_accuracy_at_4": float(accuracy_score(labels, static_predictions)) if labels else None,
        "static_cohen_kappa_at_4": float(cohen_kappa_score(labels, static_predictions)) if labels else None,
        "static_mean_pass_score": float(np.mean([s for s, y in zip(static_totals, labels) if y == 1])) if any(labels) else None,
        "static_mean_fail_score": float(np.mean([s for s, y in zip(static_totals, labels) if y == 0])) if any(y == 0 for y in labels) else None,
        "upper_bound_auc": safe_auc(labels, upper_totals),
        "upper_bound_accuracy_at_4": float(accuracy_score(labels, upper_predictions)) if labels else None,
        "upper_bound_cohen_kappa_at_4": float(cohen_kappa_score(labels, upper_predictions)) if labels else None,
        "upper_bound_mean_pass_score": float(np.mean([s for s, y in zip(upper_totals, labels) if y == 1])) if any(labels) else None,
        "upper_bound_mean_fail_score": float(np.mean([s for s, y in zip(upper_totals, labels) if y == 0])) if any(y == 0 for y in labels) else None,
        "note": "static_* uses response text only; upper_bound_* uses exact-verifier failure patterns and is not a deployable self-evaluator.",
    }

    args.scores_output.parent.mkdir(parents=True, exist_ok=True)
    with args.scores_output.open("w", encoding="utf-8") as out:
        for row in records:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
