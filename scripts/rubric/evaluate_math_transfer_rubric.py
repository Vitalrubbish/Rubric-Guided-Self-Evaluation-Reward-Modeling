#!/usr/bin/env python3
"""Evaluate GSM8K-derived/generic/MATH-derived rubrics on MATH transfer outputs."""

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
    pred = row.get("predicted_answer")
    extraction = row.get("answer_extraction_method")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    number_count = len(re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text))
    operator_count = sum(text.count(op) for op in ["+", "-", "*", "/", "=", "^"])
    has_reasoning = len(lines) >= 2 and (number_count >= 2 or operator_count >= 2)
    has_symbolic_work = operator_count >= 2 or any(tok in text for tok in ["\\frac", "\\sqrt", "^", "x", "y"])

    scores = {}
    for dim in rubric.get("dimensions", []):
        dim_id = dim.get("id")
        if dim_id in {"final_answer_format"}:
            if pred is None:
                scores[dim_id] = 1
            elif extraction in {"hash_final", "boxed"}:
                scores[dim_id] = 5
            else:
                scores[dim_id] = 3
        elif dim_id in {"stepwise_reasoning_completeness", "reasoning_completeness"}:
            if has_reasoning and len(lines) >= 3:
                scores[dim_id] = 5
            elif has_reasoning:
                scores[dim_id] = 4
            elif pred is not None:
                scores[dim_id] = 2
            else:
                scores[dim_id] = 1
        elif dim_id in {"calculation_accuracy", "symbolic_transformation_accuracy", "answer_equivalence_and_simplification"}:
            if has_symbolic_work and number_count >= 2:
                scores[dim_id] = 4
            elif has_symbolic_work or number_count >= 2:
                scores[dim_id] = 3
            else:
                scores[dim_id] = 2
        elif dim_id in {"problem_modeling", "math_problem_modeling"}:
            problem_numbers = set(re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", row.get("problem") or ""))
            used_numbers = set(re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text))
            overlap = len(problem_numbers & used_numbers)
            if problem_numbers and overlap >= max(1, len(problem_numbers) // 2) and has_reasoning:
                scores[dim_id] = 4
            elif has_reasoning:
                scores[dim_id] = 3
            else:
                scores[dim_id] = 2
        else:
            scores[dim_id] = 4 if pred is not None and has_reasoning else 2
    return scores


def upper_bound_scores(row: dict, rubric: dict) -> dict[str, int]:
    if row.get("passed"):
        return {dim.get("id"): 5 for dim in rubric.get("dimensions", [])}
    pattern = row.get("error_pattern")
    scores = {dim.get("id"): 4 for dim in rubric.get("dimensions", [])}
    for dim in rubric.get("dimensions", []):
        if pattern in set(dim.get("linked_patterns") or []):
            scores[dim.get("id")] = 1 if pattern in {"missing_final_answer", "reasoning_truncation"} else 2
    return scores


def total_score(scores: dict[str, int]) -> float:
    return float(np.mean(list(scores.values()))) if scores else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labeled", type=Path, required=True)
    parser.add_argument("--rubric", type=Path, required=True)
    parser.add_argument("--scores-output", type=Path, required=True)
    parser.add_argument("--metrics-output", type=Path, required=True)
    args = parser.parse_args()

    rows = list(read_jsonl(args.labeled))
    rubric = json.loads(args.rubric.read_text(encoding="utf-8"))
    labels = [1 if row.get("passed") else 0 for row in rows]
    static_totals = []
    upper_totals = []
    static_preds = []
    upper_preds = []
    records = []
    linked_patterns = {p for dim in rubric.get("dimensions", []) for p in dim.get("linked_patterns") or []}
    failure_count = 0
    covered = 0

    for row in rows:
        if not row.get("passed"):
            failure_count += 1
            if row.get("error_pattern") in linked_patterns:
                covered += 1
        static_scores = static_dimension_scores(row, rubric)
        upper_scores = upper_bound_scores(row, rubric)
        static_total = total_score(static_scores)
        upper_total = total_score(upper_scores)
        static_totals.append(static_total)
        upper_totals.append(upper_total)
        static_preds.append(1 if static_total >= 4.0 else 0)
        upper_preds.append(1 if upper_total >= 4.0 else 0)
        records.append({
            "id": row.get("id"),
            "passed": row.get("passed"),
            "error_pattern": row.get("error_pattern"),
            "static_dimension_scores": static_scores,
            "static_total_score": static_total,
            "static_predicted_pass": static_preds[-1],
            "upper_bound_dimension_scores": upper_scores,
            "upper_bound_total_score": upper_total,
            "upper_bound_predicted_pass": upper_preds[-1],
        })

    metrics = {
        "labeled": str(args.labeled),
        "rubric": str(args.rubric),
        "rubric_name": rubric.get("name"),
        "total": len(rows),
        "pass_count": int(sum(labels)),
        "fail_count": int(len(labels) - sum(labels)),
        "failure_pattern_coverage": covered / failure_count if failure_count else None,
        "static_auc": safe_auc(labels, static_totals),
        "static_accuracy_at_4": float(accuracy_score(labels, static_preds)) if labels else None,
        "static_cohen_kappa_at_4": float(cohen_kappa_score(labels, static_preds)) if labels else None,
        "static_mean_pass_score": float(np.mean([s for s, y in zip(static_totals, labels) if y == 1])) if any(labels) else None,
        "static_mean_fail_score": float(np.mean([s for s, y in zip(static_totals, labels) if y == 0])) if any(y == 0 for y in labels) else None,
        "upper_bound_auc": safe_auc(labels, upper_totals),
        "upper_bound_accuracy_at_4": float(accuracy_score(labels, upper_preds)) if labels else None,
        "upper_bound_cohen_kappa_at_4": float(cohen_kappa_score(labels, upper_preds)) if labels else None,
        "note": "static_* uses response text only; upper_bound_* uses verifier-derived failure patterns and is diagnostic only.",
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
