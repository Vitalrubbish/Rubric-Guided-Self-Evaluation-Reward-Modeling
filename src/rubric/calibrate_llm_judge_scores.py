#!/usr/bin/env python3
"""Calibrate LLM judge scores using validation labels only.

This script does not rerun the LLM. It fits a small regularized logistic
calibrator on validation rows from an existing judge score JSONL, then applies
the calibrated probability and threshold to every requested row.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SEMANTIC_DIMENSIONS = [
    "numeric_formula_correctness",
    "algorithmic_wrong_value",
    "edge_case_boundary_handling",
    "string_regex_pattern_logic",
]

CRITICAL_DIMENSIONS = [
    "interface_name_signature_mismatch",
    "runtime_api_type_misuse",
    "syntax_parseability_or_output_format",
    "algorithmic_wrong_value",
    "numeric_formula_correctness",
    "edge_case_boundary_handling",
    "string_regex_pattern_logic",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def dimension_ids(rows: list[dict[str, Any]]) -> list[str]:
    for row in rows:
        scores = row.get("dimension_scores")
        if isinstance(scores, dict) and scores:
            return list(scores.keys())
    raise SystemExit("No dimension_scores found in input rows")


def row_features(row: dict[str, Any], dims: list[str]) -> list[float]:
    scores = {dimension_id: float(row["dimension_scores"][dimension_id]["score"]) for dimension_id in dims}
    values = [scores[dimension_id] for dimension_id in dims]
    semantic_values = [scores[dimension_id] for dimension_id in SEMANTIC_DIMENSIONS if dimension_id in scores]
    repair = row.get("repair") or {}
    deterministic_adjustments = repair.get("deterministic_adjustments") or []
    critical_low_count = sum(1 for dimension_id in CRITICAL_DIMENSIONS if scores.get(dimension_id, 5.0) <= 2.0)
    return values + [
        float(np.mean(values)),
        float(np.min(values)),
        float(np.mean(semantic_values)) if semantic_values else 0.0,
        float(np.min(semantic_values)) if semantic_values else 0.0,
        scores.get("algorithmic_wrong_value", 0.0),
        scores.get("syntax_parseability_or_output_format", 0.0),
        float(critical_low_count),
        1.0 if repair.get("used_visible_code_fallback") else 0.0,
        float(len(deterministic_adjustments)),
    ]


def feature_names(dims: list[str]) -> list[str]:
    return dims + [
        "mean_all_dimensions",
        "min_all_dimensions",
        "mean_semantic_dimensions",
        "min_semantic_dimensions",
        "algorithmic_wrong_value_duplicate",
        "syntax_parseability_duplicate",
        "critical_low_count",
        "used_visible_code_fallback",
        "deterministic_adjustment_count",
    ]


def labels(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.array([1 if row.get("passed") else 0 for row in rows], dtype=int)


def split_rows(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    if split == "all":
        return rows
    return [row for row in rows if str(row.get("split")) == split]


def choose_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[float, dict[str, Any]]:
    best_key = None
    best_threshold = 0.5
    best_metrics: dict[str, Any] = {}
    for index in range(0, 1001):
        threshold = index / 1000.0
        predictions = (probabilities >= threshold).astype(int)
        kappa = float(cohen_kappa_score(y_true, predictions))
        accuracy = float(accuracy_score(y_true, predictions))
        # Prefer higher threshold on exact ties to avoid overly permissive pass prediction.
        key = (round(kappa, 12), round(accuracy, 12), threshold)
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = threshold
            best_metrics = compute_binary_metrics(y_true, probabilities, predictions)
    return best_threshold, best_metrics


def compute_binary_metrics(y_true: np.ndarray, probabilities: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, predictions, labels=[1], zero_division=0)
    return {
        "n": int(len(y_true)),
        "auc": float(roc_auc_score(y_true, probabilities)) if len(set(y_true.tolist())) > 1 else None,
        "accuracy": float(accuracy_score(y_true, predictions)),
        "kappa": float(cohen_kappa_score(y_true, predictions)),
        "predicted_pass_rate": float(np.mean(predictions)) if len(predictions) else 0.0,
        "true_pass_rate": float(np.mean(y_true)) if len(y_true) else 0.0,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "precision_pass": float(precision[0]),
        "recall_pass": float(recall[0]),
        "f1_pass": float(f1[0]),
    }


def metrics_by_split(rows: list[dict[str, Any]], splits: list[str]) -> dict[str, Any]:
    metrics = {}
    for split in splits:
        subset = split_rows(rows, split)
        y_true = labels(subset)
        probabilities = np.array([float(row["calibrated_probability"]) for row in subset])
        predictions = np.array([1 if row["predicted_pass"] else 0 for row in subset])
        metrics[split] = compute_binary_metrics(y_true, probabilities, predictions)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit a validation logistic calibrator for LLM judge scores.")
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--scores-output", type=Path, required=True)
    parser.add_argument("--metrics-output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--calibration-split", default="validation")
    parser.add_argument("--eval-splits", default="all,validation,test")
    parser.add_argument("--threshold", type=float, help="Use a fixed probability threshold instead of choosing one on the calibration split.")
    parser.add_argument("--c", type=float, default=0.5, help="L2 logistic regression inverse regularization strength.")
    args = parser.parse_args()

    rows = read_jsonl(args.scores)
    dims = dimension_ids(rows)
    calibration_rows = split_rows(rows, args.calibration_split)
    if not calibration_rows:
        raise SystemExit(f"No rows found for calibration split: {args.calibration_split}")

    x_train = np.array([row_features(row, dims) for row in calibration_rows], dtype=float)
    y_train = labels(calibration_rows)
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=args.c, penalty="l2", solver="liblinear", random_state=0),
    )
    model.fit(x_train, y_train)

    train_probabilities = model.predict_proba(x_train)[:, 1]
    if args.threshold is None:
        threshold, calibration_metrics = choose_threshold(y_train, train_probabilities)
    else:
        threshold = args.threshold
        calibration_predictions = (train_probabilities >= threshold).astype(int)
        calibration_metrics = compute_binary_metrics(y_train, train_probabilities, calibration_predictions)

    output_rows = []
    for row in rows:
        output = dict(row)
        features = np.array([row_features(row, dims)], dtype=float)
        probability = float(model.predict_proba(features)[0, 1])
        output["pre_calibration_predicted_pass"] = bool(row.get("predicted_pass"))
        output["calibrated_probability"] = probability
        output["calibrated_threshold"] = threshold
        output["predicted_pass"] = bool(probability >= threshold)
        output["calibration_model"] = "validation_l2_logistic_regression"
        output_rows.append(output)

    eval_splits = [split.strip() for split in args.eval_splits.split(",") if split.strip()]
    metrics = {
        "method": "validation_l2_logistic_regression_calibrator",
        "source_scores": str(args.scores),
        "calibration_split": args.calibration_split,
        "num_calibration_rows": len(calibration_rows),
        "feature_names": feature_names(dims),
        "threshold": threshold,
        "calibration_metrics": calibration_metrics,
        "splits": metrics_by_split(output_rows, eval_splits),
    }
    audit = {
        "valid": True,
        "source_scores": str(args.scores),
        "num_input_rows": len(rows),
        "num_output_rows": len(output_rows),
        "calibration_split": args.calibration_split,
        "label_distribution": dict(Counter(int(value) for value in y_train)),
        "feature_count": len(feature_names(dims)),
        "threshold": threshold,
        "model": "LogisticRegression(C=0.5, penalty=l2, solver=liblinear) scaled with StandardScaler",
    }

    write_jsonl(args.scores_output, output_rows)
    write_json(args.metrics_output, metrics)
    write_json(args.audit_output, audit)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
