#!/usr/bin/env python3
"""Evaluate generated Method 1 self-evaluation verdicts."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score


VERDICT_RE = re.compile(r"(?im)^\s*Verdict\s*:\s*(PASS|FAIL)\b")
CONFIDENCE_RE = re.compile(r"(?im)^\s*Confidence\s*:\s*(high|medium|low)\b")
PRIMARY_ERROR_RE = re.compile(r"(?im)^\s*Primary\s+error\s*:\s*([A-Za-z0-9_/-]+)")


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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_verdict(text: str) -> str | None:
    match = VERDICT_RE.search(text or "")
    if match:
        return match.group(1).upper()
    fallback = re.search(r"\b(PASS|FAIL)\b", text or "", flags=re.IGNORECASE)
    if fallback:
        return fallback.group(1).upper()
    return None


def parse_confidence(text: str) -> str | None:
    match = CONFIDENCE_RE.search(text or "")
    return match.group(1).lower() if match else None


def parse_primary_error(text: str) -> str | None:
    match = PRIMARY_ERROR_RE.search(text or "")
    return match.group(1) if match else None


def gold_passed(row: dict[str, Any]) -> bool:
    metadata = row.get("metadata") or {}
    if "passed" in metadata:
        return bool(metadata["passed"])
    gold = parse_verdict(str(row.get("completion") or ""))
    if gold == "PASS":
        return True
    if gold == "FAIL":
        return False
    raise ValueError(f"cannot infer gold label for row {row.get('id')}")


def pct(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def confusion_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": (pct(tp, tp + fn) + pct(tn, tn + fp)) / 2,
        "kappa": float(cohen_kappa_score(y_true, y_pred)) if len(set(y_pred.tolist())) > 1 else 0.0,
        "predicted_pass_rate": pct(tp + fp, len(y_true)),
        "true_pass_rate": pct(int(y_true.sum()), len(y_true)),
        "overacceptance_rate": pct(fp, fp + tn),
        "false_rejection_rate": pct(fn, fn + tp),
        "precision_pass": pct(tp, tp + fp),
        "recall_pass": pct(tp, tp + fn),
        "specificity_fail": pct(tn, tn + fp),
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def render_markdown(summary: dict[str, Any]) -> str:
    strict = summary["strict_unparsed_as_wrong"]
    parsed = summary["parsed_only"]
    lines = [
        "# Generative Self-Evaluator v1 Test",
        "",
        "## Summary",
        "",
        f"- Rows: `{summary['rows']}`",
        f"- Verdict parse rate: `{summary['verdict_parse_rate']:.4f}`",
        f"- Confidence parse rate: `{summary['confidence_parse_rate']:.4f}`",
        f"- Strict accuracy: `{strict['accuracy']:.4f}`",
        f"- Strict balanced accuracy: `{strict['balanced_accuracy']:.4f}`",
        f"- Strict Kappa: `{strict['kappa']:.4f}`",
        f"- Strict overacceptance: `{strict['overacceptance_rate']:.4f}`",
        f"- Strict false rejection: `{strict['false_rejection_rate']:.4f}`",
        "",
        "## Metrics",
        "",
        "| Metric | Strict unparsed wrong | Parsed only |",
        "| --- | ---: | ---: |",
    ]
    for key in [
        "accuracy",
        "balanced_accuracy",
        "kappa",
        "predicted_pass_rate",
        "overacceptance_rate",
        "false_rejection_rate",
        "precision_pass",
        "recall_pass",
        "specificity_fail",
    ]:
        left = strict.get(key)
        right = parsed.get(key)
        lines.append(
            f"| {key} | {left:.4f} | {right:.4f} |"
            if isinstance(left, (int, float)) and isinstance(right, (int, float))
            else f"| {key} | {left} | {right} |"
        )
    lines.extend(["", "## Gates", ""])
    for name, passed in summary["gates"].items():
        lines.append(f"- [{'x' if passed else ' '}] {name}")
    lines.extend(["", "## Full Summary", "", "```json", json.dumps(summary, ensure_ascii=False, indent=2), "```", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate generated PASS/FAIL verdicts.")
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--predictions-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--min-parse-rate", type=float, default=0.98)
    parser.add_argument("--max-overacceptance", type=float, default=0.25)
    parser.add_argument("--min-balanced-accuracy", type=float, default=0.70)
    args = parser.parse_args()

    rows = [row for row in read_jsonl(args.generations) if str(row.get("task_type")) == "judge_single"]
    if not rows:
        raise SystemExit("no judge_single generations found")

    predictions: list[dict[str, Any]] = []
    for row in rows:
        generated = str(row.get("generated_text") or "")
        parsed = parse_verdict(generated)
        gold = gold_passed(row)
        predicted_pass = None if parsed is None else parsed == "PASS"
        predictions.append(
            {
                "id": row.get("id"),
                "response_id": row.get("response_id"),
                "split": row.get("split"),
                "gold_passed": gold,
                "parsed_verdict": parsed,
                "predicted_pass": predicted_pass,
                "verdict_parseable": parsed is not None,
                "parsed_confidence": parse_confidence(generated),
                "parsed_primary_error": parse_primary_error(generated),
                "finish_reason": row.get("finish_reason"),
                "generated_token_count": row.get("generated_token_count"),
                "generated_text": generated,
            }
        )

    parseable = [row for row in predictions if row["verdict_parseable"]]
    y_true_all = np.array([1 if row["gold_passed"] else 0 for row in predictions], dtype=int)
    # Strict policy: unparseable model output is an incorrect prediction. For
    # pass gold rows it is counted as fail; for fail gold rows it is counted as pass.
    strict_pred = []
    for row in predictions:
        if row["predicted_pass"] is None:
            strict_pred.append(0 if row["gold_passed"] else 1)
        else:
            strict_pred.append(1 if row["predicted_pass"] else 0)
    y_pred_all = np.array(strict_pred, dtype=int)

    if parseable:
        y_true_parsed = np.array([1 if row["gold_passed"] else 0 for row in parseable], dtype=int)
        y_pred_parsed = np.array([1 if row["predicted_pass"] else 0 for row in parseable], dtype=int)
        parsed_metrics = confusion_metrics(y_true_parsed, y_pred_parsed)
    else:
        parsed_metrics = {
            "accuracy": 0.0,
            "balanced_accuracy": 0.0,
            "kappa": 0.0,
            "predicted_pass_rate": 0.0,
            "true_pass_rate": pct(int(y_true_all.sum()), len(y_true_all)),
            "overacceptance_rate": 1.0,
            "false_rejection_rate": 1.0,
            "precision_pass": 0.0,
            "recall_pass": 0.0,
            "specificity_fail": 0.0,
            "confusion": {"tn": 0, "fp": 0, "fn": 0, "tp": 0},
        }

    verdict_parse_rate = pct(len(parseable), len(predictions))
    confidence_parse_rate = pct(sum(row["parsed_confidence"] is not None for row in predictions), len(predictions))
    primary_error_parse_rate = pct(sum(row["parsed_primary_error"] is not None for row in predictions), len(predictions))
    strict_metrics = confusion_metrics(y_true_all, y_pred_all)
    gates = {
        "verdict_parse_rate_ge_min": verdict_parse_rate >= args.min_parse_rate,
        "strict_balanced_accuracy_ge_min": strict_metrics["balanced_accuracy"] >= args.min_balanced_accuracy,
        "strict_overacceptance_le_max": strict_metrics["overacceptance_rate"] <= args.max_overacceptance,
    }
    summary = {
        "rows": len(predictions),
        "generation_file": str(args.generations),
        "verdict_parse_rate": verdict_parse_rate,
        "confidence_parse_rate": confidence_parse_rate,
        "primary_error_parse_rate": primary_error_parse_rate,
        "gold_counts": dict(Counter("pass" if row["gold_passed"] else "fail" for row in predictions)),
        "parsed_verdict_counts": dict(Counter(str(row["parsed_verdict"]) for row in predictions)),
        "finish_reason_counts": dict(Counter(str(row.get("finish_reason")) for row in predictions)),
        "strict_unparsed_as_wrong": strict_metrics,
        "parsed_only": parsed_metrics,
        "gates": gates,
        "canary_passed": all(gates.values()),
        "policy": "held-out judge_single generation; strict gate counts unparseable verdicts as wrong",
    }

    write_json(args.summary_output, summary)
    write_jsonl(args.predictions_output, predictions)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(render_markdown(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
