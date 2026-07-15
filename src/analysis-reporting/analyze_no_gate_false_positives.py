#!/usr/bin/env python3
"""Analyze no-gate rubric judge false positives.

The main failure mode of a no-gate evaluator is over-acceptance: failed code
that the rubric judge predicts as passing. This script joins rubric judge
scores with verifier/evaluator metadata and produces a compact analysis report.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def short_text(value: Any, limit: int = 420) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 18].rstrip() + " ... [truncated]"


def mean(values: list[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def median(values: list[float]) -> float | None:
    return float(statistics.median(values)) if values else None


def pct(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def dimension_score(row: dict[str, Any], dimension: str) -> float | None:
    item = (row.get("dimension_scores") or {}).get(dimension)
    if not isinstance(item, dict):
        return None
    value = item.get("score")
    return float(value) if isinstance(value, (int, float)) else None


def dimension_applicable(row: dict[str, Any], dimension: str) -> bool:
    item = (row.get("dimension_scores") or {}).get(dimension)
    return bool(isinstance(item, dict) and item.get("applicable"))


def min_applicable_dimension(row: dict[str, Any]) -> float | None:
    scores = []
    for item in (row.get("dimension_scores") or {}).values():
        if isinstance(item, dict) and item.get("applicable") and isinstance(item.get("score"), (int, float)):
            scores.append(float(item["score"]))
    return min(scores) if scores else None


def all_applicable_dimensions_high(row: dict[str, Any], threshold: float) -> bool:
    scores = []
    for item in (row.get("dimension_scores") or {}).values():
        if isinstance(item, dict) and item.get("applicable") and isinstance(item.get("score"), (int, float)):
            scores.append(float(item["score"]))
    return bool(scores) and all(score >= threshold for score in scores)


def score_bucket(score: float | None) -> str:
    if score is None:
        return "missing"
    if score < 2:
        return "[1,2)"
    if score < 3:
        return "[2,3)"
    if score < 4:
        return "[3,4)"
    if score < 4.5:
        return "[4,4.5)"
    if score < 5:
        return "[4.5,5)"
    return "5"


def confusion_group(row: dict[str, Any]) -> str:
    passed = bool(row.get("passed"))
    predicted = bool(row.get("predicted_pass"))
    if passed and predicted:
        return "tp"
    if passed and not predicted:
        return "fn"
    if not passed and predicted:
        return "fp"
    return "tn"


def counts_by(rows: list[dict[str, Any]], key: str, metadata: dict[str, dict[str, Any]]) -> dict[str, int]:
    counter: Counter = Counter()
    for row in rows:
        meta = metadata.get(str(row.get("response_id")), {})
        value = row.get(key)
        if value is None and key in meta:
            value = meta.get(key)
        counter[str(value)] += 1
    return dict(counter)


def summarize_group(rows: list[dict[str, Any]], dimensions: list[str]) -> dict[str, Any]:
    overall = [float(row["overall_score"]) for row in rows if isinstance(row.get("overall_score"), (int, float))]
    semantic = [
        float(row["semantic_bottleneck_score"])
        for row in rows
        if isinstance(row.get("semantic_bottleneck_score"), (int, float))
    ]
    critical_error_counts = [len(row.get("critical_errors") or []) for row in rows]
    min_applicable = [
        value
        for row in rows
        for value in [min_applicable_dimension(row)]
        if value is not None
    ]

    dimension_means = {}
    dimension_high_rates = {}
    for dimension in dimensions:
        values = [dimension_score(row, dimension) for row in rows]
        values = [value for value in values if value is not None]
        applicable = [row for row in rows if dimension_applicable(row, dimension)]
        high = [row for row in rows if (dimension_score(row, dimension) or 0) >= 4]
        dimension_means[dimension] = mean(values)
        dimension_high_rates[dimension] = pct(len(high), len(rows))

    return {
        "count": len(rows),
        "overall_score_mean": mean(overall),
        "overall_score_median": median(overall),
        "semantic_bottleneck_mean": mean(semantic),
        "semantic_bottleneck_median": median(semantic),
        "min_applicable_dimension_mean": mean(min_applicable),
        "critical_errors_empty_rate": pct(sum(1 for count in critical_error_counts if count == 0), len(rows)),
        "critical_errors_mean_count": mean([float(count) for count in critical_error_counts]),
        "overall_score_buckets": dict(Counter(score_bucket(float(row["overall_score"])) for row in rows if isinstance(row.get("overall_score"), (int, float)))),
        "all_applicable_dimensions_ge_4_rate": pct(sum(all_applicable_dimensions_high(row, 4) for row in rows), len(rows)),
        "all_applicable_dimensions_eq_5_rate": pct(sum(all_applicable_dimensions_high(row, 5) for row in rows), len(rows)),
        "dimension_mean_scores": dimension_means,
        "dimension_score_ge_4_rates": dimension_high_rates,
    }


def threshold_metrics(rows: list[dict[str, Any]], score_key: str, threshold: float) -> dict[str, Any]:
    tp = tn = fp = fn = 0
    for row in rows:
        score = row.get(score_key)
        if not isinstance(score, (int, float)):
            continue
        pred = float(score) >= threshold
        passed = bool(row.get("passed"))
        if passed and pred:
            tp += 1
        elif passed and not pred:
            fn += 1
        elif not passed and pred:
            fp += 1
        else:
            tn += 1
    total = tp + tn + fp + fn
    return {
        "threshold": threshold,
        "accuracy": pct(tp + tn, total),
        "balanced_accuracy": (pct(tp, tp + fn) + pct(tn, tn + fp)) / 2,
        "predicted_pass_rate": pct(tp + fp, total),
        "overacceptance_rate": pct(fp, fp + tn),
        "false_rejection_rate": pct(fn, fn + tp),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def threshold_sweep(rows: list[dict[str, Any]], score_key: str) -> dict[str, Any]:
    thresholds = [round(1.0 + i * 0.1, 2) for i in range(41)]
    metrics = [threshold_metrics(rows, score_key, threshold) for threshold in thresholds]
    best_balanced = max(metrics, key=lambda item: (item["balanced_accuracy"], item["accuracy"]))
    best_accuracy = max(metrics, key=lambda item: (item["accuracy"], item["balanced_accuracy"]))
    constrained = [
        item for item in metrics
        if item["overacceptance_rate"] <= 0.25 and item["false_rejection_rate"] <= 0.25
    ]
    return {
        "score_key": score_key,
        "best_balanced_accuracy": best_balanced,
        "best_accuracy": best_accuracy,
        "best_with_overacceptance_le_25_false_rejection_le_25": max(constrained, key=lambda item: item["balanced_accuracy"]) if constrained else None,
        "selected_thresholds": [
            item for item in metrics
            if item["threshold"] in {2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 4.8, 5.0}
        ],
    }


def compact_case(row: dict[str, Any], meta: dict[str, Any], dimensions: list[str]) -> dict[str, Any]:
    low_dims = []
    high_dims = []
    for dimension in dimensions:
        score = dimension_score(row, dimension)
        if score is None:
            continue
        if score <= 2:
            low_dims.append(dimension)
        if score >= 4:
            high_dims.append(dimension)
    return {
        "response_id": row.get("response_id"),
        "id": row.get("id"),
        "split": row.get("split"),
        "passed": row.get("passed"),
        "predicted_pass": row.get("predicted_pass"),
        "overall_score": row.get("overall_score"),
        "semantic_bottleneck_score": row.get("semantic_bottleneck_score"),
        "quality_mean_score": row.get("quality_mean_score"),
        "confidence": row.get("confidence"),
        "critical_error_count": len(row.get("critical_errors") or []),
        "critical_errors": row.get("critical_errors") or [],
        "failure_type": meta.get("failure_type"),
        "finish_reason": meta.get("finish_reason"),
        "io_mode": meta.get("io_mode"),
        "deterministic_error_label": meta.get("deterministic_error_label"),
        "human_error_label": meta.get("human_error_label"),
        "provisional_label_reference": meta.get("provisional_label_reference"),
        "provisional_confidence_reference": meta.get("provisional_confidence_reference"),
        "low_score_dimensions": low_dims,
        "high_score_dimensions": high_dims,
        "task_excerpt": short_text(meta.get("task"), 520),
        "code_excerpt": short_text(meta.get("extracted_code") or meta.get("generated_code"), 760),
    }


def markdown_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return lines


def render_markdown(summary: dict[str, Any], fp_cases: list[dict[str, Any]]) -> str:
    confusion = summary["confusion"]
    lines = [
        "# APPS Simple No-Gate False Positive Analysis",
        "",
        "## Baseline",
        "",
        f"- Samples: `{summary['num_samples']}`",
        f"- True pass rate: `{summary['true_pass_rate']:.2%}`",
        f"- Predicted pass rate: `{summary['predicted_pass_rate']:.2%}`",
        f"- Accuracy: `{summary['accuracy']:.2%}`",
        f"- Overacceptance rate: `{summary['overacceptance_rate']:.2%}`",
        f"- False rejection rate: `{summary['false_rejection_rate']:.2%}`",
        "",
        "Confusion:",
        "",
        "```text",
        f"TN={confusion['tn']}  FP={confusion['fp']}",
        f"FN={confusion['fn']}   TP={confusion['tp']}",
        "```",
        "",
        "## False Positive Concentration",
        "",
    ]
    for title, key in [
        ("By failure type", "fp_by_failure_type"),
        ("By finish reason", "fp_by_finish_reason"),
        ("By IO mode", "fp_by_io_mode"),
        ("By deterministic label", "fp_by_deterministic_error_label"),
        ("By provisional label reference", "fp_by_provisional_label_reference"),
    ]:
        values = summary.get(key, {})
        lines.append(f"### {title}")
        lines.append("")
        lines.extend(markdown_table(["Value", "Count"], [[k, v] for k, v in sorted(values.items(), key=lambda item: (-item[1], item[0]))]))
        lines.append("")

    fp = summary["groups"]["fp"]
    tn = summary["groups"]["tn"]
    lines.extend([
        "## False Positive Shape",
        "",
        f"- FP mean overall score: `{fp['overall_score_mean']:.3f}`",
        f"- TN mean overall score: `{tn['overall_score_mean']:.3f}`",
        f"- FP empty critical-error rate: `{fp['critical_errors_empty_rate']:.2%}`",
        f"- FP all-applicable-dimensions >= 4 rate: `{fp['all_applicable_dimensions_ge_4_rate']:.2%}`",
        "",
        "### FP score buckets",
        "",
    ])
    lines.extend(markdown_table(["Overall score bucket", "FP count"], [[k, v] for k, v in sorted(fp["overall_score_buckets"].items())]))
    lines.append("")

    lines.extend([
        "## Threshold Sweep",
        "",
        "Best balanced thresholds:",
        "",
    ])
    sweep_rows = []
    for key, sweep in summary["threshold_sweeps"].items():
        best = sweep["best_balanced_accuracy"]
        sweep_rows.append([
            key,
            best["threshold"],
            f"{best['balanced_accuracy']:.3f}",
            f"{best['accuracy']:.3f}",
            f"{best['overacceptance_rate']:.3f}",
            f"{best['false_rejection_rate']:.3f}",
        ])
    lines.extend(markdown_table(["Score", "Threshold", "Balanced Acc", "Acc", "Overaccept", "False Reject"], sweep_rows))
    lines.append("")

    lines.extend([
        "## High-Confidence False Positive Examples",
        "",
        "These are failed rows accepted by the no-gate judge with high overall score and no or few critical errors.",
        "",
    ])
    for index, case in enumerate(fp_cases[:8], start=1):
        lines.extend([
            f"### {index}. `{case['response_id']}`",
            "",
            f"- failure_type: `{case.get('failure_type')}`",
            f"- overall_score: `{case.get('overall_score')}`",
            f"- semantic_bottleneck_score: `{case.get('semantic_bottleneck_score')}`",
            f"- critical_error_count: `{case.get('critical_error_count')}`",
            f"- provisional_label_reference: `{case.get('provisional_label_reference')}`",
            f"- low_score_dimensions: `{', '.join(case.get('low_score_dimensions') or []) or 'none'}`",
            "",
            "Task excerpt:",
            "",
            "```text",
            case.get("task_excerpt") or "",
            "```",
            "",
            "Code excerpt:",
            "",
            "```python",
            case.get("code_excerpt") or "",
            "```",
            "",
        ])
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze false positives from a no-gate rubric judge run.")
    parser.add_argument("--scores", type=Path, default=Path("data/rubrics/apps_simple_method1/apps_simple_no_gate_baseline_eval_v1_scores.jsonl"))
    parser.add_argument("--evaluator-rows", type=Path, default=Path("data/evaluator/apps_simple_method1_evaluator_training_rows_v1.jsonl"))
    parser.add_argument("--summary-output", type=Path, default=Path("data/rubrics/apps_simple_method1/apps_simple_no_gate_baseline_eval_v1_fp_analysis.json"))
    parser.add_argument("--cases-output", type=Path, default=Path("data/rubrics/apps_simple_method1/apps_simple_no_gate_baseline_eval_v1_false_positives.jsonl"))
    parser.add_argument("--markdown-output", type=Path, default=Path("data/rubrics/apps_simple_method1/apps_simple_no_gate_baseline_eval_v1_fp_analysis.md"))
    args = parser.parse_args()

    scores = read_jsonl(args.scores)
    metadata = {str(row.get("response_id")): row for row in read_jsonl(args.evaluator_rows)}
    dimensions = sorted((scores[0].get("dimension_scores") or {}).keys()) if scores else []

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scores:
        groups[confusion_group(row)].append(row)

    fp_rows = groups["fp"]
    compact_fp = [
        compact_case(row, metadata.get(str(row.get("response_id")), {}), dimensions)
        for row in fp_rows
    ]
    compact_fp = sorted(
        compact_fp,
        key=lambda item: (
            -(float(item.get("overall_score") or 0)),
            int(item.get("critical_error_count") or 0),
            str(item.get("response_id")),
        ),
    )

    tp, tn, fp, fn = len(groups["tp"]), len(groups["tn"]), len(groups["fp"]), len(groups["fn"])
    failed = fp + tn
    passed = tp + fn
    summary = {
        "num_samples": len(scores),
        "true_pass_rate": pct(passed, len(scores)),
        "predicted_pass_rate": pct(tp + fp, len(scores)),
        "accuracy": pct(tp + tn, len(scores)),
        "overacceptance_rate": pct(fp, failed),
        "false_rejection_rate": pct(fn, passed),
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "fp_by_failure_type": counts_by(fp_rows, "failure_type", metadata),
        "fp_by_finish_reason": counts_by(fp_rows, "finish_reason", metadata),
        "fp_by_io_mode": counts_by(fp_rows, "io_mode", metadata),
        "fp_by_deterministic_error_label": counts_by(fp_rows, "deterministic_error_label", metadata),
        "fp_by_human_error_label": counts_by(fp_rows, "human_error_label", metadata),
        "fp_by_provisional_label_reference": counts_by(fp_rows, "provisional_label_reference", metadata),
        "groups": {
            name: summarize_group(rows, dimensions)
            for name, rows in sorted(groups.items())
        },
        "failed_only_comparison": {
            "fp_count": fp,
            "tn_count": tn,
            "fp_minus_tn_dimension_mean_scores": {
                dimension: (
                    (summarize_group(fp_rows, dimensions)["dimension_mean_scores"].get(dimension) or 0)
                    - (summarize_group(groups["tn"], dimensions)["dimension_mean_scores"].get(dimension) or 0)
                )
                for dimension in dimensions
            },
        },
        "threshold_sweeps": {
            "overall_score": threshold_sweep(scores, "overall_score"),
            "semantic_bottleneck_score": threshold_sweep(scores, "semantic_bottleneck_score"),
            "quality_mean_score": threshold_sweep(scores, "quality_mean_score"),
        },
        "interpretation": {
            "primary_issue": "The no-gate judge over-accepts failed code far more often than it falsely rejects passed code.",
            "training_implication": "Use verifier pass/fail as the primary critic target and treat rubric scores as features or soft signals, not hard correctness labels.",
            "repair_implication": "Prioritize false-positive logic failures because they look structurally plausible but fail execution.",
        },
    }

    write_json(args.summary_output, summary)
    write_jsonl(args.cases_output, compact_fp)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(render_markdown(summary, compact_fp), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
