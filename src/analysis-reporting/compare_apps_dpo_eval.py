#!/usr/bin/env python3
"""Compare APPS internal held-out base and DPO verifier outcomes."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def split_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    transitions = Counter(
        f"base_{'pass' if row['base_passed'] else 'fail'}->dpo_{'pass' if row['dpo_passed'] else 'fail'}"
        for row in rows
    )
    base_passed = sum(bool(row["base_passed"]) for row in rows)
    dpo_passed = sum(bool(row["dpo_passed"]) for row in rows)
    return {
        "rows": len(rows),
        "base_passed": base_passed,
        "base_pass_rate": base_passed / len(rows) if rows else 0.0,
        "dpo_passed": dpo_passed,
        "dpo_pass_rate": dpo_passed / len(rows) if rows else 0.0,
        "net_pass_delta": dpo_passed - base_passed,
        "transitions": dict(transitions),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split-map",
        type=Path,
        default=Path("data/evaluator/apps_simple_method1_evaluator_training_rows_v1.jsonl"),
    )
    parser.add_argument(
        "--base-labeled",
        type=Path,
        default=Path("data/responses/apps_train_simple_executable_qwen25_k1_t2048_full_labeled.jsonl"),
    )
    parser.add_argument("--dpo-labeled", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    split_by_id = {str(row["id"]): str(row.get("split")) for row in read_jsonl(args.split_map)}
    base_by_id = {str(row["id"]): bool(row.get("passed")) for row in read_jsonl(args.base_labeled)}
    comparison_rows: list[dict[str, Any]] = []
    for row in read_jsonl(args.dpo_labeled):
        problem_id = str(row.get("id"))
        split = split_by_id.get(problem_id)
        if split not in {"validation", "test"}:
            raise AssertionError(f"DPO evaluation contains a non-held-out problem: {problem_id} split={split}")
        if problem_id not in base_by_id:
            raise AssertionError(f"base result missing for {problem_id}")
        comparison_rows.append(
            {
                "id": problem_id,
                "split": split,
                "base_passed": base_by_id[problem_id],
                "dpo_passed": bool(row.get("passed")),
                "dpo_failure_type": row.get("failure_type"),
            }
        )

    ids = [row["id"] for row in comparison_rows]
    if len(ids) != 523 or len(ids) != len(set(ids)):
        raise AssertionError(f"expected 523 unique DPO evaluation rows, got {len(ids)} rows/{len(set(ids))} IDs")
    summary = {
        "base_labeled": str(args.base_labeled),
        "dpo_labeled": str(args.dpo_labeled),
        "policy": "internal validation/test only; no training rows",
        "validation": split_metrics([row for row in comparison_rows if row["split"] == "validation"]),
        "test": split_metrics([row for row in comparison_rows if row["split"] == "test"]),
        "combined": split_metrics(comparison_rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# APPS Simple Method 1 DPO Held-Out Evaluation",
        "",
        "| Split | Rows | Base pass@1 | DPO pass@1 | Net passes |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for split in ("validation", "test", "combined"):
        item = summary[split]
        lines.append(
            f"| {split} | {item['rows']} | {item['base_pass_rate']:.4f} "
            f"| {item['dpo_pass_rate']:.4f} | {item['net_pass_delta']:+d} |"
        )
    lines.extend(["", "## Transitions", "", "```json", json.dumps(summary, ensure_ascii=False, indent=2), "```", ""])
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
