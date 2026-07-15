#!/usr/bin/env python3
"""Audit an APPS self-repair pool before preference construction."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {error}") from error
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_ids(paths: Iterable[Path]) -> set[str]:
    return {
        str(row.get("id"))
        for path in paths
        for row in read_jsonl(path)
        if row.get("id")
    }


def grouped_success(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, float | int]]:
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key) or "unknown")].append(row)

    result: dict[str, dict[str, float | int]] = {}
    for value, group in sorted(grouped.items()):
        tasks: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in group:
            tasks[str(row.get("id") or "")].append(row)
        passed = sum(bool(row.get("passed")) for row in group)
        successful_tasks = sum(any(bool(row.get("passed")) for row in task_rows) for task_rows in tasks.values())
        result[value] = {
            "samples": len(group),
            "passed_samples": passed,
            "sample_pass_rate": passed / len(group),
            "tasks": len(tasks),
            "successful_tasks": successful_tasks,
            "task_success_rate": successful_tasks / len(tasks),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit sample-level and task-level APPS self-repair outcomes.")
    parser.add_argument("--input", type=Path, required=True, help="Verifier-labeled repair JSONL.")
    parser.add_argument("--forbidden-ids", type=Path, action="append", default=[])
    parser.add_argument("--expected-rows", type=int)
    parser.add_argument("--expected-k", type=int)
    parser.add_argument("--min-eligible-successful-tasks", type=int, default=1)
    parser.add_argument("--fail-on-gate", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    if not rows:
        raise RuntimeError("repair pool is empty")

    by_task: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[str(row.get("id") or "")].append(row)
    task_ids = set(by_task)
    forbidden_ids = load_ids(args.forbidden_ids)
    forbidden_overlap = task_ids & forbidden_ids
    eligible_rows = [row for row in rows if str(row.get("id") or "") not in forbidden_ids]
    eligible_by_task: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in eligible_rows:
        eligible_by_task[str(row.get("id") or "")].append(row)

    response_ids = [str(row.get("response_id") or "") for row in rows]
    sample_counts = Counter(len(task_rows) for task_rows in by_task.values())
    passed_samples = sum(bool(row.get("passed")) for row in rows)
    successful_tasks = sum(any(bool(row.get("passed")) for row in task_rows) for task_rows in by_task.values())
    eligible_passed_samples = sum(bool(row.get("passed")) for row in eligible_rows)
    eligible_successful_tasks = sum(
        any(bool(row.get("passed")) for row in task_rows) for task_rows in eligible_by_task.values()
    )

    required_links_present = all(
        row.get("id") and row.get("response_id") and row.get("original_response_id") for row in rows
    )
    gates = {
        "row_count_match": args.expected_rows is None or len(rows) == args.expected_rows,
        "response_ids_unique": len(response_ids) == len(set(response_ids)) and all(response_ids),
        "required_links_present": required_links_present,
        "k_per_task_match": args.expected_k is None or set(sample_counts) == {args.expected_k},
        "eligible_successful_tasks_at_least_min": eligible_successful_tasks
        >= args.min_eligible_successful_tasks,
    }
    summary: dict[str, Any] = {
        "input": str(args.input),
        "input_sha256": sha256_file(args.input),
        "rows": len(rows),
        "unique_tasks": len(by_task),
        "samples_per_task_counts": {str(key): value for key, value in sorted(sample_counts.items())},
        "passed_samples": passed_samples,
        "sample_pass_rate": passed_samples / len(rows),
        "successful_tasks": successful_tasks,
        "task_success_rate": successful_tasks / len(by_task),
        "outcome_counts": dict(
            Counter("passed" if row.get("passed") else str(row.get("failure_type") or "unknown") for row in rows)
        ),
        "finish_reason_counts": dict(Counter(str(row.get("finish_reason") or "unknown") for row in rows)),
        "by_io_mode": grouped_success(rows, "io_mode"),
        "by_selection_reason": grouped_success(rows, "selection_reason"),
        "forbidden_files": [str(path) for path in args.forbidden_ids],
        "forbidden_id_count": len(forbidden_ids),
        "forbidden_overlap_tasks": len(forbidden_overlap),
        "forbidden_successful_tasks": sum(
            any(bool(row.get("passed")) for row in by_task[problem_id]) for problem_id in forbidden_overlap
        ),
        "eligible_rows": len(eligible_rows),
        "eligible_tasks": len(eligible_by_task),
        "eligible_passed_samples": eligible_passed_samples,
        "eligible_successful_tasks": eligible_successful_tasks,
        "minimum_eligible_successful_tasks": args.min_eligible_successful_tasks,
        "gates": gates,
        "audit_passed": all(gates.values()),
        "policy": "Report forbidden overlap, then exclude it in the v2 preference builder; never redraw DPO-dev.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.fail_on_gate and not summary["audit_passed"]:
        raise SystemExit("repair-pool audit gate failed")


if __name__ == "__main__":
    main()
