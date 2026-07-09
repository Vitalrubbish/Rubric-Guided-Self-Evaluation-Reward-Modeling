#!/usr/bin/env python3
"""Compute response-level and task-level metrics for coding benchmark labels."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def rate(num: int, den: int) -> float:
    return round(num / den, 6) if den else 0.0


def pack_task_metrics(task_passes: dict[str, list[bool]]) -> dict:
    responses = sum(len(values) for values in task_passes.values())
    passed = sum(sum(values) for values in task_passes.values())
    tasks = len(task_passes)
    any_pass = sum(any(values) for values in task_passes.values())
    all_pass = sum(all(values) for values in task_passes.values())
    all_fail = sum(not any(values) for values in task_passes.values())
    return {
        "tasks": tasks,
        "responses": responses,
        "passed_responses": passed,
        "failed_responses": responses - passed,
        "response_pass_rate": rate(passed, responses),
        "task_pass_at_k": rate(any_pass, tasks),
        "tasks_with_any_pass": any_pass,
        "tasks_with_all_pass": all_pass,
        "tasks_with_partial_pass": any_pass - all_pass,
        "tasks_with_all_fail": all_fail,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--k", type=int, default=None)
    args = parser.parse_args()

    rows = list(read_jsonl(args.input))
    by_task: dict[str, list[bool]] = defaultdict(list)
    by_split: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    by_sample_id: dict[str, Counter] = defaultdict(Counter)
    failure_types = Counter()

    for row in rows:
        task_id = str(row.get("id"))
        split = str(row.get("split") or "unknown")
        sample_id = str(row.get("sample_id", 0))
        passed = bool(row.get("passed"))
        by_task[task_id].append(passed)
        by_split[split][task_id].append(passed)
        by_sample_id[sample_id]["total"] += 1
        by_sample_id[sample_id]["passed_responses"] += int(passed)
        if not passed:
            by_sample_id[sample_id][row.get("failure_type") or "unknown"] += 1
            failure_types[row.get("failure_type") or "unknown"] += 1

    inferred_k = args.k or max((len(values) for values in by_task.values()), default=0)
    summary = {
        "input": str(args.input),
        "k": inferred_k,
        **pack_task_metrics(by_task),
        "failure_types": dict(failure_types),
        "by_split": {
            split: pack_task_metrics(tasks)
            for split, tasks in sorted(by_split.items())
        },
        "by_sample_id": {
            sample_id: {
                **dict(counter),
                "pass_rate": rate(counter["passed_responses"], counter["total"]),
            }
            for sample_id, counter in sorted(by_sample_id.items(), key=lambda item: int(item[0]))
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
