#!/usr/bin/env python3
"""Freeze APPS internal validation/test prompt files from the problem split map."""

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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluator-rows",
        type=Path,
        default=Path("data/evaluator/apps_simple_method1_evaluator_training_rows_v1.jsonl"),
    )
    parser.add_argument(
        "--source-prompts",
        type=Path,
        default=Path("data/processed/apps_train_simple_executable_prompts_unified.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/apps_simple_method1_internal_eval_prompts_v1.jsonl"),
    )
    args = parser.parse_args()

    split_by_id = {
        str(row["id"]): str(row.get("split"))
        for row in read_jsonl(args.evaluator_rows)
        if row.get("id")
    }
    selected: list[dict[str, Any]] = []
    for source in read_jsonl(args.source_prompts):
        problem_id = str(source.get("id") or "")
        eval_split = split_by_id.get(problem_id)
        if eval_split not in {"validation", "test"}:
            continue
        record = dict(source)
        record["source_split"] = source.get("split")
        record["split"] = eval_split
        record["eval_split"] = eval_split
        selected.append(record)

    counts = Counter(str(row["eval_split"]) for row in selected)
    if counts != Counter({"validation": 261, "test": 262}):
        raise AssertionError(f"unexpected internal eval split counts: {dict(counts)}")
    ids = [str(row["id"]) for row in selected]
    if len(ids) != len(set(ids)):
        raise AssertionError("duplicate internal evaluation IDs")
    write_jsonl(args.output, selected)
    print(json.dumps({"output": str(args.output), "rows": len(selected), "split_counts": dict(counts)}, indent=2))


if __name__ == "__main__":
    main()
