#!/usr/bin/env python3
"""Select prompt JSONL rows by stable metadata fields."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def matches(row: dict[str, Any], field: str, allowed: set[str] | None) -> bool:
    if not allowed:
        return True
    return str(row.get(field, "")) in allowed


def counter_to_dict(counter: Counter[tuple[str, ...]]) -> dict[str, int]:
    return {"/".join(key): value for key, value in sorted(counter.items())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--dataset", action="append")
    parser.add_argument("--split", action="append")
    parser.add_argument("--difficulty", action="append")
    parser.add_argument("--io-mode", action="append")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--min-rows", type=int)
    parser.add_argument("--max-rows", type=int)
    args = parser.parse_args()

    allowed_dataset = set(args.dataset or [])
    allowed_split = set(args.split or [])
    allowed_difficulty = set(args.difficulty or [])
    allowed_io_mode = set(args.io_mode or [])

    total = 0
    selected: list[dict[str, Any]] = []
    for row in read_jsonl(args.input):
        total += 1
        if not matches(row, "dataset", allowed_dataset):
            continue
        if not matches(row, "split", allowed_split):
            continue
        if not matches(row, "difficulty", allowed_difficulty):
            continue
        if not matches(row, "io_mode", allowed_io_mode):
            continue
        selected.append(row)
        if args.limit is not None and len(selected) >= args.limit:
            break

    if args.min_rows is not None and len(selected) < args.min_rows:
        raise SystemExit(f"selected {len(selected)} rows, below --min-rows {args.min_rows}")
    if args.max_rows is not None and len(selected) > args.max_rows:
        raise SystemExit(f"selected {len(selected)} rows, above --max-rows {args.max_rows}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as out:
        for row in selected:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    by_difficulty = Counter(str(row.get("difficulty", "unknown")) for row in selected)
    by_io_mode = Counter(str(row.get("io_mode", "unknown")) for row in selected)
    by_difficulty_io_mode = Counter(
        (str(row.get("difficulty", "unknown")), str(row.get("io_mode", "unknown")))
        for row in selected
    )
    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "total_input_rows_seen": total,
        "selected_rows": len(selected),
        "filters": {
            "dataset": sorted(allowed_dataset),
            "split": sorted(allowed_split),
            "difficulty": sorted(allowed_difficulty),
            "io_mode": sorted(allowed_io_mode),
            "limit": args.limit,
            "min_rows": args.min_rows,
            "max_rows": args.max_rows,
        },
        "by_difficulty": dict(sorted(by_difficulty.items())),
        "by_io_mode": dict(sorted(by_io_mode.items())),
        "by_difficulty_io_mode": counter_to_dict(by_difficulty_io_mode),
    }
    if args.summary_output:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
