#!/usr/bin/env python3
"""Filter prompt JSONL rows by pass/fail labels from response-style JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-passed", action="store_true")
    args = parser.parse_args()

    retained_ids = set()
    label_counts: dict[str, dict[str, int]] = {}
    for row in read_jsonl(args.labels):
        problem_id = str(row.get("id"))
        counts = label_counts.setdefault(problem_id, {"total": 0, "passed": 0})
        counts["total"] += 1
        counts["passed"] += int(bool(row.get("passed")))
        if bool(row.get("passed")) or not args.require_passed:
            retained_ids.add(problem_id)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    written = 0
    with args.output.open("w", encoding="utf-8") as out:
        for row in read_jsonl(args.prompts):
            total += 1
            if str(row.get("id")) not in retained_ids:
                continue
            row = dict(row)
            row["canonical_verifier"] = label_counts.get(str(row.get("id")), {})
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1

    print(f"filtered prompts: kept {written}/{total} rows in {args.output}")


if __name__ == "__main__":
    main()
