#!/usr/bin/env python3
"""Merge multiple preference JSONL files with basic train-only filtering."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterable


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def normalize_split(row: dict) -> str | None:
    split = row.get("split")
    if split:
        if split.startswith("mbpp/"):
            return split.split("/", 1)[1]
        return split
    parts = row.get("id", "").split("/")
    if len(parts) >= 3 and parts[0] == "mbpp":
        return parts[1]
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--dataset", type=str, default="mbpp")
    parser.add_argument("--split", type=str, default="train")
    args = parser.parse_args()

    rows = []
    skipped = Counter()
    sources = Counter()
    seen = set()
    for path in args.input:
        for row in read_jsonl(path):
            if row.get("dataset") != args.dataset or normalize_split(row) != args.split:
                skipped["not_target_split"] += 1
                continue
            if not row.get("prompt") or not row.get("chosen") or not row.get("rejected"):
                skipped["missing_fields"] += 1
                continue
            key = (row.get("id"), row.get("chosen_source"), row.get("rejected_source"))
            if key in seen:
                skipped["duplicate"] += 1
                continue
            seen.add(key)
            out = dict(row)
            out["dataset"] = args.dataset
            out["split"] = args.split
            rows.append(out)
            sources[out.get("chosen_source", "unknown")] += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "inputs": [str(path) for path in args.input],
        "output": str(args.output),
        "total_pairs": len(rows),
        "chosen_sources": dict(sources),
        "skipped": dict(skipped),
        "dataset": args.dataset,
        "split": args.split,
        "leakage": "no validation/test rows included",
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
