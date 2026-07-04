#!/usr/bin/env python3
"""Filter preference-pair JSONL by dataset and split."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--max-pairs", type=int, default=None)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    written = 0
    with args.output.open("w", encoding="utf-8") as f:
        for row in read_jsonl(args.input):
            counts[(row.get("dataset"), row.get("split"))] += 1
            if args.dataset and row.get("dataset") != args.dataset:
                continue
            if args.split and row.get("split") != args.split:
                continue
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
            if args.max_pairs is not None and written >= args.max_pairs:
                break

    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "dataset": args.dataset,
        "split": args.split,
        "written": written,
        "source_counts": {f"{dataset}/{split}": count for (dataset, split), count in sorted(counts.items())},
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
