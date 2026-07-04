#!/usr/bin/env python3
"""Filter unified coding prompt JSONL by dataset/split/id prefix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/processed/coding_prompts.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--id-prefix", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.input.open("r", encoding="utf-8") as src, args.output.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            if args.dataset and row.get("dataset") != args.dataset:
                continue
            if args.split and row.get("split") != args.split:
                continue
            if args.id_prefix and not row.get("id", "").startswith(args.id_prefix):
                continue
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
            if args.limit is not None and count >= args.limit:
                break

    print(f"wrote {count} prompts to {args.output}")


if __name__ == "__main__":
    main()
