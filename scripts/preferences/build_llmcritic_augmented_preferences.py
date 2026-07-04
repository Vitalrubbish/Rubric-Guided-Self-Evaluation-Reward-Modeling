#!/usr/bin/env python3
"""Merge train-only augmented preference pairs with LLM-critic self-play pairs."""

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
    if row.get("split"):
        split = row["split"]
        if split.startswith("mbpp/"):
            return split.split("/", 1)[1]
        return split
    row_id = row.get("id", "")
    parts = row_id.split("/")
    if len(parts) >= 3 and parts[0] == "mbpp":
        return parts[1]
    return None


def row_key(row: dict) -> tuple[str, str, str]:
    return (row.get("id", ""), row.get("chosen_source", ""), row.get("rejected_source", ""))


def keep_train_mbpp(row: dict) -> bool:
    return row.get("dataset") == "mbpp" and normalize_split(row) == "train"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--llm-critic", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--md-output", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    seen = set()
    skipped = Counter()
    sources = Counter()

    for source_name, path in [("base_augmented", args.base), ("llm_critic", args.llm_critic)]:
        for row in read_jsonl(path):
            if not keep_train_mbpp(row):
                skipped[f"{source_name}:not_mbpp_train"] += 1
                continue
            if not row.get("prompt") or not row.get("chosen") or not row.get("rejected"):
                skipped[f"{source_name}:missing_fields"] += 1
                continue
            out = dict(row)
            out["split"] = "train"
            out["dataset"] = "mbpp"
            key = row_key(out)
            if key in seen:
                skipped[f"{source_name}:duplicate_key"] += 1
                continue
            seen.add(key)
            rows.append(out)
            sources[out.get("chosen_source", "unknown")] += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "base": str(args.base),
        "llm_critic": str(args.llm_critic),
        "output": str(args.output),
        "total_pairs": len(rows),
        "chosen_sources": dict(sources),
        "skipped": dict(skipped),
        "dataset": "mbpp",
        "split": "train",
        "leakage": "no validation/test rows included",
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    md = [
        "# LLM-Critic Augmented Preference Data",
        "",
        "## Summary",
        "",
        "| Field | Value |",
        "| --- | ---: |",
        f"| Total pairs | {len(rows)} |",
        f"| Base input | `{args.base}` |",
        f"| LLM critic input | `{args.llm_critic}` |",
        f"| Output | `{args.output}` |",
        "",
        "## Chosen Sources",
        "",
        "| Source | Count |",
        "| --- | ---: |",
    ]
    for key, value in sorted(sources.items()):
        md.append(f"| {key} | {value} |")
    md.extend(
        [
            "",
            "## Leakage Check",
            "",
            "Only `mbpp/train` rows are retained. Validation/test rows are skipped by construction.",
            "",
            "## Skipped",
            "",
            "| Reason | Count |",
            "| --- | ---: |",
        ]
    )
    for key, value in sorted(skipped.items()):
        md.append(f"| {key} | {value} |")
    args.md_output.parent.mkdir(parents=True, exist_ok=True)
    args.md_output.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
