#!/usr/bin/env python3
"""Build train-only preference pairs augmented with successful rule-revised outputs."""

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


def clean_code(text: str) -> str:
    text = text or ""
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            candidate = part
            if candidate.lstrip().lower().startswith("python"):
                candidate = candidate.lstrip()[len("python") :]
            if "def " in candidate or "class " in candidate:
                return candidate.strip("\n\r")
    return text.strip("\n\r")


def split_from_id(row: dict) -> str | None:
    if row.get("split"):
        return row["split"]
    row_id = row.get("id", "")
    parts = row_id.split("/")
    if len(parts) >= 3 and parts[0] == "mbpp":
        return parts[1]
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical-pairs", type=Path, required=True)
    parser.add_argument("--original-labeled", type=Path, required=True)
    parser.add_argument("--revised-labeled", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=str, default="mbpp")
    parser.add_argument("--split", type=str, default="train")
    args = parser.parse_args()

    canonical_rows = []
    seen_keys = set()
    for row in read_jsonl(args.canonical_pairs):
        if row.get("dataset") != args.dataset or row.get("split") != args.split:
            continue
        key = (row["id"], row["chosen_source"], row["rejected_source"])
        seen_keys.add(key)
        canonical_rows.append(row)

    original = {row["id"]: row for row in read_jsonl(args.original_labeled)}
    revised = {row["id"]: row for row in read_jsonl(args.revised_labeled)}
    augmented_rows = list(canonical_rows)
    revised_success = 0
    skipped = 0

    for item_id, orig in sorted(original.items()):
        if orig.get("dataset") != args.dataset or split_from_id(orig) != args.split:
            continue
        rev = revised.get(item_id)
        if not rev:
            skipped += 1
            continue
        if orig.get("passed") or not rev.get("passed"):
            continue

        chosen = clean_code(rev.get("generated_code", ""))
        rejected = clean_code(orig.get("generated_code", ""))
        if not chosen or not rejected:
            skipped += 1
            continue

        key = (item_id, "rule_revised_success_output", "qwen25_k1_failed_output")
        if key in seen_keys:
            continue
        seen_keys.add(key)
        augmented_rows.append(
            {
                "id": item_id,
                "dataset": args.dataset,
                "split": args.split,
                "prompt": orig.get("prompt"),
                "chosen": chosen,
                "rejected": rejected,
                "chosen_source": "rule_revised_success_output",
                "rejected_source": "qwen25_k1_failed_output",
                "failure_type": orig.get("failure_type"),
                "error_pattern": None,
                "rubric_version": "auto_rubric_refined_coding_v1",
                "revision_method": rev.get("revision_method"),
                "revision_edits": rev.get("revision_edits", []),
            }
        )
        revised_success += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in augmented_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "output": str(args.output),
        "canonical_pairs": len(canonical_rows),
        "rule_revised_success_pairs": revised_success,
        "total_pairs": len(augmented_rows),
        "skipped": skipped,
        "dataset": args.dataset,
        "split": args.split,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
