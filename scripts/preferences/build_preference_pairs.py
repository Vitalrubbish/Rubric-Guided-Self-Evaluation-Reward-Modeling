#!/usr/bin/env python3
"""Build preference pairs from verified failures and canonical solutions."""

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


def canonical_map(prompts_path: Path) -> dict[str, dict]:
    result = {}
    for row in read_jsonl(prompts_path):
        result[row["id"]] = row
    return result


def clean_code(text: str) -> str:
    text = text or ""
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            candidate = part
            if candidate.lstrip().startswith("python"):
                candidate = candidate.lstrip()[len("python") :]
            if "def " in candidate or "class " in candidate:
                return candidate.strip("\n\r")
    return text.strip("\n\r")


def chosen_solution(prompt_row: dict) -> str:
    canonical = (prompt_row.get("canonical_solution") or "").strip("\n\r")
    dataset = prompt_row.get("dataset")
    if dataset == "humanevalplus":
        # HumanEval canonical solutions are function bodies. Pair the same prompt with the body completion.
        return canonical
    return canonical


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--labeled", type=Path, required=True)
    parser.add_argument("--failures", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/preferences/preference_pairs_qwen25_k1.jsonl"))
    parser.add_argument("--max-pairs", type=int, default=None)
    args = parser.parse_args()

    prompts = canonical_map(args.prompts)
    failure_patterns = {row["id"]: row for row in read_jsonl(args.failures)}
    count = 0
    skipped = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in read_jsonl(args.labeled):
            if row.get("passed"):
                continue
            prompt_row = prompts.get(row["id"])
            if not prompt_row:
                skipped += 1
                continue
            chosen = chosen_solution(prompt_row)
            rejected = clean_code(row.get("generated_code", ""))
            if not chosen or not rejected:
                skipped += 1
                continue
            failure = failure_patterns.get(row["id"], {})
            pair = {
                "id": row["id"],
                "dataset": row.get("dataset"),
                "split": prompt_row.get("split"),
                "prompt": row.get("prompt"),
                "chosen": chosen,
                "rejected": rejected,
                "chosen_source": "canonical_solution",
                "rejected_source": "qwen25_k1_failed_output",
                "failure_type": row.get("failure_type"),
                "error_pattern": failure.get("error_pattern"),
                "rubric_version": "auto_rubric_refined_coding_v1",
            }
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
            count += 1
            if args.max_pairs is not None and count >= args.max_pairs:
                break

    print(f"wrote {count} preference pairs to {args.output}; skipped={skipped}")


if __name__ == "__main__":
    main()
