#!/usr/bin/env python3
"""Merge MBPP and HumanEval+ files into one coding prompt JSONL."""

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


def mbpp_prompt(problem: dict) -> str:
    tests = "\n".join(problem.get("test_list", []))
    return (
        "You are an expert Python programmer. Solve the following task.\n"
        "Return only valid Python code, with no Markdown fences and no explanation.\n\n"
        f"Task: {problem['text']}\n\n"
        "Your code must define every function/class used by these tests:\n"
        f"{tests}\n\n"
        "Python code:\n"
    )


def convert_mbpp(raw_dir: Path):
    for split in ("train", "test", "validation"):
        path = raw_dir / f"mbpp_{split}.jsonl"
        for row in read_jsonl(path):
            task_id = row["task_id"]
            yield {
                "id": f"mbpp/{split}/{task_id}",
                "dataset": "mbpp",
                "split": split,
                "prompt": mbpp_prompt(row),
                "canonical_solution": row.get("code", ""),
                "test_list": row.get("test_list", []),
                "test_setup_code": row.get("test_setup_code", ""),
                "entry_point": None,
            }


def convert_humanevalplus(raw_dir: Path):
    path = raw_dir / "humanevalplus_test.jsonl"
    for row in read_jsonl(path):
        task_id = row["task_id"]
        prompt = (
            "You are an expert Python programmer. Complete the function below.\n"
            "Return only valid Python code, with no Markdown fences and no explanation.\n\n"
            f"{row['prompt']}"
        )
        yield {
            "id": f"humanevalplus/{task_id}",
            "dataset": "humanevalplus",
            "split": "test",
            "prompt": prompt,
            "canonical_solution": row.get("canonical_solution", ""),
            "test": row.get("test", ""),
            "entry_point": row.get("entry_point"),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/coding_prompts.jsonl"))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.output.open("w", encoding="utf-8") as f:
        for item in list(convert_mbpp(args.raw_dir)) + list(convert_humanevalplus(args.raw_dir)):
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            count += 1
            if args.limit is not None and count >= args.limit:
                break

    print(f"wrote {count} prompts to {args.output}")


if __name__ == "__main__":
    main()
