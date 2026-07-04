#!/usr/bin/env python3
"""Prepare GSM8K prompts from the original OpenAI grade-school-math JSONL files."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


FINAL_RE = re.compile(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)")


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def normalize_number(text: str) -> str:
    return text.replace(",", "").strip()


def extract_gold(answer: str) -> str:
    match = FINAL_RE.search(answer)
    if not match:
        raise ValueError(f"Could not extract GSM8K final answer from: {answer[:120]!r}")
    return normalize_number(match.group(1))


def build_prompt(question: str) -> str:
    return (
        "Solve the grade-school math problem. Write 2-5 concise reasoning steps; do not "
        "answer with only the final number. Then put the final numeric answer on a separate "
        "line in exactly this format: #### <answer>. "
        "Stop immediately after that final answer line; do not continue with another problem.\n\n"
        f"Problem:\n{question}\n"
    )


def convert(raw_path: Path, split: str, output_path: Path, limit: int | None) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as out:
        for idx, row in enumerate(read_jsonl(raw_path)):
            if limit is not None and count >= limit:
                break
            question = row["question"].strip()
            answer = row["answer"].strip()
            record = {
                "id": f"gsm8k_{split}_{idx}",
                "dataset": "gsm8k",
                "split": split,
                "question": question,
                "gold_solution": answer,
                "gold_answer": extract_gold(answer),
                "prompt": build_prompt(question),
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-train", type=Path, default=Path("data/raw/gsm8k_train.jsonl"))
    parser.add_argument("--raw-test", type=Path, default=Path("data/raw/gsm8k_test.jsonl"))
    parser.add_argument("--train-output", type=Path, default=Path("data/processed/gsm8k_train_prompts.jsonl"))
    parser.add_argument("--test-output", type=Path, default=Path("data/processed/gsm8k_test_prompts.jsonl"))
    parser.add_argument("--train-limit", type=int, default=200)
    parser.add_argument("--test-limit", type=int, default=100)
    args = parser.parse_args()

    train_count = convert(args.raw_train, "train", args.train_output, args.train_limit)
    test_count = convert(args.raw_test, "test", args.test_output, args.test_limit)
    print(json.dumps({
        "train_output": str(args.train_output),
        "test_output": str(args.test_output),
        "train_count": train_count,
        "test_count": test_count,
    }, indent=2))


if __name__ == "__main__":
    main()
