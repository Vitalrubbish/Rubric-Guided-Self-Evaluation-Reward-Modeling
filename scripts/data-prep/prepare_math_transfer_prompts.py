#!/usr/bin/env python3
"""Prepare a verifier-friendly MATH transfer subset from Hendrycks MATH parquet files."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


SUBJECTS = ("algebra", "prealgebra")
BOXED_RE = re.compile(r"\\boxed\s*{")
LEVEL_RE = re.compile(r"(\d+)")


def find_matching_brace(text: str, open_idx: int) -> int | None:
    depth = 0
    for idx in range(open_idx, len(text)):
        char = text[idx]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return idx
    return None


def extract_boxed_answer(solution: str) -> str | None:
    matches = list(BOXED_RE.finditer(solution or ""))
    if not matches:
        return None
    match = matches[-1]
    open_idx = match.end() - 1
    close_idx = find_matching_brace(solution, open_idx)
    if close_idx is None:
        return None
    return solution[open_idx + 1 : close_idx].strip()


def parse_level(level: str) -> int | None:
    match = LEVEL_RE.search(level or "")
    return int(match.group(1)) if match else None


def normalize_candidate(answer: str) -> str:
    text = answer.strip()
    text = text.replace("$", "")
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("\\,", "").replace("\\!", "")
    text = text.replace("\\%", "%")
    return text.strip()


def is_verifier_safe(answer: str) -> bool:
    text = normalize_candidate(answer)
    if not text or len(text) > 60:
        return False
    lower = text.lower()
    unsafe_tokens = [
        "\\begin",
        "\\text",
        "\\infty",
        "\\cup",
        "\\cap",
        "\\pm",
        "\\le",
        "\\ge",
        "\\lt",
        "\\gt",
        "\\mod",
        "\\equiv",
        "\\ldots",
        "...",
    ]
    if any(token in lower for token in unsafe_tokens):
        return False
    # Keep single numeric/fraction answers and short symbolic expressions. Exclude tuples,
    # sets, intervals, matrices, and answer lists for this first reliable verifier pass.
    if any(ch in text for ch in "[]();"):
        return False
    if "," in text:
        return False
    return True


def build_prompt(problem: str) -> str:
    return (
        "Solve the MATH problem. Write 2-6 concise reasoning steps. Then put the final "
        "answer on a separate line in exactly this format: #### <answer>. "
        "Stop immediately after that final answer line.\n\n"
        f"Problem:\n{problem}\n"
    )


def iter_rows(parquet_root: Path, max_level: int):
    for subject in SUBJECTS:
        path = parquet_root / subject / "test-00000-of-00001.parquet"
        df = pd.read_parquet(path)
        for idx, row in df.iterrows():
            level_num = parse_level(str(row["level"]))
            if level_num is None or level_num > max_level:
                continue
            boxed = extract_boxed_answer(str(row["solution"]))
            if boxed is None or not is_verifier_safe(boxed):
                continue
            yield {
                "id": f"math_transfer_{subject}_{idx}",
                "dataset": "math",
                "split": "test",
                "source_dataset": "EleutherAI/hendrycks_math",
                "subject": str(row["type"]),
                "subject_key": subject,
                "level": str(row["level"]),
                "level_num": level_num,
                "problem": str(row["problem"]),
                "gold_solution": str(row["solution"]),
                "gold_answer": normalize_candidate(boxed),
                "prompt": build_prompt(str(row["problem"])),
                "answer_filter": "verifier_safe_level_1_3_algebra_prealgebra",
            }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet-root", type=Path, default=Path("data/raw/hendrycks_math_parquet"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/math_transfer_prompts_n100.jsonl"))
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--max-level", type=int, default=3)
    args = parser.parse_args()

    rows = []
    for row in iter_rows(args.parquet_root, args.max_level):
        rows.append(row)
        if len(rows) >= args.limit:
            break
    if not rows:
        raise RuntimeError("No MATH transfer rows selected")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    by_subject: dict[str, int] = {}
    by_level: dict[str, int] = {}
    for row in rows:
        by_subject[row["subject"]] = by_subject.get(row["subject"], 0) + 1
        by_level[row["level"]] = by_level.get(row["level"], 0) + 1
    print(json.dumps({
        "output": str(args.output),
        "count": len(rows),
        "by_subject": by_subject,
        "by_level": by_level,
        "first_id": rows[0]["id"],
        "first_gold_answer": rows[0]["gold_answer"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
