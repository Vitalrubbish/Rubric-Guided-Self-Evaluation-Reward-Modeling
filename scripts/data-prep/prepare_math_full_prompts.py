#!/usr/bin/env python3
"""Prepare a broader MATH transfer subset across all subjects and levels."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


SUBJECTS = (
    "algebra",
    "prealgebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "precalculus",
)
BOXED_RE = re.compile(r"\\boxed\s*{")
LEVEL_RE = re.compile(r"(\d+)")


def find_matching_brace(text: str, open_idx: int) -> int | None:
    depth = 0
    for idx in range(open_idx, len(text)):
        if text[idx] == "{":
            depth += 1
        elif text[idx] == "}":
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


def normalize_answer_text(answer: str) -> str:
    text = answer.strip()
    text = text.replace("$", "")
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("\\,", "").replace("\\!", "")
    return text.strip()


def split_top_level_commas(text: str) -> list[str]:
    parts = []
    start = 0
    depth = 0
    for idx, char in enumerate(text):
        if char in "{[(":
            depth += 1
        elif char in "}])":
            depth = max(0, depth - 1)
        elif char == "," and depth == 0:
            parts.append(text[start:idx].strip())
            start = idx + 1
    parts.append(text[start:].strip())
    return [part for part in parts if part]


def answer_type(answer: str) -> str:
    text = normalize_answer_text(answer)
    if "\\pm" in text or "±" in text:
        return "plus_minus"
    if "\\infty" in text or re.match(r"^[\\\[\(].*,.*[\\\]\)]$", text):
        return "interval"
    if text.startswith(("\\{", "{")) and text.endswith(("\\}", "}")):
        return "set"
    if len(split_top_level_commas(text)) > 1:
        return "multi_answer"
    if "\\sqrt" in text or "\\pi" in text:
        return "radical_or_pi"
    if "\\frac" in text or "/" in text:
        return "fraction"
    if re.search(r"[A-Za-z]", text):
        return "symbolic"
    return "numeric"


def build_prompt(problem: str) -> str:
    return (
        "Solve the MATH problem. Write concise reasoning. Then put the final answer on a "
        "separate line in exactly this format: #### <answer>. Use exact simplified form. "
        "For multiple answers, separate them with commas. For intervals or sets, use standard "
        "mathematical notation. Do not write any #### line before the reasoning is complete. "
        "Stop immediately after the final answer line.\n\n"
        f"Problem:\n{problem}\n"
    )


def collect_rows(parquet_root: Path, max_answer_len: int) -> dict[str, dict[int, list[dict]]]:
    by_subject: dict[str, dict[int, list[dict]]] = {}
    for subject in SUBJECTS:
        path = parquet_root / subject / "test-00000-of-00001.parquet"
        df = pd.read_parquet(path)
        rows_by_level: dict[int, list[dict]] = {level: [] for level in range(1, 6)}
        for idx, row in df.iterrows():
            boxed = extract_boxed_answer(str(row["solution"]))
            if boxed is None:
                continue
            gold = normalize_answer_text(boxed)
            if not gold or len(gold) > max_answer_len:
                continue
            level_num = parse_level(str(row["level"]))
            rows_by_level.setdefault(level_num or 99, []).append({
                "id": f"math_full_{subject}_{idx}",
                "dataset": "math",
                "split": "test",
                "source_dataset": "EleutherAI/hendrycks_math",
                "subject": str(row["type"]),
                "subject_key": subject,
                "level": str(row["level"]),
                "level_num": level_num,
                "problem": str(row["problem"]),
                "gold_solution": str(row["solution"]),
                "gold_answer": gold,
                "gold_answer_type": answer_type(gold),
                "prompt": build_prompt(str(row["problem"])),
                "answer_filter": "full_math_boxed_answer_len_limited",
            })
        for rows in rows_by_level.values():
            rows.sort(key=lambda item: item["id"])
        by_subject[subject] = rows_by_level
    return by_subject


def balanced_rows(parquet_root: Path, limit: int, max_answer_len: int) -> list[dict]:
    by_subject = collect_rows(parquet_root, max_answer_len)
    levels = [1, 2, 3, 4, 5]
    cursors = {(subject, level): 0 for subject in SUBJECTS for level in levels}
    rows = []
    while len(rows) < limit:
        made_progress = False
        for level in levels:
            for subject in SUBJECTS:
                idx = cursors[(subject, level)]
                bucket = by_subject.get(subject, {}).get(level, [])
                if idx < len(bucket):
                    rows.append(bucket[idx])
                    cursors[(subject, level)] += 1
                    made_progress = True
                    if len(rows) >= limit:
                        break
            if len(rows) >= limit:
                break
        if not made_progress:
            break
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet-root", type=Path, default=Path("data/raw/hendrycks_math_parquet"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/math_full_prompts_n100.jsonl"))
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--max-answer-len", type=int, default=160)
    args = parser.parse_args()

    rows = balanced_rows(args.parquet_root, args.limit, args.max_answer_len)
    if not rows:
        raise RuntimeError("No rows selected for full MATH subset")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    by_subject: dict[str, int] = {}
    by_level: dict[str, int] = {}
    by_type: dict[str, int] = {}
    for row in rows:
        by_subject[row["subject"]] = by_subject.get(row["subject"], 0) + 1
        by_level[row["level"]] = by_level.get(row["level"], 0) + 1
        by_type[row["gold_answer_type"]] = by_type.get(row["gold_answer_type"], 0) + 1
    print(json.dumps({
        "output": str(args.output),
        "count": len(rows),
        "by_subject": by_subject,
        "by_level": by_level,
        "by_answer_type": by_type,
        "first_id": rows[0]["id"],
        "first_gold_answer": rows[0]["gold_answer"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
