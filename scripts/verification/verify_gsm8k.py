#!/usr/bin/env python3
"""Exact-answer verifier for GSM8K generated responses."""

from __future__ import annotations

import argparse
import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path


FINAL_RE = re.compile(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)")
NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def normalize_number(text: str | None) -> str | None:
    if text is None:
        return None
    cleaned = text.replace(",", "").strip().rstrip(".")
    if not cleaned:
        return None
    try:
        value = Decimal(cleaned)
    except InvalidOperation:
        return cleaned
    if value == value.to_integral():
        return str(value.quantize(Decimal(1)))
    return format(value.normalize(), "f").rstrip("0").rstrip(".")


def extract_prediction(text: str) -> tuple[str | None, str]:
    final_matches = FINAL_RE.findall(text or "")
    if final_matches:
        return normalize_number(final_matches[-1]), "hash_final"

    tail_patterns = [
        r"(?:final answer|answer is|therefore|so)\D{0,40}([-+]?\d[\d,]*(?:\.\d+)?)",
        r"=\s*([-+]?\d[\d,]*(?:\.\d+)?)\s*$",
    ]
    for pattern in tail_patterns:
        matches = re.findall(pattern, text or "", flags=re.IGNORECASE)
        if matches:
            return normalize_number(matches[-1]), "phrase_final"

    numbers = NUMBER_RE.findall(text or "")
    if numbers:
        return normalize_number(numbers[-1]), "last_number"
    return None, "no_number"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, default=None)
    args = parser.parse_args()

    rows = list(read_jsonl(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    passed = 0
    counts: dict[str, int] = {}

    with args.output.open("w", encoding="utf-8") as out:
        for row in rows:
            predicted, extraction_method = extract_prediction(row.get("generated_answer", ""))
            gold = normalize_number(row.get("gold_answer"))
            ok = predicted is not None and gold is not None and predicted == gold
            if ok:
                passed += 1
            failure_type = None if ok else ("no_answer" if predicted is None else "wrong_answer")
            counts[failure_type or "passed"] = counts.get(failure_type or "passed", 0) + 1
            row.update({
                "predicted_answer": predicted,
                "normalized_gold_answer": gold,
                "answer_extraction_method": extraction_method,
                "passed": ok,
                "failure_type": failure_type,
            })
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "total": len(rows),
        "passed": passed,
        "failed": len(rows) - passed,
        "accuracy": passed / len(rows) if rows else None,
        "counts": counts,
    }
    if args.summary_output:
        summary_path = Path(args.summary_output)
    else:
        summary_path = args.output.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
