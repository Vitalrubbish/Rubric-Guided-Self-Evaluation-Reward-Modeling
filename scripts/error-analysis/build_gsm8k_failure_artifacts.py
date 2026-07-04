#!/usr/bin/env python3
"""Build GSM8K failure records, taxonomy summary, and cluster-style artifacts."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import yaml


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def classify(row: dict) -> tuple[str, str]:
    text = row.get("generated_answer") or ""
    predicted = row.get("predicted_answer")
    extraction = row.get("answer_extraction_method")

    if predicted is None:
        return "missing_final_answer", "No parseable numeric answer was produced."
    if "####" not in text:
        return "final_format_violation", "The response omitted the requested GSM8K #### final-answer format."
    if len(text.strip()) < 80 or text.rstrip().endswith(("=", "+", "-", "*", "/")):
        return "reasoning_truncation", "The reasoning appears too short or unfinished."
    if extraction == "last_number":
        return "ambiguous_final_answer", "The verifier had to fall back to the last number instead of an explicit final answer."

    number_count = len(re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text))
    operator_count = sum(text.count(op) for op in ["+", "-", "*", "/", "="])
    if number_count >= 5 and operator_count >= 3:
        return "arithmetic_or_algebra_slip", "The response attempted a multi-step computation but ended with the wrong number."
    if any(token in text.lower() for token in ["assume", "let ", "equation", "total"]):
        return "wrong_problem_model", "The response likely modeled the quantities or relationships incorrectly."
    return "unclassified_wrong_answer", "The response is wrong but the heuristic taxonomy needs more evidence."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--failure-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--taxonomy-output", type=Path, required=True)
    args = parser.parse_args()

    rows = list(read_jsonl(args.input))
    failures = []
    counts: Counter[str] = Counter()
    examples: dict[str, list[dict]] = defaultdict(list)

    for row in rows:
        if row.get("passed"):
            continue
        pattern, rationale = classify(row)
        counts[pattern] += 1
        record = {
            "id": row.get("id"),
            "dataset": row.get("dataset"),
            "split": row.get("split"),
            "question": row.get("question"),
            "gold_answer": row.get("normalized_gold_answer") or row.get("gold_answer"),
            "predicted_answer": row.get("predicted_answer"),
            "failure_type": row.get("failure_type"),
            "error_pattern": pattern,
            "rationale": rationale,
            "generated_answer": row.get("generated_answer"),
        }
        failures.append(record)
        if len(examples[pattern]) < 3:
            examples[pattern].append({
                "id": row.get("id"),
                "gold_answer": record["gold_answer"],
                "predicted_answer": row.get("predicted_answer"),
            })

    args.failure_output.parent.mkdir(parents=True, exist_ok=True)
    with args.failure_output.open("w", encoding="utf-8") as out:
        for row in failures:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "input": str(args.input),
        "total": len(rows),
        "passed": sum(1 for row in rows if row.get("passed")),
        "failed": len(failures),
        "patterns": counts,
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    taxonomy = {
        "dataset": "gsm8k",
        "source": str(args.input),
        "num_failures": len(failures),
        "patterns": [
            {
                "id": pattern,
                "count": count,
                "description": {
                    "missing_final_answer": "No numeric final answer can be extracted.",
                    "final_format_violation": "The response does not follow the required #### final answer format.",
                    "reasoning_truncation": "The solution appears unfinished before reaching a reliable answer.",
                    "ambiguous_final_answer": "The final answer is not explicitly marked, making self-evaluation unreliable.",
                    "arithmetic_or_algebra_slip": "The setup is plausible but the arithmetic/algebra execution is wrong.",
                    "wrong_problem_model": "The quantities or relationships in the word problem are modeled incorrectly.",
                    "unclassified_wrong_answer": "The answer is wrong and needs deeper attribution.",
                }.get(pattern, pattern),
                "examples": examples[pattern],
            }
            for pattern, count in counts.most_common()
        ],
    }
    args.taxonomy_output.parent.mkdir(parents=True, exist_ok=True)
    args.taxonomy_output.write_text(yaml.safe_dump(taxonomy, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
