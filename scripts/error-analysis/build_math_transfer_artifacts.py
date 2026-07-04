#!/usr/bin/env python3
"""Build MATH transfer failure taxonomy artifacts."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import yaml


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


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
        pattern = row.get("error_pattern") or "unclassified_wrong_answer"
        counts[pattern] += 1
        record = {
            "id": row.get("id"),
            "subject": row.get("subject"),
            "level": row.get("level"),
            "problem": row.get("problem"),
            "gold_answer": row.get("gold_answer"),
            "predicted_answer": row.get("predicted_answer"),
            "failure_type": row.get("failure_type"),
            "error_pattern": pattern,
            "generated_answer": row.get("generated_answer"),
        }
        failures.append(record)
        if len(examples[pattern]) < 3:
            examples[pattern].append({
                "id": row.get("id"),
                "gold_answer": row.get("gold_answer"),
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
        "patterns": dict(counts),
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    taxonomy = {
        "dataset": "math_transfer",
        "source": str(args.input),
        "num_failures": len(failures),
        "patterns": [
            {
                "id": pattern,
                "count": count,
                "description": {
                    "missing_final_answer": "No parseable final answer was produced.",
                    "ambiguous_final_answer": "The answer was not clearly marked, so verifier had to fallback.",
                    "reasoning_truncation": "The response is too short or incomplete.",
                    "symbolic_or_arithmetic_error": "The response contains symbolic/arithmetic work but the final expression is not equivalent.",
                    "wrong_problem_model": "The response likely models the mathematical relationship incorrectly.",
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
