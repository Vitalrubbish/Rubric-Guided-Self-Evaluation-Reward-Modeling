#!/usr/bin/env python3
"""Build failure datasets, summary metrics, and an initial rule taxonomy."""

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


def classify_pattern(row: dict) -> str:
    failure_type = row.get("failure_type") or "unknown"
    error = (row.get("error") or "").lower()
    code = row.get("extracted_code") or row.get("generated_code") or ""

    if failure_type == "syntax_error":
        if "unexpected indent" in error:
            return "syntax_unexpected_indent"
        if "invalid syntax" in error and re.search(r"\)\s+def\s+", code):
            return "syntax_duplicate_function_after_return"
        if "invalid syntax" in error and "```" in row.get("generated_code", ""):
            return "syntax_markdown_or_fence_artifact"
        if "was never closed" in error or "unexpected eof" in error:
            return "syntax_truncated_or_unclosed_block"
        return "syntax_malformed_code"

    if failure_type == "runtime_error":
        for name in ("NameError", "TypeError", "AttributeError", "IndexError", "KeyError", "ValueError", "ImportError", "ZeroDivisionError"):
            if name.lower() in error:
                return "runtime_" + name.replace("Error", "").lower() + "_error"
        return "runtime_other_exception"

    if failure_type == "logic_error":
        return "logic_wrong_output"

    if failure_type == "timeout":
        return "timeout_nonterminating_or_too_slow"

    if failure_type == "generation_failure":
        return "generation_empty_or_non_code"

    return "unknown_failure"


def short(text: str, limit: int = 500) -> str:
    text = (text or "").strip()
    return text[:limit] + ("..." if len(text) > limit else "")


def infer_split(row: dict) -> str:
    if row.get("split"):
        return row["split"]
    row_id = row.get("id", "")
    if row_id.startswith("mbpp/"):
        parts = row_id.split("/")
        if len(parts) >= 3:
            return parts[1]
    if row_id.startswith("humanevalplus/"):
        return "test"
    return "unknown"


def taxonomy_item(pattern: str, rows: list[dict], total_failures: int) -> dict:
    examples = []
    for row in rows[:3]:
        examples.append(
            {
                "id": row.get("id"),
                "dataset": row.get("dataset"),
                "failure_type": row.get("failure_type"),
                "error": short(row.get("error", ""), 220),
                "snippet": short(row.get("extracted_code") or row.get("generated_code", ""), 300),
            }
        )

    descriptions = {
        "syntax_unexpected_indent": "Generated completion has indentation inconsistent with the executable context.",
        "syntax_duplicate_function_after_return": "Generated code repeats or concatenates function definitions without separators, producing invalid syntax.",
        "syntax_markdown_or_fence_artifact": "Generated output contains Markdown or formatting artifacts that interfere with execution.",
        "syntax_truncated_or_unclosed_block": "Generated code appears truncated or has an unclosed syntactic structure.",
        "syntax_malformed_code": "Generated code is syntactically invalid for other reasons.",
        "runtime_name_error": "Code references an undefined variable, function, or class.",
        "runtime_type_error": "Code uses a value with the wrong type or wrong call signature.",
        "runtime_attribute_error": "Code accesses an attribute or method that does not exist.",
        "runtime_index_error": "Code indexes outside the valid range.",
        "runtime_key_error": "Code looks up a missing key.",
        "runtime_value_error": "Code raises a ValueError during execution.",
        "runtime_import_error": "Code imports an unavailable or invalid module.",
        "runtime_zerodivision_error": "Code divides by zero.",
        "runtime_other_exception": "Code parses but raises an uncategorized runtime exception.",
        "logic_wrong_output": "Code executes but fails one or more reference tests.",
        "timeout_nonterminating_or_too_slow": "Code exceeds the execution timeout.",
        "generation_empty_or_non_code": "Model output is empty or does not contain executable code.",
        "unknown_failure": "Failure does not match an existing rule.",
    }

    return {
        "pattern": pattern,
        "description": descriptions.get(pattern, "Automatically discovered rule bucket."),
        "frequency": len(rows),
        "ratio_among_failures": round(len(rows) / total_failures, 4) if total_failures else 0,
        "linked_failure_types": sorted({row.get("failure_type") for row in rows}),
        "example_ids": [row.get("id") for row in rows[:10]],
        "examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--failure-output", type=Path, default=Path("data/analysis/coding_failures_qwen25_k1.jsonl"))
    parser.add_argument("--summary-output", type=Path, default=Path("data/analysis/coding_baseline_summary_qwen25_k1.json"))
    parser.add_argument("--taxonomy-output", type=Path, default=Path("data/analysis/coding_error_taxonomy_initial.yaml"))
    args = parser.parse_args()

    rows = list(read_jsonl(args.input))
    failures = []
    by_dataset = defaultdict(Counter)
    by_split = defaultdict(Counter)
    by_failure_type = defaultdict(Counter)
    by_pattern = defaultdict(list)

    for row in rows:
        dataset = row.get("dataset", "unknown")
        split = infer_split(row)
        by_dataset[dataset]["total"] += 1
        by_split[f"{dataset}/{split}"]["total"] += 1
        if row.get("passed"):
            by_dataset[dataset]["passed"] += 1
            by_split[f"{dataset}/{split}"]["passed"] += 1
            continue

        by_dataset[dataset]["failed"] += 1
        by_split[f"{dataset}/{split}"]["failed"] += 1
        by_failure_type[dataset][row.get("failure_type") or "unknown"] += 1

        pattern = classify_pattern(row)
        failure = {
            "id": row.get("id"),
            "dataset": dataset,
            "split": split,
            "prompt": row.get("prompt"),
            "generated_code": row.get("generated_code"),
            "extracted_code": row.get("extracted_code"),
            "passed": False,
            "failure_type": row.get("failure_type"),
            "error_pattern": pattern,
            "error": row.get("error"),
            "entry_point": row.get("entry_point"),
            "test_list": row.get("test_list"),
            "test": row.get("test"),
        }
        failures.append(failure)
        by_pattern[pattern].append(failure)

    summary = {
        "input": str(args.input),
        "total": len(rows),
        "passed": sum(1 for row in rows if row.get("passed")),
        "failed": len(failures),
        "pass_rate": round(sum(1 for row in rows if row.get("passed")) / len(rows), 6) if rows else 0,
        "by_dataset": {
            key: {
                "total": value["total"],
                "passed": value["passed"],
                "failed": value["failed"],
                "pass_rate": round(value["passed"] / value["total"], 6) if value["total"] else 0,
            }
            for key, value in sorted(by_dataset.items())
        },
        "by_split": {
            key: {
                "total": value["total"],
                "passed": value["passed"],
                "failed": value["failed"],
                "pass_rate": round(value["passed"] / value["total"], 6) if value["total"] else 0,
            }
            for key, value in sorted(by_split.items())
        },
        "failure_types": {key: dict(value) for key, value in sorted(by_failure_type.items())},
        "error_patterns": {key: len(value) for key, value in sorted(by_pattern.items(), key=lambda item: (-len(item[1]), item[0]))},
    }

    taxonomy = {
        "name": "coding_error_taxonomy_initial_rules",
        "source": str(args.input),
        "total_failures": len(failures),
        "note": "Initial rule-based taxonomy for Phase 1. Later clustering/LLM attribution should refine these buckets.",
        "patterns": [
            taxonomy_item(pattern, pattern_rows, len(failures))
            for pattern, pattern_rows in sorted(by_pattern.items(), key=lambda item: (-len(item[1]), item[0]))
        ],
    }

    args.failure_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.taxonomy_output.parent.mkdir(parents=True, exist_ok=True)

    with args.failure_output.open("w", encoding="utf-8") as f:
        for failure in failures:
            f.write(json.dumps(failure, ensure_ascii=False) + "\n")
    args.summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    args.taxonomy_output.write_text(yaml.safe_dump(taxonomy, allow_unicode=True, sort_keys=False), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote failures to {args.failure_output}")
    print(f"wrote taxonomy to {args.taxonomy_output}")


if __name__ == "__main__":
    main()
