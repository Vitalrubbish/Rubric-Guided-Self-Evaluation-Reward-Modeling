#!/usr/bin/env python3
"""Build SFT rows from quality-gated executable-rubric test suites."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from src.self_play.executable_rubric_utils import read_jsonl, sha256_file, stable_hash, write_json, write_jsonl


def suite_completion(row: dict[str, Any]) -> str:
    return json.dumps(
        {
            "fn_name": row.get("fn_name") or "",
            "tests": row.get("tests") or [],
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def build_source_index(source_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for row in source_rows:
        row_id = str(row.get("id") or "")
        if row_id and row_id not in index:
            index[row_id] = row
    return index


def build_training_rows(
    suite_rows: list[dict[str, Any]],
    source_rows: list[dict[str, Any]],
    min_tests: int,
    source_tag: str,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    source_by_id = build_source_index(source_rows)
    counts: Counter[str] = Counter()
    output_rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    for suite in suite_rows:
        counts["suite_input"] += 1
        if not suite.get("quality_gate_passed"):
            counts[f"skipped:quality_status:{suite.get('quality_status') or 'unknown'}"] += 1
            continue
        tests = suite.get("tests") or []
        if len(tests) < min_tests:
            counts["skipped:too_few_tests"] += 1
            continue
        suite_id = str(suite.get("id") or "")
        source = source_by_id.get(suite_id)
        if source is None:
            counts["skipped:missing_source_prompt"] += 1
            continue
        completion = suite_completion(suite)
        key = (str(suite.get("problem_id") or ""), completion)
        if key in seen:
            counts["skipped:duplicate_problem_suite"] += 1
            continue
        seen.add(key)
        metadata = dict(source.get("metadata") or {})
        metadata.update(
            {
                "source": source_tag,
                "problem_id": suite.get("problem_id"),
                "source_row_id": suite.get("source_row_id"),
                "suite_id": suite.get("id"),
                "suite_response_id": suite.get("response_id"),
                "quality_status": suite.get("quality_status"),
                "test_count": len(tests),
                "raw_test_count": suite.get("raw_test_count"),
                "canonical_valid_test_count": suite.get("canonical_valid_test_count"),
                "source_failed_pass_count": suite.get("source_failed_pass_count"),
                "source_failure_caught": suite.get("source_failure_caught"),
            }
        )
        out = dict(source)
        out.update(
            {
                "id": f"{suite_id}__{source_tag}_{stable_hash(completion)}",
                "split": "train",
                "completion": completion,
                "source": source_tag,
                "metadata": metadata,
            }
        )
        output_rows.append(out)
        counts["accepted"] += 1

    output_rows.sort(key=lambda row: str(row.get("id") or ""))
    return output_rows, counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suites", type=Path, nargs="+", required=True)
    parser.add_argument("--source-inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--min-tests", type=int, default=2)
    parser.add_argument("--source-tag", default="executable_rubric_quality_gated_testgen")
    parser.add_argument("--allow-empty", action="store_true")
    args = parser.parse_args()

    if args.min_tests < 1:
        raise ValueError("--min-tests must be positive")

    suite_rows: list[dict[str, Any]] = []
    for path in args.suites:
        suite_rows.extend(read_jsonl(path))
    source_rows: list[dict[str, Any]] = []
    for path in args.source_inputs:
        source_rows.extend(read_jsonl(path))

    output_rows, counts = build_training_rows(suite_rows, source_rows, args.min_tests, args.source_tag)
    if not output_rows and not args.allow_empty:
        raise SystemExit("no quality-gated suites matched the requested filters")

    write_jsonl(args.output, output_rows)
    summary = {
        "suites": [str(path) for path in args.suites],
        "suites_sha256": {str(path): sha256_file(path) for path in args.suites},
        "source_inputs": [str(path) for path in args.source_inputs],
        "source_inputs_sha256": {str(path): sha256_file(path) for path in args.source_inputs},
        "output": str(args.output),
        "output_sha256": sha256_file(args.output),
        "min_tests": args.min_tests,
        "source_tag": args.source_tag,
        "suite_rows_input": len(suite_rows),
        "source_rows_input": len(source_rows),
        "sft_rows": len(output_rows),
        "unique_problem_count": len({row.get("metadata", {}).get("problem_id") for row in output_rows}),
        "test_count_distribution": dict(Counter(str(row.get("metadata", {}).get("test_count") or 0) for row in output_rows)),
        "counts": dict(counts),
        "policy": {
            "suite_filter": "only quality_gate_passed suites are eligible",
            "completion": "canonical-valid retained tests are serialized as compact JSON",
            "hidden_labels": "canonical and failed-code execution are used only as offline quality gates",
        },
    }
    write_json(args.summary_output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
