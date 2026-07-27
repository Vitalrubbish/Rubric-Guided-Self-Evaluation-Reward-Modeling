#!/usr/bin/env python3
"""Extract and quality-gate self-written executable rubric tests."""

from __future__ import annotations

import argparse
import ast
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from src.self_play.executable_rubric_utils import (
    case_key,
    execute_function_tests,
    normalize_case,
    parse_apps_input_output,
    passed_count,
    read_jsonl,
    sha256_file,
    write_json,
    write_jsonl,
)


FENCED_RE = re.compile(r"```(?:json|python)?\s*(.*?)```", flags=re.DOTALL | re.IGNORECASE)


def balanced_object_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    start: int | None = None
    depth = 0
    in_string = False
    escape = False
    for index, char in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                candidates.append(text[start : index + 1])
                start = None
    return candidates


def parse_object_candidate(candidate: str) -> tuple[dict[str, Any] | None, str]:
    try:
        parsed = json.loads(candidate)
        return (parsed, "json") if isinstance(parsed, dict) else (None, "not_object")
    except json.JSONDecodeError:
        pass
    try:
        parsed = ast.literal_eval(candidate)
        return (parsed, "python_literal") if isinstance(parsed, dict) else (None, "not_object")
    except Exception:
        pass
    return None, "parse_failed"


def looks_like_suite(parsed: dict[str, Any]) -> bool:
    return any(key in parsed for key in ("tests", "cases", "test_cases"))


def extract_json_object(text: str) -> tuple[dict[str, Any] | None, str]:
    text = text.strip()
    fenced = FENCED_RE.search(text)
    candidates: list[tuple[str, str]] = []
    if fenced:
        candidates.append((fenced.group(1).strip(), "fenced"))
    for candidate in reversed(balanced_object_candidates(text)):
        candidates.append((candidate.strip(), "balanced"))
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidates.append((text[start : end + 1], "outer_span"))
    candidates.append((text, "whole_text"))
    fallback: tuple[dict[str, Any], str] | None = None
    for candidate, source in candidates:
        if not candidate.strip():
            continue
        parsed, parse_status = parse_object_candidate(candidate)
        if parsed is None:
            continue
        status = f"{parse_status}:{source}"
        if looks_like_suite(parsed):
            return parsed, status
        if fallback is None:
            fallback = (parsed, status)
    if fallback is not None:
        return fallback
    return None, "parse_failed"


def parse_tests(
    raw_text: str,
    expected_fn_name: str,
    min_tests: int,
    max_tests: int,
) -> tuple[list[dict[str, Any]], str, list[str]]:
    parsed, parse_status = extract_json_object(raw_text)
    notes = [f"parse:{parse_status}"]
    if parsed is None:
        return [], "parse_failed", notes
    fn_name = str(parsed.get("fn_name") or parsed.get("function") or "").strip()
    if fn_name and fn_name != expected_fn_name:
        notes.append(f"fn_name_mismatch:{fn_name}")
        return [], "fn_name_mismatch", notes
    if not fn_name:
        notes.append("missing_fn_name_filled_from_source")
    raw_cases = parsed.get("tests")
    if raw_cases is None:
        raw_cases = parsed.get("cases") or parsed.get("test_cases")
    if not isinstance(raw_cases, list):
        return [], "missing_tests", notes

    cases: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_case in raw_cases:
        case = normalize_case(raw_case)
        if case is None:
            notes.append("dropped_malformed_case")
            continue
        key = case_key(case)
        if key in seen:
            notes.append("dropped_duplicate_case")
            continue
        seen.add(key)
        cases.append(case)
        if len(cases) >= max_tests:
            break
    if len(cases) < min_tests:
        return cases, "too_few_tests", notes
    return cases, "ok", notes


def passed_flags(result: dict[str, Any], total: int) -> list[bool]:
    if result.get("setup_error"):
        return [False] * total
    flags = [False] * total
    for item in result.get("results") or []:
        index = item.get("test_index")
        if isinstance(index, int) and 0 <= index < total:
            flags[index] = bool(item.get("passed"))
    return flags


def evaluate_generation(
    generation: dict[str, Any],
    source: dict[str, Any],
    min_tests: int,
    max_tests: int,
    timeout: float,
) -> dict[str, Any]:
    metadata = source.get("metadata") or {}
    fn_name = str(metadata.get("fn_name") or parse_apps_input_output(source).get("fn_name") or "")
    raw_text = str(generation.get("generated_code") or generation.get("completion") or "")
    cases, extraction_status, notes = parse_tests(raw_text, fn_name, min_tests, max_tests)
    row: dict[str, Any] = {
        "id": generation.get("id"),
        "response_id": generation.get("response_id"),
        "source_row_id": source.get("source_row_id"),
        "problem_id": source.get("problem_id"),
        "fn_name": fn_name,
        "tests": cases,
        "test_count": len(cases),
        "raw_test_count": len(cases),
        "extraction_status": extraction_status,
        "extraction_notes": notes,
        "quality_gate_passed": False,
        "canonical_passed_all_tests": False,
        "source_failure_caught": False,
    }
    if extraction_status != "ok":
        row["quality_status"] = extraction_status
        return row

    canonical = execute_function_tests(str(source.get("canonical_solution") or ""), fn_name, cases, timeout=timeout)
    total = len(cases)
    canonical_flags = passed_flags(canonical, total)
    usable_cases = [case for case, ok in zip(cases, canonical_flags) if ok]
    dropped_indices = [index for index, ok in enumerate(canonical_flags) if not ok]
    usable_total = len(usable_cases)
    row.update(
        {
            "raw_tests": cases,
            "raw_canonical_result": canonical,
            "raw_canonical_pass_count": passed_count(canonical, total),
            "canonical_valid_test_count": usable_total,
            "dropped_test_indices": dropped_indices,
            "tests": usable_cases,
            "test_count": usable_total,
            "canonical_passed_all_tests": total > 0 and usable_total == total,
        }
    )
    if usable_total < min_tests:
        row.update(
            {
                "quality_status": "too_few_canonical_valid_tests",
                "canonical_result": canonical,
                "canonical_pass_count": usable_total,
            }
        )
        return row

    canonical_filtered = execute_function_tests(
        str(source.get("canonical_solution") or ""),
        fn_name,
        usable_cases,
        timeout=timeout,
    )
    failed = execute_function_tests(str(source.get("failed_code") or ""), fn_name, usable_cases, timeout=timeout)
    failed_pass_count = passed_count(failed, usable_total)
    source_failure_caught = bool(failed.get("setup_error")) or failed_pass_count < usable_total
    row.update(
        {
            "canonical_result": canonical_filtered,
            "source_failed_result": failed,
            "canonical_pass_count": passed_count(canonical_filtered, usable_total),
            "source_failed_pass_count": failed_pass_count,
            "source_failure_caught": source_failure_caught,
            "quality_gate_passed": source_failure_caught,
            "quality_status": "ok" if source_failure_caught else "failed_to_catch_source_failure",
        }
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--source-input", type=Path, default=Path("data/self_play/executable_rubric_probe_testgen_input.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/self_play/executable_rubric_probe_tests_extracted.jsonl"))
    parser.add_argument("--summary-output", type=Path, default=Path("data/self_play/executable_rubric_probe_tests_summary.json"))
    parser.add_argument("--min-tests", type=int, default=3)
    parser.add_argument("--max-tests", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args()

    source_rows = read_jsonl(args.source_input)
    source_by_id = {str(row.get("id")): row for row in source_rows}
    generations = read_jsonl(args.generations)

    output_rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for generation in generations:
        source = source_by_id.get(str(generation.get("id")))
        if source is None:
            counts["skipped:missing_source"] += 1
            continue
        evaluated = evaluate_generation(generation, source, args.min_tests, args.max_tests, args.timeout)
        output_rows.append(evaluated)
        counts[f"quality_status:{evaluated['quality_status']}"] += 1
        counts[f"extraction_status:{evaluated['extraction_status']}"] += 1
        if evaluated.get("quality_gate_passed"):
            counts["quality_gate_passed"] += 1

    write_jsonl(args.output, output_rows)
    total = len(output_rows)
    summary = {
        "generations": str(args.generations),
        "generations_sha256": sha256_file(args.generations),
        "source_input": str(args.source_input),
        "source_input_sha256": sha256_file(args.source_input),
        "output": str(args.output),
        "output_sha256": sha256_file(args.output),
        "rows_scored": total,
        "quality_gate_passed": counts.get("quality_gate_passed", 0),
        "quality_gate_pass_rate": counts.get("quality_gate_passed", 0) / total if total else 0.0,
        "canonical_pass_all_rate": sum(1 for row in output_rows if row.get("canonical_passed_all_tests")) / total if total else 0.0,
        "canonical_valid_test_rate": (
            sum(int(row.get("canonical_valid_test_count") or 0) for row in output_rows)
            / sum(int(row.get("raw_test_count") or 0) for row in output_rows)
            if sum(int(row.get("raw_test_count") or 0) for row in output_rows)
            else 0.0
        ),
        "source_failure_recall": sum(1 for row in output_rows if row.get("source_failure_caught")) / total if total else 0.0,
        "min_tests": args.min_tests,
        "max_tests": args.max_tests,
        "counts": dict(counts),
        "policy": {
            "quality_gate": "self-written tests are filtered individually by canonical_solution; a suite is usable when at least min_tests canonical-valid tests remain and the known failed code fails at least one retained test",
            "hidden_suite": "not used by this script; it validates self-written tests against canonical and known failed code only",
        },
    }
    write_json(args.summary_output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
