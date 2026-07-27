#!/usr/bin/env python3
"""Score candidate repairs with quality-gated executable rubric tests."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.self_play.executable_rubric_utils import (
    execute_function_tests,
    passed_count,
    read_jsonl,
    sha256_file,
    write_json,
    write_jsonl,
)


PROBLEM_ID_RE = re.compile(r"apps/(?:train|test|validation)/\d+")


def problem_id(row: dict[str, Any]) -> str:
    for value in (
        row.get("problem_id"),
        (row.get("metadata") or {}).get("problem_id"),
        row.get("id"),
    ):
        text = str(value or "")
        match = PROBLEM_ID_RE.search(text)
        if match:
            return match.group(0)
        if text:
            return text
    return ""


def candidate_code(row: dict[str, Any]) -> str:
    return str(row.get("extracted_code") or row.get("generated_code") or row.get("previous_repair_code") or "").strip()


def suite_rank(row: dict[str, Any]) -> tuple[int, int, str]:
    return (
        int(row.get("test_count") or 0),
        int(row.get("source_failed_pass_count") is not None),
        str(row.get("response_id") or row.get("id") or ""),
    )


def select_suites(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_problem: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if not row.get("quality_gate_passed"):
            continue
        pid = problem_id(row)
        if pid:
            by_problem[pid].append(row)
    return {pid: sorted(suites, key=suite_rank, reverse=True)[0] for pid, suites in by_problem.items()}


def score_candidate(candidate: dict[str, Any], suite: dict[str, Any], timeout: float) -> dict[str, Any]:
    tests = suite.get("tests") or []
    total = len(tests)
    result = execute_function_tests(candidate_code(candidate), str(suite.get("fn_name") or ""), tests, timeout=timeout)
    pass_count = passed_count(result, total)
    test_score = pass_count / total if total else 0.0
    predicted_pass = total > 0 and pass_count == total and not result.get("setup_error")
    return {
        "id": candidate.get("id"),
        "response_id": candidate.get("response_id"),
        "problem_id": problem_id(candidate),
        "generated_code": candidate.get("generated_code"),
        "extracted_code": candidate.get("extracted_code"),
        "gold_passed": bool(candidate.get("passed")),
        "passed": bool(candidate.get("passed")),
        "gold_failure_type": candidate.get("failure_type"),
        "failure_type": candidate.get("failure_type"),
        "suite_response_id": suite.get("response_id"),
        "test_count": total,
        "test_pass_count": pass_count,
        "test_score": test_score,
        "predicted_pass_by_tests": predicted_pass,
        "test_execution_result": result,
    }


def confusion(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tp = sum(1 for row in rows if row["predicted_pass_by_tests"] and row["gold_passed"])
    fp = sum(1 for row in rows if row["predicted_pass_by_tests"] and not row["gold_passed"])
    tn = sum(1 for row in rows if not row["predicted_pass_by_tests"] and not row["gold_passed"])
    fn = sum(1 for row in rows if not row["predicted_pass_by_tests"] and row["gold_passed"])
    total = len(rows)
    return {
        "rows": total,
        "tp_pass": tp,
        "fp_pass": fp,
        "tn_fail": tn,
        "fn_fail_rejection": fn,
        "accuracy": (tp + tn) / total if total else 0.0,
        "precision_pass": tp / (tp + fp) if tp + fp else 0.0,
        "recall_pass": tp / (tp + fn) if tp + fn else 0.0,
        "precision_fail": tn / (tn + fn) if tn + fn else 0.0,
        "recall_fail": tn / (tn + fp) if tn + fp else 0.0,
        "false_rejection_rate": fn / (tp + fn) if tp + fn else 0.0,
        "overacceptance_rate": fp / (fp + tn) if fp + tn else 0.0,
    }


def selection_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["problem_id"])].append(row)
    if not groups:
        return {"problem_count": 0}
    selected = []
    first = []
    oracle_pass = 0
    for pid, group in groups.items():
        ordered = sorted(group, key=lambda row: (-float(row["test_score"]), str(row.get("response_id") or row.get("id") or "")))
        selected.append(ordered[0])
        first.append(sorted(group, key=lambda row: str(row.get("response_id") or row.get("id") or ""))[0])
        if any(row["gold_passed"] for row in group):
            oracle_pass += 1
    return {
        "problem_count": len(groups),
        "candidate_rows": len(rows),
        "selected_by_tests_passed": sum(1 for row in selected if row["gold_passed"]),
        "selected_by_tests_pass_rate": sum(1 for row in selected if row["gold_passed"]) / len(selected),
        "first_candidate_passed": sum(1 for row in first if row["gold_passed"]),
        "first_candidate_pass_rate": sum(1 for row in first if row["gold_passed"]) / len(first),
        "oracle_best_of_k_passed": oracle_pass,
        "oracle_best_of_k_pass_rate": oracle_pass / len(groups),
    }


def select_rows_by_tests(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["problem_id"])].append(row)
    selected: list[dict[str, Any]] = []
    for _pid, group in sorted(groups.items()):
        ordered = sorted(group, key=lambda row: (-float(row["test_score"]), str(row.get("response_id") or row.get("id") or "")))
        chosen = dict(ordered[0])
        chosen["selection_policy"] = "max executable-rubric test_score; tie by response_id"
        selected.append(chosen)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suites", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/self_play/executable_rubric_candidate_scores.jsonl"))
    parser.add_argument("--summary-output", type=Path, default=Path("data/self_play/executable_rubric_candidate_scores_summary.json"))
    parser.add_argument("--selected-output", type=Path, default=Path("data/self_play/executable_rubric_selected_candidates.jsonl"))
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args()

    suites = select_suites(read_jsonl(args.suites))
    candidates = read_jsonl(args.candidates)
    output_rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for candidate in candidates:
        pid = problem_id(candidate)
        suite = suites.get(pid)
        if suite is None:
            counts["skipped:no_suite"] += 1
            continue
        if not candidate_code(candidate):
            counts["skipped:empty_candidate_code"] += 1
            continue
        scored = score_candidate(candidate, suite, args.timeout)
        output_rows.append(scored)
        counts["scored"] += 1

    write_jsonl(args.output, output_rows)
    selected_rows = select_rows_by_tests(output_rows)
    write_jsonl(args.selected_output, selected_rows)
    summary = {
        "suites": str(args.suites),
        "suites_sha256": sha256_file(args.suites),
        "candidates": str(args.candidates),
        "candidates_sha256": sha256_file(args.candidates),
        "output": str(args.output),
        "output_sha256": sha256_file(args.output),
        "selected_output": str(args.selected_output),
        "selected_output_sha256": sha256_file(args.selected_output),
        "usable_suite_count": len(suites),
        "candidate_rows_input": len(candidates),
        "candidate_rows_scored": len(output_rows),
        "selected_rows": len(selected_rows),
        "gold_counts": dict(Counter("pass" if row["gold_passed"] else "fail" for row in output_rows)),
        "predicted_counts": dict(Counter("pass" if row["predicted_pass_by_tests"] else "fail" for row in output_rows)),
        "confusion": confusion(output_rows),
        "selection": selection_report(output_rows),
        "counts": dict(counts),
        "policy": {
            "suite_selection": "one quality-gated suite per problem; prefer more tests, then deterministic response id",
            "candidate_prediction": "candidate passes executable rubric only if all self-written tests pass",
            "hidden_labels": "candidate .passed is used only for reporting, never for selecting by tests",
        },
    }
    write_json(args.summary_output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
