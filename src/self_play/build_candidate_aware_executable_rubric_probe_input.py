#!/usr/bin/env python3
"""Build candidate-aware adversarial test-generation prompts.

The rendered prompt contains the public task, existing model-written tests,
and one visible suspect repair candidate that currently passes those tests.
Canonical solutions are retained only for offline quality-gating.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from src.self_play.build_executable_rubric_probe_input import extract_public_prompt
from src.self_play.executable_rubric_utils import read_jsonl, sha256_file, write_json, write_jsonl
from src.self_play.score_executable_rubric_tests import candidate_code, problem_id, select_suites


def build_source_index(source_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for row in source_rows:
        pid = problem_id(row)
        if pid and pid not in index:
            index[pid] = row
    return index


def existing_suite_tests(suite_paths: list[Path], aggregation: str) -> dict[str, list[dict[str, Any]]]:
    if not suite_paths:
        return {}
    rows: list[dict[str, Any]] = []
    for path in suite_paths:
        rows.extend(read_jsonl(path))
    suites = select_suites(rows, aggregation=aggregation)
    return {pid: list(suite.get("tests") or []) for pid, suite in suites.items()}


def compact_tests_for_prompt(tests: list[dict[str, Any]], max_tests: int) -> str:
    return json.dumps(tests[:max_tests], ensure_ascii=False, sort_keys=True)


def suite_filter_allows(row: dict[str, Any], suite_filter: str) -> bool:
    has_suite = bool(row.get("suite_response_id"))
    if suite_filter == "all":
        return True
    if suite_filter == "no_suite":
        return not has_suite
    if suite_filter == "with_suite":
        return has_suite
    raise ValueError(f"unsupported suite filter: {suite_filter}")


def build_prompt(
    public_prompt: str,
    suspect_code: str,
    fn_name: str,
    existing_tests: list[dict[str, Any]],
    target_tests: int,
    max_existing_tests_in_prompt: int,
) -> str:
    existing_json = compact_tests_for_prompt(existing_tests, max_existing_tests_in_prompt)
    return (
        "You are refining an executable rubric for a Python coding task.\n"
        "Use only the public task text, required callable name, existing model-written tests, and visible suspect code. "
        "Do not rely on hidden tests or private verifier messages.\n"
        f"Write exactly {target_tests} NEW focused function-call tests that a correct solution should pass and that are likely to falsify the suspect code if it is wrong.\n"
        "Do not duplicate the existing tests. Only include a test when you can manually derive the expected value from the task statement. "
        "Prefer small, hand-checkable inputs over broad random coverage.\n"
        "Adversarial coverage requirements:\n"
        "- Target behavior not already covered by the existing tests.\n"
        "- Include a branch, boundary, ordering direction, state transition, or invalid/sentinel case that the suspect code may mishandle.\n"
        "- For parsers, interpreters, regex-like tasks, or pointer/state-machine tasks, include a multi-step case that changes direction or state at least once.\n"
        "- For bit, mask, numeric range, or encoding tasks, include a hand-checkable boundary or out-of-range case if the statement defines one.\n"
        "The executor will call the function as fn(*args). Therefore args must be the JSON array of positional arguments. "
        "If the function takes one list-valued argument, wrap that list once, for example {\"args\": [[1, 2, 3]], \"expected\": 3}.\n"
        "Return JSON only. The first character must be { and the last character must be }. No Markdown, no prose.\n"
        "Schema:\n"
        "{\n"
        f'  "fn_name": "{fn_name}",\n'
        '  "tests": [\n'
        '    {"args": [arg1, arg2], "expected": expected_value}\n'
        "  ]\n"
        "}\n\n"
        f"Required callable name: {fn_name}\n\n"
        f"Public task prompt:\n{public_prompt.strip()}\n\n"
        f"Existing executable rubric tests already tried:\n{existing_json}\n\n"
        f"Visible suspect code:\n{suspect_code.strip()}\n\n"
        "JSON:"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--source-input", type=Path, required=True)
    parser.add_argument("--suites", type=Path, nargs="*", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--suite-aggregation", choices=("best", "union"), default="union")
    parser.add_argument("--target-tests", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-existing-tests-in-prompt", type=int, default=12)
    parser.add_argument("--predicted-pass-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--suite-filter",
        choices=("all", "no_suite", "with_suite"),
        default="all",
        help="Which scored candidates to render based on whether they already had a quality-gated suite.",
    )
    args = parser.parse_args()

    scores = read_jsonl(args.scores)
    source_rows = read_jsonl(args.source_input)
    source_by_problem = build_source_index(source_rows)
    existing_tests_by_problem = existing_suite_tests(args.suites, args.suite_aggregation)

    counts: Counter[str] = Counter()
    selected_scores: list[dict[str, Any]] = []
    for row in scores:
        if args.predicted_pass_only and not row.get("predicted_pass_by_tests"):
            counts["skipped:not_predicted_pass"] += 1
            continue
        if not suite_filter_allows(row, args.suite_filter):
            counts[f"skipped:suite_filter:{args.suite_filter}"] += 1
            continue
        pid = problem_id(row)
        if not pid:
            counts["skipped:missing_problem_id"] += 1
            continue
        if pid not in source_by_problem:
            counts["skipped:missing_source_row"] += 1
            continue
        if not candidate_code(row):
            counts["skipped:empty_candidate_code"] += 1
            continue
        source = source_by_problem[pid]
        metadata = source.get("metadata") or {}
        fn_name = str(metadata.get("fn_name") or (source.get("interface_names") or [""])[0] or "")
        if not fn_name:
            counts["skipped:missing_fn_name"] += 1
            continue
        if not str(source.get("canonical_solution") or "").strip():
            counts["skipped:missing_canonical_solution"] += 1
            continue
        selected_scores.append(row)
        counts["selected"] += 1

    selected_scores.sort(key=lambda row: (problem_id(row), str(row.get("response_id") or row.get("id") or "")))
    if args.limit:
        selected_scores = selected_scores[: args.limit]

    output_rows: list[dict[str, Any]] = []
    for index, score in enumerate(selected_scores, start=1):
        pid = problem_id(score)
        source = source_by_problem[pid]
        metadata = source.get("metadata") or {}
        fn_name = str(metadata.get("fn_name") or (source.get("interface_names") or [""])[0] or "")
        suspect_code = candidate_code(score)
        candidate_response_id = str(score.get("response_id") or score.get("id") or f"candidate-{index}")
        record_id = f"{pid}__candidate_aware_exec_rubric_testgen_{index:05d}"
        public_prompt = extract_public_prompt(str(source.get("prompt") or ""))
        existing_tests = existing_tests_by_problem.get(pid) or []
        output_rows.append(
            {
                "id": record_id,
                "source_row_id": source.get("source_row_id") or source.get("id"),
                "problem_id": pid,
                "dataset": "apps",
                "split": source.get("split") or "validation",
                "task_type": "candidate_aware_executable_rubric_test_generation",
                "prompt": build_prompt(
                    public_prompt,
                    suspect_code,
                    fn_name,
                    existing_tests,
                    args.target_tests,
                    args.max_existing_tests_in_prompt,
                ),
                "completion": "",
                "source": "strategy_v2_candidate_aware_executable_rubric_probe",
                "io_mode": source.get("io_mode"),
                "interface_names": source.get("interface_names") or [fn_name],
                "input_output": source.get("input_output"),
                "canonical_solution": source.get("canonical_solution"),
                "failed_code": suspect_code,
                "metadata": {
                    "fn_name": fn_name,
                    "source_row_id": source.get("source_row_id") or source.get("id"),
                    "source_input": str(args.source_input),
                    "score_source": str(args.scores),
                    "candidate_response_id": candidate_response_id,
                    "candidate_test_score": score.get("test_score"),
                    "candidate_suite_response_id": score.get("suite_response_id"),
                    "existing_suite_aggregation": args.suite_aggregation,
                    "existing_test_count": len(existing_tests),
                    "target_tests": args.target_tests,
                },
            }
        )
        counts["written"] += 1

    write_jsonl(args.output, output_rows)
    summary = {
        "scores": str(args.scores),
        "scores_sha256": sha256_file(args.scores),
        "source_input": str(args.source_input),
        "source_input_sha256": sha256_file(args.source_input),
        "suites": [str(path) for path in args.suites],
        "suites_sha256": {str(path): sha256_file(path) for path in args.suites},
        "output": str(args.output),
        "output_sha256": sha256_file(args.output),
        "rows_input": len(scores),
        "rows_written": len(output_rows),
        "suite_aggregation": args.suite_aggregation,
        "target_tests": args.target_tests,
        "limit": args.limit,
        "predicted_pass_only": args.predicted_pass_only,
        "suite_filter": args.suite_filter,
        "counts": dict(counts),
        "policy": {
            "prompt_inputs": "public task, existing model-written executable tests, and one visible suspect candidate only",
            "offline_fields": "canonical_solution is retained only for validation/scoring and is not rendered in prompt",
            "quality_gate": "extractor keeps tests only when canonical passes and the visible suspect candidate fails at least one retained test",
        },
    }
    write_json(args.summary_output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
