#!/usr/bin/env python3
"""Build test-generation prompts for the executable-rubric probe.

The output keeps canonical solutions and verifier I/O for offline validation,
but the rendered model prompt contains only the public task/interface and the
visible failed code.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from src.self_play.executable_rubric_utils import (
    parse_apps_input_output,
    read_jsonl,
    sha256_file,
    stable_hash,
    write_json,
    write_jsonl,
)


def source_id(row: dict[str, Any]) -> str:
    return str(row.get("gate_row_id") or row.get("id") or row.get("problem_id") or stable_hash(json.dumps(row, sort_keys=True)))


def extract_public_prompt(prompt: str) -> str:
    marker = "Public task prompt:\n"
    if marker in prompt:
        tail = prompt.split(marker, 1)[1]
        if "\n\nPrevious failed code:" in tail:
            tail = tail.split("\n\nPrevious failed code:", 1)[0]
        return tail.strip()
    return prompt.strip()


def build_prompt(public_prompt: str, failed_code: str, fn_name: str, target_tests: int) -> str:
    return (
        "You are writing an executable rubric for a Python coding task.\n"
        "Use only the public task text, public examples, required callable name, and visible failed code. "
        "Do not rely on hidden tests or private verifier messages.\n"
        f"Write exactly {target_tests} focused function-call tests that a correct solution should pass and that are likely to expose the visible bug.\n"
        "Only include a test when you can manually derive the expected value from the task statement. "
        "Prefer small, hand-checkable inputs over broad random coverage.\n"
        "Coverage requirements:\n"
        "- Include at least one non-trivial public example or close variant when the task provides examples.\n"
        "- Do not use only empty inputs, smallest numbers, identity cases, or base cases.\n"
        "- Include one visible-bug-targeted case that exercises the control-flow branch, ordering rule, boundary, or state transition the failed code is likely to miss.\n"
        "- If the task defines invalid inputs, overflow, out-of-range values, or a None/False/error sentinel, include one hand-checkable boundary case for that behavior.\n"
        "- For sequence, parser, state-machine, or pointer-style tasks, include a case with more than one transition or segment.\n"
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
        "Argument examples:\n"
        '- For def add(a, b), use {"args": [2, 3], "expected": 5}.\n'
        '- For def count_items(items), use {"args": [["a", "b"]], "expected": 2}.\n'
        '- For def normalize(s), use {"args": ["  hi  "], "expected": "hi"}.\n\n'
        f"Required callable name: {fn_name}\n\n"
        f"Public task prompt:\n{public_prompt.strip()}\n\n"
        f"Visible failed code:\n{failed_code.strip()}\n\n"
        "JSON:"
    )


def eligible(row: dict[str, Any]) -> tuple[bool, str]:
    if row.get("io_mode") != "function_call":
        return False, "io_mode_not_function_call"
    io_spec = parse_apps_input_output(row)
    if not io_spec.get("fn_name"):
        return False, "missing_fn_name"
    if not str(row.get("canonical_solution") or "").strip():
        return False, "missing_canonical_solution"
    if not str(row.get("previous_repair_code") or "").strip():
        return False, "missing_previous_repair_code"
    return True, "ok"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/self_play/exec_feedback_probe_round1_rows.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/self_play/executable_rubric_probe_testgen_input.jsonl"))
    parser.add_argument("--summary-output", type=Path, default=Path("data/self_play/executable_rubric_probe_testgen_input_summary.json"))
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-tests", type=int, default=5)
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    counts: Counter[str] = Counter()
    candidates: list[dict[str, Any]] = []
    seen_source_ids: set[str] = set()
    for row in rows:
        ok, reason = eligible(row)
        if not ok:
            counts[f"skipped:{reason}"] += 1
            continue
        sid = source_id(row)
        if sid in seen_source_ids:
            counts["skipped:duplicate_source_id"] += 1
            continue
        seen_source_ids.add(sid)
        candidates.append(row)
        counts["eligible"] += 1

    candidates.sort(key=source_id)
    rng = random.Random(args.seed)
    rng.shuffle(candidates)
    selected = candidates[: args.limit] if args.limit else candidates

    output_rows: list[dict[str, Any]] = []
    for row in selected:
        io_spec = parse_apps_input_output(row)
        fn_name = str(io_spec["fn_name"])
        sid = source_id(row)
        public_prompt = extract_public_prompt(str(row.get("prompt") or ""))
        failed_code = str(row.get("previous_repair_code") or "").strip()
        output_rows.append(
            {
                "id": f"{sid}__exec_rubric_testgen",
                "source_row_id": sid,
                "problem_id": row.get("problem_id") or sid,
                "dataset": "apps",
                "split": "validation",
                "task_type": "executable_rubric_test_generation",
                "prompt": build_prompt(public_prompt, failed_code, fn_name, args.target_tests),
                "completion": "",
                "source": "strategy_v2_executable_rubric_probe",
                "io_mode": row.get("io_mode"),
                "interface_names": row.get("interface_names") or [fn_name],
                "input_output": row.get("input_output"),
                "canonical_solution": row.get("canonical_solution"),
                "failed_code": failed_code,
                "metadata": {
                    "fn_name": fn_name,
                    "source_row_id": sid,
                    "source_file": str(args.input),
                    "target_tests": args.target_tests,
                    "previous_failure_type": row.get("previous_failure_type"),
                },
            }
        )
        counts["written"] += 1

    write_jsonl(args.output, output_rows)
    summary = {
        "input": str(args.input),
        "input_sha256": sha256_file(args.input),
        "output": str(args.output),
        "output_sha256": sha256_file(args.output),
        "rows_input": len(rows),
        "rows_written": len(output_rows),
        "limit": args.limit,
        "seed": args.seed,
        "target_tests": args.target_tests,
        "counts": dict(counts),
        "policy": {
            "prompt_inputs": "public task/interface plus visible failed code only",
            "offline_fields": "canonical_solution and input_output are retained only for local validation/scoring and are not rendered in prompt",
            "scope": "APPS function_call rows only for the first executable-rubric probe",
        },
    }
    write_json(args.summary_output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
