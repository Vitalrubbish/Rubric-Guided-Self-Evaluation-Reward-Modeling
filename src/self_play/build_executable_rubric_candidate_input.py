#!/usr/bin/env python3
"""Build repair-candidate prompts for executable-rubric best-of-K scoring."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from src.self_play.build_executable_rubric_probe_input import source_id
from src.self_play.executable_rubric_utils import (
    parse_apps_input_output,
    read_jsonl,
    sha256_file,
    write_json,
    write_jsonl,
)


def problem_id(row: dict[str, Any]) -> str:
    return str(row.get("problem_id") or (row.get("metadata") or {}).get("problem_id") or row.get("id") or "")


def usable_suite_sources(suite_paths: list[Path]) -> tuple[dict[str, set[str]], Counter[str]]:
    by_problem: dict[str, set[str]] = {}
    counts: Counter[str] = Counter()
    for path in suite_paths:
        rows = read_jsonl(path)
        counts[f"suite_rows:{path}"] += len(rows)
        for row in rows:
            if not row.get("quality_gate_passed"):
                counts["suite_skipped:not_quality_gated"] += 1
                continue
            pid = problem_id(row)
            sid = str(row.get("source_row_id") or "")
            if not pid:
                counts["suite_skipped:missing_problem_id"] += 1
                continue
            by_problem.setdefault(pid, set())
            if sid:
                by_problem[pid].add(sid)
            counts["suite_usable"] += 1
    return by_problem, counts


def interface_names(row: dict[str, Any]) -> list[str]:
    names = row.get("interface_names")
    if isinstance(names, list) and names:
        return [str(name) for name in names]
    fn_name = parse_apps_input_output(row).get("fn_name")
    return [str(fn_name)] if fn_name else []


def build_row(row: dict[str, Any], record_id: str, source_path: Path) -> dict[str, Any]:
    return {
        "id": record_id,
        "problem_id": row.get("problem_id"),
        "dataset": "apps",
        "split": "validation",
        "task_type": "executable_rubric_best_of_k_repair",
        "prompt": row.get("prompt"),
        "completion": "",
        "source": "strategy_v2_executable_rubric_best_of_k_probe",
        "io_mode": row.get("io_mode"),
        "interface_names": interface_names(row),
        "interface_signatures": row.get("interface_signatures") or [],
        "starter_code": row.get("starter_code"),
        "input_output": row.get("input_output"),
        "difficulty": row.get("difficulty"),
        "metadata": {
            "problem_id": row.get("problem_id"),
            "source_row_id": source_id(row),
            "source_file": str(source_path),
            "previous_failure_type": row.get("previous_failure_type"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-rows", type=Path, default=Path("data/self_play/exec_feedback_probe_round1_rows.jsonl"))
    parser.add_argument("--suites", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, default=Path("data/self_play/executable_rubric_best_of_k_candidate_input.jsonl"))
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("data/self_play/executable_rubric_best_of_k_candidate_input_summary.json"),
    )
    args = parser.parse_args()

    suite_sources, counts = usable_suite_sources(args.suites)
    source_rows = read_jsonl(args.source_rows)
    counts["source_rows"] = len(source_rows)
    selected: dict[str, dict[str, Any]] = {}
    for row in source_rows:
        pid = problem_id(row)
        if pid not in suite_sources:
            counts["source_skipped:no_usable_suite"] += 1
            continue
        sid = source_id(row)
        preferred_sources = suite_sources.get(pid) or set()
        if preferred_sources and sid not in preferred_sources:
            counts["source_skipped:source_id_not_matched"] += 1
            continue
        if pid in selected:
            counts["source_skipped:duplicate_problem"] += 1
            continue
        selected[pid] = row
        counts["source_selected"] += 1

    output_rows: list[dict[str, Any]] = []
    for index, (pid, row) in enumerate(sorted(selected.items()), start=1):
        record_id = f"{pid}__exec_rubric_best_of_k_{index:05d}"
        output_rows.append(build_row(row, record_id, args.source_rows))
    write_jsonl(args.output, output_rows)
    summary = {
        "source_rows": str(args.source_rows),
        "source_rows_sha256": sha256_file(args.source_rows),
        "suites": [str(path) for path in args.suites],
        "suites_sha256": {str(path): sha256_file(path) for path in args.suites},
        "output": str(args.output),
        "output_sha256": sha256_file(args.output),
        "usable_suite_problem_count": len(suite_sources),
        "rows_written": len(output_rows),
        "counts": dict(counts),
        "policy": {
            "prompt_inputs": "reuse the existing Method 2 repair prompt containing public task/interface and visible failed code only",
            "suite_filter": "only problems with at least one quality-gated executable-rubric suite are selected",
            "hidden_suite": "input_output is retained for external verifier scoring and is not rendered by this script",
        },
    }
    write_json(args.summary_output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
