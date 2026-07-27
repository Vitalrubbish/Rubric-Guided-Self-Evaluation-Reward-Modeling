#!/usr/bin/env python3
"""Build SFT rows from high-quality executable-rubric selected repairs."""

from __future__ import annotations

import argparse
import ast
import json
from collections import Counter
from pathlib import Path
from typing import Any

from src.self_play.executable_rubric_utils import read_jsonl, sha256_file, stable_hash, write_json, write_jsonl
from src.self_play.score_executable_rubric_tests import candidate_code, problem_id


def normalize_code(code: Any) -> str:
    return "\n".join(str(code or "").strip().splitlines()).strip()


def parseable(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def build_source_index(source_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for row in source_rows:
        pid = problem_id(row)
        if pid and pid not in index:
            index[pid] = row
    return index


def verifier_passed(row: dict[str, Any]) -> bool:
    if "gold_passed" in row:
        return bool(row.get("gold_passed"))
    return bool(row.get("passed"))


def has_quality_suite(row: dict[str, Any]) -> bool:
    return bool(row.get("suite_response_id")) and int(row.get("test_count") or 0) > 0


def selected_row_allowed(
    row: dict[str, Any],
    quality_tier: str,
    min_test_score: float,
    require_predicted_pass: bool,
) -> tuple[bool, str]:
    if not verifier_passed(row):
        return False, f"not_verifier_passed:{row.get('failure_type') or row.get('gold_failure_type') or 'unknown'}"
    if require_predicted_pass and not row.get("predicted_pass_by_tests"):
        return False, "not_predicted_pass_by_tests"
    if float(row.get("test_score") or 0.0) < min_test_score:
        return False, "test_score_too_low"
    if quality_tier == "strict_rubric" and not has_quality_suite(row):
        return False, "missing_quality_suite"
    if quality_tier == "verifier_pass":
        return True, "ok"
    if quality_tier == "strict_rubric":
        return True, "ok"
    raise ValueError(f"unsupported quality tier: {quality_tier}")


def build_sft_row(
    source: dict[str, Any],
    selected: dict[str, Any],
    code: str,
    source_tag: str,
    quality_tier: str,
    selected_path: Path,
) -> dict[str, Any]:
    source_id = str(source.get("id") or selected.get("problem_id") or "")
    response_id = str(selected.get("response_id") or selected.get("id") or "")
    out = dict(source)
    metadata = dict(source.get("metadata") or {})
    metadata.update(
        {
            "source": source_tag,
            "quality_tier": quality_tier,
            "base_problem_id": selected.get("problem_id"),
            "selected_candidate_id": selected.get("id"),
            "selected_response_id": response_id,
            "selected_sample_id": selected.get("sample_id"),
            "selection_policy": selected.get("selection_policy"),
            "suite_response_id": selected.get("suite_response_id"),
            "test_count": selected.get("test_count"),
            "test_pass_count": selected.get("test_pass_count"),
            "test_score": selected.get("test_score"),
            "predicted_pass_by_tests": selected.get("predicted_pass_by_tests"),
            "verifier_passed": verifier_passed(selected),
            "verifier_failure_type": selected.get("failure_type") or selected.get("gold_failure_type"),
            "selected_candidates_source": str(selected_path),
        }
    )
    out.update(
        {
            "id": f"{source_id}__{source_tag}_{stable_hash(response_id + code)}",
            "split": "train",
            "completion": code,
            "source": source_tag,
            "metadata": metadata,
        }
    )
    return out


def build_training_rows(
    selected_rows: list[dict[str, Any]],
    source_rows: list[dict[str, Any]],
    quality_tier: str,
    min_test_score: float,
    require_predicted_pass: bool,
    max_rows: int | None,
    source_tag: str,
    selected_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Counter[str]]:
    source_by_problem = build_source_index(source_rows)
    counts: Counter[str] = Counter()
    sft_rows: list[dict[str, Any]] = []
    accepted_rows: list[dict[str, Any]] = []
    seen_code: set[tuple[str, str]] = set()

    for selected in selected_rows:
        counts["selected_input"] += 1
        allowed, reason = selected_row_allowed(selected, quality_tier, min_test_score, require_predicted_pass)
        if not allowed:
            counts[f"skipped:{reason}"] += 1
            continue
        pid = problem_id(selected)
        source = source_by_problem.get(pid)
        if source is None:
            counts["skipped:missing_source_row"] += 1
            continue
        if source.get("split") != "train":
            counts["skipped:source_not_train"] += 1
            continue
        code = normalize_code(candidate_code(selected))
        if not code:
            counts["skipped:empty_code"] += 1
            continue
        if not parseable(code):
            counts["skipped:not_parseable"] += 1
            continue
        code_key = (pid, code)
        if code_key in seen_code:
            counts["skipped:duplicate_problem_code"] += 1
            continue
        seen_code.add(code_key)

        sft_row = build_sft_row(source, selected, code, source_tag, quality_tier, selected_path)
        accepted = dict(selected)
        accepted["sft_id"] = sft_row["id"]
        accepted["quality_tier"] = quality_tier
        sft_rows.append(sft_row)
        accepted_rows.append(accepted)
        counts["accepted"] += 1
        if selected.get("suite_response_id"):
            counts["accepted:with_suite"] += 1
        else:
            counts["accepted:no_suite"] += 1
        if max_rows is not None and len(sft_rows) >= max_rows:
            counts["stopped:max_rows"] += 1
            break

    sft_rows.sort(key=lambda row: str(row.get("id") or ""))
    accepted_rows.sort(key=lambda row: str(row.get("sft_id") or ""))
    return sft_rows, accepted_rows, counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-candidates", type=Path, required=True)
    parser.add_argument("--source-input", type=Path, required=True)
    parser.add_argument("--sft-output", type=Path, required=True)
    parser.add_argument("--accepted-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--quality-tier", choices=("strict_rubric", "verifier_pass"), default="strict_rubric")
    parser.add_argument("--min-test-score", type=float, default=1.0)
    parser.add_argument("--require-predicted-pass", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--source-tag", default="executable_rubric_selected_verifier_pass")
    parser.add_argument("--allow-empty", action="store_true")
    args = parser.parse_args()

    if args.min_test_score < 0.0 or args.min_test_score > 1.0:
        raise ValueError("--min-test-score must be in [0, 1]")
    if args.max_rows is not None and args.max_rows < 1:
        raise ValueError("--max-rows must be positive")

    selected_rows = read_jsonl(args.selected_candidates)
    source_rows = read_jsonl(args.source_input)
    sft_rows, accepted_rows, counts = build_training_rows(
        selected_rows=selected_rows,
        source_rows=source_rows,
        quality_tier=args.quality_tier,
        min_test_score=args.min_test_score,
        require_predicted_pass=args.require_predicted_pass,
        max_rows=args.max_rows,
        source_tag=args.source_tag,
        selected_path=args.selected_candidates,
    )
    if not sft_rows and not args.allow_empty:
        raise SystemExit("no high-quality selected repairs matched the requested filters")

    write_jsonl(args.sft_output, sft_rows)
    write_jsonl(args.accepted_output, accepted_rows)
    summary = {
        "selected_candidates": str(args.selected_candidates),
        "selected_candidates_sha256": sha256_file(args.selected_candidates),
        "source_input": str(args.source_input),
        "source_input_sha256": sha256_file(args.source_input),
        "sft_output": str(args.sft_output),
        "sft_output_sha256": sha256_file(args.sft_output),
        "accepted_output": str(args.accepted_output),
        "accepted_output_sha256": sha256_file(args.accepted_output),
        "quality_tier": args.quality_tier,
        "min_test_score": args.min_test_score,
        "require_predicted_pass": args.require_predicted_pass,
        "max_rows": args.max_rows,
        "source_tag": args.source_tag,
        "selected_rows_input": len(selected_rows),
        "sft_rows": len(sft_rows),
        "unique_problem_count": len({row.get("metadata", {}).get("base_problem_id") for row in sft_rows}),
        "io_mode_counts": dict(Counter(str(row.get("io_mode") or "unknown") for row in sft_rows)),
        "suite_counts": dict(Counter("with_suite" if row.get("metadata", {}).get("suite_response_id") else "no_suite" for row in sft_rows)),
        "counts": dict(counts),
        "policy": {
            "repair_rows": "only verifier-passing selected repairs are eligible for SFT",
            "strict_rubric": "also requires a quality-gated suite and a passing executable-rubric score",
            "verifier_pass": "keeps verifier-passing selected repairs even when no executable suite covered the problem",
            "hidden_labels": "verifier labels are used only as an offline safety filter for training data",
        },
    }
    write_json(args.summary_output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
