#!/usr/bin/env python3
"""Compare a DPO adapter with base on the frozen train-derived APPS DPO-dev."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def failure_counts(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter("passed" if row.get("passed") else str(row.get("failure_type")) for row in rows))


def normalize_decoding_value(field: str, value: Any) -> Any:
    if field == "repetition_penalty" and value is None:
        return 1.0
    return value


def decoding_signature(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fields = ("temperature", "top_p", "repetition_penalty", "max_tokens", "seed")
    signature: dict[str, Any] = {}
    for field in fields:
        values = {normalize_decoding_value(field, row.get(field)) for row in rows}
        if len(values) != 1:
            raise AssertionError(f"DPO-dev {field} is not constant: {sorted(map(str, values))}")
        signature[field] = next(iter(values))
    return signature


def paired_verification_signature(rows: list[dict[str, Any]], expected_variant: str) -> dict[str, Any]:
    fields = ("paired_verification_run_id", "paired_verification_timeout", "paired_verification_workers")
    signature: dict[str, Any] = {}
    for field in fields:
        values = {row.get(field) for row in rows}
        if len(values) != 1 or None in values:
            raise AssertionError(f"paired DPO-dev {field} is missing or not constant: {sorted(map(str, values))}")
        signature[field] = next(iter(values))
    variants = {row.get("paired_verification_variant") for row in rows}
    if variants != {expected_variant}:
        raise AssertionError(f"expected paired variant {expected_variant}, got {sorted(map(str, variants))}")
    return signature


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate APPS DPO-v2 canary gates on DPO-dev.")
    parser.add_argument(
        "--base-labeled",
        type=Path,
        default=Path("data/responses/apps_simple_method1_dpo_dev_v2_base_greedy_labeled.jsonl"),
    )
    parser.add_argument("--candidate-labeled", type=Path, required=True)
    parser.add_argument("--training-preferences", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=160)
    parser.add_argument("--require-paired-verification", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    base_rows = read_jsonl(args.base_labeled)
    candidate_rows = read_jsonl(args.candidate_labeled)
    preference_rows = read_jsonl(args.training_preferences)
    base_by_id = {str(row.get("id")): row for row in base_rows}
    candidate_by_id = {str(row.get("id")): row for row in candidate_rows}
    if len(base_rows) != args.expected_rows or len(base_by_id) != args.expected_rows:
        raise AssertionError(f"expected {args.expected_rows} unique base rows, got {len(base_rows)}/{len(base_by_id)}")
    if len(candidate_rows) != args.expected_rows or len(candidate_by_id) != args.expected_rows:
        raise AssertionError(
            f"expected {args.expected_rows} unique candidate rows, got {len(candidate_rows)}/{len(candidate_by_id)}"
        )
    if set(base_by_id) != set(candidate_by_id):
        raise AssertionError("base and candidate DPO-dev IDs differ")
    if any(row.get("source_split") != "train" or row.get("eval_split") != "dpo_dev" for row in base_rows):
        raise AssertionError("base DPO-dev rows do not carry the frozen split contract")
    if any(row.get("source_split") != "train" or row.get("eval_split") != "dpo_dev" for row in candidate_rows):
        raise AssertionError("candidate DPO-dev rows do not carry the frozen split contract")
    base_decoding = decoding_signature(base_rows)
    candidate_decoding = decoding_signature(candidate_rows)
    if base_decoding != candidate_decoding:
        raise AssertionError(
            f"base/candidate decoding mismatch: base={base_decoding}, candidate={candidate_decoding}"
        )
    paired_verification = None
    if args.require_paired_verification:
        base_paired = paired_verification_signature(base_rows, "base")
        candidate_paired = paired_verification_signature(candidate_rows, "candidate")
        if base_paired != candidate_paired:
            raise AssertionError(
                f"base/candidate paired verification mismatch: base={base_paired}, candidate={candidate_paired}"
            )
        paired_verification = base_paired

    training_ids = {str(row.get("id")) for row in preference_rows}
    overlap = training_ids & set(base_by_id)
    if overlap:
        raise AssertionError(f"DPO-dev overlaps training preferences: {sorted(overlap)[:5]}")

    transitions = Counter()
    for problem_id in sorted(base_by_id):
        base_passed = bool(base_by_id[problem_id].get("passed"))
        candidate_passed = bool(candidate_by_id[problem_id].get("passed"))
        transitions[
            f"base_{'pass' if base_passed else 'fail'}->candidate_{'pass' if candidate_passed else 'fail'}"
        ] += 1

    base_passed = sum(bool(row.get("passed")) for row in base_rows)
    candidate_passed = sum(bool(row.get("passed")) for row in candidate_rows)
    base_syntax = sum(row.get("failure_type") == "syntax_error" for row in base_rows)
    candidate_syntax = sum(row.get("failure_type") == "syntax_error" for row in candidate_rows)
    base_length = sum(row.get("finish_reason") == "length" for row in base_rows)
    candidate_length = sum(row.get("finish_reason") == "length" for row in candidate_rows)
    positive_transitions = transitions["base_fail->candidate_pass"]
    negative_transitions = transitions["base_pass->candidate_fail"]
    gates = {
        "pass_at_1_not_lower": candidate_passed >= base_passed,
        "syntax_errors_not_higher": candidate_syntax <= base_syntax,
        "length_finishes_not_higher": candidate_length <= base_length,
        "positive_transition_present": positive_transitions >= 1,
        "positive_transitions_not_lower_than_regressions": positive_transitions >= negative_transitions,
        "train_overlap_zero": len(overlap) == 0,
    }
    summary: dict[str, Any] = {
        "rows": args.expected_rows,
        "base_passed": base_passed,
        "base_pass_rate": base_passed / args.expected_rows,
        "candidate_passed": candidate_passed,
        "candidate_pass_rate": candidate_passed / args.expected_rows,
        "net_pass_delta": candidate_passed - base_passed,
        "transitions": dict(transitions),
        "positive_transitions": positive_transitions,
        "negative_transitions": negative_transitions,
        "base_failure_counts": failure_counts(base_rows),
        "candidate_failure_counts": failure_counts(candidate_rows),
        "base_length_finishes": base_length,
        "candidate_length_finishes": candidate_length,
        "training_overlap_count": len(overlap),
        "decoding": base_decoding,
        "paired_verification": paired_verification,
        "gates": gates,
        "canary_passed": all(gates.values()),
        "policy": "train-derived DPO-dev; excluded from preference training and final 523 held-out",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# APPS DPO-v2 Canary Evaluation",
        "",
        f"- Rows: {args.expected_rows}",
        f"- Base pass@1: {summary['base_pass_rate']:.4f}",
        f"- Candidate pass@1: {summary['candidate_pass_rate']:.4f}",
        f"- Net passes: {summary['net_pass_delta']:+d}",
        f"- Canary gate: {'PASS' if summary['canary_passed'] else 'FAIL'}",
        "",
        "## Gates",
        "",
    ]
    for name, passed in gates.items():
        lines.append(f"- [{'x' if passed else ' '}] {name}")
    lines.extend(["", "## Full Summary", "", "```json", json.dumps(summary, ensure_ascii=False, indent=2), "```", ""])
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
