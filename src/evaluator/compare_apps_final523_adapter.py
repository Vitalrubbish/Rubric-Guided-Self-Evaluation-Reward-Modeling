#!/usr/bin/env python3
"""Compare a generic adapter with base on APPS final523."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def outcome_counts(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter("passed" if row.get("passed") else str(row.get("failure_type")) for row in rows))


def decoding_signature(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fields = ("temperature", "top_p", "repetition_penalty", "max_tokens", "seed")
    result: dict[str, Any] = {}
    for field in fields:
        values = {row.get(field) for row in rows}
        if len(values) != 1:
            raise AssertionError(f"final523 {field} is not constant: {sorted(map(str, values))}")
        result[field] = next(iter(values))
    return result


def paired_signature(rows: list[dict[str, Any]], expected_variant: str) -> dict[str, Any]:
    fields = ("paired_verification_run_id", "paired_verification_timeout", "paired_verification_workers")
    result: dict[str, Any] = {}
    for field in fields:
        values = {row.get(field) for row in rows}
        if len(values) != 1 or None in values:
            raise AssertionError(f"paired final523 {field} is missing or non-constant")
        result[field] = next(iter(values))
    variants = {row.get("paired_verification_variant") for row in rows}
    if variants != {expected_variant}:
        raise AssertionError(f"expected paired variant {expected_variant}, got {variants}")
    return result


def train_problem_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    for row in read_jsonl(path):
        if str(row.get("split")) != "train":
            continue
        metadata = row.get("metadata") or {}
        problem_id = metadata.get("problem_id") or row.get("id")
        if problem_id:
            ids.add(str(problem_id))
    return ids


def row_ids(path: Path) -> set[str]:
    return {str(row.get("id")) for row in read_jsonl(path)}


def metric_block(
    problem_ids: list[str],
    base_by_id: dict[str, dict[str, Any]],
    candidate_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    transitions: Counter[str] = Counter()
    for problem_id in problem_ids:
        base_passed = bool(base_by_id[problem_id].get("passed"))
        candidate_passed = bool(candidate_by_id[problem_id].get("passed"))
        transitions[
            f"base_{'pass' if base_passed else 'fail'}->candidate_{'pass' if candidate_passed else 'fail'}"
        ] += 1
    base_rows = [base_by_id[problem_id] for problem_id in problem_ids]
    candidate_rows = [candidate_by_id[problem_id] for problem_id in problem_ids]
    base_passed = sum(bool(row.get("passed")) for row in base_rows)
    candidate_passed = sum(bool(row.get("passed")) for row in candidate_rows)
    base_syntax = sum(row.get("failure_type") == "syntax_error" for row in base_rows)
    candidate_syntax = sum(row.get("failure_type") == "syntax_error" for row in candidate_rows)
    return {
        "rows": len(problem_ids),
        "base_passed": base_passed,
        "base_pass_rate": base_passed / len(problem_ids),
        "candidate_passed": candidate_passed,
        "candidate_pass_rate": candidate_passed / len(problem_ids),
        "net_pass_delta": candidate_passed - base_passed,
        "positive_transitions": transitions["base_fail->candidate_pass"],
        "negative_transitions": transitions["base_pass->candidate_fail"],
        "transitions": dict(transitions),
        "base_failure_counts": outcome_counts(base_rows),
        "candidate_failure_counts": outcome_counts(candidate_rows),
        "base_syntax_errors": base_syntax,
        "candidate_syntax_errors": candidate_syntax,
        "base_length_finishes": sum(row.get("finish_reason") == "length" for row in base_rows),
        "candidate_length_finishes": sum(row.get("finish_reason") == "length" for row in candidate_rows),
    }


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Generative SFT v1 Final523 Preservation",
        "",
        "| Split | Rows | Base pass@1 | Candidate pass@1 | Net passes | + / - transitions | Syntax base/candidate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split in ("validation", "test", "combined"):
        item = summary[split]
        lines.append(
            f"| {split} | {item['rows']} | {item['base_pass_rate']:.4f} | "
            f"{item['candidate_pass_rate']:.4f} | {item['net_pass_delta']:+d} | "
            f"{item['positive_transitions']} / {item['negative_transitions']} | "
            f"{item['base_syntax_errors']} / {item['candidate_syntax_errors']} |"
        )
    lines.extend(["", "## Gates", ""])
    for name, passed in summary["gates"].items():
        lines.append(f"- [{'x' if passed else ' '}] {name}")
    lines.extend(["", "## Full Summary", "", "```json", json.dumps(summary, ensure_ascii=False, indent=2), "```", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-labeled", type=Path, required=True)
    parser.add_argument("--candidate-labeled", type=Path, required=True)
    parser.add_argument("--sft-data", type=Path, required=True)
    parser.add_argument("--dpo-dev", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    base_rows = read_jsonl(args.base_labeled)
    candidate_rows = read_jsonl(args.candidate_labeled)
    if len(base_rows) != 523 or len(candidate_rows) != 523:
        raise AssertionError(f"expected 523 rows, got {len(base_rows)}/{len(candidate_rows)}")
    base_by_id = {str(row.get("id") or ""): row for row in base_rows}
    candidate_by_id = {str(row.get("id") or ""): row for row in candidate_rows}
    if len(base_by_id) != 523 or len(candidate_by_id) != 523 or set(base_by_id) != set(candidate_by_id):
        raise AssertionError("final523 base/candidate IDs are missing, duplicated, or mismatched")

    split_by_id = {problem_id: str(row.get("eval_split")) for problem_id, row in base_by_id.items()}
    split_counts = Counter(split_by_id.values())
    if split_counts != Counter({"validation": 261, "test": 262}):
        raise AssertionError(f"unexpected final523 split counts: {dict(split_counts)}")
    if any(row.get("source_split") != "train" for row in base_rows + candidate_rows):
        raise AssertionError("final523 source split contract is broken")
    for problem_id in base_by_id:
        if candidate_by_id[problem_id].get("eval_split") != split_by_id[problem_id]:
            raise AssertionError(f"base/candidate split mismatch for {problem_id}")

    base_decoding = decoding_signature(base_rows)
    candidate_decoding = decoding_signature(candidate_rows)
    if base_decoding != candidate_decoding:
        raise AssertionError(f"decoding mismatch: {base_decoding} vs {candidate_decoding}")
    base_paired = paired_signature(base_rows, "base")
    candidate_paired = paired_signature(candidate_rows, "candidate")
    if base_paired != candidate_paired:
        raise AssertionError("base/candidate paired verification signatures differ")

    final_ids = set(base_by_id)
    sft_train_overlap = final_ids & train_problem_ids(args.sft_data)
    dpo_dev_overlap = final_ids & row_ids(args.dpo_dev)
    if sft_train_overlap:
        raise AssertionError(f"final523 overlaps SFT train: {sorted(sft_train_overlap)[:5]}")
    if dpo_dev_overlap:
        raise AssertionError(f"final523 overlaps DPO-dev: {sorted(dpo_dev_overlap)[:5]}")

    ids_by_split = {
        split: sorted(problem_id for problem_id, value in split_by_id.items() if value == split)
        for split in ("validation", "test")
    }
    summary = {
        "status": "completed",
        "policy": "final523 generation preservation; no SFT train or DPO-dev overlap",
        "split_counts": dict(split_counts),
        "sft_train_overlap_count": 0,
        "dpo_dev_overlap_count": 0,
        "decoding": base_decoding,
        "paired_verification": base_paired,
        "validation": metric_block(ids_by_split["validation"], base_by_id, candidate_by_id),
        "test": metric_block(ids_by_split["test"], base_by_id, candidate_by_id),
        "combined": metric_block(sorted(final_ids), base_by_id, candidate_by_id),
    }
    combined = summary["combined"]
    summary["gates"] = {
        "pass_at_1_not_lower": combined["candidate_passed"] >= combined["base_passed"],
        "syntax_errors_not_higher": combined["candidate_syntax_errors"] <= combined["base_syntax_errors"],
        "positive_transitions_not_lower_than_regressions": (
            combined["positive_transitions"] >= combined["negative_transitions"]
        ),
        "sft_train_overlap_zero": True,
        "dpo_dev_overlap_zero": True,
        "paired_protocol_valid": True,
    }
    summary["canary_passed"] = all(summary["gates"].values())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render_report(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
