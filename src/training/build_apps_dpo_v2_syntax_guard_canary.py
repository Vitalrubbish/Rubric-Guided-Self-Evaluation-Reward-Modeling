#!/usr/bin/env python3
"""Add a small set of real, length-controlled syntax failures to semantic DPO data."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .build_apps_dpo_v2_preferences import PRIVATE_KEYS, code_audit, normalize_code
from .build_apps_dpo_v2_semantic_canary import canonical_fence, read_jsonl, sha256_file, write_jsonl


def symmetric_length_ratio(chosen: str, rejected: str) -> float:
    shorter = min(len(chosen), len(rejected))
    return float("inf") if shorter == 0 else max(len(chosen), len(rejected)) / shorter


def build_syntax_guarded_pairs(
    semantic_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    *,
    max_length_ratio: float,
    completion_format: str = "fenced",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Counter[str]]:
    if completion_format not in {"fenced", "raw"}:
        raise ValueError("completion_format must be fenced or raw")
    skipped: Counter[str] = Counter()
    semantic_ids = {str(row.get("id") or "") for row in semantic_rows}
    if "" in semantic_ids or len(semantic_ids) != len(semantic_rows):
        raise AssertionError("semantic rows must have unique non-empty problem IDs")

    guards: list[dict[str, Any]] = []
    for row in sorted(candidate_rows, key=lambda item: str(item.get("pair_id"))):
        if row.get("original_failure_type") != "syntax_error":
            skipped["not_syntax_error"] += 1
            continue
        problem_id = str(row.get("id") or "")
        if not problem_id or problem_id in semantic_ids:
            skipped["missing_or_duplicate_problem_id"] += 1
            continue

        chosen = normalize_code(str(row.get("chosen") or ""))
        rejected = normalize_code(str(row.get("rejected") or ""))
        if not chosen or not rejected or chosen == rejected:
            skipped["empty_or_identical"] += 1
            continue
        if "```" in chosen or "```" in rejected:
            skipped["residual_fence"] += 1
            continue

        ratio = symmetric_length_ratio(chosen, rejected)
        if ratio > max_length_ratio:
            skipped["length_ratio_too_large"] += 1
            continue

        chosen_audit = code_audit(chosen, [])
        rejected_audit = code_audit(rejected, [])
        if not chosen_audit["parseable"] or chosen_audit["top_level_demo_count"]:
            skipped["chosen_not_clean"] += 1
            continue
        if rejected_audit["parseable"]:
            skipped["rejected_actually_parseable"] += 1
            continue

        chosen_output = canonical_fence(chosen) if completion_format == "fenced" else chosen
        rejected_output = canonical_fence(rejected) if completion_format == "fenced" else rejected
        guard = {
            **row,
            "pair_version": f"apps_simple_method1_dpo_v2_real_syntax_guard_{completion_format}",
            "chosen": chosen_output,
            "rejected": rejected_output,
            "chosen_parseable": True,
            "rejected_parseable": False,
            "completion_char_ratio": ratio,
            "completion_format": (
                "matched_python_fence" if completion_format == "fenced" else "matched_raw_python"
            ),
            "syntax_guard": {
                "source": "same_model_verifier_gated_self_repair",
                "rejected_syntax_error": rejected_audit.get("syntax_error"),
                "max_completion_char_ratio": max_length_ratio,
                "observed_completion_char_ratio": ratio,
            },
        }
        leaked = PRIVATE_KEYS.intersection(guard)
        if leaked:
            raise AssertionError(f"private fields leaked into {guard.get('pair_id')}: {sorted(leaked)}")
        if completion_format == "fenced":
            if guard["chosen"].count("```") != 2 or guard["rejected"].count("```") != 2:
                raise AssertionError(f"unmatched canonical fence in {guard.get('pair_id')}")
        elif "```" in guard["chosen"] or "```" in guard["rejected"]:
            raise AssertionError(f"residual fence in raw syntax guard {guard.get('pair_id')}")
        guards.append(guard)

    output = sorted(semantic_rows + guards, key=lambda row: str(row.get("pair_id")))
    pair_ids = [str(row.get("pair_id") or "") for row in output]
    problem_ids = [str(row.get("id") or "") for row in output]
    if "" in pair_ids or len(pair_ids) != len(set(pair_ids)):
        raise AssertionError("syntax-guard data has missing or duplicate pair IDs")
    if "" in problem_ids or len(problem_ids) != len(set(problem_ids)):
        raise AssertionError("syntax-guard data has missing or duplicate problem IDs")
    return output, guards, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description="Build APPS DPO-v2 real syntax-guard canary data.")
    parser.add_argument("--semantic-input", type=Path, required=True)
    parser.add_argument("--candidate-input", type=Path, required=True)
    parser.add_argument("--forbidden-prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--max-length-ratio", type=float, default=1.5)
    parser.add_argument("--min-syntax-guards", type=int, default=10)
    parser.add_argument("--completion-format", choices=("fenced", "raw"), default="fenced")
    args = parser.parse_args()

    semantic_rows = read_jsonl(args.semantic_input)
    candidate_rows = read_jsonl(args.candidate_input)
    output, guards, skipped = build_syntax_guarded_pairs(
        semantic_rows,
        candidate_rows,
        max_length_ratio=args.max_length_ratio,
        completion_format=args.completion_format,
    )
    if len(guards) < args.min_syntax_guards:
        raise RuntimeError(f"only {len(guards)} syntax guards; require {args.min_syntax_guards}")

    forbidden_ids = {str(row.get("id") or "") for row in read_jsonl(args.forbidden_prompts)}
    training_ids = {str(row.get("id") or "") for row in output}
    overlap = training_ids & forbidden_ids
    if overlap:
        raise AssertionError(f"syntax-guard training data overlaps DPO-dev: {sorted(overlap)[:5]}")

    write_jsonl(args.output, output)
    summary = {
        "status": "frozen",
        "semantic_input": str(args.semantic_input),
        "semantic_input_sha256": sha256_file(args.semantic_input),
        "candidate_input": str(args.candidate_input),
        "candidate_input_sha256": sha256_file(args.candidate_input),
        "output": str(args.output),
        "output_sha256": sha256_file(args.output),
        "semantic_pair_count": len(semantic_rows),
        "syntax_guard_count": len(guards),
        "pair_count": len(output),
        "unique_problem_count": len(training_ids),
        "max_length_ratio": args.max_length_ratio,
        "completion_format": args.completion_format,
        "skipped_counts": dict(skipped),
        "failure_type_counts": dict(Counter(str(row.get("original_failure_type")) for row in output)),
        "forbidden_prompt_count": len(forbidden_ids),
        "forbidden_overlap_count": len(overlap),
        "policy": (
            "semantic pairs plus real same-model syntax failures; strict length control; "
            f"matched {args.completion_format} completions; no DPO-dev rows"
        ),
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
