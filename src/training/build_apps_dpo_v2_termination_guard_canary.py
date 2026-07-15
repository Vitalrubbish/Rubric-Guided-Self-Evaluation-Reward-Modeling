#!/usr/bin/env python3
"""Build a length-matched semantic canary with explicit termination guards."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .build_apps_dpo_v2_preferences import PRIVATE_KEYS, code_audit, normalize_code
from .build_apps_dpo_v2_semantic_canary import canonical_fence, read_jsonl, sha256_file, write_jsonl


def symmetric_length_ratio(chosen: str, rejected: str) -> float:
    shorter = min(len(chosen), len(rejected))
    return float("inf") if shorter == 0 else max(len(chosen), len(rejected)) / shorter


def build_guarded_pairs(
    rows: list[dict[str, Any]],
    *,
    max_length_ratio: float,
    termination_guards: int,
    guard_max_code_chars: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Counter[str]]:
    real_pairs: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    for row in rows:
        chosen = str(row.get("chosen") or "")
        rejected = str(row.get("rejected") or "")
        ratio = symmetric_length_ratio(chosen, rejected)
        if ratio > max_length_ratio:
            skipped["real_length_ratio_too_large"] += 1
            continue
        real_pairs.append(
            {
                **row,
                "pair_version": "apps_simple_method1_dpo_v2_length_matched_v4",
                "length_control": {
                    "kind": "symmetric_real_pair_filter",
                    "max_completion_char_ratio": max_length_ratio,
                    "observed_completion_char_ratio": ratio,
                },
            }
        )

    guard_candidates: list[dict[str, Any]] = []
    for row in real_pairs:
        code = normalize_code(str(row.get("chosen") or ""))
        audit = code_audit(code, [])
        if not code or len(code) > guard_max_code_chars:
            skipped["guard_code_too_long_or_empty"] += 1
            continue
        if not audit["parseable"] or audit["top_level_demo_count"]:
            skipped["guard_code_not_clean"] += 1
            continue
        guard_candidates.append({**row, "_guard_code": code})

    guard_candidates.sort(
        key=lambda row: hashlib.sha256(f"{seed}:{row.get('pair_id')}".encode("utf-8")).hexdigest()
    )
    if len(guard_candidates) < termination_guards:
        raise RuntimeError(
            f"only {len(guard_candidates)} clean termination-guard candidates; require {termination_guards}"
        )

    guards: list[dict[str, Any]] = []
    for row in guard_candidates[:termination_guards]:
        code = str(row.pop("_guard_code"))
        repeated_code = f"{code.rstrip()}\n\n{code.rstrip()}"
        source_pair_id = str(row.get("pair_id"))
        guard = {
            **row,
            "pair_id": f"{source_pair_id}::termination_guard_v1",
            "source_pair_id": source_pair_id,
            "pair_version": "apps_simple_method1_dpo_v2_termination_guard_v4",
            "chosen": canonical_fence(code),
            "rejected": canonical_fence(repeated_code),
            "repair_method": "protected_termination_guard_v1",
            "original_failure_type": "reward_hacking_repetition",
            "chosen_parseable": True,
            "rejected_parseable": True,
            "completion_format": "matched_python_fence",
            "termination_guard": {
                "dimension": "single_complete_implementation_and_stop",
                "chosen_implementations": 1,
                "rejected_implementations": 2,
                "rationale": "Repeated equivalent implementations are a verbosity/reward-hacking failure.",
            },
        }
        leaked = PRIVATE_KEYS.intersection(guard)
        if leaked:
            raise AssertionError(f"private fields leaked into {guard['pair_id']}: {sorted(leaked)}")
        guards.append(guard)

    output = sorted(real_pairs + guards, key=lambda row: str(row.get("pair_id")))
    pair_ids = [str(row.get("pair_id")) for row in output]
    if len(pair_ids) != len(set(pair_ids)):
        raise AssertionError("termination-guard canary contains duplicate pair IDs")
    return output, guards, skipped


def length_direction_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        delta = len(str(row.get("chosen") or "")) - len(str(row.get("rejected") or ""))
        counts["chosen_longer" if delta > 0 else "rejected_longer" if delta < 0 else "equal"] += 1
    return dict(counts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build APPS DPO-v2 v4 length and termination guard data.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--forbidden-prompts", type=Path, required=True)
    parser.add_argument("--max-length-ratio", type=float, default=1.1)
    parser.add_argument("--termination-guards", type=int, default=4)
    parser.add_argument("--guard-max-code-chars", type=int, default=240)
    parser.add_argument("--min-real-pairs", type=int, default=35)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    output, guards, skipped = build_guarded_pairs(
        rows,
        max_length_ratio=args.max_length_ratio,
        termination_guards=args.termination_guards,
        guard_max_code_chars=args.guard_max_code_chars,
        seed=args.seed,
    )
    real_pair_count = len(output) - len(guards)
    if real_pair_count < args.min_real_pairs:
        raise RuntimeError(f"only {real_pair_count} length-matched real pairs; require {args.min_real_pairs}")

    forbidden_ids = {str(row.get("id")) for row in read_jsonl(args.forbidden_prompts)}
    training_ids = {str(row.get("id")) for row in output}
    overlap = training_ids & forbidden_ids
    if overlap:
        raise AssertionError(f"v4 training data overlaps DPO-dev: {sorted(overlap)[:5]}")

    write_jsonl(args.output, output)
    summary = {
        "status": "frozen",
        "source": str(args.input),
        "source_sha256": sha256_file(args.input),
        "output": str(args.output),
        "output_sha256": sha256_file(args.output),
        "source_pairs": len(rows),
        "pair_count": len(output),
        "real_pair_count": real_pair_count,
        "termination_guard_count": len(guards),
        "unique_problem_count": len(training_ids),
        "max_length_ratio": args.max_length_ratio,
        "guard_max_code_chars": args.guard_max_code_chars,
        "skipped_counts": dict(skipped),
        "repair_method_counts": dict(Counter(str(row.get("repair_method")) for row in output)),
        "failure_type_counts": dict(Counter(str(row.get("original_failure_type")) for row in output)),
        "length_direction_counts": length_direction_counts(output),
        "forbidden_prompt_count": len(forbidden_ids),
        "forbidden_overlap_count": len(overlap),
        "policy": "real semantic pairs are length matched; small protected guard rejects repeated equivalent code",
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
