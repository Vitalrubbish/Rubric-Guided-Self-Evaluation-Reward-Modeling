#!/usr/bin/env python3
"""Merge a primary preference set with deterministic, non-overlapping supplements."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .build_apps_dpo_v2_preferences import PRIVATE_KEYS
from .build_apps_dpo_v2_semantic_canary import read_jsonl, sha256_file, write_jsonl


def merge_rows(
    primary: list[dict[str, Any]],
    supplement: list[dict[str, Any]],
    *,
    target_size: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if target_size < len(primary):
        raise ValueError("target size cannot be smaller than the primary set")
    primary_ids = {str(row.get("id") or "") for row in primary}
    primary_pair_ids = {str(row.get("pair_id") or "") for row in primary}
    if "" in primary_ids or len(primary_ids) != len(primary):
        raise ValueError("primary set must have unique non-empty problem IDs")
    if "" in primary_pair_ids or len(primary_pair_ids) != len(primary):
        raise ValueError("primary set must have unique non-empty pair IDs")

    candidates = [
        row
        for row in supplement
        if str(row.get("id") or "") not in primary_ids
        and str(row.get("pair_id") or "") not in primary_pair_ids
    ]
    candidates.sort(
        key=lambda row: hashlib.sha256(f"{seed}:{row.get('pair_id')}".encode("utf-8")).hexdigest()
    )
    needed = target_size - len(primary)
    if len(candidates) < needed:
        raise RuntimeError(f"only {len(candidates)} non-overlapping supplements; require {needed}")
    selected_supplements = candidates[:needed]
    output = sorted(primary + selected_supplements, key=lambda row: str(row.get("pair_id")))
    return output, selected_supplements


def validate_rows(rows: list[dict[str, Any]], forbidden_ids: set[str]) -> None:
    problem_ids = [str(row.get("id") or "") for row in rows]
    pair_ids = [str(row.get("pair_id") or "") for row in rows]
    if "" in problem_ids or len(problem_ids) != len(set(problem_ids)):
        raise AssertionError("merged preferences have missing or duplicate problem IDs")
    if "" in pair_ids or len(pair_ids) != len(set(pair_ids)):
        raise AssertionError("merged preferences have missing or duplicate pair IDs")
    overlap = set(problem_ids) & forbidden_ids
    if overlap:
        raise AssertionError(f"merged preferences overlap held-out IDs: {sorted(overlap)[:5]}")
    formats = {str(row.get("completion_format") or "") for row in rows}
    if len(formats) != 1 or formats.pop() not in {"matched_python_fence", "matched_raw_python"}:
        raise AssertionError("merged preferences must use one explicit matched completion format")
    for row in rows:
        leaked = PRIVATE_KEYS.intersection(row)
        if leaked:
            raise AssertionError(f"private fields leaked into {row.get('pair_id')}: {sorted(leaked)}")
        if not row.get("chosen_parseable") or not row.get("rejected_parseable"):
            raise AssertionError(f"non-semantic pair in merged data: {row.get('pair_id')}")
        if row.get("original_failure_type") == "syntax_error":
            raise AssertionError(f"syntax negative in merged data: {row.get('pair_id')}")
        chosen_fences = str(row.get("chosen") or "").count("```")
        rejected_fences = str(row.get("rejected") or "").count("```")
        if row.get("completion_format") == "matched_python_fence":
            if chosen_fences != 2 or rejected_fences != 2:
                raise AssertionError(f"fenced envelope mismatch: {row.get('pair_id')}")
        elif chosen_fences or rejected_fences:
            raise AssertionError(f"residual fence in raw pair: {row.get('pair_id')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge deterministic APPS preference supplements.")
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--supplement", type=Path, required=True)
    parser.add_argument("--forbidden-ids", type=Path, action="append", default=[])
    parser.add_argument("--target-size", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    primary = read_jsonl(args.primary)
    supplement = read_jsonl(args.supplement)
    forbidden_ids = {
        str(row.get("id") or "")
        for path in args.forbidden_ids
        for row in read_jsonl(path)
        if row.get("id")
    }
    output, selected_supplements = merge_rows(
        primary,
        supplement,
        target_size=args.target_size,
        seed=args.seed,
    )
    validate_rows(output, forbidden_ids)
    write_jsonl(args.output, output)
    manifest = {
        "status": "frozen",
        "primary": str(args.primary),
        "primary_sha256": sha256_file(args.primary),
        "supplement": str(args.supplement),
        "supplement_sha256": sha256_file(args.supplement),
        "output": str(args.output),
        "output_sha256": sha256_file(args.output),
        "seed": args.seed,
        "target_size": args.target_size,
        "primary_count": len(primary),
        "supplement_count": len(selected_supplements),
        "selected_supplement_pair_ids": [str(row.get("pair_id")) for row in selected_supplements],
        "unique_problem_count": len({str(row.get("id")) for row in output}),
        "repair_method_counts": dict(Counter(str(row.get("repair_method")) for row in output)),
        "failure_type_counts": dict(Counter(str(row.get("original_failure_type")) for row in output)),
        "forbidden_id_count": len(forbidden_ids),
        "forbidden_overlap_count": 0,
        "policy": "retain all primary semantic pairs; deterministic non-overlapping semantic supplement only",
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
