#!/usr/bin/env python3
"""Build preference pairs from verified repair outputs.

Pairs use the original task prompt:

chosen = repaired code that passes the verifier
rejected = original failed model code

This creates a conservative Method 1 preference signal without exposing hidden
verifier details to the model prompt.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_code_for_dedup(code: str) -> str:
    return "\n".join(line.rstrip() for line in code.strip().splitlines()).strip()


def strip_markdown_fences(code: str) -> str:
    cleaned = []
    for line in code.strip().splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            continue
        cleaned.append(line)
    return "\n".join(cleaned).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build preference pairs from passed repair outputs.")
    parser.add_argument("--repair-labeled", type=Path, default=Path("data/repair/apps_simple_method1_repair_smoke20_labeled.jsonl"))
    parser.add_argument("--evaluator-rows", type=Path, default=Path("data/evaluator/apps_simple_method1_evaluator_training_rows_v1.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/preferences/apps_simple_method1_repair_smoke20_pairs.jsonl"))
    parser.add_argument("--summary-output", type=Path, default=Path("data/preferences/apps_simple_method1_repair_smoke20_pairs_summary.json"))
    parser.add_argument("--source", default="apps_simple_method1_repair_smoke20")
    parser.add_argument("--max-pairs-per-original", type=int, default=2, help="Use 0 for no per-original cap.")
    args = parser.parse_args()

    original_by_response = {str(row.get("response_id")): row for row in read_jsonl(args.evaluator_rows)}
    repair_rows = read_jsonl(args.repair_labeled)
    pairs = []
    skipped = Counter()
    chosen_seen_by_original: dict[str, set[str]] = defaultdict(set)
    pair_count_by_original: Counter[str] = Counter()

    for repair in repair_rows:
        if not repair.get("passed"):
            skipped["repair_failed"] += 1
            continue
        original_response_id = str(repair.get("original_response_id") or "")
        original = original_by_response.get(original_response_id)
        if not original:
            skipped["missing_original"] += 1
            continue
        if original.get("passed"):
            skipped["original_not_failed"] += 1
            continue
        chosen = strip_markdown_fences(str(repair.get("extracted_code") or repair.get("generated_code") or ""))
        rejected = strip_markdown_fences(str(original.get("extracted_code") or original.get("generated_code") or ""))
        prompt = str(original.get("prompt") or "").strip()
        if not chosen or not rejected or not prompt:
            skipped["missing_prompt_or_code"] += 1
            continue
        if chosen == rejected:
            skipped["identical_code"] += 1
            continue
        normalized_chosen = normalize_code_for_dedup(chosen)
        if normalized_chosen in chosen_seen_by_original[original_response_id]:
            skipped["duplicate_repair_for_original"] += 1
            continue
        if args.max_pairs_per_original > 0 and pair_count_by_original[original_response_id] >= args.max_pairs_per_original:
            skipped["max_pairs_per_original"] += 1
            continue
        pair_id = f"repair_pair_{len(pairs) + 1:04d}"
        pairs.append(
            {
                "pair_id": pair_id,
                "source": args.source,
                "id": original.get("id"),
                "dataset": original.get("dataset"),
                "split": original.get("split"),
                "difficulty": original.get("difficulty"),
                "io_mode": original.get("io_mode"),
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "chosen_response_id": repair.get("response_id"),
                "rejected_response_id": original_response_id,
                "repair_candidate_id": repair.get("repair_candidate_id"),
                "critic_pass_probability": repair.get("critic_pass_probability"),
                "selection_reason": repair.get("selection_reason"),
                "chosen_verifier_passed": True,
                "rejected_verifier_passed": False,
            }
        )
        chosen_seen_by_original[original_response_id].add(normalized_chosen)
        pair_count_by_original[original_response_id] += 1

    summary = {
        "num_repair_rows": len(repair_rows),
        "num_pairs": len(pairs),
        "num_originals_with_pairs": len(pair_count_by_original),
        "source": args.source,
        "max_pairs_per_original": args.max_pairs_per_original,
        "skipped": dict(skipped),
        "selection_reason_counts": dict(Counter(str(row.get("selection_reason")) for row in pairs)),
        "io_mode_counts": dict(Counter(str(row.get("io_mode")) for row in pairs)),
        "policy": {
            "prompt": "original public task prompt",
            "chosen": "verified passing repair code",
            "rejected": "original failed model code",
            "private_verifier_details": "excluded",
        },
    }
    write_jsonl(args.output, pairs)
    write_json(args.summary_output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
