#!/usr/bin/env python3
"""Expand the Method 2 repair gate beyond the original 38 validation rows.

The 38-row gate has a +-15pp Wilson half-width, so no realistic iteration
effect is detectable on it. This script builds a larger validation-only file
by combining:

1. the original validation rows from an existing Method 2 SFT build
   (kept verbatim, so historical gate numbers remain a comparable subset);
2. additional failed original responses that were never used as training
   problems by any Method 2 version (v0.3 base SFT, v0.4/v0.5 generated
   rows all derive from the same base train prompts).

Leakage rule: a failed original is eligible only when its problem id does
not appear anywhere in the SFT build (train or validation rows).
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from src.self_play.build_method2_bootstrap_data import (
    method2_prompt,
    normalize_code,
    sha256_file,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft-input", type=Path, required=True, help="Method 2 SFT build to take validation rows and the exclusion set from")
    parser.add_argument("--responses", type=Path, required=True, help="labeled non-length original responses (failed rows form the expansion pool)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--target-rows", type=int, default=200)
    parser.add_argument("--response-prefix", default="Repair response:")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sft_rows = read_jsonl(args.sft_input)
    original_validation = [row for row in sft_rows if row.get("split") == "validation"]
    excluded_problem_ids = {str(row.get("problem_id")) for row in sft_rows}

    responses = read_jsonl(args.responses)
    pool_by_problem: dict[str, dict[str, Any]] = {}
    for row in responses:
        if row.get("passed"):
            continue
        problem_id = str(row.get("id") or "")
        if not problem_id or problem_id in excluded_problem_ids:
            continue
        if problem_id not in pool_by_problem:
            pool_by_problem[problem_id] = row

    pool = sorted(pool_by_problem.values(), key=lambda row: str(row.get("id")))
    rng = random.Random(args.seed)
    rng.shuffle(pool)

    n_new = args.target_rows - len(original_validation)
    if n_new < 0:
        raise ValueError("--target-rows is smaller than the original validation set")
    if len(pool) < n_new:
        raise ValueError(f"expansion pool too small: need {n_new}, have {len(pool)}")
    selected = pool[:n_new]

    new_rows: list[dict[str, Any]] = []
    for index, row in enumerate(selected, start=1):
        problem_id = str(row.get("id"))
        record_id = f"{problem_id}__method2_gate_expansion_{index:05d}"
        failed_code = normalize_code(row.get("generated_code") or row.get("extracted_code"))
        prompt = method2_prompt(
            str(row.get("prompt") or ""),
            failed_code,
            response_prefix=args.response_prefix,
        )
        new_rows.append(
            {
                "id": record_id,
                "problem_id": problem_id,
                "dataset": row.get("dataset"),
                "split": "validation",
                "task_type": "method2_self_play_critic_repair",
                "prompt": prompt,
                "completion": "",
                "source": "gate_expansion_failed_original",
                "interface_names": row.get("interface_names") or [],
                "interface_signatures": row.get("interface_signatures") or [],
                "starter_code": row.get("starter_code"),
                "input_output": row.get("input_output"),
                "difficulty": row.get("difficulty"),
                "io_mode": row.get("io_mode"),
                "chosen_response_id": None,
                "rejected_response_id": row.get("response_id"),
                "metadata": {
                    "problem_id": problem_id,
                    "dataset": row.get("dataset"),
                    "source_split": row.get("split"),
                    "source_file": str(args.responses),
                    "source_kind": "gate_expansion_failed_original",
                    "failure_type": row.get("failure_type"),
                    "original_finish_reason": row.get("finish_reason"),
                    "end_marker": None,
                },
            }
        )

    out_rows = [*original_validation, *new_rows]
    write_jsonl(args.output, out_rows)

    summary = {
        "sft_input": str(args.sft_input),
        "responses": str(args.responses),
        "output": str(args.output),
        "output_sha256": sha256_file(args.output),
        "seed": args.seed,
        "original_validation_rows": len(original_validation),
        "expansion_pool_size": len(pool),
        "new_rows": len(new_rows),
        "total_rows": len(out_rows),
        "excluded_problem_count": len(excluded_problem_ids),
        "new_rows_io_mode_counts": dict(Counter(str(r.get("io_mode") or "unknown") for r in new_rows)),
        "new_rows_failure_type_counts": dict(
            Counter(str((r.get("metadata") or {}).get("failure_type") or "unknown") for r in new_rows)
        ),
        "original_validation_problem_ids": sorted(str(r.get("problem_id")) for r in original_validation),
        "policy": {
            "leakage_rule": "expansion problems never appear in the Method 2 SFT build (train or validation); v0.4/v0.5 generated rows derive from the same base train prompts",
            "distribution_note": "expansion rows are raw verifier failures and include problems where no passing repair was found, so the expanded gate is harder than the original 38-row subset; always report the original-38 subset alongside the full set",
        },
    }
    write_json(args.summary_output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
