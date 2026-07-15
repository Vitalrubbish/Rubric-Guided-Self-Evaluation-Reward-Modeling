#!/usr/bin/env python3
"""Build rejection-sampling SFT rows from Method 1 loop-v0 repair pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_bucket(value: str, modulo: int) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:8], 16) % modulo


def build_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    pairs = read_jsonl(args.pairs)
    seen: set[tuple[str, str]] = set()
    records = []
    for pair in sorted(pairs, key=lambda row: (str(row.get("id")), str(row.get("pair_id")))):
        problem_id = str(pair.get("id") or "")
        prompt = str(pair.get("prompt") or "").strip()
        chosen = str(pair.get("chosen") or "").strip()
        if not problem_id or not prompt or not chosen:
            continue
        key = (problem_id, chosen)
        if args.deduplicate_problem_chosen and key in seen:
            continue
        seen.add(key)
        split = "validation" if stable_bucket(problem_id, 100) < args.validation_percent else "train"
        records.append(
            {
                "id": f"{problem_id}__rs_sft_{len(records) + 1:04d}",
                "problem_id": problem_id,
                "split": split,
                "task_type": "solve_rejection_sampled_repair",
                "prompt": prompt,
                "completion": chosen,
                "source": "same_problem_k5_verifier_passing_repair",
                "metadata": {
                    "pair_id": pair.get("pair_id"),
                    "chosen_response_id": pair.get("chosen_response_id"),
                    "chosen_rubric_score": pair.get("chosen_rubric_score"),
                    "rejected_response_id": pair.get("rejected_response_id"),
                    "rejected_failure_type": pair.get("rejected_failure_type"),
                    "preference_source": pair.get("preference_source"),
                },
            }
        )
    if not any(row["split"] == "train" for row in records) or not any(row["split"] == "validation" for row in records):
        raise RuntimeError("RS-SFT split must contain both train and validation rows")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Build APPS Method 1 loop-v0 RS-SFT rows.")
    parser.add_argument(
        "--pairs",
        type=Path,
        default=Path("data/preferences/apps_simple_method1_loop_v0_same_problem_only_dpo_pairs.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/sft/apps_simple_method1_loop_v0_same_problem_rs_sft.jsonl"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("data/sft/apps_simple_method1_loop_v0_same_problem_rs_sft_summary.json"),
    )
    parser.add_argument("--validation-percent", type=int, default=10)
    parser.add_argument("--deduplicate-problem-chosen", action="store_true", default=True)
    args = parser.parse_args()
    if not 1 <= args.validation_percent <= 50:
        raise ValueError("--validation-percent must be in [1, 50]")

    records = build_rows(args)
    write_jsonl(args.output, records)
    summary = {
        "input_pairs": str(args.pairs),
        "input_pairs_sha256": sha256_file(args.pairs),
        "output": str(args.output),
        "output_sha256": sha256_file(args.output),
        "rows": len(records),
        "unique_problem_count": len({row["problem_id"] for row in records}),
        "split_counts": dict(Counter(row["split"] for row in records)),
        "task_type_counts": dict(Counter(row["task_type"] for row in records)),
        "source_counts": dict(Counter(row["source"] for row in records)),
        "rejected_failure_counts": dict(
            Counter(str((row.get("metadata") or {}).get("rejected_failure_type")) for row in records)
        ),
        "policy": {
            "role": "stable rejection-sampling SFT ablation for Method 1 loop-v0",
            "target": "verifier-passing same-problem k=5 repair code only",
            "rejected_code_in_loss": False,
            "split_unit": "problem id hash",
        },
    }
    write_json(args.summary_output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
