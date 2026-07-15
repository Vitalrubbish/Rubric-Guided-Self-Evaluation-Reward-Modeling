#!/usr/bin/env python3
"""Freeze a train-derived APPS DPO-dev set that never enters preference training."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ids_from_files(paths: Iterable[Path]) -> set[str]:
    result: set[str] = set()
    for path in paths:
        for row in read_jsonl(path):
            problem_id = str(row.get("id") or "")
            if problem_id:
                result.add(problem_id)
    return result


def stable_rank(problem_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{problem_id}".encode("utf-8")).hexdigest()


def freeze_dev_rows(
    source_rows: Iterable[dict[str, Any]],
    base_by_id: dict[str, dict[str, Any]],
    excluded_ids: set[str],
    *,
    size: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    eligible: list[dict[str, Any]] = []
    for source in source_rows:
        problem_id = str(source.get("id") or "")
        if source.get("split") != "train" or not problem_id or problem_id in excluded_ids:
            continue
        if problem_id not in base_by_id:
            continue
        eligible.append(source)

    eligible.sort(key=lambda row: (stable_rank(str(row["id"]), seed), str(row["id"])))
    if len(eligible) < size:
        raise ValueError(f"only {len(eligible)} eligible train rows for requested DPO-dev size {size}")
    selected = eligible[:size]

    prompts: list[dict[str, Any]] = []
    base_labeled: list[dict[str, Any]] = []
    for source in selected:
        problem_id = str(source["id"])
        prompt_row = dict(source)
        prompt_row["source_split"] = "train"
        prompt_row["split"] = "dpo_dev"
        prompt_row["eval_split"] = "dpo_dev"
        prompts.append(prompt_row)

        base_row = dict(base_by_id[problem_id])
        base_row["source_split"] = "train"
        base_row["split"] = "dpo_dev"
        base_row["eval_split"] = "dpo_dev"
        base_labeled.append(base_row)
    return prompts, base_labeled


def distribution(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    return {
        "rows": len(rows),
        "io_mode_counts": dict(Counter(str(row.get("io_mode")) for row in rows)),
        "difficulty_counts": dict(Counter(str(row.get("difficulty")) for row in rows)),
        "base_passed": sum(bool(row.get("passed")) for row in rows if "passed" in row),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze a leakage-safe APPS DPO-dev split from train rows.")
    parser.add_argument(
        "--source-prompts",
        type=Path,
        default=Path("data/processed/apps_train_simple_executable_prompts_unified.jsonl"),
    )
    parser.add_argument(
        "--base-labeled",
        type=Path,
        default=Path("data/responses/apps_train_simple_executable_qwen25_k1_t2048_full_labeled.jsonl"),
    )
    parser.add_argument(
        "--exclude-file",
        type=Path,
        action="append",
        default=[Path("data/repair/apps_simple_method1_repair_candidates_v1.jsonl")],
    )
    parser.add_argument(
        "--final-heldout",
        type=Path,
        default=Path("data/processed/apps_simple_method1_internal_eval_prompts_v1.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/apps_simple_method1_dpo_dev_v2_prompts.jsonl"),
    )
    parser.add_argument(
        "--base-output",
        type=Path,
        default=Path("data/responses/apps_simple_method1_dpo_dev_v2_base_labeled.jsonl"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("data/eval/apps_simple_method1_dpo_dev_v2_manifest.json"),
    )
    parser.add_argument("--size", type=int, default=160)
    parser.add_argument("--seed", type=int, default=20260713)
    args = parser.parse_args()

    source_rows = read_jsonl(args.source_prompts)
    base_rows = read_jsonl(args.base_labeled)
    base_by_id = {str(row.get("id")): row for row in base_rows if row.get("id")}
    if len(base_by_id) != len(base_rows):
        raise ValueError("base labeled IDs are missing or duplicated")

    excluded_ids = ids_from_files(args.exclude_file)
    final_ids = ids_from_files([args.final_heldout])
    prompts, selected_base = freeze_dev_rows(
        source_rows,
        base_by_id,
        excluded_ids | final_ids,
        size=args.size,
        seed=args.seed,
    )
    prompt_ids = {str(row["id"]) for row in prompts}
    if prompt_ids & excluded_ids:
        raise AssertionError("DPO-dev overlaps repair candidates")
    if prompt_ids & final_ids:
        raise AssertionError("DPO-dev overlaps final validation/test held-out")
    if any(row.get("source_split") != "train" for row in prompts):
        raise AssertionError("DPO-dev contains a non-train source row")

    write_jsonl(args.output, prompts)
    write_jsonl(args.base_output, selected_base)
    summary = {
        "source_prompts": str(args.source_prompts),
        "base_labeled": str(args.base_labeled),
        "exclude_files": [str(path) for path in args.exclude_file],
        "final_heldout": str(args.final_heldout),
        "output": str(args.output),
        "base_output": str(args.base_output),
        "seed": args.seed,
        "requested_size": args.size,
        "selected": distribution(selected_base),
        "excluded_id_count": len(excluded_ids),
        "final_heldout_id_count": len(final_ids),
        "preference_overlap_count": 0,
        "final_heldout_overlap_count": 0,
        "policy": "source split train only; excluded from all DPO-v2 preference construction",
        "source_prompts_sha256": sha256_file(args.source_prompts),
        "base_labeled_sha256": sha256_file(args.base_labeled),
    }
    summary["output_sha256"] = sha256_file(args.output)
    summary["base_output_sha256"] = sha256_file(args.base_output)
    write_json(args.summary_output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
