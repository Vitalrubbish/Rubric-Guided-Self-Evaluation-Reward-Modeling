#!/usr/bin/env python3
"""Freeze an auditable APPS DPO-v2 canary preference subset."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Iterable


PRIVATE_KEYS = {
    "input_output",
    "private_diagnostics",
    "expected",
    "got",
    "failing_case",
    "hidden_tests",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {error}") from error
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def freeze_canary_rows(rows: list[dict[str, Any]], size: int, seed: int) -> list[dict[str, Any]]:
    if size <= 0:
        raise ValueError("size must be positive")
    problem_ids = [str(row.get("id") or "") for row in rows]
    pair_ids = [str(row.get("pair_id") or "") for row in rows]
    if not all(problem_ids) or len(problem_ids) != len(set(problem_ids)):
        raise ValueError("input must contain one row per non-empty problem ID")
    if not all(pair_ids) or len(pair_ids) != len(set(pair_ids)):
        raise ValueError("input pair IDs are missing or duplicated")

    rng = random.Random(seed)
    priority = [row for row in rows if "two_stage" in str(row.get("repair_method") or "")]
    remainder = [row for row in rows if "two_stage" not in str(row.get("repair_method") or "")]
    rng.shuffle(priority)
    selected = priority[:size]

    buckets: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in remainder:
        buckets[str(row.get("original_failure_type") or "unknown")].append(row)
    queues: dict[str, deque[dict[str, Any]]] = {}
    for name, bucket in buckets.items():
        rng.shuffle(bucket)
        queues[name] = deque(bucket)

    while len(selected) < min(size, len(rows)):
        made_progress = False
        for name in sorted(queues):
            if queues[name] and len(selected) < size:
                selected.append(queues[name].popleft())
                made_progress = True
        if not made_progress:
            break
    return sorted(selected, key=lambda row: str(row["pair_id"]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze a deterministic APPS DPO-v2 canary set.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--forbidden-ids", type=Path, action="append", default=[])
    parser.add_argument("--size", type=int, default=400)
    parser.add_argument("--min-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260713)
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    if len(rows) < args.min_size:
        raise RuntimeError(f"only {len(rows)} strict pairs; require at least {args.min_size}")
    forbidden_ids = {
        str(row.get("id"))
        for path in args.forbidden_ids
        for row in read_jsonl(path)
        if row.get("id")
    }
    overlap = {str(row.get("id")) for row in rows} & forbidden_ids
    if overlap:
        raise AssertionError(f"input preferences overlap forbidden IDs: {sorted(overlap)[:5]}")
    for row in rows:
        leaked = PRIVATE_KEYS.intersection(row)
        if leaked:
            raise AssertionError(f"private fields in {row.get('pair_id')}: {sorted(leaked)}")
        if "```" in str(row.get("chosen") or "") or "```" in str(row.get("rejected") or ""):
            raise AssertionError(f"fenced completion in {row.get('pair_id')}")
        if not row.get("chosen_parseable") or row.get("split") != "train":
            raise AssertionError(f"invalid strict pair contract in {row.get('pair_id')}")

    selected = freeze_canary_rows(rows, args.size, args.seed)
    write_jsonl(args.output, selected)
    manifest = {
        "source": str(args.input),
        "source_sha256": sha256_file(args.input),
        "output": str(args.output),
        "output_sha256": sha256_file(args.output),
        "seed": args.seed,
        "requested_size": args.size,
        "minimum_size": args.min_size,
        "source_pairs": len(rows),
        "selected_pairs": len(selected),
        "selected_unique_problems": len({str(row["id"]) for row in selected}),
        "repair_method_counts": dict(Counter(str(row.get("repair_method")) for row in selected)),
        "original_failure_type_counts": dict(
            Counter(str(row.get("original_failure_type") or "unknown") for row in selected)
        ),
        "forbidden_id_count": len(forbidden_ids),
        "forbidden_overlap_count": 0,
        "policy": "retain all available two-stage pairs first, then deterministic round-robin by original failure type",
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
