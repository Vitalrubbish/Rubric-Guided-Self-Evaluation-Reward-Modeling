#!/usr/bin/env python3
"""Freeze a sanitized APPS final523 prompt set and audit all ID boundaries."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


PRIVATE_FIELDS = {"canonical_solution", "canonical_solutions", "canonical_verifier"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


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


def row_ids(rows: Iterable[dict[str, Any]]) -> set[str]:
    return {str(row.get("id") or "") for row in rows}


def freeze_rows(
    source_rows: list[dict[str, Any]],
    training_rows: list[dict[str, Any]],
    dev_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(source_rows) != 523:
        raise AssertionError(f"expected 523 source rows, got {len(source_rows)}")
    source_ids = [str(row.get("id") or "") for row in source_rows]
    if "" in source_ids or len(source_ids) != len(set(source_ids)):
        raise AssertionError("final523 source IDs are missing or duplicated")

    split_counts = Counter(str(row.get("eval_split")) for row in source_rows)
    if split_counts != Counter({"validation": 261, "test": 262}):
        raise AssertionError(f"unexpected final523 split counts: {dict(split_counts)}")
    if any(row.get("source_split") != "train" for row in source_rows):
        raise AssertionError("final523 source rows must originate from the APPS train split")

    training_overlap = set(source_ids) & row_ids(training_rows)
    dev_overlap = set(source_ids) & row_ids(dev_rows)
    if training_overlap:
        raise AssertionError(f"final523 overlaps DPO training IDs: {sorted(training_overlap)[:5]}")
    if dev_overlap:
        raise AssertionError(f"final523 overlaps DPO-dev IDs: {sorted(dev_overlap)[:5]}")

    frozen = [{key: value for key, value in row.items() if key not in PRIVATE_FIELDS} for row in source_rows]
    residual = [
        str(row.get("id"))
        for row in frozen
        if PRIVATE_FIELDS.intersection(row)
    ]
    if residual:
        raise AssertionError(f"private fields remain in final523 rows: {residual[:5]}")
    if any(not str(row.get("prompt") or "").strip() for row in frozen):
        raise AssertionError("final523 contains an empty public prompt")

    audit = {
        "rows": len(frozen),
        "unique_ids": len(set(source_ids)),
        "split_counts": dict(split_counts),
        "source_split_counts": dict(Counter(str(row.get("source_split")) for row in frozen)),
        "removed_private_fields": sorted(PRIVATE_FIELDS),
        "training_id_count": len(row_ids(training_rows)),
        "training_overlap_count": 0,
        "dpo_dev_id_count": len(row_ids(dev_rows)),
        "dpo_dev_overlap_count": 0,
        "policy": "frozen final523; public prompt only; canonical solution/verifier metadata removed",
    }
    return frozen, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--training-preferences", type=Path, required=True)
    parser.add_argument("--dpo-dev", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    source_rows = read_jsonl(args.source)
    training_rows = read_jsonl(args.training_preferences)
    dev_rows = read_jsonl(args.dpo_dev)
    frozen, audit = freeze_rows(source_rows, training_rows, dev_rows)
    write_jsonl(args.output, frozen)
    manifest = {
        "status": "frozen",
        "source": str(args.source),
        "source_sha256": sha256_file(args.source),
        "training_preferences": str(args.training_preferences),
        "training_preferences_sha256": sha256_file(args.training_preferences),
        "dpo_dev": str(args.dpo_dev),
        "dpo_dev_sha256": sha256_file(args.dpo_dev),
        "output": str(args.output),
        "output_sha256": sha256_file(args.output),
        **audit,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
