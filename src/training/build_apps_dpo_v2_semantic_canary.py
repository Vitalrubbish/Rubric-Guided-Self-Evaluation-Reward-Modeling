#!/usr/bin/env python3
"""Build a semantic-only, format-matched APPS DPO-v2 canary set."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .build_apps_dpo_v2_preferences import PRIVATE_KEYS, code_audit


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


def canonical_fence(code: str) -> str:
    return f"```python\n{code.strip()}\n```"


def build_semantic_pairs(
    rows: list[dict[str, Any]],
    max_length_ratio: float,
    completion_format: str = "fenced",
) -> tuple[list[dict[str, Any]], Counter[str]]:
    if completion_format not in {"fenced", "raw"}:
        raise ValueError("completion_format must be fenced or raw")
    selected: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    for row in rows:
        if row.get("original_failure_type") == "syntax_error":
            skipped["syntax_error_negative"] += 1
            continue
        if not row.get("rejected_parseable"):
            skipped["rejected_not_parseable"] += 1
            continue
        if not row.get("rejected_required_interface_present"):
            skipped["rejected_interface_missing"] += 1
            continue
        if float(row.get("completion_char_ratio") or 0.0) > max_length_ratio:
            skipped["length_ratio_too_large"] += 1
            continue
        rejected_audit = code_audit(str(row.get("rejected") or ""), [])
        if rejected_audit["top_level_demo_count"]:
            skipped["rejected_top_level_demo"] += 1
            continue

        chosen = str(row.get("chosen") or "").strip()
        rejected = str(row.get("rejected") or "").strip()
        if not chosen or not rejected or chosen == rejected:
            skipped["empty_or_identical"] += 1
            continue
        chosen_output = canonical_fence(chosen) if completion_format == "fenced" else chosen
        rejected_output = canonical_fence(rejected) if completion_format == "fenced" else rejected
        semantic = {
            **row,
            "pair_version": f"apps_simple_method1_dpo_v2_semantic_{completion_format}",
            "chosen": chosen_output,
            "rejected": rejected_output,
            "completion_format": "matched_python_fence" if completion_format == "fenced" else "matched_raw_python",
            "semantic_filter": {
                "rejected_parseable": True,
                "rejected_required_interface_present": True,
                "rejected_top_level_demo_count": 0,
                "syntax_error_negative_excluded": True,
                "max_completion_char_ratio": max_length_ratio,
            },
        }
        leaked = PRIVATE_KEYS.intersection(semantic)
        if leaked:
            raise AssertionError(f"private fields leaked into {semantic.get('pair_id')}: {sorted(leaked)}")
        if completion_format == "fenced":
            if semantic["chosen"].count("```") != 2 or semantic["rejected"].count("```") != 2:
                raise AssertionError(f"unmatched canonical fence in {semantic.get('pair_id')}")
        elif "```" in semantic["chosen"] or "```" in semantic["rejected"]:
            raise AssertionError(f"residual fence in raw semantic pair {semantic.get('pair_id')}")
        selected.append(semantic)
    return sorted(selected, key=lambda row: str(row.get("pair_id"))), skipped


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter semantic APPS repairs and apply a matched code envelope.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--max-length-ratio", type=float, default=3.0)
    parser.add_argument("--min-pairs", type=int, default=100)
    parser.add_argument("--completion-format", choices=("fenced", "raw"), default="fenced")
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    pairs, skipped = build_semantic_pairs(rows, args.max_length_ratio, args.completion_format)
    if len(pairs) < args.min_pairs:
        raise RuntimeError(f"only {len(pairs)} semantic pairs; require at least {args.min_pairs}")
    ids = [str(row.get("id")) for row in pairs]
    if len(ids) != len(set(ids)) or not all(ids):
        raise AssertionError("semantic canary problem IDs are missing or duplicated")

    write_jsonl(args.output, pairs)
    summary = {
        "source": str(args.input),
        "source_sha256": sha256_file(args.input),
        "output": str(args.output),
        "output_sha256": sha256_file(args.output),
        "source_pairs": len(rows),
        "pair_count": len(pairs),
        "unique_problem_count": len(set(ids)),
        "max_length_ratio": args.max_length_ratio,
        "minimum_pairs": args.min_pairs,
        "completion_format": args.completion_format,
        "skipped_counts": dict(skipped),
        "original_failure_type_counts": dict(Counter(str(row.get("original_failure_type")) for row in pairs)),
        "repair_method_counts": dict(Counter(str(row.get("repair_method")) for row in pairs)),
        "io_mode_counts": dict(Counter(str(row.get("io_mode")) for row in pairs)),
        "completion_format_counts": {
            "chosen_fenced": sum("```" in str(row["chosen"]) for row in pairs),
            "rejected_fenced": sum("```" in str(row["rejected"]) for row in pairs),
        },
        "policy": f"semantic-only negatives; both completions use matched {args.completion_format} Python format",
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
