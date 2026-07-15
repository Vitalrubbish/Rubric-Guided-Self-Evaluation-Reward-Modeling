#!/usr/bin/env python3
"""Summarize Method 2 repair-gate verifier results."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


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


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Method 2 repair gate.")
    parser.add_argument("--labeled", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-pass-rate", type=float, default=0.20)
    parser.add_argument("--max-syntax-rate", type=float, default=0.30)
    args = parser.parse_args()

    rows = read_jsonl(args.labeled)
    total = len(rows)
    passed = sum(bool(row.get("passed")) for row in rows)
    failure_counts = Counter("passed" if row.get("passed") else str(row.get("failure_type") or "unknown") for row in rows)
    extraction_counts = Counter(str(row.get("method2_extraction_status") or "unknown") for row in rows)
    finish_counts = Counter(str(row.get("finish_reason") or "unknown") for row in rows)
    io_counts = Counter(str(row.get("io_mode") or "unknown") for row in rows)

    pass_rate = passed / total if total else 0.0
    syntax_rate = failure_counts.get("syntax_error", 0) / total if total else 0.0
    summary = {
        "labeled": str(args.labeled),
        "rows": total,
        "passed": passed,
        "pass_rate": pass_rate,
        "failure_counts": dict(failure_counts),
        "extraction_counts": dict(extraction_counts),
        "finish_counts": dict(finish_counts),
        "io_mode_counts": dict(io_counts),
        "gates": {
            "pass_rate_ge_min": pass_rate >= args.min_pass_rate,
            "syntax_rate_le_max": syntax_rate <= args.max_syntax_rate,
            "all_extractions_ok": set(extraction_counts) <= {"ok"},
        },
        "thresholds": {
            "min_pass_rate": args.min_pass_rate,
            "max_syntax_rate": args.max_syntax_rate,
        },
        "policy": "validation repair gate for Method 2 critic+repair adapter; originals are known failed by pair construction",
    }
    summary["gate_passed"] = all(summary["gates"].values())
    write_json(args.output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
