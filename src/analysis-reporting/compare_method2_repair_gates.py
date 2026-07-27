#!/usr/bin/env python3
"""Compare two Method 2 repair gate labeled JSONL files by sample id."""

from __future__ import annotations

import argparse
import hashlib
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


def code_sha(row: dict[str, Any]) -> str:
    code = str(row.get("extracted_code") or row.get("generated_code") or "")
    return hashlib.sha256(code.encode("utf-8")).hexdigest()[:12]


def short_error(row: dict[str, Any], limit: int = 240) -> str | None:
    error = row.get("error")
    if error is None:
        return None
    text = " ".join(str(error).split())
    return text[:limit]


def status(row: dict[str, Any]) -> str:
    return "P" if row.get("passed") else "F"


def case_summary(sample_id: str, baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    baseline_sha = code_sha(baseline)
    candidate_sha = code_sha(candidate)
    return {
        "id": sample_id,
        "baseline_passed": bool(baseline.get("passed")),
        "candidate_passed": bool(candidate.get("passed")),
        "baseline_failure_type": baseline.get("failure_type"),
        "candidate_failure_type": candidate.get("failure_type"),
        "baseline_finish_reason": baseline.get("finish_reason"),
        "candidate_finish_reason": candidate.get("finish_reason"),
        "baseline_generated_token_count": baseline.get("method2_generated_token_count"),
        "candidate_generated_token_count": candidate.get("method2_generated_token_count"),
        "baseline_code_sha": baseline_sha,
        "candidate_code_sha": candidate_sha,
        "same_extracted_code": baseline_sha == candidate_sha,
        "candidate_error": short_error(candidate),
        "likely_timeout_flake": (
            bool(baseline.get("passed"))
            and not candidate.get("passed")
            and candidate.get("failure_type") == "timeout"
            and baseline_sha == candidate_sha
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Method 2 repair gate transitions.")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    baseline_by_id = {str(row.get("id")): row for row in read_jsonl(args.baseline)}
    candidate_by_id = {str(row.get("id")): row for row in read_jsonl(args.candidate)}
    common_ids = sorted(set(baseline_by_id) & set(candidate_by_id))
    if not common_ids:
        raise SystemExit("no common sample ids")

    transitions: Counter[str] = Counter()
    regressions: list[dict[str, Any]] = []
    improvements: list[dict[str, Any]] = []
    still_failed: list[dict[str, Any]] = []
    for sample_id in common_ids:
        baseline = baseline_by_id[sample_id]
        candidate = candidate_by_id[sample_id]
        transition = f"{status(baseline)}->{status(candidate)}"
        transitions[transition] += 1
        item = case_summary(sample_id, baseline, candidate)
        if transition == "P->F":
            regressions.append(item)
        elif transition == "F->P":
            improvements.append(item)
        elif transition == "F->F":
            still_failed.append(item)

    summary = {
        "baseline": str(args.baseline),
        "candidate": str(args.candidate),
        "common_rows": len(common_ids),
        "baseline_only_rows": len(set(baseline_by_id) - set(candidate_by_id)),
        "candidate_only_rows": len(set(candidate_by_id) - set(baseline_by_id)),
        "transitions": dict(transitions),
        "candidate_failure_counts": dict(
            Counter(str(candidate_by_id[sample_id].get("failure_type") or "passed") for sample_id in common_ids)
        ),
        "likely_timeout_flake_count": sum(1 for item in regressions if item["likely_timeout_flake"]),
        "regressions": regressions,
        "improvements": improvements,
        "still_failed_failure_counts": dict(Counter(str(item["candidate_failure_type"]) for item in still_failed)),
    }

    if args.output:
        write_json(args.output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
