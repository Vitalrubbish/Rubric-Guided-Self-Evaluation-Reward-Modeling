#!/usr/bin/env python3
"""Filter Method 1 loop-v0 APPS DPO pairs for a lower-noise canary."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from collections import Counter
from pathlib import Path
from statistics import mean
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


def parseable_python(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def nonempty_line_count(text: str) -> int:
    return sum(1 for line in text.splitlines() if line.strip())


def safe_ratio(left: int, right: int) -> float:
    return max(left, right) / max(1, min(left, right))


def pair_audit(row: dict[str, Any]) -> dict[str, Any]:
    chosen = str(row.get("chosen") or "").strip()
    rejected = str(row.get("rejected") or "").strip()
    chosen_chars = len(chosen)
    rejected_chars = len(rejected)
    chosen_tokens = len(chosen.split())
    rejected_tokens = len(rejected.split())
    chosen_lines = nonempty_line_count(chosen)
    rejected_lines = nonempty_line_count(rejected)
    chosen_score = row.get("chosen_rubric_score")
    rejected_score = row.get("rejected_rubric_score")
    try:
        score_margin = float(chosen_score) - float(rejected_score)
    except (TypeError, ValueError):
        score_margin = 0.0
    return {
        "chosen_chars": chosen_chars,
        "rejected_chars": rejected_chars,
        "chosen_whitespace_tokens": chosen_tokens,
        "rejected_whitespace_tokens": rejected_tokens,
        "chosen_nonempty_lines": chosen_lines,
        "rejected_nonempty_lines": rejected_lines,
        "char_ratio": safe_ratio(chosen_chars, rejected_chars),
        "whitespace_token_ratio": safe_ratio(chosen_tokens, rejected_tokens),
        "line_ratio": safe_ratio(chosen_lines, rejected_lines),
        "chosen_parseable": parseable_python(chosen),
        "rejected_parseable": parseable_python(rejected),
        "score_margin": score_margin,
    }


def numeric_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    ordered = sorted(values)

    def pct(fraction: float) -> float:
        return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * fraction))]

    return {
        "min": ordered[0],
        "mean": mean(ordered),
        "p50": pct(0.50),
        "p95": pct(0.95),
        "max": ordered[-1],
    }


def failure_type(row: dict[str, Any]) -> str:
    return str(row.get("rejected_failure_type") or "unknown_failure")


def quality_key(row: dict[str, Any]) -> tuple[Any, ...]:
    audit = row["_filter_audit"]
    return (
        audit["char_ratio"],
        audit["whitespace_token_ratio"],
        audit["line_ratio"],
        -audit["score_margin"],
        str(row.get("id") or ""),
        str(row.get("pair_id") or ""),
    )


def build_candidates(args: argparse.Namespace, rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], Counter[str]]:
    candidates: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    excluded_failure_types = set(args.exclude_rejected_failure_type or [])
    for row in rows:
        chosen = str(row.get("chosen") or "").strip()
        rejected = str(row.get("rejected") or "").strip()
        if args.required_preference_source and row.get("preference_source") != args.required_preference_source:
            skipped["preference_source_mismatch"] += 1
            continue
        if failure_type(row) in excluded_failure_types:
            skipped["excluded_rejected_failure_type"] += 1
            continue
        if not chosen or not rejected or chosen == rejected:
            skipped["empty_or_identical_completion"] += 1
            continue
        audit = pair_audit(row)
        if not args.allow_unparseable_chosen and not audit["chosen_parseable"]:
            skipped["chosen_not_parseable"] += 1
            continue
        if max(audit["chosen_chars"], audit["rejected_chars"]) > args.max_completion_chars:
            skipped["completion_too_long"] += 1
            continue
        if audit["char_ratio"] > args.max_char_ratio:
            skipped["char_ratio_too_large"] += 1
            continue
        if audit["whitespace_token_ratio"] > args.max_whitespace_token_ratio:
            skipped["whitespace_token_ratio_too_large"] += 1
            continue
        if audit["line_ratio"] > args.max_line_ratio:
            skipped["line_ratio_too_large"] += 1
            continue
        candidate = dict(row)
        candidate["_filter_audit"] = audit
        candidates.append(candidate)
    return candidates, skipped


def select_pairs(args: argparse.Namespace, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    source_failure_counts = Counter(failure_type(row) for row in candidates)
    selected: list[dict[str, Any]] = []
    problem_counts: Counter[str] = Counter()
    failure_counts: Counter[str] = Counter()

    ordered = sorted(
        candidates,
        key=lambda row: (
            source_failure_counts[failure_type(row)],
            failure_type(row) == "syntax_error",
            failure_type(row),
            *quality_key(row),
        ),
    )
    for row in ordered:
        problem_id = str(row.get("id") or "")
        rejected_type = failure_type(row)
        if not problem_id:
            continue
        if args.max_pairs and len(selected) >= args.max_pairs:
            break
        if problem_counts[problem_id] >= args.max_pairs_per_problem:
            continue
        if args.max_pairs_per_rejected_failure_type and (
            failure_counts[rejected_type] >= args.max_pairs_per_rejected_failure_type
        ):
            continue
        selected.append(row)
        problem_counts[problem_id] += 1
        failure_counts[rejected_type] += 1
    return selected


def output_pair(row: dict[str, Any], index: int, pair_version: str) -> dict[str, Any]:
    audit = row["_filter_audit"]
    output = {key: value for key, value in row.items() if key != "_filter_audit"}
    source_pair_id = str(output.get("pair_id") or f"source_pair_{index:04d}")
    output.update(
        {
            "pair_id": f"{source_pair_id}__balanced_filter_v1",
            "source_pair_id": source_pair_id,
            "pair_version": pair_version,
            "rubric_preference_role": "strong_same_problem_anchor_balanced_filtered",
            "selection_policy": "same_problem_length_balanced_filter_v1",
            "filter_audit": {
                "char_ratio": round(float(audit["char_ratio"]), 4),
                "whitespace_token_ratio": round(float(audit["whitespace_token_ratio"]), 4),
                "line_ratio": round(float(audit["line_ratio"]), 4),
                "chosen_parseable": audit["chosen_parseable"],
                "rejected_parseable": audit["rejected_parseable"],
                "score_margin": round(float(audit["score_margin"]), 4),
            },
        }
    )
    return output


def summarize_lengths(rows: list[dict[str, Any]]) -> dict[str, Any]:
    audits = [row["_filter_audit"] for row in rows]
    return {
        "char_ratio": numeric_stats([float(audit["char_ratio"]) for audit in audits]),
        "whitespace_token_ratio": numeric_stats([float(audit["whitespace_token_ratio"]) for audit in audits]),
        "line_ratio": numeric_stats([float(audit["line_ratio"]) for audit in audits]),
        "chosen_chars": numeric_stats([float(audit["chosen_chars"]) for audit in audits]),
        "rejected_chars": numeric_stats([float(audit["rejected_chars"]) for audit in audits]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter APPS Method 1 loop-v0 same-problem DPO pairs.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/preferences/apps_simple_method1_loop_v0_same_problem_only_dpo_pairs.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/preferences/apps_simple_method1_loop_v0_same_problem_balanced_dpo_pairs.jsonl"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("data/preferences/apps_simple_method1_loop_v0_same_problem_balanced_dpo_pairs_summary.json"),
    )
    parser.add_argument(
        "--pair-version",
        default="apps_simple_method1_loop_v0_same_problem_balanced_dpo",
    )
    parser.add_argument(
        "--required-preference-source",
        default="external_verifier_pass_over_same_problem_repair_fail",
    )
    parser.add_argument("--exclude-rejected-failure-type", action="append", default=[])
    parser.add_argument("--max-completion-chars", type=int, default=6000)
    parser.add_argument("--max-char-ratio", type=float, default=3.0)
    parser.add_argument("--max-whitespace-token-ratio", type=float, default=3.0)
    parser.add_argument("--max-line-ratio", type=float, default=8.0)
    parser.add_argument("--max-pairs-per-problem", type=int, default=2)
    parser.add_argument("--max-pairs-per-rejected-failure-type", type=int, default=40)
    parser.add_argument("--max-pairs", type=int, default=160)
    parser.add_argument("--allow-unparseable-chosen", action="store_true")
    args = parser.parse_args()

    if args.max_pairs_per_problem < 1:
        raise ValueError("--max-pairs-per-problem must be >= 1")
    if args.max_pairs_per_rejected_failure_type < 0:
        raise ValueError("--max-pairs-per-rejected-failure-type must be >= 0")
    if args.max_pairs < 0:
        raise ValueError("--max-pairs must be >= 0")

    rows = read_jsonl(args.input)
    candidates, skipped = build_candidates(args, rows)
    selected = select_pairs(args, candidates)
    output_rows = [output_pair(row, index, args.pair_version) for index, row in enumerate(selected, start=1)]
    write_jsonl(args.output, output_rows)

    summary = {
        "input": str(args.input),
        "input_sha256": sha256_file(args.input),
        "output": str(args.output),
        "output_sha256": sha256_file(args.output),
        "input_pair_count": len(rows),
        "candidate_pair_count": len(candidates),
        "selected_pair_count": len(output_rows),
        "input_unique_problem_count": len({str(row.get("id") or "") for row in rows if row.get("id")}),
        "candidate_unique_problem_count": len({str(row.get("id") or "") for row in candidates if row.get("id")}),
        "selected_unique_problem_count": len({str(row.get("id") or "") for row in output_rows if row.get("id")}),
        "input_rejected_failure_counts": dict(Counter(failure_type(row) for row in rows)),
        "candidate_rejected_failure_counts": dict(Counter(failure_type(row) for row in candidates)),
        "selected_rejected_failure_counts": dict(Counter(failure_type(row) for row in output_rows)),
        "skipped_counts": dict(skipped),
        "candidate_length_stats": summarize_lengths(candidates),
        "selected_length_stats": summarize_lengths(selected),
        "policy": {
            "role": "stable Method 1 same-problem DPO canary data",
            "kept_preference_source": args.required_preference_source,
            "chosen_must_be_parseable": not args.allow_unparseable_chosen,
            "max_completion_chars": args.max_completion_chars,
            "max_char_ratio": args.max_char_ratio,
            "max_whitespace_token_ratio": args.max_whitespace_token_ratio,
            "max_line_ratio": args.max_line_ratio,
            "max_pairs_per_problem": args.max_pairs_per_problem,
            "max_pairs_per_rejected_failure_type": args.max_pairs_per_rejected_failure_type,
            "max_pairs": args.max_pairs,
        },
    }
    write_json(args.summary_output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
