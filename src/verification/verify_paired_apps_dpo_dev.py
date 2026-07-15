#!/usr/bin/env python3
"""Verify base and candidate APPS DPO-dev outputs with a shared code cache."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from src.verification.verify_mbpp_smoke import evaluate_one, extract_code, read_jsonl


RESULT_FIELDS = (
    "extracted_code",
    "passed",
    "failure_type",
    "error",
    "source_mode",
    "safe_diagnostics",
    "private_diagnostics",
)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_key(row: dict[str, Any]) -> str:
    problem_id = str(row.get("id"))
    return f"{problem_id}:{sha256_text(extract_code(str(row.get('generated_code') or '')))}"


def index_rows(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    indexed = {str(row.get("id")): row for row in rows}
    if len(indexed) != len(rows):
        raise AssertionError(f"{label} contains duplicate problem IDs")
    return indexed


def build_unique_rows(
    base_rows: list[dict[str, Any]], candidate_rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], int]:
    base_by_id = index_rows(base_rows, "base")
    candidate_by_id = index_rows(candidate_rows, "candidate")
    if set(base_by_id) != set(candidate_by_id):
        raise AssertionError("base and candidate problem IDs differ")

    unique: dict[str, dict[str, Any]] = {}
    identical = 0
    for problem_id in sorted(base_by_id):
        if code_key(base_by_id[problem_id]) == code_key(candidate_by_id[problem_id]):
            identical += 1
        for variant, row in (("base", base_by_id[problem_id]), ("candidate", candidate_by_id[problem_id])):
            key = code_key(row)
            if key not in unique:
                representative = dict(row)
                representative["paired_code_key"] = key
                representative["response_id"] = f"{problem_id}__paired_{key.rsplit(':', 1)[-1][:16]}"
                representative["paired_first_variant"] = variant
                unique[key] = representative
    return [unique[key] for key in sorted(unique)], identical


def verify_unique_rows(
    rows: list[dict[str, Any]], timeout: float, workers: int, start_method: str
) -> dict[str, dict[str, Any]]:
    if workers <= 1:
        results = [evaluate_one(row, timeout, start_method) for row in rows]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(evaluate_one, row, timeout, start_method) for row in rows]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
    indexed = {str(row["paired_code_key"]): row for row in results}
    if len(indexed) != len(rows):
        raise AssertionError("paired verifier lost or duplicated a unique code result")
    return indexed


def materialize_rows(
    rows: list[dict[str, Any]],
    results: dict[str, dict[str, Any]],
    *,
    variant: str,
    run_id: str,
    timeout: float,
    workers: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        key = code_key(row)
        result = results[key]
        merged = dict(row)
        for field in RESULT_FIELDS:
            merged[field] = result.get(field)
        merged["extracted_code"] = extract_code(str(row.get("generated_code") or ""))
        merged["paired_code_sha256"] = key.rsplit(":", 1)[-1]
        merged["paired_verification_variant"] = variant
        merged["paired_verification_run_id"] = run_id
        merged["paired_verification_timeout"] = timeout
        merged["paired_verification_workers"] = workers
        output.append(merged)
    return output


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def outcome_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter("passed" if row.get("passed") else str(row.get("failure_type")) for row in rows))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify base/candidate APPS DPO-dev outputs once per unique (problem, code) pair."
    )
    parser.add_argument("--base-input", type=Path, required=True)
    parser.add_argument("--candidate-input", type=Path, required=True)
    parser.add_argument("--base-output", type=Path, required=True)
    parser.add_argument("--candidate-output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=160)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--process-start-method", choices=("spawn", "fork", "forkserver"), default="spawn"
    )
    args = parser.parse_args()

    base_rows = list(read_jsonl(args.base_input))
    candidate_rows = list(read_jsonl(args.candidate_input))
    if len(base_rows) != args.expected_rows or len(candidate_rows) != args.expected_rows:
        raise AssertionError(
            f"expected {args.expected_rows} base/candidate rows, got {len(base_rows)}/{len(candidate_rows)}"
        )

    unique_rows, identical_count = build_unique_rows(base_rows, candidate_rows)
    run_payload = {
        "base_input_sha256": sha256_file(args.base_input),
        "candidate_input_sha256": sha256_file(args.candidate_input),
        "timeout": args.timeout,
        "workers": args.workers,
        "process_start_method": args.process_start_method,
    }
    run_id = sha256_text(json.dumps(run_payload, sort_keys=True, separators=(",", ":")))
    results = verify_unique_rows(unique_rows, args.timeout, args.workers, args.process_start_method)
    base_output = materialize_rows(
        base_rows,
        results,
        variant="base",
        run_id=run_id,
        timeout=args.timeout,
        workers=args.workers,
    )
    candidate_output = materialize_rows(
        candidate_rows,
        results,
        variant="candidate",
        run_id=run_id,
        timeout=args.timeout,
        workers=args.workers,
    )
    write_jsonl(args.base_output, base_output)
    write_jsonl(args.candidate_output, candidate_output)

    manifest = {
        "status": "completed",
        "run_id": run_id,
        **run_payload,
        "expected_rows_per_variant": args.expected_rows,
        "requested_evaluations": len(base_rows) + len(candidate_rows),
        "unique_problem_code_evaluations": len(unique_rows),
        "reused_evaluations": len(base_rows) + len(candidate_rows) - len(unique_rows),
        "identical_base_candidate_codes": identical_count,
        "base_outcomes": outcome_counts(base_output),
        "candidate_outcomes": outcome_counts(candidate_output),
        "base_output": str(args.base_output),
        "base_output_sha256": sha256_file(args.base_output),
        "candidate_output": str(args.candidate_output),
        "candidate_output_sha256": sha256_file(args.candidate_output),
        "policy": "one execution per unique (problem_id, extracted_code); shared result for identical outputs",
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
