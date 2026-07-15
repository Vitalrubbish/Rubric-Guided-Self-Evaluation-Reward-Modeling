#!/usr/bin/env python3
"""Route APPS base/adapter responses with verifier-free protected rubric rules."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from src.training.build_apps_dpo_v2_preferences import code_audit
from src.verification.verify_mbpp_smoke import extract_code


FORBIDDEN_LABEL_FIELDS = {
    "passed",
    "failure_type",
    "private_diagnostics",
    "safe_diagnostics",
    "paired_verification_variant",
    "paired_verification_run_id",
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


def assert_unlabeled(rows: list[dict[str, Any]], label: str) -> None:
    for index, row in enumerate(rows):
        leaked = FORBIDDEN_LABEL_FIELDS.intersection(row)
        if leaked:
            raise AssertionError(
                f"{label} row {index} contains verifier labels: {sorted(leaked)}"
            )


def protected_score(row: dict[str, Any]) -> dict[str, Any]:
    code = extract_code(str(row.get("generated_code") or "")).strip()
    if not code:
        return {
            "score": 1,
            "fatal_reason": "empty_code",
            "parseable": False,
            "required_interface_present": False,
        }

    audit = code_audit(code, row.get("interface_names") or [])
    if not audit["parseable"]:
        return {
            "score": 1,
            "fatal_reason": "syntax_error",
            "parseable": False,
            "required_interface_present": False,
        }
    if not audit["required_interface_present"]:
        return {
            "score": 2,
            "fatal_reason": "required_interface_missing",
            "parseable": True,
            "required_interface_present": False,
        }
    return {
        "score": 5,
        "fatal_reason": None,
        "parseable": True,
        "required_interface_present": True,
    }


def index_rows(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    indexed = {str(row.get("id") or ""): row for row in rows}
    if "" in indexed or len(indexed) != len(rows):
        raise AssertionError(f"{label} has missing or duplicate problem IDs")
    return indexed


def route_rows(
    base_rows: list[dict[str, Any]], candidate_rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    assert_unlabeled(base_rows, "base")
    assert_unlabeled(candidate_rows, "candidate")
    base_by_id = index_rows(base_rows, "base")
    candidate_by_id = index_rows(candidate_rows, "candidate")
    if set(base_by_id) != set(candidate_by_id):
        raise AssertionError("base and candidate problem IDs differ")

    output: list[dict[str, Any]] = []
    selection_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    selected_candidate_ids: list[str] = []
    for problem_id in sorted(base_by_id):
        base = base_by_id[problem_id]
        candidate = candidate_by_id[problem_id]
        if base.get("prompt") != candidate.get("prompt"):
            raise AssertionError(f"prompt mismatch for {problem_id}")

        base_rubric = protected_score(base)
        candidate_rubric = protected_score(candidate)
        use_candidate = candidate_rubric["score"] > base_rubric["score"]
        selected_variant = "candidate" if use_candidate else "base"
        selected = dict(candidate if use_candidate else base)
        if use_candidate:
            selected_candidate_ids.append(problem_id)
            reason = f"{base_rubric['fatal_reason']}->protected_valid"
        elif candidate_rubric["score"] < base_rubric["score"]:
            reason = "candidate_hard_regression_blocked"
        else:
            reason = "tie_preserves_base"

        selected["protected_rubric_route"] = {
            "version": "apps_protected_rubric_router_v1",
            "selected_variant": selected_variant,
            "selection_reason": reason,
            "base": base_rubric,
            "candidate": candidate_rubric,
            "policy": "candidate only when its protected score is strictly higher; ties preserve base",
        }
        output.append(selected)
        selection_counts[selected_variant] += 1
        reason_counts[reason] += 1

    audit = {
        "rows": len(output),
        "selection_counts": dict(selection_counts),
        "reason_counts": dict(reason_counts),
        "selected_candidate_ids": selected_candidate_ids,
        "forbidden_label_fields": sorted(FORBIDDEN_LABEL_FIELDS),
        "rubric": {
            "empty_or_syntax_invalid": 1,
            "required_interface_missing": 2,
            "parseable_with_required_interface": 5,
        },
        "policy": "verifier-free protected rubric; no execution result or reference answer is read",
    }
    return output, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-input", type=Path, required=True)
    parser.add_argument("--candidate-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, required=True)
    args = parser.parse_args()

    base_rows = read_jsonl(args.base_input)
    candidate_rows = read_jsonl(args.candidate_input)
    if len(base_rows) != args.expected_rows or len(candidate_rows) != args.expected_rows:
        raise AssertionError(
            f"expected {args.expected_rows} rows, got {len(base_rows)}/{len(candidate_rows)}"
        )
    output, audit = route_rows(base_rows, candidate_rows)
    write_jsonl(args.output, output)
    manifest = {
        "status": "completed",
        "base_input": str(args.base_input),
        "base_input_sha256": sha256_file(args.base_input),
        "candidate_input": str(args.candidate_input),
        "candidate_input_sha256": sha256_file(args.candidate_input),
        "output": str(args.output),
        "output_sha256": sha256_file(args.output),
        **audit,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
