#!/usr/bin/env python3
"""Build Method 1 evaluator seed labels and training rows.

The output intentionally separates reliable supervision sources:

- verifier pass/fail for binary correctness;
- deterministic structural labels for syntax/runtime/timeout/truncation/interface;
- human seed labels for fine-grained error attribution.

Provisional logic labels are copied as metadata only. They are not emitted as
hard training targets because the pilot and round-2 audits found them noisy.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.annotation.build_error_annotation_review_set import (
    PILOT_LABELS,
    extract_task,
    response_id,
    verifier_summary,
)


DEFAULT_ANNOTATION_DIRS = [
    Path("data/annotation/apps_simple_error_review_pilot_v1"),
    Path("data/annotation/apps_simple_error_review_round2_v1"),
]

PRIVATE_OR_LEAKY_KEYS = {
    "canonical_solution",
    "canonical_solutions",
    "canonical_verifier",
    "input_output",
    "private_diagnostics",
    "test",
    "test_list",
    "test_setup_code",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def can_parse(code: str) -> bool:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            ast.parse(code or "")
        return True
    except SyntaxError:
        return False


def interface_names_from_signatures(signatures: list[Any]) -> set[str]:
    names: set[str] = set()
    for signature in signatures or []:
        match = re.search(r"\b(?:def|class)\s+([A-Za-z_][A-Za-z0-9_]*)\b", str(signature))
        if match:
            names.add(match.group(1))
    return names


def required_interface_names(row: dict[str, Any]) -> set[str]:
    names = {str(name) for name in row.get("interface_names") or [] if str(name).strip()}
    names |= interface_names_from_signatures(row.get("interface_signatures") or [])
    return names


def has_required_interface(row: dict[str, Any]) -> bool:
    names = required_interface_names(row)
    if not names:
        return True
    code = str(row.get("extracted_code") or row.get("generated_code") or "")
    return all(re.search(rf"\b(?:def|class)\s+{re.escape(name)}\b", code) for name in names)


def problem_hash_key(problem_id: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}:{problem_id}".encode("utf-8")).hexdigest()


def build_problem_splits(rows: list[dict[str, Any]], train_ratio: float, validation_ratio: float, salt: str) -> dict[str, str]:
    problem_ids = sorted({str(row.get("id")) for row in rows if row.get("id")})
    ordered = sorted(problem_ids, key=lambda item: problem_hash_key(item, salt))
    total = len(ordered)
    train_count = round(total * train_ratio)
    validation_count = round(total * validation_ratio)
    splits: dict[str, str] = {}
    for index, problem_id in enumerate(ordered):
        if index < train_count:
            split = "train"
        elif index < train_count + validation_count:
            split = "validation"
        else:
            split = "test"
        splits[problem_id] = split
    return splits


def annotation_key(row: dict[str, Any]) -> str:
    return str(row.get("response_id") or row.get("item_id") or row.get("id"))


def load_human_seed(annotation_dirs: list[Path]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    seed_rows: list[dict[str, Any]] = []
    for annotation_dir in annotation_dirs:
        review_items = {str(row.get("response_id")): row for row in read_jsonl(annotation_dir / "review_items.jsonl")}
        annotations = read_jsonl(annotation_dir / "annotations_working.jsonl")
        round_name = annotation_dir.name
        for annotation in annotations:
            rid = annotation_key(annotation)
            human_label = annotation.get("human_primary_label")
            if human_label not in PILOT_LABELS:
                continue
            item = review_items.get(rid, {})
            seed_rows.append(
                {
                    "response_id": rid,
                    "id": item.get("id"),
                    "dataset": item.get("dataset"),
                    "source_split": item.get("split"),
                    "sample_id": item.get("sample_id"),
                    "difficulty": item.get("difficulty"),
                    "io_mode": item.get("io_mode"),
                    "failure_type": item.get("failure_type"),
                    "diagnostic_kind": (item.get("verifier_summary") or {}).get("diagnostic_kind"),
                    "error_pattern": item.get("error_pattern"),
                    "annotation_round": round_name,
                    "sampling_reason": item.get("sampling_reason"),
                    "current_label": item.get("current_label"),
                    "current_label_quality": annotation.get("current_label_quality"),
                    "provisional_label": item.get("provisional_label") or annotation.get("provisional_label"),
                    "provisional_confidence": item.get("provisional_confidence") or annotation.get("provisional_confidence"),
                    "human_primary_label": human_label,
                    "human_secondary_label": annotation.get("human_secondary_label"),
                    "confidence": annotation.get("confidence"),
                    "evidence": annotation.get("evidence") or "",
                    "notes": annotation.get("notes") or "",
                }
            )

    by_response: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    for row in seed_rows:
        rid = str(row["response_id"])
        if rid in by_response:
            duplicates.append(rid)
            previous = by_response[rid]
            if previous.get("human_primary_label") != row.get("human_primary_label"):
                previous["label_conflict"] = True
        by_response[rid] = row
    if duplicates:
        duplicate_counts = Counter(duplicates)
        raise ValueError(f"duplicate human annotations found: {dict(duplicate_counts)}")
    return seed_rows, by_response


def deterministic_error_label(row: dict[str, Any]) -> tuple[str | None, str | None, str | None]:
    if bool(row.get("passed")):
        return None, None, None

    finish_reason = str(row.get("finish_reason") or "")
    failure_type = str(row.get("failure_type") or "")
    code = str(row.get("extracted_code") or row.get("generated_code") or "")

    if finish_reason == "length":
        return "truncation_or_overgeneration", "deterministic_finish_reason", "high"
    if failure_type == "generation_failure":
        return "unclear_other_or_not_failure", "deterministic_generation_failure", "medium"
    if failure_type == "syntax_error" or not can_parse(code):
        return "syntax_or_parse_error", "deterministic_parse_or_syntax", "high"
    if not has_required_interface(row):
        return "interface_contract_error", "deterministic_interface_missing", "high"
    if failure_type == "runtime_error":
        return "runtime_exception_or_timeout", "deterministic_runtime_error", "high"
    if failure_type == "timeout":
        return "runtime_exception_or_timeout", "deterministic_timeout", "medium"
    return None, None, None


def safe_public_row(
    row: dict[str, Any],
    eval_split: str,
    human_seed: dict[str, Any] | None,
    provisional: dict[str, Any] | None,
) -> dict[str, Any]:
    deterministic_label, deterministic_source, deterministic_confidence = deterministic_error_label(row)
    hard_label = deterministic_label
    hard_source = deterministic_source
    hard_confidence = deterministic_confidence

    if human_seed:
        hard_label = human_seed.get("human_primary_label")
        hard_source = "human_seed"
        hard_confidence = human_seed.get("confidence")

    record = {
        "response_id": response_id(row),
        "id": row.get("id"),
        "dataset": row.get("dataset"),
        "source_split": row.get("split"),
        "split": eval_split,
        "eval_split": eval_split,
        "prompt_mode": row.get("prompt_mode"),
        "difficulty": row.get("difficulty"),
        "io_mode": row.get("io_mode"),
        "sample_id": row.get("sample_id", 0),
        "model": row.get("model"),
        "temperature": row.get("temperature"),
        "top_p": row.get("top_p"),
        "max_tokens": row.get("max_tokens"),
        "finish_reason": row.get("finish_reason"),
        "generated_token_count": row.get("generated_token_count"),
        "prompt": row.get("prompt"),
        "task": extract_task(row.get("prompt")),
        "public_interface": row.get("interface_signatures") or row.get("interface_names") or [],
        "generated_code": row.get("generated_code"),
        "extracted_code": row.get("extracted_code") or row.get("generated_code"),
        "correctness_label": "pass" if bool(row.get("passed")) else "fail",
        "passed": bool(row.get("passed")),
        "failure_type": row.get("failure_type"),
        "verifier_summary": verifier_summary(row, include_raw_error=False),
        "deterministic_error_label": deterministic_label,
        "deterministic_label_source": deterministic_source,
        "deterministic_label_confidence": deterministic_confidence,
        "human_error_label": human_seed.get("human_primary_label") if human_seed else None,
        "human_error_confidence": human_seed.get("confidence") if human_seed else None,
        "human_evidence": human_seed.get("evidence") if human_seed else None,
        "error_attribution_label": hard_label,
        "error_attribution_source": hard_source,
        "error_attribution_confidence": hard_confidence,
        "error_attribution_trainable": bool(hard_label and hard_source),
        "provisional_label_reference": provisional.get("provisional_label") if provisional else None,
        "provisional_confidence_reference": provisional.get("provisional_confidence") if provisional else None,
        "provisional_method_reference": provisional.get("provisional_method") if provisional else None,
    }
    return {key: value for key, value in record.items() if key not in PRIVATE_OR_LEAKY_KEYS}


def split_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped = {"train": [], "validation": [], "test": []}
    for row in rows:
        grouped.setdefault(str(row.get("eval_split")), []).append(row)
    return grouped


def build_summary(
    training_rows: list[dict[str, Any]],
    human_seed_rows: list[dict[str, Any]],
    problem_splits: dict[str, str],
) -> dict[str, Any]:
    failed_rows = [row for row in training_rows if row["correctness_label"] == "fail"]
    trainable_attr_rows = [row for row in training_rows if row.get("error_attribution_trainable")]
    logic_failures = [row for row in failed_rows if row.get("failure_type") == "logic_error"]
    logic_with_human = [row for row in logic_failures if row.get("error_attribution_source") == "human_seed"]
    structural_rows = [
        row for row in trainable_attr_rows
        if str(row.get("error_attribution_source") or "").startswith("deterministic_")
    ]
    human_eval_split_counts = Counter()
    for row in training_rows:
        if row.get("human_error_label"):
            human_eval_split_counts[str(row.get("eval_split"))] += 1

    return {
        "total_rows": len(training_rows),
        "total_problems": len(problem_splits),
        "problem_split_counts": dict(Counter(problem_splits.values())),
        "row_split_counts": dict(Counter(str(row.get("eval_split")) for row in training_rows)),
        "correctness_counts": dict(Counter(row["correctness_label"] for row in training_rows)),
        "failure_type_counts": dict(Counter(str(row.get("failure_type")) for row in failed_rows)),
        "finish_reason_counts": dict(Counter(str(row.get("finish_reason")) for row in training_rows)),
        "human_seed_count": len(human_seed_rows),
        "human_seed_label_counts": dict(Counter(row["human_primary_label"] for row in human_seed_rows)),
        "human_seed_confidence_counts": dict(Counter(str(row.get("confidence")) for row in human_seed_rows)),
        "human_seed_eval_split_counts": dict(human_eval_split_counts),
        "error_attribution_trainable_rows": len(trainable_attr_rows),
        "error_attribution_label_counts": dict(Counter(str(row.get("error_attribution_label")) for row in trainable_attr_rows)),
        "error_attribution_source_counts": dict(Counter(str(row.get("error_attribution_source")) for row in trainable_attr_rows)),
        "deterministic_structural_rows": len(structural_rows),
        "logic_failure_rows": len(logic_failures),
        "logic_failure_rows_with_human_seed": len(logic_with_human),
        "logic_failure_rows_without_hard_attribution": len(logic_failures) - len(logic_with_human),
        "policy": {
            "binary_correctness_target": "all rows use verifier pass/fail",
            "hard_error_attribution": "human seed labels plus deterministic structural labels only",
            "provisional_logic_labels": "metadata only; excluded from hard training targets",
            "split_unit": "problem id",
            "private_fields_excluded": sorted(PRIVATE_OR_LEAKY_KEYS),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Method 1 evaluator seed labels and training rows.")
    parser.add_argument("--labeled", type=Path, default=Path("data/responses/apps_train_simple_executable_qwen25_k1_t2048_full_labeled.jsonl"))
    parser.add_argument("--provisional", type=Path, default=Path("data/annotation/apps_simple_provisional_coarse_labels_v1.jsonl"))
    parser.add_argument("--annotation-dir", action="append", type=Path, dest="annotation_dirs")
    parser.add_argument("--human-seed-output", type=Path, default=Path("data/annotation/apps_simple_human_seed_labels_v1.jsonl"))
    parser.add_argument("--human-seed-summary-output", type=Path, default=Path("data/annotation/apps_simple_human_seed_labels_v1_summary.json"))
    parser.add_argument("--training-output", type=Path, default=Path("data/evaluator/apps_simple_method1_evaluator_training_rows_v1.jsonl"))
    parser.add_argument("--split-output-dir", type=Path, default=Path("data/evaluator/apps_simple_method1_evaluator_training_rows_v1"))
    parser.add_argument("--summary-output", type=Path, default=Path("data/evaluator/apps_simple_method1_evaluator_training_rows_v1_summary.json"))
    parser.add_argument("--train-ratio", type=float, default=0.80)
    parser.add_argument("--validation-ratio", type=float, default=0.10)
    parser.add_argument("--split-salt", default="apps_simple_method1_evaluator_v1")
    args = parser.parse_args()

    if args.train_ratio <= 0 or args.validation_ratio < 0 or args.train_ratio + args.validation_ratio >= 1:
        raise SystemExit("--train-ratio and --validation-ratio must leave a non-empty test split")

    annotation_dirs = args.annotation_dirs or DEFAULT_ANNOTATION_DIRS
    labeled_rows = read_jsonl(args.labeled)
    human_seed_rows, human_by_response = load_human_seed(annotation_dirs)
    provisional_by_response = {str(row.get("response_id")): row for row in read_jsonl(args.provisional)}
    problem_splits = build_problem_splits(labeled_rows, args.train_ratio, args.validation_ratio, args.split_salt)

    enriched_seed_rows = []
    for row in human_seed_rows:
        problem_id = str(row.get("id") or "")
        enriched_seed_rows.append({**row, "eval_split": problem_splits.get(problem_id)})

    training_rows = []
    for row in sorted(labeled_rows, key=lambda item: response_id(item)):
        rid = response_id(row)
        problem_id = str(row.get("id"))
        training_rows.append(
            safe_public_row(
                row=row,
                eval_split=problem_splits[problem_id],
                human_seed=human_by_response.get(rid),
                provisional=provisional_by_response.get(rid),
            )
        )

    summary = build_summary(training_rows, enriched_seed_rows, problem_splits)
    human_summary = {
        "human_seed_count": len(enriched_seed_rows),
        "label_counts": dict(Counter(row["human_primary_label"] for row in enriched_seed_rows)),
        "confidence_counts": dict(Counter(str(row.get("confidence")) for row in enriched_seed_rows)),
        "annotation_round_counts": dict(Counter(str(row.get("annotation_round")) for row in enriched_seed_rows)),
        "eval_split_counts": dict(Counter(str(row.get("eval_split")) for row in enriched_seed_rows)),
        "label_options": PILOT_LABELS,
    }

    write_jsonl(args.human_seed_output, enriched_seed_rows)
    write_json(args.human_seed_summary_output, human_summary)
    write_jsonl(args.training_output, training_rows)
    for split_name, rows in split_rows(training_rows).items():
        write_jsonl(args.split_output_dir / f"{split_name}.jsonl", rows)
    write_json(args.summary_output, summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
