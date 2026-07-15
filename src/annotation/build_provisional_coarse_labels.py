#!/usr/bin/env python3
"""Build provisional coarse labels and a targeted second annotation round.

This script deliberately avoids treating the original taxonomy assignment as
ground truth. It uses verifier-observed structural failure modes first, then
weak logic heuristics for the remaining wrong-output rows. The output is a
provisional routing layer, not a training gold label.
"""

from __future__ import annotations

import argparse
import ast
import json
import random
import re
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.annotation.build_error_annotation_review_set import (
    PILOT_LABELS,
    extract_task,
    markdown_code_block,
    render_html_index,
    render_packet,
    render_schema,
    response_id,
    short_text,
    verifier_summary,
    write_json,
    write_jsonl,
)


BAD_CURRENT_LABELS = {
    "interface_name_signature_mismatch",
    "edge_case_handling",
    "runtime_api_type_misuse",
}

CURRENT_TO_COARSE = {
    "syntax_parseability_truncation": "syntax_or_parse_error",
    "interface_name_signature_mismatch": "interface_contract_error",
    "runtime_api_type_misuse": "runtime_exception_or_timeout",
    "output_type_or_container_shape": "output_format_or_type_error",
    "numeric_formula_arithmetic_error": "numeric_formula_or_counting_error",
    "sequence_collection_transformation_error": "sequence_or_state_transformation_error",
    "predicate_branch_condition_error": "predicate_condition_or_edge_case_error",
    "string_regex_pattern_logic": "string_pattern_or_text_error",
    "edge_case_handling": "predicate_condition_or_edge_case_error",
}

STABLE_LOGIC_CURRENT_LABELS = {
    "sequence_collection_transformation_error",
    "predicate_branch_condition_error",
    "string_regex_pattern_logic",
}

KEYWORDS = {
    "string_pattern_or_text_error": {
        "regex", "regular expression", "string", "text", "character", "char",
        "word", "sentence", "substring", "case", "uppercase", "lowercase",
        "replace", "split", "join", "parse", "pattern", "match", "token",
    },
    "numeric_formula_or_counting_error": {
        "formula", "calculate", "calculation", "count", "counting", "sum",
        "average", "mean", "median", "number", "numeric", "integer", "float",
        "round", "rounding", "cost", "score", "distance", "seconds", "hours",
        "conversion", "rate", "prime", "square", "modulo", "multipl", "divide",
        "off-by-one", "off by one",
    },
    "sequence_or_state_transformation_error": {
        "list", "array", "sequence", "matrix", "permutation", "element",
        "duplicate", "order", "sort", "append", "remove", "filter", "flatten",
        "nested", "index", "indices", "state", "stack", "queue", "set",
        "dictionary", "dict", "mapping", "group", "row", "column",
    },
    "predicate_condition_or_edge_case_error": {
        "condition", "predicate", "branch", "if", "else", "true", "false",
        "boolean", "compare", "comparison", "tie", "empty", "boundary",
        "edge", "all", "any", "minimum", "maximum", "less", "greater",
        "accept", "reject", "valid", "invalid",
    },
    "output_format_or_type_error": {
        "return type", "output format", "container", "shape", "list instead",
        "string instead", "tuple", "dict", "dictionary", "print", "return",
        "format", "stdout", "newline",
    },
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def can_parse(code: str) -> bool:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            ast.parse(code or "")
        return True
    except SyntaxError:
        return False


def interface_names_from_signatures(signatures: list[Any]) -> set[str]:
    names = set()
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


def keyword_scores(text: str) -> Counter:
    normalized = text.lower()
    scores: Counter = Counter()
    for label, terms in KEYWORDS.items():
        for term in terms:
            if term in normalized:
                scores[label] += 1
    return scores


def choose_logic_label(row: dict[str, Any], assignment: dict[str, Any]) -> tuple[str, str, str, list[str]]:
    current = str(assignment.get("taxonomy_category_id") or "")
    summary = str(assignment.get("llm_summary") or "")
    task_head = extract_task(row.get("prompt"))[:800]
    function_names = " ".join(required_interface_names(row))
    text = " ".join([summary, current, function_names, task_head])
    scores = keyword_scores(text)
    reasons: list[str] = []

    current_coarse = CURRENT_TO_COARSE.get(current)
    if current_coarse:
        scores[current_coarse] += 2
        reasons.append(f"current_label_prior={current}->{current_coarse}")

    if not scores:
        return (
            "unclear_other_or_not_failure",
            "low",
            "logic_heuristic_no_signal",
            ["no coarse keyword or reliable current-label signal"],
        )

    ranked = scores.most_common()
    label, top_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0
    margin = top_score - second_score

    if current in STABLE_LOGIC_CURRENT_LABELS and current_coarse == label and top_score >= 2:
        confidence = "high"
    elif top_score >= 4 and margin >= 2:
        confidence = "high"
    elif top_score >= 2 and margin >= 1:
        confidence = "medium"
    else:
        confidence = "low"
    reasons.append(f"keyword_scores={dict(scores)}")
    return label, confidence, "logic_weak_heuristic", reasons


def provisional_label(row: dict[str, Any], assignment: dict[str, Any]) -> dict[str, Any]:
    failure_type = str(row.get("failure_type") or "")
    current = str(assignment.get("taxonomy_category_id") or "")
    code = str(row.get("extracted_code") or row.get("generated_code") or "")
    reasons: list[str] = []

    if row.get("finish_reason") == "length":
        label, confidence, method = "truncation_or_overgeneration", "high", "deterministic_finish_reason"
        reasons.append("finish_reason=length")
    elif failure_type == "generation_failure":
        label, confidence, method = "unclear_other_or_not_failure", "high", "deterministic_generation_failure"
        reasons.append("generation failure is not a code mechanism label")
    elif failure_type == "syntax_error" or not can_parse(code):
        label, confidence, method = "syntax_or_parse_error", "high", "deterministic_parse_or_syntax"
        reasons.append(f"failure_type={failure_type}; ast_parse_ok={can_parse(code)}")
    elif not has_required_interface(row):
        label, confidence, method = "interface_contract_error", "high", "deterministic_interface_missing"
        reasons.append(f"required_interface={sorted(required_interface_names(row))}")
    elif failure_type == "runtime_error":
        label, confidence, method = "runtime_exception_or_timeout", "high", "deterministic_runtime_error"
        reasons.append("verifier classified row as runtime_error")
    elif failure_type == "timeout":
        label, confidence, method = "runtime_exception_or_timeout", "medium", "deterministic_timeout"
        reasons.append("verifier classified row as timeout; timeout mechanism still needs spot-checking")
    elif failure_type == "logic_error":
        label, confidence, method, reasons = choose_logic_label(row, assignment)
    else:
        label, confidence, method = "unclear_other_or_not_failure", "low", "unknown_failure_type"
        reasons.append(f"unrecognized failure_type={failure_type}")

    review_reasons = []
    if confidence != "high":
        review_reasons.append("non_high_confidence")
    if current in BAD_CURRENT_LABELS:
        review_reasons.append(f"bad_current_label={current}")
    if failure_type == "timeout":
        review_reasons.append("timeout_needs_mechanism_check")
    if row.get("io_mode") == "stdin_stdout":
        review_reasons.append("stdin_stdout_coverage")
    if label == "unclear_other_or_not_failure":
        review_reasons.append("unclear_or_other")

    return {
        "response_id": response_id(row),
        "id": row.get("id"),
        "dataset": row.get("dataset"),
        "split": row.get("split"),
        "io_mode": row.get("io_mode"),
        "failure_type": failure_type,
        "error_pattern": assignment.get("error_pattern"),
        "current_label": current,
        "llm_summary": assignment.get("llm_summary"),
        "provisional_label": label,
        "provisional_confidence": confidence,
        "provisional_method": method,
        "provisional_reason": "; ".join(reasons),
        "review_priority": bool(review_reasons),
        "review_reasons": review_reasons,
    }


def build_review_item(
    index: int,
    row: dict[str, Any],
    assignment: dict[str, Any],
    provisional: dict[str, Any],
    sampling_reason: str,
    task_chars: int,
    code_chars: int,
) -> dict[str, Any]:
    return {
        "review_index": index,
        "response_id": response_id(row),
        "id": row.get("id"),
        "sample_id": row.get("sample_id", 0),
        "dataset": row.get("dataset"),
        "split": row.get("split"),
        "difficulty": row.get("difficulty"),
        "io_mode": row.get("io_mode"),
        "failure_type": row.get("failure_type"),
        "error_pattern": assignment.get("error_pattern"),
        "current_label": assignment.get("taxonomy_category_id"),
        "current_label_name": assignment.get("taxonomy_category_name"),
        "current_rubric_dimension": assignment.get("rubric_dimension"),
        "llm_summary": assignment.get("llm_summary"),
        "provisional_label": provisional.get("provisional_label"),
        "provisional_confidence": provisional.get("provisional_confidence"),
        "provisional_reason": provisional.get("provisional_reason"),
        "review_reasons": provisional.get("review_reasons") or [],
        "verifier_summary": verifier_summary(row, include_raw_error=False),
        "sampling_reason": sampling_reason,
        "task": short_text(extract_task(row.get("prompt")), task_chars),
        "public_interface": row.get("interface_signatures") or row.get("interface_names") or [],
        "generated_code": short_text(row.get("generated_code"), code_chars),
        "extracted_code": short_text(row.get("extracted_code") or row.get("generated_code"), code_chars),
    }


def annotation_stub(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "response_id": item["response_id"],
        "current_label": item.get("current_label"),
        "provisional_label": item.get("provisional_label"),
        "provisional_confidence": item.get("provisional_confidence"),
        "human_primary_label": item.get("provisional_label"),
        "human_secondary_label": None,
        "confidence": item.get("provisional_confidence"),
        "evidence": "",
        "notes": "",
    }


def select_round2_items(
    provisional_rows: list[dict[str, Any]],
    reviewed_ids: set[str],
    size: int,
    seed: int,
) -> list[tuple[str, dict[str, Any]]]:
    rng = random.Random(seed)
    candidates = [row for row in provisional_rows if row["response_id"] not in reviewed_ids]
    buckets: list[tuple[str, int, list[dict[str, Any]]]] = [
        ("low_or_unclear", 12, [r for r in candidates if r["provisional_confidence"] == "low" or r["provisional_label"] == "unclear_other_or_not_failure"]),
        ("bad_current_label", 18, [r for r in candidates if any(str(reason).startswith("bad_current_label=") for reason in r.get("review_reasons", []))]),
        ("timeout_mechanism", 10, [r for r in candidates if "timeout_needs_mechanism_check" in r.get("review_reasons", [])]),
        ("stdin_stdout_coverage", 8, [r for r in candidates if "stdin_stdout_coverage" in r.get("review_reasons", [])]),
        ("medium_confidence_logic", 12, [r for r in candidates if r["provisional_confidence"] == "medium" and r["failure_type"] == "logic_error"]),
    ]

    selected: list[tuple[str, dict[str, Any]]] = []
    seen = set()
    for bucket_name, limit, rows in buckets:
        rows = sorted(rows, key=lambda row: row["response_id"])
        if len(rows) > limit:
            rows = sorted(rng.sample(rows, limit), key=lambda row: row["response_id"])
        for row in rows:
            if row["response_id"] in seen:
                continue
            selected.append((bucket_name, row))
            seen.add(row["response_id"])
            if len(selected) >= size:
                return selected

    remaining = [row for row in candidates if row["response_id"] not in seen]
    remaining = sorted(remaining, key=lambda row: row["response_id"])
    if len(selected) < size and remaining:
        needed = min(size - len(selected), len(remaining))
        for row in sorted(rng.sample(remaining, needed), key=lambda row: row["response_id"]):
            selected.append(("coverage_fill", row))
    return selected[:size]


def render_round2_packet(items: list[dict[str, Any]], labels: list[str]) -> str:
    return render_packet(items, labels)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build provisional coarse labels and a targeted second annotation round.")
    parser.add_argument("--labeled", type=Path, default=Path("data/responses/apps_train_simple_executable_qwen25_k1_t2048_full_labeled_nonlength.jsonl"))
    parser.add_argument("--assignments", type=Path, default=Path("data/analysis/apps_simple_phase1/apps_train_simple_qwen25_k1_t2048_taxonomy_refined_response_assignments.jsonl"))
    parser.add_argument("--pilot-annotations", type=Path, default=Path("data/annotation/apps_simple_error_review_pilot_v1/annotations_working.jsonl"))
    parser.add_argument("--provisional-output", type=Path, default=Path("data/annotation/apps_simple_provisional_coarse_labels_v1.jsonl"))
    parser.add_argument("--summary-output", type=Path, default=Path("data/annotation/apps_simple_provisional_coarse_labels_v1_summary.json"))
    parser.add_argument("--round2-output-dir", type=Path, default=Path("data/annotation/apps_simple_error_review_round2_v1"))
    parser.add_argument("--round2-size", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--task-chars", type=int, default=4500)
    parser.add_argument("--code-chars", type=int, default=3500)
    args = parser.parse_args()

    labeled_rows = {response_id(row): row for row in read_jsonl(args.labeled)}
    assignments = {str(row["response_id"]): row for row in read_jsonl(args.assignments)}
    reviewed_ids = {str(row.get("response_id")) for row in read_jsonl(args.pilot_annotations)}

    provisional_rows = []
    for rid, assignment in sorted(assignments.items()):
        row = labeled_rows.get(rid)
        if not row or bool(row.get("passed")):
            continue
        provisional_rows.append(provisional_label(row, assignment))

    write_jsonl(args.provisional_output, provisional_rows)

    round2_selected = select_round2_items(provisional_rows, reviewed_ids, args.round2_size, args.seed)
    round2_items = []
    for index, (sampling_reason, provisional) in enumerate(round2_selected, start=1):
        rid = provisional["response_id"]
        round2_items.append(
            build_review_item(
                index=index,
                row=labeled_rows[rid],
                assignment=assignments[rid],
                provisional=provisional,
                sampling_reason=sampling_reason,
                task_chars=args.task_chars,
                code_chars=args.code_chars,
            )
        )

    output_dir = args.round2_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "review_items.jsonl", round2_items)
    write_jsonl(output_dir / "annotation_template.jsonl", [annotation_stub(item) for item in round2_items])
    (output_dir / "review_packet.md").write_text(render_round2_packet(round2_items, PILOT_LABELS), encoding="utf-8")
    (output_dir / "annotation_schema.md").write_text(render_schema(PILOT_LABELS), encoding="utf-8")
    (output_dir / "review_index.html").write_text(render_html_index(round2_items), encoding="utf-8")

    summary = {
        "num_provisional_rows": len(provisional_rows),
        "provisional_label_counts": dict(Counter(row["provisional_label"] for row in provisional_rows)),
        "provisional_confidence_counts": dict(Counter(row["provisional_confidence"] for row in provisional_rows)),
        "provisional_method_counts": dict(Counter(row["provisional_method"] for row in provisional_rows)),
        "review_priority_count": sum(bool(row.get("review_priority")) for row in provisional_rows),
        "review_reason_counts": dict(Counter(reason for row in provisional_rows for reason in row.get("review_reasons", []))),
        "pilot_reviewed_ids": len(reviewed_ids),
        "round2_output_dir": str(output_dir),
        "round2_size": len(round2_items),
        "round2_sampling_counts": dict(Counter(item["sampling_reason"] for item in round2_items)),
        "round2_current_label_counts": dict(Counter(str(item.get("current_label")) for item in round2_items)),
        "round2_provisional_label_counts": dict(Counter(str(item.get("provisional_label")) for item in round2_items)),
        "label_options": PILOT_LABELS,
    }
    write_json(args.summary_output, summary)
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
