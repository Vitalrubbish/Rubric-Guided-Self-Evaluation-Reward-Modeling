#!/usr/bin/env python3
"""Build audited APPS DPO-v2 pairs from verifier-passing self-repairs."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


PRIVATE_KEYS = {
    "canonical_solutions",
    "canonical_verifier",
    "input_output",
    "private_diagnostics",
    "safe_diagnostics",
    "test",
    "test_list",
    "test_setup_code",
}


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


def extract_code(text: str) -> str:
    fenced = re.search(r"```(?:python)?\s*(.*?)```", text or "", flags=re.DOTALL | re.IGNORECASE)
    return fenced.group(1) if fenced else str(text or "")


def normalize_code(text: str) -> str:
    code = extract_code(text).replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in code.strip().splitlines()]
    if lines and re.fullmatch(r"\s*```(?:python)?\s*", lines[0], flags=re.IGNORECASE):
        lines.pop(0)
    if lines and re.fullmatch(r"\s*```\s*", lines[-1]):
        lines.pop()
    return "\n".join(lines).strip()


def _is_main_guard(node: ast.If) -> bool:
    test = node.test
    return (
        isinstance(test, ast.Compare)
        and isinstance(test.left, ast.Name)
        and test.left.id == "__name__"
        and len(test.comparators) == 1
        and isinstance(test.comparators[0], ast.Constant)
        and test.comparators[0].value == "__main__"
    )


def code_audit(code: str, interface_names: Iterable[str]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "parseable": False,
        "required_interface_present": False,
        "top_level_demo_count": 0,
        "defined_functions": [],
        "solution_methods": [],
    }
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        result["syntax_error"] = f"{exc.msg} at line {exc.lineno}"
        return result

    functions = {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    solution_methods: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Solution":
            solution_methods.update(
                child.name for child in node.body if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            )

    demos = 0
    for node in tree.body:
        if isinstance(node, ast.Assert):
            demos += 1
        elif isinstance(node, ast.If) and _is_main_guard(node):
            demos += 1
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            demos += 1

    required = {str(name) for name in interface_names if str(name)}
    result.update(
        {
            "parseable": True,
            "required_interface_present": not required or bool(required & (functions | solution_methods)),
            "top_level_demo_count": demos,
            "defined_functions": sorted(functions),
            "solution_methods": sorted(solution_methods),
        }
    )
    return result


def numeric_stats(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    if not ordered:
        return {"min": 0.0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}

    def percentile(fraction: float) -> float:
        return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * fraction))]

    return {
        "min": ordered[0],
        "mean": mean(ordered),
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "max": ordered[-1],
    }


def load_forbidden_ids(paths: Iterable[Path]) -> set[str]:
    forbidden: set[str] = set()
    for path in paths:
        for row in read_jsonl(path):
            problem_id = str(row.get("id") or "")
            if problem_id:
                forbidden.add(problem_id)
    return forbidden


def build_pairs(
    repair_rows: Iterable[dict[str, Any]],
    original_by_response: dict[str, dict[str, Any]],
    *,
    forbidden_ids: set[str] | None = None,
    max_length_ratio: float = 8.0,
    max_pairs_per_problem: int = 1,
) -> tuple[list[dict[str, Any]], Counter[str], dict[str, Any]]:
    forbidden_ids = forbidden_ids or set()
    pairs: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    per_problem: defaultdict[str, int] = defaultdict(int)
    length_ratios: list[float] = []
    zero_format_counts = {"chosen_fenced": 0, "rejected_fenced": 0}
    candidate_raw_format = Counter(zero_format_counts)
    accepted_raw_format = Counter(zero_format_counts)
    accepted_normalized_format = Counter(zero_format_counts)

    ordered_repairs = sorted(
        repair_rows,
        key=lambda row: (
            str(row.get("id") or ""),
            0 if "two_stage" in str(row.get("prompt_mode") or "") else 1,
            str(row.get("response_id") or ""),
        ),
    )
    for repair in ordered_repairs:
        if not bool(repair.get("passed")):
            skipped["repair_failed_verifier"] += 1
            continue

        problem_id = str(repair.get("id") or "")
        if not problem_id:
            skipped["missing_problem_id"] += 1
            continue
        if problem_id in forbidden_ids:
            skipped["forbidden_problem_id"] += 1
            continue
        if per_problem[problem_id] >= max_pairs_per_problem:
            skipped["max_pairs_per_problem"] += 1
            continue

        original_response_id = str(repair.get("original_response_id") or "")
        original = original_by_response.get(original_response_id)
        if original is None:
            skipped["missing_original"] += 1
            continue
        if str(original.get("id") or "") != problem_id:
            skipped["original_problem_mismatch"] += 1
            continue
        if original.get("split") != "train" or repair.get("split") != "train":
            skipped["non_train_split"] += 1
            continue
        if bool(original.get("passed")):
            skipped["original_not_failed"] += 1
            continue

        repair_model = str(repair.get("model") or "")
        original_model = str(original.get("model") or "")
        if repair_model and original_model and repair_model != original_model:
            skipped["model_mismatch"] += 1
            continue
        if repair.get("finish_reason") == "length":
            skipped["chosen_length_finish"] += 1
            continue

        chosen_raw = str(repair.get("extracted_code") or repair.get("generated_code") or "")
        rejected_raw = str(original.get("extracted_code") or original.get("generated_code") or "")
        chosen_raw_fenced = int("```" in str(repair.get("generated_code") or ""))
        rejected_raw_fenced = int("```" in str(original.get("generated_code") or ""))
        candidate_raw_format["chosen_fenced"] += chosen_raw_fenced
        candidate_raw_format["rejected_fenced"] += rejected_raw_fenced
        chosen = normalize_code(chosen_raw)
        rejected = normalize_code(rejected_raw)
        chosen_normalized_fenced = int("```" in chosen)
        rejected_normalized_fenced = int("```" in rejected)
        if chosen_normalized_fenced or rejected_normalized_fenced:
            skipped["normalized_fence_remains"] += 1
            continue
        if not chosen or not rejected:
            skipped["empty_normalized_code"] += 1
            continue
        if chosen == rejected:
            skipped["identical_normalized_code"] += 1
            continue

        interface_names = repair.get("interface_names") or []
        chosen_audit = code_audit(chosen, interface_names)
        rejected_audit = code_audit(rejected, interface_names)
        if not chosen_audit["parseable"]:
            skipped["chosen_not_parseable"] += 1
            continue
        if repair.get("io_mode") == "function_call" and not chosen_audit["required_interface_present"]:
            skipped["chosen_interface_missing"] += 1
            continue
        if repair.get("io_mode") == "function_call" and chosen_audit["top_level_demo_count"]:
            skipped["chosen_top_level_demo"] += 1
            continue

        ratio = max(len(chosen), len(rejected)) / max(1, min(len(chosen), len(rejected)))
        if ratio > max_length_ratio:
            skipped["length_ratio_too_large"] += 1
            continue

        prompt = str(original.get("prompt") or "").strip()
        if not prompt:
            skipped["missing_original_prompt"] += 1
            continue

        pair = {
            "pair_id": f"{problem_id}__self_repair_v2_{per_problem[problem_id] + 1}",
            "pair_version": "apps_simple_method1_dpo_v2",
            "id": problem_id,
            "dataset": "apps",
            "split": "train",
            "source_split": original.get("source_split", "train"),
            "difficulty": original.get("difficulty"),
            "io_mode": original.get("io_mode"),
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "chosen_source": "same_model_verifier_passing_repair",
            "rejected_source": "same_model_verifier_failed_original",
            "preference_source": "external_verifier_pass_over_fail",
            "repair_method": repair.get("prompt_mode"),
            "chosen_response_id": repair.get("response_id"),
            "rejected_response_id": original_response_id,
            "repair_candidate_id": repair.get("repair_candidate_id"),
            "original_failure_type": original.get("failure_type"),
            "critic_pass_probability": repair.get("critic_pass_probability"),
            "selection_reason": repair.get("selection_reason"),
            "chosen_parseable": True,
            "rejected_parseable": bool(rejected_audit["parseable"]),
            "chosen_required_interface_present": bool(chosen_audit["required_interface_present"]),
            "rejected_required_interface_present": bool(rejected_audit["required_interface_present"]),
            "completion_char_ratio": ratio,
        }
        leaked = sorted(PRIVATE_KEYS.intersection(pair))
        if leaked:
            raise AssertionError(f"private keys leaked into {pair['pair_id']}: {leaked}")
        pairs.append(pair)
        per_problem[problem_id] += 1
        length_ratios.append(ratio)
        accepted_raw_format["chosen_fenced"] += chosen_raw_fenced
        accepted_raw_format["rejected_fenced"] += rejected_raw_fenced
        accepted_normalized_format["chosen_fenced"] += chosen_normalized_fenced
        accepted_normalized_format["rejected_fenced"] += rejected_normalized_fenced

    audit = {
        "candidate_raw_format_counts": dict(candidate_raw_format),
        "raw_format_counts": dict(accepted_raw_format),
        "normalized_format_counts": dict(accepted_normalized_format),
        "completion_char_ratio": numeric_stats(length_ratios),
        "chosen_parseable_count": len(pairs),
        "rejected_parseable_count": sum(bool(row["rejected_parseable"]) for row in pairs),
        "unique_problem_count": len({str(row["id"]) for row in pairs}),
    }
    return pairs, skipped, audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Build strictly audited APPS DPO-v2 self-repair pairs.")
    parser.add_argument(
        "--repair-labeled",
        type=Path,
        action="append",
        default=None,
        help="May be repeated; two-stage repairs take priority per problem.",
    )
    parser.add_argument(
        "--evaluator-rows",
        type=Path,
        default=Path("data/evaluator/apps_simple_method1_evaluator_training_rows_v1.jsonl"),
    )
    parser.add_argument(
        "--forbidden-ids",
        type=Path,
        action="append",
        default=[],
        help="JSONL files whose problem IDs must not appear in DPO training.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/preferences/apps_simple_method1_self_repair_dpo_v2.jsonl"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("data/preferences/apps_simple_method1_self_repair_dpo_v2_summary.json"),
    )
    parser.add_argument("--max-length-ratio", type=float, default=8.0)
    parser.add_argument("--max-pairs-per-problem", type=int, default=1)
    args = parser.parse_args()

    evaluator_rows = read_jsonl(args.evaluator_rows)
    original_by_response = {str(row.get("response_id")): row for row in evaluator_rows}
    if len(original_by_response) != len(evaluator_rows):
        raise ValueError("evaluator response IDs are missing or duplicated")
    repair_paths = args.repair_labeled or [Path("data/repair/apps_simple_method1_repair_v1_labeled.jsonl")]
    repair_rows = [row for path in repair_paths for row in read_jsonl(path)]
    forbidden_ids = load_forbidden_ids(args.forbidden_ids)
    pairs, skipped, audit = build_pairs(
        repair_rows,
        original_by_response,
        forbidden_ids=forbidden_ids,
        max_length_ratio=args.max_length_ratio,
        max_pairs_per_problem=args.max_pairs_per_problem,
    )
    if not pairs:
        raise RuntimeError("no DPO-v2 preference pairs passed the audit")

    pair_ids = [str(row["pair_id"]) for row in pairs]
    if len(pair_ids) != len(set(pair_ids)):
        raise AssertionError("duplicate DPO-v2 pair IDs")
    if any(str(row["id"]) in forbidden_ids for row in pairs):
        raise AssertionError("forbidden ID leaked into DPO-v2 preferences")

    write_jsonl(args.output, pairs)
    summary = {
        "repair_labeled": [str(path) for path in repair_paths],
        "evaluator_rows": str(args.evaluator_rows),
        "output": str(args.output),
        "repair_labeled_sha256": {str(path): sha256_file(path) for path in repair_paths},
        "evaluator_rows_sha256": sha256_file(args.evaluator_rows),
        "pair_count": len(pairs),
        "forbidden_id_count": len(forbidden_ids),
        "forbidden_overlap_count": 0,
        "split_counts": dict(Counter(str(row["split"]) for row in pairs)),
        "io_mode_counts": dict(Counter(str(row.get("io_mode")) for row in pairs)),
        "repair_method_counts": dict(Counter(str(row.get("repair_method")) for row in pairs)),
        "skipped_counts": dict(skipped),
        "audit": audit,
        "policy": {
            "chosen": "same-model repair that passed the exact APPS verifier",
            "rejected": "same-model original response that failed the exact APPS verifier",
            "normalization": "same code-block extraction and whitespace normalization on both sides",
            "chosen_gates": "parseable, required interface present, no top-level demo, no length finish",
            "private_verifier_details": "excluded",
        },
    }
    summary["output_sha256"] = sha256_file(args.output)
    write_json(args.summary_output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
