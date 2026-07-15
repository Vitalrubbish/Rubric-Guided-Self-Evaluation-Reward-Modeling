#!/usr/bin/env python3
"""Build cleaned repair + preservation SFT rows for APPS Method 1 loop-v0."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
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


def stable_hash(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:16], 16)


def stable_bucket(value: str, modulo: int) -> int:
    return stable_hash(value) % modulo


def load_forbidden_ids(paths: Iterable[Path]) -> set[str]:
    forbidden: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        for row in read_jsonl(path):
            problem_id = str(row.get("id") or row.get("problem_id") or "")
            if problem_id:
                forbidden.add(problem_id)
    return forbidden


def infer_io_mode(row: dict[str, Any], prompt: str) -> str:
    value = row.get("io_mode") or (row.get("metadata") or {}).get("io_mode")
    if value:
        return str(value)
    if "Define the callable name(s) expected by the evaluator:" in prompt:
        return "function_call"
    return "stdin_stdout"


def extract_interface_names(row: dict[str, Any], prompt: str) -> list[str]:
    values = row.get("interface_names") or (row.get("metadata") or {}).get("interface_names")
    if isinstance(values, list) and values:
        return [str(value) for value in values if str(value).strip()]

    marker = "Define the callable name(s) expected by the evaluator:"
    if marker not in prompt:
        return []
    tail = prompt.split(marker, 1)[1].split("Python code:", 1)[0]
    names: list[str] = []
    for token in re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", tail):
        if token not in names:
            names.append(token)
    return names


def is_main_guard(node: ast.If) -> bool:
    test = node.test
    return (
        isinstance(test, ast.Compare)
        and isinstance(test.left, ast.Name)
        and test.left.id == "__name__"
        and len(test.comparators) == 1
        and isinstance(test.comparators[0], ast.Constant)
        and test.comparators[0].value == "__main__"
    )


def parseable(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def tree_has_call_named(tree: ast.AST, names: set[str]) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in names:
                return True
    return False


def top_level_callable_counts(tree: ast.Module) -> Counter[str]:
    counts: Counter[str] = Counter()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            counts[node.name] += 1
    return counts


def strict_completion_reject_reasons(
    row: dict[str, Any],
    prompt: str,
    completion: str,
    audit: dict[str, Any],
) -> list[str]:
    reasons: list[str] = []
    io_mode = str(audit.get("io_mode") or infer_io_mode(row, prompt))
    if "```" in completion:
        reasons.append("markdown_fence")
    if not audit.get("parseable_after"):
        reasons.append("not_parseable")
        return reasons
    try:
        tree = ast.parse(completion)
    except SyntaxError:
        reasons.append("not_parseable")
        return reasons

    if io_mode != "function_call":
        return reasons

    if tree_has_call_named(tree, {"input"}):
        reasons.append("function_call_uses_input")
    if tree_has_call_named(tree, {"print"}):
        reasons.append("function_call_uses_print")
    if any(isinstance(node, ast.Assert) for node in tree.body):
        reasons.append("top_level_assert")
    if any(isinstance(node, ast.If) and is_main_guard(node) for node in tree.body):
        reasons.append("main_guard")
    if any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.lower() in {"check", "test", "testing"}
        for node in tree.body
    ):
        reasons.append("demo_function")

    interface_names = extract_interface_names(row, prompt)
    if interface_names:
        counts = top_level_callable_counts(tree)
        expected_count = sum(counts[name] for name in interface_names)
        if expected_count == 0:
            reasons.append("missing_expected_callable")
        repeated = [name for name in interface_names if counts[name] > 1]
        if repeated:
            reasons.append("repeated_expected_callable")
    return reasons


def clean_function_call_completion(code: str) -> tuple[str, dict[str, Any]]:
    """Remove public/demo harnesses from function-call answers only."""
    original = code.strip()
    audit = {
        "parseable_before": False,
        "parseable_after": False,
        "removed_top_level_nodes": [],
        "changed": False,
    }
    try:
        tree = ast.parse(original)
    except SyntaxError:
        return original, audit
    audit["parseable_before"] = True

    drop_lines: set[int] = set()
    for node in tree.body:
        reason = None
        if isinstance(node, ast.Assert):
            reason = "top_level_assert"
        elif isinstance(node, ast.If) and is_main_guard(node):
            reason = "main_guard"
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.lower() in {"check", "test"}:
            reason = f"demo_function:{node.name}"
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            reason = "top_level_call"
        if reason and getattr(node, "lineno", None) and getattr(node, "end_lineno", None):
            audit["removed_top_level_nodes"].append(reason)
            drop_lines.update(range(node.lineno, node.end_lineno + 1))

    if not drop_lines:
        audit["parseable_after"] = True
        return original, audit

    lines = original.splitlines()
    kept = [line for index, line in enumerate(lines, start=1) if index not in drop_lines]
    cleaned = "\n".join(kept).strip()
    audit["changed"] = cleaned != original
    audit["parseable_after"] = parseable(cleaned)
    if not cleaned or not audit["parseable_after"]:
        return original, audit
    return cleaned, audit


def clean_completion(row: dict[str, Any], prompt: str, completion: str) -> tuple[str, dict[str, Any]]:
    io_mode = infer_io_mode(row, prompt)
    if io_mode != "function_call":
        return completion.strip(), {
            "io_mode": io_mode,
            "parseable_before": parseable(completion.strip()),
            "parseable_after": parseable(completion.strip()),
            "removed_top_level_nodes": [],
            "changed": False,
        }
    cleaned, audit = clean_function_call_completion(completion)
    audit["io_mode"] = io_mode
    return cleaned, audit


def split_for_problem(problem_id: str, validation_percent: int) -> str:
    return "validation" if stable_bucket(problem_id, 100) < validation_percent else "train"


def build_repair_rows(args: argparse.Namespace, rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], Counter[str]]:
    output = []
    counts: Counter[str] = Counter()
    seen: set[tuple[str, str, str]] = set()
    for row in sorted(rows, key=lambda item: str(item.get("id") or "")):
        problem_id = str(row.get("problem_id") or row.get("id") or "").split("__rs_sft_", 1)[0]
        prompt = str(row.get("prompt") or "").strip()
        completion = str(row.get("completion") or "").strip()
        if not problem_id or not prompt or not completion:
            counts["missing_required_field"] += 1
            continue
        cleaned, audit = clean_completion(row, prompt, completion)
        if args.require_parseable_completion and not audit["parseable_after"]:
            counts["completion_not_parseable"] += 1
            continue
        if args.strict_interface_filter:
            reject_reasons = strict_completion_reject_reasons(row, prompt, cleaned, audit)
            if reject_reasons:
                counts["strict_interface_reject"] += 1
                for reason in reject_reasons:
                    counts[f"strict_interface_reject:{reason}"] += 1
                continue
        key = (problem_id, prompt, cleaned)
        if key in seen:
            counts["duplicate"] += 1
            continue
        seen.add(key)
        if audit["changed"]:
            counts["cleaned_completion"] += 1
        output.append(
            {
                "id": f"{problem_id}__clean_rs_sft_{len(output) + 1:04d}",
                "problem_id": problem_id,
                "split": row.get("split") or split_for_problem(problem_id, args.validation_percent),
                "task_type": "solve_rejection_sampled_repair_cleaned",
                "prompt": prompt,
                "completion": cleaned,
                "source": "same_problem_k5_verifier_passing_repair_cleaned",
                "metadata": {
                    **(row.get("metadata") or {}),
                    "io_mode": audit["io_mode"],
                    "original_sft_id": row.get("id"),
                    "cleanup_audit": audit,
                },
            }
        )
    return output, counts


def build_preservation_rows(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    forbidden_ids: set[str],
    existing_keys: set[tuple[str, str, str]],
) -> tuple[list[dict[str, Any]], Counter[str]]:
    output = []
    counts: Counter[str] = Counter()
    candidates = []
    for row in rows:
        problem_id = str(row.get("id") or "")
        if not problem_id or problem_id in forbidden_ids:
            counts["forbidden_or_missing_problem"] += 1
            continue
        if not bool(row.get("passed")):
            counts["not_passed"] += 1
            continue
        if args.require_preservation_stop and row.get("finish_reason") != "stop":
            counts["non_stop_finish"] += 1
            continue
        prompt = str(row.get("prompt") or "").strip()
        completion = str(row.get("extracted_code") or row.get("generated_code") or "").strip()
        if not prompt or not completion:
            counts["missing_prompt_or_completion"] += 1
            continue
        cleaned, audit = clean_completion(row, prompt, completion)
        if args.require_parseable_completion and infer_io_mode(row, prompt) == "function_call" and not audit["parseable_after"]:
            counts["completion_not_parseable"] += 1
            continue
        if args.strict_interface_filter:
            reject_reasons = strict_completion_reject_reasons(row, prompt, cleaned, audit)
            if reject_reasons:
                counts["strict_interface_reject"] += 1
                for reason in reject_reasons:
                    counts[f"strict_interface_reject:{reason}"] += 1
                continue
        key = (problem_id, prompt, cleaned)
        if key in existing_keys:
            counts["duplicate_with_repair"] += 1
            continue
        candidate = (stable_hash(problem_id), row, prompt, cleaned, audit, key)
        candidates.append(candidate)
    candidates.sort(key=lambda item: (item[0], str(item[1].get("response_id") or "")))
    for _, row, prompt, cleaned, audit, key in candidates[: args.max_preservation_rows]:
        problem_id = str(row.get("id") or "")
        existing_keys.add(key)
        if audit["changed"]:
            counts["cleaned_completion"] += 1
        output.append(
            {
                "id": f"{problem_id}__preserve_base_pass_{len(output) + 1:04d}",
                "problem_id": problem_id,
                "split": split_for_problem(problem_id, args.validation_percent),
                "task_type": "solve_base_pass_preservation",
                "prompt": prompt,
                "completion": cleaned,
                "source": "base_model_verifier_passing_preservation",
                "metadata": {
                    "response_id": row.get("response_id"),
                    "difficulty": row.get("difficulty"),
                    "io_mode": infer_io_mode(row, prompt),
                    "finish_reason": row.get("finish_reason"),
                    "cleanup_audit": audit,
                },
            }
        )
    counts["selected_preservation_rows"] = len(output)
    counts["available_preservation_candidates"] = len(candidates)
    return output, counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Build cleaned Method 1 mixed RS-SFT data.")
    parser.add_argument("--repair-sft", type=Path, default=Path("data/sft/apps_simple_method1_loop_v0_same_problem_rs_sft.jsonl"))
    parser.add_argument(
        "--preservation-rows",
        type=Path,
        default=Path("data/responses/apps_train_simple_executable_qwen25_k1_t2048_full_labeled_nonlength.jsonl"),
    )
    parser.add_argument("--forbidden-ids", type=Path, action="append", default=[])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/sft/apps_simple_method1_loop_v0_mixed_clean_rs_sft.jsonl"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("data/sft/apps_simple_method1_loop_v0_mixed_clean_rs_sft_summary.json"),
    )
    parser.add_argument("--max-preservation-rows", type=int, default=356)
    parser.add_argument("--validation-percent", type=int, default=10)
    parser.add_argument("--require-preservation-stop", action="store_true", default=True)
    parser.add_argument("--require-parseable-completion", action="store_true", default=True)
    parser.add_argument("--strict-interface-filter", action="store_true")
    args = parser.parse_args()

    if not 1 <= args.validation_percent <= 50:
        raise ValueError("--validation-percent must be in [1, 50]")
    if args.max_preservation_rows < 0:
        raise ValueError("--max-preservation-rows must be >= 0")

    repair_source = read_jsonl(args.repair_sft)
    repair_rows, repair_counts = build_repair_rows(args, repair_source)
    existing_keys = {(row["problem_id"], row["prompt"], row["completion"]) for row in repair_rows}
    preservation_source = read_jsonl(args.preservation_rows)
    forbidden_ids = load_forbidden_ids(args.forbidden_ids)
    preservation_rows, preservation_counts = build_preservation_rows(
        args, preservation_source, forbidden_ids, existing_keys
    )
    rows = sorted(repair_rows + preservation_rows, key=lambda row: (row["split"], row["source"], row["id"]))
    write_jsonl(args.output, rows)

    summary = {
        "repair_sft": str(args.repair_sft),
        "repair_sft_sha256": sha256_file(args.repair_sft),
        "preservation_rows": str(args.preservation_rows),
        "preservation_rows_sha256": sha256_file(args.preservation_rows),
        "output": str(args.output),
        "output_sha256": sha256_file(args.output),
        "rows": len(rows),
        "unique_problem_count": len({row["problem_id"] for row in rows}),
        "split_counts": dict(Counter(row["split"] for row in rows)),
        "task_type_counts": dict(Counter(row["task_type"] for row in rows)),
        "source_counts": dict(Counter(row["source"] for row in rows)),
        "repair_counts": dict(repair_counts),
        "preservation_counts": dict(preservation_counts),
        "policy": {
            "role": "cleaned repair SFT with base-pass preservation",
            "function_call_cleanup": "remove top-level asserts, demo check/test functions, top-level calls, and __main__ guards",
            "strict_interface_filter": args.strict_interface_filter,
            "stdin_stdout_cleanup": "preserve executable stdin/stdout programs",
            "max_preservation_rows": args.max_preservation_rows,
            "validation_percent": args.validation_percent,
            "forbidden_ids": [str(path) for path in args.forbidden_ids],
        },
    }
    write_json(args.summary_output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
