#!/usr/bin/env python3
"""Build Method 1 loop-v0 rubric scores and DPO pairs for APPS.

This script intentionally follows the task.md loop rather than a pass/fail
judge-only pipeline:

1. summarize model failures into a rubric,
2. score candidates with that rubric,
3. build verifier-anchored and small weak rubric preference pairs.

The scoring is deterministic for loop-v0 so the data build is reproducible and
does not require a GPU. The generated rubric and score schema can later be
swapped to an LLM-generated rubric without changing the downstream DPO format.
"""

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

from sklearn.metrics import cohen_kappa_score, roc_auc_score


PRIVATE_KEYS = {
    "canonical_solution",
    "canonical_solutions",
    "canonical_verifier",
    "input_output",
    "private_diagnostics",
    "safe_diagnostics",
    "test",
    "test_list",
    "test_setup_code",
}

DIMENSIONS = [
    {
        "id": "syntax_parseability_truncation",
        "name": "Syntax, Parseability, and Truncation",
        "critical_failure": True,
        "failure_types": {"syntax_error", "generation_failure"},
    },
    {
        "id": "public_interface_contract",
        "name": "Public Interface Contract",
        "critical_failure": True,
        "failure_types": {"interface_error"},
    },
    {
        "id": "runtime_safety",
        "name": "Runtime Safety and Termination",
        "critical_failure": True,
        "failure_types": {"runtime_error", "timeout"},
    },
    {
        "id": "output_contract_type_shape",
        "name": "Output Contract, Type, and Shape",
        "critical_failure": False,
        "failure_types": {"output_format_error"},
    },
    {
        "id": "algorithmic_logic_edge_cases",
        "name": "Algorithmic Logic and Edge Cases",
        "critical_failure": False,
        "failure_types": {"logic_error"},
    },
    {
        "id": "code_only_response_discipline",
        "name": "Code-Only Response Discipline",
        "critical_failure": True,
        "failure_types": {"generation_failure"},
    },
]

RUBRIC_ANCHORS = {
    "1": "Fatal violation: the candidate is very likely unusable for this dimension.",
    "2": "Major violation: the candidate shows a concrete defect that can fail ordinary valid cases.",
    "3": "Uncertain or partial: the candidate has visible risk but the defect is not fully localized.",
    "4": "Mostly satisfies the dimension with only narrow or cosmetic risk.",
    "5": "Fully satisfies the dimension based on public task/interface/code evidence.",
}

PROSE_MARKERS = {
    "the code is",
    "the repair",
    "this version",
    "here is",
    "should pass",
    "probability of passing",
    "no changes are needed",
    "explanation",
}

TYPING_NAMES = {"List", "Dict", "Tuple", "Set", "Optional", "Deque", "DefaultDict"}


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


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_response_id(row: dict[str, Any]) -> str:
    if row.get("response_id"):
        return str(row["response_id"])
    return f"{row.get('id')}__sample{row.get('sample_id', 0)}"


def extract_code(text: Any) -> str:
    value = str(text or "")
    fence = re.search(r"```(?:python)?\s*(.*?)```", value, flags=re.DOTALL | re.IGNORECASE)
    return fence.group(1) if fence else value


def normalize_code(text: Any) -> str:
    code = extract_code(text).replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in code.strip().splitlines()]
    if lines and re.fullmatch(r"\s*```(?:python)?\s*", lines[0], flags=re.IGNORECASE):
        lines.pop(0)
    if lines and re.fullmatch(r"\s*```\s*", lines[-1]):
        lines.pop()
    return "\n".join(lines).strip()


def code_for_row(row: dict[str, Any]) -> str:
    return normalize_code(row.get("extracted_code") or row.get("generated_code") or "")


def interface_names(row: dict[str, Any]) -> list[str]:
    names = row.get("interface_names")
    if isinstance(names, list):
        return [str(name) for name in names if str(name)]
    public = row.get("public_interface")
    if isinstance(public, list):
        return [str(name) for name in public if str(name)]
    return []


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


def ast_audit(code: str, required_names: Iterable[str]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "parseable": False,
        "required_interface_present": False,
        "top_level_demo_count": 0,
        "duplicate_defs": [],
        "defined_functions": [],
        "solution_methods": [],
        "typing_names_without_import": [],
    }
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        result["syntax_error"] = f"{exc.msg} at line {exc.lineno}"
        return result

    functions = [node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    solution_methods: list[str] = []
    imported: set[str] = set()
    used_names: set[str] = set()
    demos = 0
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "typing":
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ClassDef) and node.name == "Solution":
            solution_methods.extend(
                child.name for child in node.body if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            )
        elif isinstance(node, ast.Assert):
            demos += 1
        elif isinstance(node, ast.If) and is_main_guard(node):
            demos += 1
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            demos += 1
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            used_names.add(node.id)

    required = {str(name) for name in required_names if str(name)}
    all_callables = set(functions) | set(solution_methods)
    duplicates = sorted(name for name, count in Counter(functions + solution_methods).items() if count > 1)
    missing_typing = sorted((used_names & TYPING_NAMES) - imported)
    result.update(
        {
            "parseable": True,
            "required_interface_present": not required or bool(required & all_callables),
            "top_level_demo_count": demos,
            "duplicate_defs": duplicates,
            "defined_functions": sorted(set(functions)),
            "solution_methods": sorted(set(solution_methods)),
            "typing_names_without_import": missing_typing,
        }
    )
    return result


def visible_features(row: dict[str, Any], code: str) -> dict[str, Any]:
    generated = str(row.get("generated_code") or "")
    lower_code = code.lower()
    nonempty_lines = [line for line in code.splitlines() if line.strip()]
    audit = ast_audit(code, interface_names(row))
    return {
        **audit,
        "has_markdown_fence": "```" in generated or "```" in code,
        "has_prose_marker": any(marker in lower_code for marker in PROSE_MARKERS),
        "finish_reason_length": row.get("finish_reason") == "length",
        "line_count": len(nonempty_lines),
        "many_lines": len(nonempty_lines) >= 120,
        "empty_code": not bool(code.strip()),
    }


def score_row(row: dict[str, Any], source: str) -> dict[str, Any]:
    code = code_for_row(row)
    features = visible_features(row, code)
    passed = bool(row.get("passed"))
    failure_type = str(row.get("failure_type") or ("passed" if passed else "unknown_failure"))
    io_mode = str(row.get("io_mode") or "")

    scores = {dimension["id"]: 5 for dimension in DIMENSIONS}
    reasons: list[str] = []

    if features["empty_code"]:
        scores["syntax_parseability_truncation"] = 1
        scores["code_only_response_discipline"] = 1
        reasons.append("empty code")
    elif not features["parseable"]:
        scores["syntax_parseability_truncation"] = 1
        reasons.append("Python parser rejects the code")

    if features["finish_reason_length"] or features["many_lines"]:
        scores["syntax_parseability_truncation"] = min(scores["syntax_parseability_truncation"], 2)
        scores["code_only_response_discipline"] = min(scores["code_only_response_discipline"], 2)
        reasons.append("length finish or unusually long completion")

    if features["has_markdown_fence"]:
        scores["code_only_response_discipline"] = min(scores["code_only_response_discipline"], 2)
        reasons.append("Markdown fence visible in generated response")
    if features["has_prose_marker"]:
        scores["code_only_response_discipline"] = min(scores["code_only_response_discipline"], 2)
        reasons.append("non-code prose marker visible inside code")
    if features["duplicate_defs"]:
        scores["syntax_parseability_truncation"] = min(scores["syntax_parseability_truncation"], 3)
        scores["code_only_response_discipline"] = min(scores["code_only_response_discipline"], 3)
        reasons.append("duplicate callable definitions")
    if features["typing_names_without_import"]:
        scores["runtime_safety"] = min(scores["runtime_safety"], 3)
        reasons.append("typing annotation names used without visible typing import")
    if features["top_level_demo_count"] and not passed:
        scores["public_interface_contract"] = min(scores["public_interface_contract"], 3)
        scores["code_only_response_discipline"] = min(scores["code_only_response_discipline"], 3)
        reasons.append("top-level demo/assert/call may interfere with evaluation")

    if io_mode == "function_call" and not features["required_interface_present"]:
        scores["public_interface_contract"] = 1
        reasons.append("required callable interface is not visible")

    if not passed:
        if failure_type == "syntax_error":
            scores["syntax_parseability_truncation"] = min(scores["syntax_parseability_truncation"], 1)
        elif failure_type == "runtime_error":
            scores["runtime_safety"] = min(scores["runtime_safety"], 1)
        elif failure_type == "timeout":
            scores["runtime_safety"] = min(scores["runtime_safety"], 2)
        elif failure_type == "logic_error":
            scores["algorithmic_logic_edge_cases"] = min(scores["algorithmic_logic_edge_cases"], 2)
            scores["output_contract_type_shape"] = min(scores["output_contract_type_shape"], 4)
        elif failure_type == "generation_failure":
            scores["syntax_parseability_truncation"] = min(scores["syntax_parseability_truncation"], 1)
            scores["code_only_response_discipline"] = min(scores["code_only_response_discipline"], 1)
        else:
            scores["algorithmic_logic_edge_cases"] = min(scores["algorithmic_logic_edge_cases"], 3)
        reasons.append(f"external verifier outcome: {failure_type}")

    weights = {
        "syntax_parseability_truncation": 1.25,
        "public_interface_contract": 1.20,
        "runtime_safety": 1.10,
        "output_contract_type_shape": 0.90,
        "algorithmic_logic_edge_cases": 1.25,
        "code_only_response_discipline": 0.85,
    }
    weighted = sum(scores[key] * weight for key, weight in weights.items()) / sum(weights.values())
    critical_floor = min(
        scores["syntax_parseability_truncation"],
        scores["public_interface_contract"],
        scores["runtime_safety"],
        scores["code_only_response_discipline"],
    )
    if critical_floor <= 1:
        weighted = min(weighted, 2.0)
    elif critical_floor == 2:
        weighted = min(weighted, 3.0)

    overall_score = round(float(weighted), 4)
    return {
        "score_id": stable_response_id(row),
        "source": source,
        "id": row.get("id"),
        "split": row.get("eval_split") or row.get("split") or row.get("source_split"),
        "dataset": row.get("dataset", "apps"),
        "response_id": stable_response_id(row),
        "sample_id": row.get("sample_id"),
        "repair_candidate_id": row.get("repair_candidate_id"),
        "original_response_id": row.get("original_response_id"),
        "selection_reason": row.get("selection_reason"),
        "io_mode": row.get("io_mode"),
        "passed": passed,
        "failure_type": None if passed else failure_type,
        "overall_score": overall_score,
        "predicted_pass": overall_score >= 4.0,
        "dimension_scores": scores,
        "visible_features": {
            "parseable": features["parseable"],
            "required_interface_present": features["required_interface_present"],
            "has_markdown_fence": features["has_markdown_fence"],
            "has_prose_marker": features["has_prose_marker"],
            "finish_reason_length": features["finish_reason_length"],
            "many_lines": features["many_lines"],
            "line_count": features["line_count"],
            "top_level_demo_count": features["top_level_demo_count"],
            "duplicate_defs": features["duplicate_defs"],
            "typing_names_without_import": features["typing_names_without_import"],
        },
        "rubric_reasons": reasons[:6],
    }


def numeric_stats(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    if not ordered:
        return {"min": 0.0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}

    def pct(frac: float) -> float:
        return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * frac))]

    return {
        "min": ordered[0],
        "mean": mean(ordered),
        "p50": pct(0.50),
        "p95": pct(0.95),
        "max": ordered[-1],
    }


def binary_metrics(scores: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [row for row in scores if isinstance(row.get("passed"), bool)]
    if not rows:
        return {}
    y_true = [1 if row["passed"] else 0 for row in rows]
    y_score = [float(row["overall_score"]) for row in rows]
    thresholds = sorted(set(y_score), reverse=True)
    if min(y_true) == max(y_true):
        auc = None
    else:
        auc = float(roc_auc_score(y_true, y_score))

    def at_threshold(threshold: float) -> dict[str, Any]:
        pred = [1 if value >= threshold else 0 for value in y_score]
        tp = sum(1 for gold, guess in zip(y_true, pred) if gold == 1 and guess == 1)
        tn = sum(1 for gold, guess in zip(y_true, pred) if gold == 0 and guess == 0)
        fp = sum(1 for gold, guess in zip(y_true, pred) if gold == 0 and guess == 1)
        fn = sum(1 for gold, guess in zip(y_true, pred) if gold == 1 and guess == 0)
        pos = tp + fn
        neg = tn + fp
        recall = tp / pos if pos else 0.0
        specificity = tn / neg if neg else 0.0
        return {
            "threshold": threshold,
            "accuracy": (tp + tn) / len(rows),
            "balanced_accuracy": (recall + specificity) / 2,
            "kappa": float(cohen_kappa_score(y_true, pred)) if len(set(pred)) > 1 else 0.0,
            "overacceptance_rate": fp / neg if neg else 0.0,
            "false_rejection_rate": fn / pos if pos else 0.0,
            "predicted_pass_rate": sum(pred) / len(pred),
            "true_pass_rate": sum(y_true) / len(y_true),
            "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        }

    swept = [at_threshold(threshold) for threshold in thresholds]
    best_ba = max(swept, key=lambda item: (item["balanced_accuracy"], item["accuracy"])) if swept else None
    best_safe = [
        item for item in swept if item["overacceptance_rate"] <= 0.25
    ]
    return {
        "rows": len(rows),
        "auc": auc,
        "score_stats": numeric_stats(y_score),
        "default_threshold_4": at_threshold(4.0),
        "best_balanced_accuracy": best_ba,
        "best_with_overacceptance_le_0_25": max(
            best_safe, key=lambda item: (item["balanced_accuracy"], item["accuracy"])
        )
        if best_safe
        else None,
    }


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


def extract_between(text: str, start: str, end: str) -> str:
    if start not in text:
        return ""
    tail = text.split(start, 1)[1]
    if end in tail:
        return tail.split(end, 1)[0].strip()
    return tail.strip()


def solve_prompt(row: dict[str, Any]) -> str:
    prompt = str(row.get("prompt") or "").strip()
    if prompt.startswith("You are an expert Python programmer. Solve"):
        return prompt

    task = extract_between(prompt, "\nTask:\n", "\n\nStarting code:")
    starting = extract_between(prompt, "\nStarting code:\n", "\n\nDefine the callable")
    names = interface_names(row)
    parts = [
        "You are an expert Python programmer. Solve the following task.",
        "Return only valid Python code, with no Markdown fences and no explanation.",
        "",
        "Task:",
        task or str(row.get("task") or "").strip(),
    ]
    if starting:
        parts.extend(["", "Starting code:", starting])
    if names:
        parts.extend(["", "Define the callable name(s) expected by the evaluator:", "\n".join(names)])
    parts.extend(["", "Python code:"])
    return "\n".join(parts).strip()


def pair_length_ok(chosen: str, rejected: str, max_completion_chars: int, max_length_ratio: float) -> bool:
    if not chosen or not rejected or chosen == rejected:
        return False
    if len(chosen) > max_completion_chars or len(rejected) > max_completion_chars:
        return False
    ratio = max(len(chosen), len(rejected)) / max(1, min(len(chosen), len(rejected)))
    return ratio <= max_length_ratio


def sanitized_pair(row: dict[str, Any]) -> dict[str, Any]:
    leaked = PRIVATE_KEYS.intersection(row)
    if leaked:
        raise AssertionError(f"private fields leaked into pair {row.get('pair_id')}: {sorted(leaked)}")
    return row


def build_strict_pairs(
    rows: list[dict[str, Any]],
    score_by_response: dict[str, dict[str, Any]],
    forbidden_ids: set[str],
    max_completion_chars: int,
    max_length_ratio: float,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    pairs = []
    skipped: Counter[str] = Counter()
    for source in rows:
        problem_id = str(source.get("id") or "")
        if not problem_id or problem_id in forbidden_ids:
            skipped["forbidden_or_missing_problem_id"] += 1
            continue
        chosen = normalize_code(source.get("chosen"))
        rejected = normalize_code(source.get("rejected"))
        if not pair_length_ok(chosen, rejected, max_completion_chars, max_length_ratio):
            skipped["length_or_identity_gate"] += 1
            continue
        chosen_score = score_by_response.get(str(source.get("chosen_response_id") or ""), {})
        rejected_score = score_by_response.get(str(source.get("rejected_response_id") or ""), {})
        pair = {
            "pair_id": f"loop_v0_strict__{source.get('pair_id')}",
            "pair_version": "apps_simple_method1_loop_v0_rubric_dpo",
            "id": problem_id,
            "dataset": "apps",
            "split": "train",
            "source_split": source.get("source_split", "train"),
            "difficulty": source.get("difficulty"),
            "io_mode": source.get("io_mode"),
            "prompt": str(source.get("prompt") or "").strip(),
            "chosen": chosen,
            "rejected": rejected,
            "chosen_source": source.get("chosen_source", "same_model_verifier_passing_repair"),
            "rejected_source": source.get("rejected_source", "same_model_verifier_failed_original"),
            "preference_source": "external_verifier_pass_over_fail_strict_reused",
            "rubric_preference_role": "strong_verifier_anchor",
            "chosen_response_id": source.get("chosen_response_id"),
            "rejected_response_id": source.get("rejected_response_id"),
            "chosen_rubric_score": chosen_score.get("overall_score"),
            "rejected_rubric_score": rejected_score.get("overall_score"),
            "original_failure_type": source.get("original_failure_type"),
            "selection_reason": source.get("selection_reason"),
        }
        pairs.append(sanitized_pair(pair))
    return pairs, skipped


def build_repair_contrast_pairs(
    repair_rows: list[dict[str, Any]],
    score_by_response: dict[str, dict[str, Any]],
    forbidden_ids: set[str],
    max_pairs_per_problem: int,
    max_completion_chars: int,
    max_length_ratio: float,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in repair_rows:
        problem_id = str(row.get("id") or "")
        if not problem_id or problem_id in forbidden_ids:
            continue
        if str(row.get("split") or "train") != "train":
            continue
        groups[problem_id].append(row)

    pairs = []
    skipped: Counter[str] = Counter()
    for problem_id, rows in sorted(groups.items()):
        passed = [row for row in rows if bool(row.get("passed"))]
        failed = [row for row in rows if not bool(row.get("passed"))]
        if not passed or not failed:
            skipped["no_same_problem_pass_fail_contrast"] += 1
            continue
        passed = sorted(
            passed,
            key=lambda row: (
                -float(score_by_response.get(stable_response_id(row), {}).get("overall_score") or 0.0),
                len(code_for_row(row)),
                stable_response_id(row),
            ),
        )
        failed = sorted(
            failed,
            key=lambda row: (
                float(score_by_response.get(stable_response_id(row), {}).get("overall_score") or 0.0),
                stable_response_id(row),
            ),
        )
        made = 0
        for chosen_row in passed:
            chosen = code_for_row(chosen_row)
            chosen_features = score_by_response.get(stable_response_id(chosen_row), {}).get("visible_features", {})
            if chosen_row.get("finish_reason") == "length":
                skipped["chosen_length_finish"] += 1
                continue
            if not chosen_features.get("parseable", False):
                skipped["chosen_not_parseable"] += 1
                continue
            if chosen_row.get("io_mode") == "function_call" and not chosen_features.get("required_interface_present", False):
                skipped["chosen_interface_missing"] += 1
                continue
            for rejected_row in failed:
                rejected = code_for_row(rejected_row)
                if not pair_length_ok(chosen, rejected, max_completion_chars, max_length_ratio):
                    skipped["length_or_identity_gate"] += 1
                    continue
                chosen_score = score_by_response.get(stable_response_id(chosen_row), {})
                rejected_score = score_by_response.get(stable_response_id(rejected_row), {})
                pair = {
                    "pair_id": f"{problem_id}__loop_v0_repair_contrast_{made + 1}",
                    "pair_version": "apps_simple_method1_loop_v0_rubric_dpo",
                    "id": problem_id,
                    "dataset": "apps",
                    "split": "train",
                    "source_split": "train",
                    "difficulty": chosen_row.get("difficulty"),
                    "io_mode": chosen_row.get("io_mode"),
                    "prompt": solve_prompt(chosen_row),
                    "chosen": chosen,
                    "rejected": rejected,
                    "chosen_source": "same_model_k5_repair_verifier_pass",
                    "rejected_source": "same_model_k5_repair_verifier_fail",
                    "preference_source": "external_verifier_pass_over_same_problem_repair_fail",
                    "rubric_preference_role": "strong_same_problem_anchor",
                    "chosen_response_id": stable_response_id(chosen_row),
                    "rejected_response_id": stable_response_id(rejected_row),
                    "chosen_rubric_score": chosen_score.get("overall_score"),
                    "rejected_rubric_score": rejected_score.get("overall_score"),
                    "rejected_failure_type": rejected_row.get("failure_type"),
                    "selection_reason": chosen_row.get("selection_reason"),
                }
                pairs.append(sanitized_pair(pair))
                made += 1
                break
            if made >= max_pairs_per_problem:
                break
    return pairs, skipped


def build_weak_rubric_pairs(
    repair_rows: list[dict[str, Any]],
    score_by_response: dict[str, dict[str, Any]],
    forbidden_ids: set[str],
    max_pairs: int,
    max_pairs_per_problem: int,
    min_margin: float,
    max_completion_chars: int,
    max_length_ratio: float,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    if max_pairs <= 0:
        return [], Counter()
    groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in repair_rows:
        problem_id = str(row.get("id") or "")
        if not problem_id or problem_id in forbidden_ids or bool(row.get("passed")):
            continue
        if str(row.get("split") or "train") != "train":
            continue
        groups[problem_id].append(row)

    pairs = []
    skipped: Counter[str] = Counter()
    for problem_id, rows in sorted(groups.items()):
        scored = [
            (float(score_by_response.get(stable_response_id(row), {}).get("overall_score") or 0.0), row)
            for row in rows
        ]
        if len(scored) < 2:
            skipped["too_few_failed_candidates"] += 1
            continue
        scored.sort(key=lambda item: (item[0], stable_response_id(item[1])))
        low_score, low_row = scored[0]
        high_score, high_row = scored[-1]
        if high_score - low_score < min_margin:
            skipped["margin_too_small"] += 1
            continue
        high_features = score_by_response.get(stable_response_id(high_row), {}).get("visible_features", {})
        if not high_features.get("parseable", False):
            skipped["weak_chosen_not_parseable"] += 1
            continue
        if high_row.get("io_mode") == "function_call" and not high_features.get("required_interface_present", False):
            skipped["weak_chosen_interface_missing"] += 1
            continue
        chosen = code_for_row(high_row)
        rejected = code_for_row(low_row)
        if not pair_length_ok(chosen, rejected, max_completion_chars, max_length_ratio):
            skipped["length_or_identity_gate"] += 1
            continue
        for index in range(max_pairs_per_problem):
            if index > 0:
                break
            pair = {
                "pair_id": f"{problem_id}__loop_v0_weak_rubric_{index + 1}",
                "pair_version": "apps_simple_method1_loop_v0_rubric_dpo",
                "id": problem_id,
                "dataset": "apps",
                "split": "train",
                "source_split": "train",
                "difficulty": high_row.get("difficulty"),
                "io_mode": high_row.get("io_mode"),
                "prompt": solve_prompt(high_row),
                "chosen": chosen,
                "rejected": rejected,
                "chosen_source": "same_model_k5_repair_verifier_fail_higher_rubric_score",
                "rejected_source": "same_model_k5_repair_verifier_fail_lower_rubric_score",
                "preference_source": "weak_rubric_score_fail_over_worse_fail",
                "rubric_preference_role": "weak_rubric_reward",
                "chosen_response_id": stable_response_id(high_row),
                "rejected_response_id": stable_response_id(low_row),
                "chosen_rubric_score": high_score,
                "rejected_rubric_score": low_score,
                "rubric_margin": round(high_score - low_score, 4),
                "chosen_failure_type": high_row.get("failure_type"),
                "rejected_failure_type": low_row.get("failure_type"),
            }
            pairs.append(sanitized_pair(pair))
        if len(pairs) >= max_pairs:
            break
    return pairs[:max_pairs], skipped


def example_refs(rows: list[dict[str, Any]], failure_types: set[str], passed: bool, limit: int = 3) -> list[dict[str, Any]]:
    refs = []
    for row in rows:
        if bool(row.get("passed")) != passed:
            continue
        if not passed and str(row.get("failure_type") or "") not in failure_types:
            continue
        refs.append(
            {
                "id": row.get("id"),
                "response_id": stable_response_id(row),
                "failure_type": row.get("failure_type"),
                "selection_reason": row.get("selection_reason"),
            }
        )
        if len(refs) >= limit:
            break
    return refs


def negative_example_refs(rows: list[dict[str, Any]], dimension_id: str, failure_types: set[str], limit: int = 3) -> list[dict[str, Any]]:
    refs = example_refs(rows, failure_types, False, limit)
    if len(refs) >= limit:
        return refs

    for row in rows:
        if bool(row.get("passed")):
            continue
        code = code_for_row(row)
        features = visible_features(row, code)
        failure_type = str(row.get("failure_type") or "")
        include = False
        if dimension_id == "public_interface_contract":
            include = str(row.get("io_mode") or "") == "function_call" and not features["required_interface_present"]
        elif dimension_id == "output_contract_type_shape":
            include = failure_type in {"logic_error", "runtime_error"}
        elif dimension_id == "code_only_response_discipline":
            include = (
                features["has_markdown_fence"]
                or features["has_prose_marker"]
                or features["finish_reason_length"]
                or failure_type == "generation_failure"
            )
        elif dimension_id == "algorithmic_logic_edge_cases":
            include = failure_type == "logic_error"
        elif dimension_id == "runtime_safety":
            include = failure_type in {"runtime_error", "timeout"}
        elif dimension_id == "syntax_parseability_truncation":
            include = failure_type in {"syntax_error", "generation_failure"} or not features["parseable"]
        if not include:
            continue
        ref = {
            "id": row.get("id"),
            "response_id": stable_response_id(row),
            "failure_type": row.get("failure_type"),
            "selection_reason": row.get("selection_reason"),
        }
        if ref not in refs:
            refs.append(ref)
        if len(refs) >= limit:
            break

    if refs:
        return refs
    return example_refs(rows, set(), False, limit)


def build_rubric(repair_rows: list[dict[str, Any]], eligible_rows: list[dict[str, Any]]) -> dict[str, Any]:
    outcome_counts = Counter("passed" if row.get("passed") else str(row.get("failure_type")) for row in eligible_rows)
    selection_counts = Counter(str(row.get("selection_reason")) for row in eligible_rows)
    dimensions = []
    for dimension in DIMENSIONS:
        dimensions.append(
            {
                "id": dimension["id"],
                "name": dimension["name"],
                "definition": {
                    "syntax_parseability_truncation": "Code must be valid Python and must not be a truncated or repeated fragment.",
                    "public_interface_contract": "Code must expose the public callable or stdin/stdout behavior requested by the task.",
                    "runtime_safety": "Code should avoid predictable exceptions, undefined names, API misuse, and nontermination.",
                    "output_contract_type_shape": "Returned or printed values must match the task's required format, type, and container shape.",
                    "algorithmic_logic_edge_cases": "The implementation must satisfy the core algorithm and edge cases implied by the task.",
                    "code_only_response_discipline": "The completion should be executable Python code only, without Markdown or explanatory prose.",
                }[dimension["id"]],
                "score_anchors": RUBRIC_ANCHORS,
                "critical_failure": dimension["critical_failure"],
                "weight": {
                    "syntax_parseability_truncation": 1.25,
                    "public_interface_contract": 1.20,
                    "runtime_safety": 1.10,
                    "output_contract_type_shape": 0.90,
                    "algorithmic_logic_edge_cases": 1.25,
                    "code_only_response_discipline": 0.85,
                }[dimension["id"]],
                "positive_examples": example_refs(repair_rows, dimension["failure_types"], True),
                "negative_examples": negative_example_refs(repair_rows, dimension["id"], dimension["failure_types"]),
            }
        )
    return {
        "name": "apps_simple_method1_loop_v0_failure_derived_rubric",
        "generation_method": "deterministic_from_k5_failure_statistics",
        "task_type": "Python code generation",
        "intended_use": "Rubric-guided self-evaluation and DPO pair construction for Method 1 loop-v0.",
        "source_statistics": {
            "eligible_repair_rows": len(eligible_rows),
            "outcome_counts": dict(outcome_counts),
            "selection_reason_counts": dict(selection_counts),
        },
        "global_judging_instructions": [
            "Judge only public task, public interface, and visible code.",
            "Use verifier pass/fail only as an external audit label, not as prompt text.",
            "Treat syntax, interface, runtime, and code-only violations as critical because they block reliable execution.",
        ],
        "dimensions": dimensions,
    }


def source_metrics(scores: list[dict[str, Any]]) -> dict[str, Any]:
    buckets: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scores:
        buckets[f"{row.get('source')}:{row.get('split')}"].append(row)
    return {key: binary_metrics(rows) for key, rows in sorted(buckets.items())}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Method 1 APPS loop-v0 rubric scores and DPO pairs.")
    parser.add_argument(
        "--repair-rows",
        type=Path,
        default=Path("data/repair/apps_simple_method1_repair_all_train_failures_k5_v1_labeled.jsonl"),
    )
    parser.add_argument(
        "--evaluator-rows",
        type=Path,
        default=Path("data/evaluator/apps_simple_method1_evaluator_training_rows_v1.jsonl"),
    )
    parser.add_argument(
        "--strict-verifier-pairs",
        type=Path,
        default=Path("data/preferences/apps_simple_method1_all_train_failures_k5_dpo_v2.jsonl"),
    )
    parser.add_argument("--forbidden-ids", type=Path, action="append", default=[])
    parser.add_argument(
        "--rubric-output",
        type=Path,
        default=Path("data/rubrics/apps_simple_method1_loop_v0_rubric.json"),
    )
    parser.add_argument(
        "--scores-output",
        type=Path,
        default=Path("data/rubrics/apps_simple_method1_loop_v0_rubric_scores.jsonl"),
    )
    parser.add_argument(
        "--pairs-output",
        type=Path,
        default=Path("data/preferences/apps_simple_method1_loop_v0_rubric_dpo_pairs.jsonl"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("data/preferences/apps_simple_method1_loop_v0_rubric_dpo_pairs_summary.json"),
    )
    parser.add_argument("--max-contrast-pairs-per-problem", type=int, default=2)
    parser.add_argument("--max-weak-rubric-pairs", type=int, default=120)
    parser.add_argument("--max-weak-pairs-per-problem", type=int, default=1)
    parser.add_argument("--min-weak-rubric-margin", type=float, default=1.0)
    parser.add_argument("--max-completion-chars", type=int, default=6000)
    parser.add_argument("--max-length-ratio", type=float, default=8.0)
    args = parser.parse_args()

    repair_rows = read_jsonl(args.repair_rows)
    evaluator_rows = read_jsonl(args.evaluator_rows)
    forbidden_ids = load_forbidden_ids(args.forbidden_ids)
    eligible_repair_rows = [row for row in repair_rows if str(row.get("id") or "") not in forbidden_ids]

    repair_scores = [score_row(row, "k5_repair") for row in eligible_repair_rows]
    evaluator_scores = [score_row(row, "evaluator_k1") for row in evaluator_rows]
    all_scores = repair_scores + evaluator_scores
    score_by_response = {str(row["response_id"]): row for row in all_scores}

    rubric = build_rubric(repair_rows, eligible_repair_rows)
    write_json(args.rubric_output, rubric)
    write_jsonl(args.scores_output, all_scores)

    strict_rows = read_jsonl(args.strict_verifier_pairs) if args.strict_verifier_pairs.exists() else []
    strict_pairs, strict_skipped = build_strict_pairs(
        strict_rows,
        score_by_response,
        forbidden_ids,
        args.max_completion_chars,
        args.max_length_ratio,
    )
    contrast_pairs, contrast_skipped = build_repair_contrast_pairs(
        eligible_repair_rows,
        score_by_response,
        forbidden_ids,
        args.max_contrast_pairs_per_problem,
        args.max_completion_chars,
        args.max_length_ratio,
    )
    weak_pairs, weak_skipped = build_weak_rubric_pairs(
        eligible_repair_rows,
        score_by_response,
        forbidden_ids,
        args.max_weak_rubric_pairs,
        args.max_weak_pairs_per_problem,
        args.min_weak_rubric_margin,
        args.max_completion_chars,
        args.max_length_ratio,
    )

    pairs = strict_pairs + contrast_pairs + weak_pairs
    pair_ids = [str(row["pair_id"]) for row in pairs]
    if len(pair_ids) != len(set(pair_ids)):
        duplicates = [key for key, value in Counter(pair_ids).items() if value > 1]
        raise AssertionError(f"duplicate pair IDs: {duplicates[:5]}")
    if any(str(row.get("id") or "") in forbidden_ids for row in pairs):
        raise AssertionError("forbidden problem leaked into loop-v0 DPO pairs")
    if not pairs:
        raise RuntimeError("no loop-v0 DPO pairs were produced")

    write_jsonl(args.pairs_output, pairs)
    summary = {
        "inputs": {
            "repair_rows": str(args.repair_rows),
            "repair_rows_sha256": sha256_file(args.repair_rows),
            "evaluator_rows": str(args.evaluator_rows),
            "evaluator_rows_sha256": sha256_file(args.evaluator_rows),
            "strict_verifier_pairs": str(args.strict_verifier_pairs),
            "strict_verifier_pairs_sha256": sha256_file(args.strict_verifier_pairs),
        },
        "outputs": {
            "rubric": str(args.rubric_output),
            "rubric_sha256": sha256_file(args.rubric_output),
            "scores": str(args.scores_output),
            "scores_sha256": sha256_file(args.scores_output),
            "pairs": str(args.pairs_output),
            "pairs_sha256": sha256_file(args.pairs_output),
        },
        "forbidden_id_count": len(forbidden_ids),
        "repair_rows": len(repair_rows),
        "eligible_repair_rows": len(eligible_repair_rows),
        "eligible_repair_outcome_counts": dict(
            Counter("passed" if row.get("passed") else str(row.get("failure_type")) for row in eligible_repair_rows)
        ),
        "score_metrics": source_metrics(all_scores),
        "pair_count": len(pairs),
        "pair_source_counts": dict(Counter(str(row.get("preference_source")) for row in pairs)),
        "pair_role_counts": dict(Counter(str(row.get("rubric_preference_role")) for row in pairs)),
        "io_mode_counts": dict(Counter(str(row.get("io_mode")) for row in pairs)),
        "unique_problem_count": len({str(row.get("id")) for row in pairs}),
        "skipped": {
            "strict_pairs": dict(strict_skipped),
            "repair_contrast_pairs": dict(contrast_skipped),
            "weak_rubric_pairs": dict(weak_skipped),
        },
        "policy": {
            "method1_loop": "failure statistics -> rubric -> rubric scores -> verifier/rubric DPO pairs",
            "strong_pairs": "external verifier pass over fail, including same-problem k=5 repair contrasts",
            "weak_pairs": "small capped set of verifier-failed candidates ranked by rubric score margin",
            "private_fields": "excluded from rubric, scores, and DPO pairs",
            "heldout_use": "evaluator validation/test rows are scored for diagnostics only, never used in DPO pairs",
        },
    }
    write_json(args.summary_output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
