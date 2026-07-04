#!/usr/bin/env python3
"""Evaluate rubric discriminability with static and verifier-informed scorers."""

from __future__ import annotations

import argparse
import ast
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, roc_auc_score


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def extract_code(text: str) -> str:
    fenced = re.search(r"```(?:python)?\s*(.*?)```", text or "", flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip("\n\r")
    return (text or "").strip("\n\r")


def raw_humaneval_prompt(prompt: str) -> str:
    marker = "Return only valid Python code, with no Markdown fences and no explanation.\n\n"
    if marker in prompt:
        return prompt.split(marker, 1)[1]
    return prompt


def can_parse(row: dict, code: str) -> bool:
    candidates = [code]
    if row.get("dataset") == "humanevalplus":
        prefix = raw_humaneval_prompt(row.get("prompt", ""))
        if prefix.strip():
            candidates.append(prefix.rstrip() + "\n" + code)
    for candidate in candidates:
        try:
            ast.parse(candidate)
            return True
        except SyntaxError:
            pass
    return False


def expected_names(row: dict) -> set[str]:
    if row.get("dataset") == "humanevalplus" and row.get("entry_point"):
        return {row["entry_point"]}
    names = set()
    for test in row.get("test_list") or []:
        names.update(re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", test))
    names -= {"assert", "len", "range", "list", "tuple", "set", "dict", "str", "int", "float", "sum", "max", "min", "abs", "sorted"}
    return names


def has_interface(row: dict, code: str) -> bool:
    names = expected_names(row)
    if not names:
        return True
    if row.get("dataset") == "humanevalplus" and code.startswith((" ", "\t")):
        return True
    return all(re.search(rf"\b(def|class)\s+{re.escape(name)}\b", code) for name in names)


def obvious_missing_dependency(code: str) -> bool:
    checks = [
        ("re.", "import re"),
        ("reduce(", "from functools import reduce"),
        ("heapq.", "import heapq"),
        ("math.", "import math"),
        ("itertools.", "import itertools"),
        ("collections.", "import collections"),
    ]
    return any(token in code and import_text not in code for token, import_text in checks)


def duplicate_or_artifact(text: str, code: str) -> bool:
    return bool(
        "```" in text
        or re.search(r"\)\s+def\s+", code)
        or re.search(r"return [^\n]+ def\s+", code)
        or "This solution is" in text
        or "Explanation" in text
    )


def static_scores(row: dict, rubric: dict) -> dict[str, int]:
    text = row.get("generated_code") or ""
    code = row.get("extracted_code") or extract_code(text)
    parse_ok = can_parse(row, code)
    interface_ok = has_interface(row, code)
    missing_dep = obvious_missing_dependency(code)
    dirty = duplicate_or_artifact(text, code)
    stub = bool(re.search(r"\b(pass|TODO|NotImplementedError)\b", code))
    risky_loop = "while True" in code or code.count("while ") >= 3

    scores = {}
    for dim in rubric.get("dimensions", []):
        dim_id = dim.get("id")
        if dim_id == "syntax_parseability":
            scores[dim_id] = 5 if parse_ok else (2 if dirty else 1)
        elif dim_id == "interface_contract_compliance":
            scores[dim_id] = 5 if interface_ok else 1
        elif dim_id == "runtime_dependency_safety":
            scores[dim_id] = 2 if missing_dep else (3 if "input(" in code else 5)
        elif dim_id == "termination_complexity":
            scores[dim_id] = 2 if risky_loop else 5
        elif dim_id == "output_format_cleanliness":
            scores[dim_id] = 2 if dirty else 5
        elif dim_id == "functional_correctness":
            if not parse_ok or not interface_ok:
                scores[dim_id] = 2
            elif stub:
                scores[dim_id] = 1
            else:
                scores[dim_id] = 4
        else:
            # Generic rubric dimensions get a weak static proxy.
            scores[dim_id] = 4 if parse_ok and interface_ok and not dirty else 2
    return scores


def verifier_scores(row: dict, rubric: dict, failure_pattern: str | None) -> dict[str, int]:
    if row.get("passed"):
        return {dim.get("id"): 5 for dim in rubric.get("dimensions", [])}

    scores = {dim.get("id"): 4 for dim in rubric.get("dimensions", [])}
    for dim in rubric.get("dimensions", []):
        dim_id = dim.get("id")
        linked = set(dim.get("linked_patterns") or [])
        if failure_pattern and failure_pattern in linked:
            if row.get("failure_type") == "syntax_error":
                scores[dim_id] = 1
            elif row.get("failure_type") == "logic_error":
                scores[dim_id] = 2
            elif row.get("failure_type") == "runtime_error":
                scores[dim_id] = 2
            elif row.get("failure_type") == "timeout":
                scores[dim_id] = 1
            else:
                scores[dim_id] = 2
    return scores


def total_score(scores: dict[str, int]) -> float:
    return float(np.mean(list(scores.values()))) if scores else 0.0


def safe_auc(labels: list[int], scores: list[float]) -> float | None:
    if len(set(labels)) < 2:
        return None
    return float(roc_auc_score(labels, scores))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labeled", type=Path, required=True)
    parser.add_argument("--failures", type=Path, required=True)
    parser.add_argument("--rubric", type=Path, required=True)
    parser.add_argument("--scores-output", type=Path, default=Path("data/rubrics/auto_rubric_scores_static.jsonl"))
    parser.add_argument("--metrics-output", type=Path, default=Path("data/rubrics/auto_rubric_eval_metrics.json"))
    args = parser.parse_args()

    rubric = json.loads(args.rubric.read_text(encoding="utf-8"))
    pattern_by_id = {row["id"]: row.get("error_pattern") for row in read_jsonl(args.failures)}
    rows = list(read_jsonl(args.labeled))
    labels = [1 if row.get("passed") else 0 for row in rows]

    records = []
    static_totals = []
    verifier_totals = []
    covered_failures = 0
    failure_count = 0
    linked_patterns = {pattern for dim in rubric.get("dimensions", []) for pattern in dim.get("linked_patterns") or []}

    for row in rows:
        pattern = pattern_by_id.get(row.get("id"))
        if not row.get("passed"):
            failure_count += 1
            if pattern in linked_patterns:
                covered_failures += 1
        static = static_scores(row, rubric)
        upper = verifier_scores(row, rubric, pattern)
        static_total = total_score(static)
        upper_total = total_score(upper)
        static_totals.append(static_total)
        verifier_totals.append(upper_total)
        records.append(
            {
                "id": row.get("id"),
                "dataset": row.get("dataset"),
                "passed": row.get("passed"),
                "failure_type": row.get("failure_type"),
                "error_pattern": pattern,
                "static_dimension_scores": static,
                "static_total_score": round(static_total, 4),
                "verifier_informed_dimension_scores": upper,
                "verifier_informed_total_score": round(upper_total, 4),
            }
        )

    static_pred = [1 if score >= 4.0 else 0 for score in static_totals]
    upper_pred = [1 if score >= 4.0 else 0 for score in verifier_totals]
    metrics = {
        "rubric": rubric.get("name"),
        "num_dimensions": len(rubric.get("dimensions", [])),
        "num_samples": len(rows),
        "coverage": round(covered_failures / failure_count, 6) if failure_count else 0,
        "static_auc": safe_auc(labels, static_totals),
        "static_kappa": float(cohen_kappa_score(labels, static_pred)),
        "static_accuracy": float(accuracy_score(labels, static_pred)),
        "static_mean_score_passed": float(np.mean([s for s, y in zip(static_totals, labels) if y == 1])),
        "static_mean_score_failed": float(np.mean([s for s, y in zip(static_totals, labels) if y == 0])),
        "verifier_informed_auc_upper_bound": safe_auc(labels, verifier_totals),
        "verifier_informed_kappa_upper_bound": float(cohen_kappa_score(labels, upper_pred)),
        "verifier_informed_accuracy_upper_bound": float(accuracy_score(labels, upper_pred)),
    }

    args.scores_output.parent.mkdir(parents=True, exist_ok=True)
    with args.scores_output.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    args.metrics_output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
