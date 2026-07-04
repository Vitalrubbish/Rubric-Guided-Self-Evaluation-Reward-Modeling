#!/usr/bin/env python3
"""Verifier for the MATH transfer subset."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import sympy as sp


BOXED_RE = re.compile(r"\\boxed\s*{")
HASH_RE = re.compile(r"####\s*(.+?)(?:\n|$)")
NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def find_matching_brace(text: str, open_idx: int) -> int | None:
    depth = 0
    for idx in range(open_idx, len(text)):
        if text[idx] == "{":
            depth += 1
        elif text[idx] == "}":
            depth -= 1
            if depth == 0:
                return idx
    return None


def extract_boxed(text: str) -> str | None:
    matches = list(BOXED_RE.finditer(text or ""))
    if not matches:
        return None
    match = matches[-1]
    open_idx = match.end() - 1
    close_idx = find_matching_brace(text, open_idx)
    if close_idx is None:
        return None
    return text[open_idx + 1 : close_idx].strip()


def extract_prediction(text: str) -> tuple[str | None, str]:
    hash_matches = HASH_RE.findall(text or "")
    if hash_matches:
        return hash_matches[-1].strip().rstrip("."), "hash_final"
    boxed = extract_boxed(text or "")
    if boxed is not None:
        return boxed, "boxed"
    phrase_patterns = [
        r"(?:final answer|answer is|therefore|so)\D{0,40}(.+?)(?:\n|$)",
        r"=\s*([^=\n]+?)\s*$",
    ]
    for pattern in phrase_patterns:
        matches = re.findall(pattern, text or "", flags=re.IGNORECASE)
        if matches:
            candidate = matches[-1].strip().rstrip(".")
            if candidate:
                return candidate, "phrase_final"
    numbers = NUMBER_RE.findall(text or "")
    if numbers:
        return numbers[-1].replace(",", ""), "last_number"
    return None, "no_answer"


def latex_to_sympy_text(text: str) -> str:
    s = (text or "").strip()
    s = s.replace("$", "")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\,", "").replace("\\!", "")
    s = s.replace(",", "")
    s = s.replace("^", "**")
    s = re.sub(r"\\dfrac\s*{([^{}]+)}\s*{([^{}]+)}", r"(\1)/(\2)", s)
    s = re.sub(r"\\frac\s*{([^{}]+)}\s*{([^{}]+)}", r"(\1)/(\2)", s)
    s = re.sub(r"\\sqrt\s*{([^{}]+)}", r"sqrt(\1)", s)
    s = s.replace("\\pi", "pi")
    s = s.replace("\\cdot", "*").replace("\\times", "*")
    s = s.replace("%", "/100")
    return s.strip()


def normalize_text(text: str | None) -> str | None:
    if text is None:
        return None
    s = latex_to_sympy_text(text)
    s = s.strip().rstrip(".")
    s = re.sub(r"\s+", "", s)
    return s or None


def equivalent(pred: str | None, gold: str | None) -> tuple[bool, str]:
    pred_norm = normalize_text(pred)
    gold_norm = normalize_text(gold)
    if pred_norm is None or gold_norm is None:
        return False, "missing"
    if pred_norm == gold_norm:
        return True, "exact_norm"
    try:
        pred_expr = sp.sympify(pred_norm)
        gold_expr = sp.sympify(gold_norm)
        if bool(sp.simplify(pred_expr - gold_expr) == 0):
            return True, "sympy_equal"
    except Exception:
        pass
    try:
        pred_float = float(sp.N(sp.sympify(pred_norm)))
        gold_float = float(sp.N(sp.sympify(gold_norm)))
        if abs(pred_float - gold_float) <= 1e-8 * max(1.0, abs(gold_float)):
            return True, "float_close"
    except Exception:
        pass
    return False, "not_equal"


def classify_failure(row: dict) -> str | None:
    if row.get("passed"):
        return None
    pred = row.get("predicted_answer")
    method = row.get("answer_extraction_method")
    text = row.get("generated_answer") or row.get("gold_solution") or ""
    if pred is None:
        return "missing_final_answer"
    if method not in {"hash_final", "boxed"}:
        return "ambiguous_final_answer"
    if len(text.strip()) < 80:
        return "reasoning_truncation"
    if re.search(r"\\frac|/|\*|\+|-|=", text):
        return "symbolic_or_arithmetic_error"
    return "wrong_problem_model"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--text-field", type=str, default="generated_answer")
    args = parser.parse_args()

    rows = list(read_jsonl(args.input))
    passed = 0
    counts: dict[str, int] = {}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as out:
        for row in rows:
            text = row.get(args.text_field) or ""
            pred, method = extract_prediction(text)
            ok, eq_method = equivalent(pred, row.get("gold_answer"))
            if ok:
                passed += 1
            row.update({
                "predicted_answer": pred,
                "answer_extraction_method": method,
                "equivalence_method": eq_method,
                "passed": ok,
                "failure_type": None if ok else "wrong_answer",
            })
            row["error_pattern"] = classify_failure(row)
            counts[row["error_pattern"] or "passed"] = counts.get(row["error_pattern"] or "passed", 0) + 1
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "text_field": args.text_field,
        "total": len(rows),
        "passed": passed,
        "failed": len(rows) - passed,
        "accuracy": passed / len(rows) if rows else None,
        "counts": counts,
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
