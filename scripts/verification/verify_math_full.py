#!/usr/bin/env python3
"""Broader verifier for MATH answers with LaTeX, sets, intervals, and multi-answer support."""

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


def strip_outer_math(text: str) -> str:
    s = (text or "").strip()
    s = s.replace("$", "")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\,", "").replace("\\!", "")
    s = s.replace("\\;", "")
    s = s.replace("\\mathrm", "")
    s = s.replace("\\text", "")
    s = re.sub(r"\\(?:degree|circ)\b|\^\\circ|\^\{\\circ\}", "", s)
    return s.strip().rstrip(".")


def replace_frac(text: str) -> str:
    s = text
    pattern = re.compile(r"\\(?:dfrac|frac)\s*{([^{}]+)}\s*{([^{}]+)}")
    last = None
    while last != s:
        last = s
        s = pattern.sub(r"(\1)/(\2)", s)
    return s


def latex_to_expr_text(text: str) -> str:
    s = strip_outer_math(text)
    if re.fullmatch(r"[-+]?\d{1,3}(,\d{3})+(?:\.\d+)?", s):
        s = s.replace(",", "")
    s = replace_frac(s)
    s = re.sub(r"\\sqrt\s*{([^{}]+)}", r"sqrt(\1)", s)
    s = s.replace("\\pi", "pi")
    s = s.replace("\\cdot", "*").replace("\\times", "*")
    s = s.replace("^", "**")
    s = s.replace("\\%", "%").replace("%", "/100")
    s = s.replace("−", "-")
    s = re.sub(r"\s+", "", s)
    return s


def trim_equation(text: str) -> str:
    s = text.strip()
    if "=" not in s:
        return s
    left, right = s.split("=", 1)
    left_clean = left.strip().lower()
    if re.fullmatch(r"[a-z][_\d]*", left_clean):
        return right.strip()
    return s


def split_top_level(text: str, separators: str = ",;") -> list[str]:
    stripped = strip_outer_math(text)
    if re.fullmatch(r"[-+]?\d{1,3}(,\d{3})+(?:\.\d+)?", stripped):
        return [stripped]
    text = re.sub(r"\s+(?:or|and)\s+", ",", text, flags=re.IGNORECASE)
    parts = []
    start = 0
    depth = 0
    for idx, char in enumerate(text):
        if char in "{[(":
            depth += 1
        elif char in "}])":
            depth = max(0, depth - 1)
        elif char in separators and depth == 0:
            parts.append(text[start:idx].strip())
            start = idx + 1
    parts.append(text[start:].strip())
    return [part for part in parts if part]


def looks_interval_like(text: str) -> bool:
    s = strip_outer_math(text)
    if "\\cup" in s or "∪" in s:
        return bool(re.search(r"[\(\[][^\n,]+,[^\n]+[\)\]]", s))
    return bool(re.fullmatch(r"[\(\[][^\n]+,[^\n]+[\)\]]", s))


def unwrap_container(text: str) -> tuple[str, str | None]:
    s = strip_outer_math(text)
    if s.startswith("\\{") and s.endswith("\\}"):
        return s[2:-2].strip(), "set"
    if s.startswith("{") and s.endswith("}"):
        return s[1:-1].strip(), "set"
    if (s.startswith("(") and s.endswith(")")) or (s.startswith("[") and s.endswith("]")):
        if "," in s:
            if looks_interval_like(s):
                return s, "interval"
            return s[1:-1].strip(), "tuple"
    return s, None


def canonical_interval(text: str) -> str:
    s = latex_to_expr_text(text).lower()
    s = s.replace("\\infty", "oo").replace("infty", "oo").replace("∞", "oo")
    s = s.replace("\\cup", "U").replace("∪", "U")
    s = s.replace("+oo", "oo")
    return s


def expand_pm(text: str) -> list[str] | None:
    s = strip_outer_math(text)
    if "\\pm" not in s and "±" not in s:
        return None
    s = s.replace("\\pm", "±")
    if s.startswith("±"):
        val = s[1:].strip()
        return [val, f"-({val})"]
    return None


def sympy_expr(text: str):
    s = trim_equation(latex_to_expr_text(text))
    return sp.sympify(s)


def single_equiv(pred: str, gold: str) -> tuple[bool, str]:
    pred_norm = trim_equation(latex_to_expr_text(pred))
    gold_norm = trim_equation(latex_to_expr_text(gold))
    if pred_norm == gold_norm:
        return True, "exact_norm"
    if "=" in pred_norm or "=" in gold_norm:
        return (pred_norm == gold_norm), "equation_string"
    try:
        if bool(sp.simplify(sympy_expr(pred) - sympy_expr(gold)) == 0):
            return True, "sympy_equal"
    except Exception:
        pass
    try:
        pred_float = float(sp.N(sympy_expr(pred)))
        gold_float = float(sp.N(sympy_expr(gold)))
        if abs(pred_float - gold_float) <= 1e-8 * max(1.0, abs(gold_float)):
            return True, "float_close"
    except Exception:
        pass
    return False, "not_equal"


def list_equiv(pred_items: list[str], gold_items: list[str], ordered: bool) -> tuple[bool, str]:
    if len(pred_items) != len(gold_items):
        return False, "length_mismatch"
    if ordered:
        for pred, gold in zip(pred_items, gold_items):
            ok, _ = single_equiv(pred, gold)
            if not ok:
                return False, "ordered_item_mismatch"
        return True, "ordered_list_equal"

    used = [False] * len(pred_items)
    for gold in gold_items:
        found = False
        for idx, pred in enumerate(pred_items):
            if used[idx]:
                continue
            ok, _ = single_equiv(pred, gold)
            if ok:
                used[idx] = True
                found = True
                break
        if not found:
            return False, "unordered_item_mismatch"
    return True, "unordered_list_equal"


def equivalent(pred: str | None, gold: str | None) -> tuple[bool, str]:
    if pred is None or gold is None:
        return False, "missing"
    pred_pm = expand_pm(pred)
    gold_pm = expand_pm(gold)
    if pred_pm is not None or gold_pm is not None:
        return list_equiv(pred_pm or split_top_level(strip_outer_math(pred)), gold_pm or split_top_level(strip_outer_math(gold)), ordered=False)

    pred_inner, pred_kind = unwrap_container(pred)
    gold_inner, gold_kind = unwrap_container(gold)
    if pred_kind == "interval" or gold_kind == "interval":
        return (canonical_interval(pred_inner) == canonical_interval(gold_inner)), "interval_string"

    pred_parts = split_top_level(pred_inner)
    gold_parts = split_top_level(gold_inner)
    if pred_kind == "set" or gold_kind == "set" or len(pred_parts) > 1 or len(gold_parts) > 1:
        ordered = pred_kind == "tuple" or gold_kind == "tuple"
        return list_equiv(pred_parts, gold_parts, ordered=ordered)

    return single_equiv(pred_inner, gold_inner)


def classify_failure(row: dict) -> str | None:
    if row.get("passed"):
        return None
    pred = row.get("predicted_answer")
    method = row.get("answer_extraction_method")
    gold_type = row.get("gold_answer_type")
    if pred is None:
        return "missing_final_answer"
    if method not in {"hash_final", "boxed"}:
        return "ambiguous_final_answer"
    if gold_type in {"set", "multi_answer", "plus_minus"}:
        return "multi_answer_error"
    if gold_type == "interval":
        return "interval_answer_error"
    if gold_type in {"radical_or_pi", "symbolic"}:
        return "symbolic_equivalence_error"
    return "numeric_or_algebra_error"


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
            pred, method = extract_prediction(row.get(args.text_field) or "")
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
