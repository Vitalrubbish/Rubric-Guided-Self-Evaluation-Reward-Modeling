#!/usr/bin/env python3
"""Extract REVISED_CODE blocks from Method 2 critic/repair generations."""

from __future__ import annotations

import argparse
import ast
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


COPY_FIELDS = (
    "dataset",
    "prompt",
    "split",
    "eval_split",
    "source_split",
    "prompt_mode",
    "interface_names",
    "interface_signatures",
    "starter_code",
    "code_prompt",
    "libs",
    "input_output",
    "difficulty",
    "io_mode",
    "chosen_response_id",
    "rejected_response_id",
)

REVISED_CODE_RE = re.compile(r"(?im)^\s*REVISED[_ ]CODE\s*:\s*")
END_MARKER_RE = re.compile(r"(?im)^\s*END_REVISED_CODE\s*$")
PYTHON_START_RE = re.compile(
    r"(?m)^(?:from\s+\S+\s+import\s+|import\s+\S+|def\s+\w+\s*\(|async\s+def\s+\w+\s*\(|class\s+\w+\s*[\(:]|@\w|[A-Z_][A-Za-z0-9_]*\s*=)"
)


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


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def strip_fence(text: str) -> str:
    fenced = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()
    return "\n".join(line for line in text.splitlines() if not line.strip().startswith("```")).strip()


def split_revised_code_marker(text: str) -> tuple[str, bool]:
    match = REVISED_CODE_RE.search(text)
    if not match:
        return text, False
    return text[match.end() :].strip(), True


def split_end_marker(text: str) -> tuple[str, bool]:
    match = END_MARKER_RE.search(text)
    if match:
        return text[: match.start()].strip(), True
    if "END_REVISED_CODE" in text:
        return text.split("END_REVISED_CODE", 1)[0].strip(), True
    return text, False


def direct_code_fallback(text: str) -> tuple[str, bool]:
    fenced = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip(), True
    match = PYTHON_START_RE.search(text)
    if not match:
        return text, False
    return text[match.start() :].strip(), True


def parseable(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


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


def remove_inline_prose(text: str) -> str:
    markers = (
        "\nEND_REVISED_CODE",
        " END_REVISED_CODE",
        "\nERROR_FINDINGS:",
        " ERROR_FINDINGS:",
        "\nREVISED_CODE:",
        " REVISED_CODE:",
        "\nREVISED CODE:",
        " REVISED CODE:",
        "\nPublic task prompt:",
        " Public task prompt:",
        "\nPrevious failed code:",
        " Previous failed code:",
        "\nThe provided code",
        " The provided code",
        "\nThe original code",
        " The original code",
        "\nNo changes",
        " No changes",
        "\nHere is",
        " Here is",
    )
    cleaned = text
    for marker in markers:
        if marker in cleaned:
            cleaned = cleaned.split(marker, 1)[0]
    return cleaned.strip()


def strip_placeholders(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^Python code:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^<valid Python code only>\s*", "", text, flags=re.IGNORECASE)
    text, _ = split_revised_code_marker(text)
    return text


def longest_parseable_prefix(code: str) -> tuple[str, bool]:
    code = code.strip()
    if not code or parseable(code):
        return code, False
    lines = code.splitlines()
    for end in range(len(lines) - 1, 0, -1):
        candidate = "\n".join(lines[:end]).rstrip()
        if candidate and parseable(candidate):
            return candidate, True
    return code, False


def clean_function_call_harness(code: str, expected_names: set[str]) -> tuple[str, bool]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code, False
    drop_lines: set[int] = set()
    for node in tree.body:
        drop = False
        if isinstance(node, ast.Assert):
            drop = True
        elif isinstance(node, ast.If) and is_main_guard(node):
            drop = True
        elif isinstance(node, ast.Expr):
            # Top-level example calls, comparisons, and stray strings after the implementation.
            drop = not isinstance(node.value, ast.Constant)
        elif (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name.lower() in {"check", "test", "testing"}
            and node.name not in expected_names
        ):
            drop = True
        if drop and getattr(node, "lineno", None) and getattr(node, "end_lineno", None):
            drop_lines.update(range(node.lineno, node.end_lineno + 1))
    if not drop_lines:
        return code, False
    lines = code.splitlines()
    cleaned = "\n".join(line for index, line in enumerate(lines, start=1) if index not in drop_lines).strip()
    return (cleaned, True) if cleaned and parseable(cleaned) else (code, False)


def extract_revised_code(row: dict[str, Any]) -> tuple[str, str, list[str]]:
    text = str(row.get("generated_code") or "")
    notes: list[str] = []
    tail, has_revised_code_marker = split_revised_code_marker(text)
    if has_revised_code_marker:
        status = "ok"
    else:
        status = "missing_revised_code_marker"
        tail, used_fallback = direct_code_fallback(text)
        if used_fallback:
            status = "ok"
            notes.append("recovered_direct_code_without_revised_code_marker")
    if tail.lstrip().startswith("<valid Python code only>"):
        new_tail, skipped_marker = split_revised_code_marker(tail)
        if skipped_marker:
            tail = new_tail
            notes.append("skipped_placeholder_marker")
        else:
            tail = re.sub(r"^\s*<valid Python code only>\s*", "", tail, flags=re.IGNORECASE)
            notes.append("removed_placeholder")
    tail, found_end_marker = split_end_marker(tail)
    if found_end_marker:
        notes.append("split_marker:END_REVISED_CODE")
    for marker in ("\nEND_REVISED_CODE", "\nERROR_FINDINGS:", "\nPublic task prompt:", "\nPrevious failed code:"):
        if marker in tail:
            tail = tail.split(marker, 1)[0].strip()
            notes.append(f"split_marker:{marker.strip()}")
    code = strip_placeholders(strip_fence(tail))
    prose_trimmed = remove_inline_prose(code)
    if prose_trimmed != code:
        notes.append("trimmed_inline_prose")
        code = prose_trimmed
    code, prefix_trimmed = longest_parseable_prefix(code)
    if prefix_trimmed:
        notes.append("trimmed_parseable_prefix")
    if row.get("io_mode") == "function_call":
        expected_names = {str(name) for name in row.get("interface_names") or []}
        code, harness_cleaned = clean_function_call_harness(code, expected_names)
        if harness_cleaned:
            notes.append("cleaned_function_call_harness")
    if not code:
        status = "empty_revised_code"
    return code, status, notes


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Method 2 REVISED_CODE into verifier-ready JSONL.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    output_rows = []
    extraction_counts: Counter[str] = Counter()
    finish_counts: Counter[str] = Counter()
    for row in rows:
        raw_completion = str(row.get("generated_code") or "")
        revised_code, status, notes = extract_revised_code(row)
        extraction_counts[status] += 1
        for note in notes:
            extraction_counts[f"note:{note}"] += 1
        finish_counts[str(row.get("finish_reason"))] += 1
        out = {
            "response_id": row.get("response_id"),
            "id": row.get("id"),
            "generated_code": revised_code,
            "method2_raw_completion": raw_completion,
            "method2_extraction_status": status,
            "method2_extraction_notes": notes,
            "method2_generated_token_count": row.get("generated_token_count"),
            "model": row.get("model"),
            "adapter": row.get("adapter"),
            "sample_id": row.get("sample_id"),
            "seed": row.get("seed"),
            "temperature": row.get("temperature"),
            "top_p": row.get("top_p"),
            "repetition_penalty": row.get("repetition_penalty"),
            "max_tokens": row.get("max_tokens"),
            "finish_reason": row.get("finish_reason"),
        }
        for key in COPY_FIELDS:
            if key in row:
                out[key] = row.get(key)
        output_rows.append(out)

    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "rows": len(output_rows),
        "extraction_counts": dict(extraction_counts),
        "finish_counts": dict(finish_counts),
        "policy": "extract text after REVISED_CODE and pass only revised Python code to verifier",
    }
    write_jsonl(args.output, output_rows)
    write_json(args.summary_output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
