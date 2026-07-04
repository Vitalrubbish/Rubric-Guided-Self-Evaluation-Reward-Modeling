#!/usr/bin/env python3
"""Deterministic rubric-guided cleanup baseline for failed code outputs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def extract_fenced(text: str) -> str:
    fenced = re.search(r"```(?:python)?\s*(.*?)```", text or "", flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip("\n\r")
    return (text or "").strip("\n\r")


def truncate_duplicate_functions(code: str) -> tuple[str, bool]:
    patterns = [
        r"(\n\s*return[^\n]*?)\s+def\s+",
        r"(\n\s*return[^\n]*?)\s+class\s+",
        r"(\))\s+def\s+",
    ]
    for pattern in patterns:
        match = re.search(pattern, code)
        if match:
            return code[: match.end(1)].rstrip() + "\n", True
    return code, False


def drop_trailing_prose(code: str) -> tuple[str, bool]:
    lines = code.splitlines()
    kept = []
    changed = False
    prose_prefixes = (
        "# This solution",
        "This solution",
        "Explanation",
        "Note:",
        "The function",
        "Output:",
    )
    for line in lines:
        if line.strip().startswith(prose_prefixes):
            changed = True
            break
        kept.append(line)
    return "\n".join(kept).rstrip() + ("\n" if kept else ""), changed


def remove_print_examples(code: str) -> tuple[str, bool]:
    lines = code.splitlines()
    new_lines = [line for line in lines if not line.lstrip().startswith("print(")]
    return "\n".join(new_lines).rstrip() + ("\n" if new_lines else ""), len(new_lines) != len(lines)


def revise_code(row: dict) -> tuple[str, list[str]]:
    code = extract_fenced(row.get("generated_code", ""))
    edits = []
    revised, changed = truncate_duplicate_functions(code)
    if changed:
        edits.append("truncate_duplicate_function_body")
    revised, changed = drop_trailing_prose(revised)
    if changed:
        edits.append("drop_trailing_prose")
    revised, changed = remove_print_examples(revised)
    if changed:
        edits.append("remove_print_examples")
    return revised.strip("\n\r"), edits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labeled", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/responses/coding_all_qwen25_vllm_k1_revised.jsonl"))
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    revised_count = 0
    with args.output.open("w", encoding="utf-8") as f:
        for row in read_jsonl(args.labeled):
            revised, edits = revise_code(row)
            if edits:
                revised_count += 1
            out = dict(row)
            out["original_generated_code"] = row.get("generated_code")
            out["generated_code"] = revised
            out["revision_edits"] = edits
            out["revision_method"] = "deterministic_rubric_guided_cleanup_v1"
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            total += 1
    print(f"wrote {total} revised responses to {args.output}; edited={revised_count}")


if __name__ == "__main__":
    main()
