#!/usr/bin/env python3
"""Protected deterministic cleanup that avoids modifying already-passing outputs."""

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


def should_attempt_revision(row: dict, only_failed: bool, allowed_failure_types: set[str] | None) -> tuple[bool, str | None]:
    if only_failed and row.get("passed"):
        return False, "already_passed"
    failure_type = row.get("failure_type")
    if allowed_failure_types is not None and failure_type not in allowed_failure_types:
        return False, f"failure_type_not_allowed:{failure_type}"
    return True, None


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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--only-failed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--allowed-failure-types",
        type=str,
        default=None,
        help="Comma-separated allowlist such as syntax_error,runtime_error. Defaults to all failed rows.",
    )
    args = parser.parse_args()

    allowed = None
    if args.allowed_failure_types:
        allowed = {item.strip() for item in args.allowed_failure_types.split(",") if item.strip()}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    attempted = 0
    revised_count = 0
    skipped_passed = 0
    skipped_other = 0
    with args.output.open("w", encoding="utf-8") as f:
        for row in read_jsonl(args.labeled):
            total += 1
            out = dict(row)
            should_revise, reason = should_attempt_revision(row, args.only_failed, allowed)
            out["original_generated_code"] = row.get("generated_code")
            out["revision_edits"] = []
            out["revision_method"] = "protected_deterministic_cleanup_v1"
            out["revision_skipped_reason"] = reason
            if should_revise:
                attempted += 1
                revised, edits = revise_code(row)
                out["generated_code"] = revised
                out["revision_edits"] = edits
                out["revision_skipped_reason"] = None
                if edits:
                    revised_count += 1
            elif reason == "already_passed":
                skipped_passed += 1
            else:
                skipped_other += 1
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    summary = {
        "input": str(args.labeled),
        "output": str(args.output),
        "total": total,
        "attempted": attempted,
        "edited": revised_count,
        "skipped_passed": skipped_passed,
        "skipped_other": skipped_other,
        "only_failed": args.only_failed,
        "allowed_failure_types": sorted(allowed) if allowed else None,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
