#!/usr/bin/env python3
"""Build a Method 2 iterative SFT dataset from verifier-passing self-generated repairs."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


REVISED_CODE_RE = re.compile(r"(?im)^\s*REVISED[_ ]CODE\s*:\s*")
ERROR_FINDINGS_RE = re.compile(r"(?im)^\s*ERROR_FINDINGS\s*:\s*")


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


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def parseable(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def normalize_code(code: Any) -> str:
    return "\n".join(str(code or "").strip().splitlines()).strip()


def code_from_completion(completion: str) -> str:
    match = REVISED_CODE_RE.search(completion)
    if not match:
        return normalize_code(completion)
    return normalize_code(completion[match.end() :])


def clean_finding(line: str) -> str:
    line = line.strip()
    line = re.sub(r"^[-*\d.)\s]+", "", line).strip()
    return line.rstrip(".") + "." if line and not line.endswith((".", "!", "?")) else line


def extract_findings(raw_completion: str, count: int) -> list[str]:
    findings_text = ""
    error_match = ERROR_FINDINGS_RE.search(raw_completion)
    revised_match = REVISED_CODE_RE.search(raw_completion)
    if error_match and revised_match and error_match.end() <= revised_match.start():
        findings_text = raw_completion[error_match.end() : revised_match.start()]
    findings: list[str] = []
    for line in findings_text.splitlines():
        cleaned = clean_finding(line)
        if not cleaned:
            continue
        if cleaned in {"...", "None.", "None found.", "No errors found."}:
            continue
        if cleaned not in findings:
            findings.append(cleaned)
        if len(findings) >= count:
            break
    fallback = [
        "The previous solution failed the public task behavior for this APPS prompt.",
        "The revised implementation preserves the requested interface and fixes the observed failure.",
    ]
    for item in fallback:
        if len(findings) >= count:
            break
        findings.append(item)
    return findings[:count]


def canonical_completion(findings: list[str], code: str) -> str:
    bullets = "\n".join(f"- {finding}" for finding in findings)
    return f"ERROR_FINDINGS:\n{bullets}\nREVISED_CODE:\n{code.strip()}"


def rank_generated(row: dict[str, Any]) -> tuple[int, int, int, str]:
    notes = row.get("method2_extraction_notes") or []
    note_penalty = len(notes)
    finish_penalty = 0 if row.get("finish_reason") == "stop" else 1
    token_count = int(row.get("method2_generated_token_count") or row.get("generated_token_count") or 0)
    return (finish_penalty, note_penalty, token_count, str(row.get("response_id") or ""))


def generated_token_count(row: dict[str, Any]) -> int:
    return int(row.get("method2_generated_token_count") or row.get("generated_token_count") or 0)


def build_generated_row(
    base_row: dict[str, Any],
    generated_row: dict[str, Any],
    findings_count: int,
    source_tag: str,
) -> dict[str, Any] | None:
    code = normalize_code(generated_row.get("generated_code"))
    if not code or not parseable(code):
        return None
    raw_completion = str(generated_row.get("method2_raw_completion") or "")
    findings = extract_findings(raw_completion, findings_count)
    response_id = str(generated_row.get("response_id") or stable_hash(raw_completion + code))
    generated_id = f"{base_row['id']}__{source_tag}_{stable_hash(response_id + code)}"
    metadata = dict(base_row.get("metadata") or {})
    metadata.update(
        {
            "source": source_tag,
            "base_sft_id": base_row.get("id"),
            "generated_response_id": response_id,
            "generated_sample_id": generated_row.get("sample_id"),
            "generated_finish_reason": generated_row.get("finish_reason"),
            "generated_token_count": generated_row.get("method2_generated_token_count"),
            "generated_extraction_notes": generated_row.get("method2_extraction_notes") or [],
            "verifier_source_mode": generated_row.get("source_mode"),
        }
    )
    out = dict(base_row)
    out.update(
        {
            "id": generated_id,
            "split": "train",
            "completion": canonical_completion(findings, code),
            "source": source_tag,
            "metadata": metadata,
        }
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Method 2 v0.4 iterative SFT data.")
    parser.add_argument("--base-sft", type=Path, required=True)
    parser.add_argument("--generated-labeled", type=Path, required=True)
    parser.add_argument("--sft-output", type=Path, required=True)
    parser.add_argument("--accepted-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--max-generated-per-id", type=int, default=1)
    parser.add_argument("--max-generated-total", type=int, default=None)
    parser.add_argument(
        "--require-finish-reason",
        action="append",
        default=[],
        help="Only keep generated repairs with this finish_reason. May be passed more than once.",
    )
    parser.add_argument("--max-generated-tokens", type=int, default=None)
    parser.add_argument("--max-extraction-notes", type=int, default=None)
    parser.add_argument("--findings-count", type=int, default=2)
    parser.add_argument("--source-tag", default="method2_v0_4_self_generated_pass")
    parser.add_argument("--allow-empty-generated", action="store_true")
    args = parser.parse_args()

    if args.max_generated_per_id < 1:
        raise ValueError("--max-generated-per-id must be at least 1")
    if args.findings_count < 1:
        raise ValueError("--findings-count must be at least 1")
    required_finish_reasons = {str(value) for value in args.require_finish_reason}
    if args.max_generated_tokens is not None and args.max_generated_tokens < 1:
        raise ValueError("--max-generated-tokens must be positive")
    if args.max_extraction_notes is not None and args.max_extraction_notes < 0:
        raise ValueError("--max-extraction-notes cannot be negative")

    base_rows = read_jsonl(args.base_sft)
    generated_rows = read_jsonl(args.generated_labeled)
    base_by_id = {str(row.get("id")): row for row in base_rows}

    counts: Counter[str] = Counter()
    candidates_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    base_code_keys = {
        (str(row.get("id")), code_from_completion(str(row.get("completion") or "")))
        for row in base_rows
        if row.get("split") == "train"
    }
    seen_generated_code: set[tuple[str, str]] = set()

    for row in generated_rows:
        counts["generated_input"] += 1
        base_id = str(row.get("id") or "")
        if base_id not in base_by_id:
            counts["skipped:missing_base_id"] += 1
            continue
        if base_by_id[base_id].get("split") != "train":
            counts["skipped:base_not_train"] += 1
            continue
        if not row.get("passed"):
            counts[f"skipped:not_passed:{row.get('failure_type') or 'unknown'}"] += 1
            continue
        if row.get("method2_extraction_status") != "ok":
            counts[f"skipped:bad_extraction:{row.get('method2_extraction_status') or 'unknown'}"] += 1
            continue
        finish_reason = str(row.get("finish_reason") or "unknown")
        if required_finish_reasons and finish_reason not in required_finish_reasons:
            counts[f"skipped:finish_reason:{finish_reason}"] += 1
            continue
        token_count = generated_token_count(row)
        if args.max_generated_tokens is not None and token_count > args.max_generated_tokens:
            counts["skipped:generated_tokens_too_high"] += 1
            continue
        extraction_notes = row.get("method2_extraction_notes") or []
        if args.max_extraction_notes is not None and len(extraction_notes) > args.max_extraction_notes:
            counts["skipped:too_many_extraction_notes"] += 1
            continue
        code = normalize_code(row.get("generated_code"))
        if not code:
            counts["skipped:empty_code"] += 1
            continue
        if not parseable(code):
            counts["skipped:not_parseable"] += 1
            continue
        if (base_id, code) in base_code_keys:
            counts["skipped:duplicate_base_code"] += 1
            continue
        if (base_id, code) in seen_generated_code:
            counts["skipped:duplicate_generated_code"] += 1
            continue
        seen_generated_code.add((base_id, code))
        candidates_by_id[base_id].append(row)
        counts["candidate_accepted_pre_cap"] += 1

    generated_sft_rows: list[dict[str, Any]] = []
    accepted_labeled_rows: list[dict[str, Any]] = []
    for base_id in sorted(candidates_by_id):
        selected = sorted(candidates_by_id[base_id], key=rank_generated)[: args.max_generated_per_id]
        for row in selected:
            built = build_generated_row(base_by_id[base_id], row, args.findings_count, args.source_tag)
            if not built:
                counts["skipped:build_failed"] += 1
                continue
            generated_sft_rows.append(built)
            accepted_labeled_rows.append(row)
            counts["generated_sft_selected"] += 1
            if args.max_generated_total is not None and len(generated_sft_rows) >= args.max_generated_total:
                break
        if args.max_generated_total is not None and len(generated_sft_rows) >= args.max_generated_total:
            break

    if not generated_sft_rows and not args.allow_empty_generated:
        raise SystemExit("no verifier-passing generated rows selected; run generation/verification first or pass --allow-empty-generated")

    combined_rows = [*base_rows, *generated_sft_rows]
    combined_rows.sort(key=lambda row: (str(row.get("split")), str(row.get("source")), str(row.get("id"))))
    write_jsonl(args.sft_output, combined_rows)
    write_jsonl(args.accepted_output, accepted_labeled_rows)

    summary = {
        "base_sft": str(args.base_sft),
        "base_sft_sha256": sha256_file(args.base_sft),
        "generated_labeled": str(args.generated_labeled),
        "generated_labeled_sha256": sha256_file(args.generated_labeled),
        "sft_output": str(args.sft_output),
        "sft_output_sha256": sha256_file(args.sft_output),
        "accepted_output": str(args.accepted_output),
        "accepted_output_sha256": sha256_file(args.accepted_output),
        "base_rows": len(base_rows),
        "generated_selected_rows": len(generated_sft_rows),
        "combined_rows": len(combined_rows),
        "unique_generated_problem_count": len({row.get("id") for row in accepted_labeled_rows}),
        "split_counts": dict(Counter(str(row.get("split") or "unknown") for row in combined_rows)),
        "source_counts": dict(Counter(str(row.get("source") or "unknown") for row in combined_rows)),
        "generated_finish_counts": dict(Counter(str(row.get("finish_reason") or "unknown") for row in accepted_labeled_rows)),
        "generated_io_mode_counts": dict(Counter(str(row.get("io_mode") or "unknown") for row in accepted_labeled_rows)),
        "max_generated_per_id": args.max_generated_per_id,
        "max_generated_total": args.max_generated_total,
        "required_finish_reasons": sorted(required_finish_reasons),
        "max_generated_tokens": args.max_generated_tokens,
        "max_extraction_notes": args.max_extraction_notes,
        "counts": dict(counts),
        "policy": {
            "route": "Method 2 iterative self-play repair",
            "base": "v0.3 no-end-marker SFT",
            "augment": "only verifier-passing self-generated train repairs",
            "validation": "base validation rows are preserved; generated repairs are train-only",
        },
    }
    write_json(args.summary_output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
