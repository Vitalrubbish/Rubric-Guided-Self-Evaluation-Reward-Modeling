#!/usr/bin/env python3
"""Build Method 2 self-play critic/repair bootstrap datasets.

Method 2 is not plain solver imitation. Each row trains or evaluates the model
on: inspect a failed answer, state concrete error findings, then produce a
revised implementation. Verifier-passing revisions can also form conservative
preferences for the same critic/repair prompt.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
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


def stable_split(row_id: str, validation_percent: int) -> str:
    return "validation" if stable_hash(row_id) % 100 < validation_percent else "train"


def normalize_code(code: Any) -> str:
    return "\n".join(str(code or "").strip().splitlines()).strip()


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


def clean_function_call_code(code: str) -> tuple[str, list[str]]:
    """Remove public/demo harnesses from function-call repaired code."""
    original = code.strip()
    try:
        tree = ast.parse(original)
    except SyntaxError:
        return original, ["not_parseable"]

    drop_lines: set[int] = set()
    reasons: list[str] = []
    for node in tree.body:
        reason = None
        if isinstance(node, ast.Assert):
            reason = "top_level_assert"
        elif isinstance(node, ast.If) and is_main_guard(node):
            reason = "main_guard"
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.lower() in {"check", "test", "testing"}:
            reason = f"demo_function:{node.name}"
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            reason = "top_level_call"
        if reason and getattr(node, "lineno", None) and getattr(node, "end_lineno", None):
            reasons.append(reason)
            drop_lines.update(range(node.lineno, node.end_lineno + 1))

    if not drop_lines:
        return original, []
    lines = original.splitlines()
    cleaned = "\n".join(line for index, line in enumerate(lines, start=1) if index not in drop_lines).strip()
    if cleaned and parseable(cleaned):
        return cleaned, reasons
    return original, ["cleanup_failed", *reasons]


def default_findings(row: dict[str, Any], source_kind: str) -> list[str]:
    failure_type = str(row.get("failure_type") or row.get("rejected_failure_type") or "").strip()
    error_pattern = str(row.get("error_pattern") or "").strip()
    selection_reason = str(row.get("selection_reason") or "").strip()
    findings = []
    if failure_type:
        findings.append(f"The previous solution failed as {failure_type}; identify the public-task behavior that caused it.")
    if error_pattern:
        findings.append(f"The visible failure pattern is {error_pattern}; remove that pattern while preserving the required interface.")
    if selection_reason:
        findings.append(f"The repair source was selected as {selection_reason}; verify the fix against public examples and edge cases.")
    if source_kind == "apps_repair_self_play":
        findings.append(
            "Use the public task, starting code, callable interface, and visible failed code to justify the revised APPS solution."
        )
    if not findings:
        findings.append("The previous solution failed the public task contract; repair the concrete bug visible from the prompt and code.")
    return findings


def normalize_findings(value: Any, row: dict[str, Any], source_kind: str) -> list[str]:
    if isinstance(value, list):
        findings = [str(item).strip() for item in value if str(item).strip()]
        return findings or default_findings(row, source_kind)
    text = str(value or "").strip()
    if not text:
        return default_findings(row, source_kind)
    return [line.strip("- ").strip() for line in text.splitlines() if line.strip()] or default_findings(row, source_kind)


def is_train_source(row: dict[str, Any]) -> bool:
    split = str(row.get("split") or "")
    return split.endswith("/train") or split.endswith("_train") or split == "train" or "/train/" in split


def method2_prompt(
    public_prompt: str,
    failed_code: str,
    end_marker: str | None = None,
    response_prefix: str | None = None,
) -> str:
    end_instruction = ""
    if end_marker:
        end_instruction = f"{end_marker}\n"
    response_prefix_text = ""
    if response_prefix:
        response_prefix_text = f"\n\n{response_prefix.strip()}\n"
    return (
        "You are a Python self-play critic and repair model.\n"
        "Use only the public task, public tests/examples, public interface, and visible failed code. "
        "Do not rely on hidden tests or private verifier messages.\n"
        "First identify concrete errors in the previous code. Then write one corrected implementation.\n"
        "Begin your answer with ERROR_FINDINGS: exactly. Do not write prefaces, placeholders, Markdown, or extra sections.\n"
        "Write exactly two short bullet points under ERROR_FINDINGS.\n"
        "Write REVISED_CODE: exactly once, then Python code only. Do not repeat the task prompt or previous failed code.\n"
        f"{'End the answer with exactly ' + end_marker + ' and no text after it.' if end_marker else 'Stop immediately after the final implementation statement.'}\n"
        "Return exactly this format:\n"
        "ERROR_FINDINGS:\n"
        "- ...\n"
        "REVISED_CODE:\n"
        "<valid Python code only>\n"
        f"{end_instruction}"
        f"Public task prompt:\n{public_prompt.strip()}\n\n"
        f"Previous failed code:\n{failed_code.strip()}\n"
        f"{response_prefix_text}"
    )


def method2_completion(findings: list[str], revised_code: str, end_marker: str | None = None) -> str:
    bullet_lines = "\n".join(f"- {finding}" for finding in findings)
    completion = f"ERROR_FINDINGS:\n{bullet_lines}\nREVISED_CODE:\n{revised_code.strip()}"
    if end_marker:
        completion = f"{completion}\n{end_marker}"
    return completion


def convert_pair(
    row: dict[str, Any],
    source_file: Path,
    source_kind: str,
    index: int,
    args: argparse.Namespace,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, str | None]:
    public_prompt = str(row.get("prompt") or "").strip()
    rejected = normalize_code(row.get("response_a") or row.get("rejected"))
    chosen = normalize_code(row.get("response_b") or row.get("chosen"))
    if not public_prompt or not rejected or not chosen:
        return None, None, "missing_prompt_or_code"
    if rejected == chosen:
        return None, None, "identical_code"
    cleanup_reasons: list[str] = []
    if row.get("dataset") == "apps" and row.get("io_mode") == "function_call":
        chosen, cleanup_reasons = clean_function_call_code(chosen)
    if args.max_revised_code_chars is not None and len(chosen) > args.max_revised_code_chars:
        return None, None, "revised_code_too_long"

    findings = normalize_findings(row.get("critique"), row, source_kind)
    row_id = str(row.get("id") or f"{source_file.stem}:{index}")
    record_id = f"{row_id}__method2_{source_kind}_{index:05d}"
    prompt = method2_prompt(public_prompt, rejected, args.end_marker, args.response_prefix)
    chosen_completion = method2_completion(findings, chosen, args.end_marker)
    rejected_completion = method2_completion(["No successful correction was made."], rejected, args.end_marker)
    split = stable_split(record_id, args.validation_percent)

    common_metadata = {
        "problem_id": row.get("id"),
        "dataset": row.get("dataset"),
        "source_split": row.get("split"),
        "source_file": str(source_file),
        "source_kind": source_kind,
        "self_discovery_source": row.get("self_discovery_source"),
        "llm_critic_generated": bool(row.get("llm_critic_generated")),
        "failure_type": row.get("failure_type"),
        "error_pattern": row.get("error_pattern"),
        "rubric_version": row.get("rubric_version"),
        "selection_reason": row.get("selection_reason"),
        "critic_pass_probability": row.get("critic_pass_probability"),
        "cleanup_reasons": cleanup_reasons,
        "end_marker": args.end_marker,
    }
    sft_row = {
        "id": record_id,
        "problem_id": row_id,
        "dataset": row.get("dataset"),
        "split": split,
        "task_type": "method2_self_play_critic_repair",
        "prompt": prompt,
        "completion": chosen_completion,
        "source": source_kind,
        "interface_names": row.get("interface_names") or [],
        "interface_signatures": row.get("interface_signatures") or [],
        "starter_code": row.get("starter_code"),
        "input_output": row.get("input_output"),
        "difficulty": row.get("difficulty"),
        "io_mode": row.get("io_mode"),
        "chosen_response_id": row.get("chosen_response_id"),
        "rejected_response_id": row.get("rejected_response_id"),
        "metadata": common_metadata,
    }
    dpo_row = {
        "pair_id": record_id,
        "id": row_id,
        "dataset": row.get("dataset"),
        "split": split,
        "difficulty": row.get("difficulty"),
        "io_mode": row.get("io_mode"),
        "task_type": "method2_self_play_critic_repair_preference",
        "prompt": prompt,
        "chosen": chosen_completion,
        "rejected": rejected_completion,
        "chosen_code": chosen,
        "rejected_code": rejected,
        "interface_names": row.get("interface_names") or [],
        "interface_signatures": row.get("interface_signatures") or [],
        "starter_code": row.get("starter_code"),
        "input_output": row.get("input_output"),
        "preference": "failed_code < self_play_revised_code",
        "source": source_kind,
        "metadata": common_metadata,
    }
    return sft_row, dpo_row, None


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Method 2 self-play bootstrap SFT and DPO data.")
    parser.add_argument(
        "--llm-critic-pairs",
        type=Path,
        action="append",
        default=[],
    )
    parser.add_argument(
        "--proxy-pairs",
        type=Path,
        action="append",
        default=[],
    )
    parser.add_argument("--apps-repair-pairs", type=Path, action="append", default=[])
    parser.add_argument("--apps-repair-labeled", type=Path, action="append", default=[])
    parser.add_argument("--sft-output", type=Path, default=Path("data/sft/method2_self_play_critic_repair_v0.jsonl"))
    parser.add_argument(
        "--dpo-output",
        type=Path,
        default=Path("data/preferences/method2_self_play_critic_repair_pairs_v0.jsonl"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("data/self_play/method2_self_play_bootstrap_v0_summary.json"),
    )
    parser.add_argument("--validation-percent", type=int, default=20)
    parser.add_argument("--end-marker", default=None)
    parser.add_argument("--response-prefix", default=None)
    parser.add_argument("--max-revised-code-chars", type=int, default=None)
    parser.add_argument("--train-source-only", action="store_true", default=True)
    parser.add_argument("--include-proxy", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if not 1 <= args.validation_percent <= 50:
        raise ValueError("--validation-percent must be in [1, 50]")
    if not args.llm_critic_pairs and not args.proxy_pairs and not args.apps_repair_pairs:
        args.llm_critic_pairs = [Path("data/self_play/llm_critic_pairs_mbpp_train_logic_n20_k5.jsonl")]
        args.proxy_pairs = [Path("data/self_play/self_play_pairs_from_protected_revision.jsonl")]

    sources: list[tuple[str, Path]] = [("llm_critic_self_play", path) for path in args.llm_critic_pairs]
    if args.include_proxy:
        sources.extend(("protected_proxy_self_play", path) for path in args.proxy_pairs)
    sources.extend(("apps_repair_self_play", path) for path in args.apps_repair_pairs)

    input_sha256: dict[str, str] = {}
    apps_labeled_by_response: dict[str, dict[str, Any]] = {}
    for path in args.apps_repair_labeled:
        if path.exists():
            input_sha256[str(path)] = sha256_file(path)
            for row in read_jsonl(path):
                response_id = str(row.get("response_id") or "")
                if response_id:
                    apps_labeled_by_response[response_id] = row

    sft_rows: list[dict[str, Any]] = []
    dpo_rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    failure_counts: Counter[str] = Counter()
    seen: set[tuple[str, str, str]] = set()

    for source_kind, path in sources:
        if not path.exists():
            counts[f"missing_source:{path}"] += 1
            continue
        input_sha256[str(path)] = sha256_file(path)
        for index, row in enumerate(read_jsonl(path), start=1):
            counts[f"input:{source_kind}"] += 1
            if source_kind == "apps_repair_self_play":
                labeled = apps_labeled_by_response.get(str(row.get("chosen_response_id") or ""))
                if labeled:
                    row = {
                        **row,
                        "interface_names": labeled.get("interface_names") or [],
                        "interface_signatures": labeled.get("interface_signatures") or [],
                        "starter_code": labeled.get("starter_code"),
                        "input_output": labeled.get("input_output"),
                    }
                    counts["apps_labeled_metadata_joined"] += 1
            if args.train_source_only and not is_train_source(row):
                counts[f"skipped_non_train_source:{source_kind}"] += 1
                continue
            sft_row, dpo_row, reason = convert_pair(row, path, source_kind, index, args)
            if reason:
                counts[f"skipped:{reason}"] += 1
                continue
            assert sft_row is not None and dpo_row is not None
            key = (str(sft_row["problem_id"]), str(dpo_row["chosen_code"]), str(dpo_row["rejected_code"]))
            if key in seen:
                counts["skipped:duplicate_pair"] += 1
                continue
            seen.add(key)
            sft_rows.append(sft_row)
            dpo_rows.append(dpo_row)
            source_counts[source_kind] += 1
            failure_counts[str((sft_row.get("metadata") or {}).get("failure_type") or "unknown")] += 1

    sft_rows.sort(key=lambda row: (row["split"], row["source"], row["id"]))
    dpo_rows.sort(key=lambda row: (row["split"], row["source"], row["pair_id"]))
    write_jsonl(args.sft_output, sft_rows)
    write_jsonl(args.dpo_output, dpo_rows)

    summary = {
        "sft_output": str(args.sft_output),
        "sft_output_sha256": sha256_file(args.sft_output),
        "dpo_output": str(args.dpo_output),
        "dpo_output_sha256": sha256_file(args.dpo_output),
        "input_sha256": input_sha256,
        "rows": len(sft_rows),
        "preference_pairs": len(dpo_rows),
        "unique_problem_count": len({row["problem_id"] for row in sft_rows}),
        "split_counts": dict(Counter(row["split"] for row in sft_rows)),
        "source_counts": dict(source_counts),
        "failure_type_counts": dict(failure_counts),
        "io_mode_counts": dict(Counter(str(row.get("io_mode") or "unknown") for row in sft_rows)),
        "counts": dict(counts),
        "policy": {
            "route": "Method 2: Self-Play Error Discovery",
            "unit": "failed response -> model critique -> revised response",
            "train_source_only": args.train_source_only,
            "include_proxy": args.include_proxy,
            "proxy_role": "bootstrap only; deterministic protected edits are tracked separately from real LLM critic rows",
            "next_gate": "repair-rate and harmful-edit preservation before any solver DPO",
        },
    }
    write_json(args.summary_output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
