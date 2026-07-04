#!/usr/bin/env python3
"""Build self-play-style error discovery pairs from protected revisions.

The generated pairs are a verifier-grounded proxy for the assignment's
"A -> find error -> B -> A<B preference" setting. The critique is derived from
the observed failure metadata and protected revision edit type, not from a
separate LLM critic.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


EDIT_CRITIQUES = {
    "truncate_duplicate_function_body": (
        "The response contains duplicated or concatenated solution fragments after the first "
        "valid implementation. Keep one parseable implementation that matches the required interface."
    ),
    "remove_print_examples": (
        "The response includes executable examples or print calls after the solution. Remove benchmark-"
        "unrelated execution so the submitted code only defines the requested function/classes."
    ),
    "drop_trailing_prose": (
        "The response mixes code with trailing prose. Keep only runnable Python code for the verifier."
    ),
}


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def clean_code(text: str) -> str:
    text = text or ""
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            candidate = part.strip("\n\r")
            if candidate.lstrip().lower().startswith("python"):
                candidate = candidate.lstrip()[len("python") :].strip("\n\r")
            if "def " in candidate or "class " in candidate:
                return candidate.strip("\n\r")
    return text.strip("\n\r")


def split_from_id(row: dict) -> str:
    row_id = row.get("id", "")
    if row.get("split"):
        return f"{row.get('dataset')}/{row['split']}"
    parts = row_id.split("/")
    if len(parts) >= 3 and parts[0] == "mbpp":
        return f"mbpp/{parts[1]}"
    if row.get("dataset") == "humanevalplus" or row_id.startswith("humanevalplus/"):
        return "humanevalplus/test"
    return row.get("dataset") or "unknown"


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def safe_rate(num: int, den: int) -> float:
    return round(num / den, 6) if den else 0.0


def build_critique(failure: dict, revised: dict) -> str:
    edits = revised.get("revision_edits") or []
    edit_notes = [EDIT_CRITIQUES.get(edit, f"The protected reviser applied `{edit}`.") for edit in edits]
    pattern = failure.get("error_pattern") or "unknown_error_pattern"
    failure_type = failure.get("failure_type") or "unknown_failure_type"
    error = failure.get("error")
    parts = [
        f"Detected failure type: {failure_type}.",
        f"Detected error pattern: {pattern}.",
    ]
    if error:
        parts.append(f"Verifier error message: {error}")
    parts.extend(edit_notes)
    return " ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-labeled", type=Path, required=True)
    parser.add_argument("--protected-revised-labeled", type=Path, required=True)
    parser.add_argument("--failures", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/self_play/self_play_pairs_from_protected_revision.jsonl"))
    parser.add_argument("--metrics-output", type=Path, default=Path("data/self_play/self_play_error_discovery_metrics.json"))
    parser.add_argument("--md-output", type=Path, default=Path("docs/self_play_error_discovery.md"))
    args = parser.parse_args()

    original = {row["id"]: row for row in read_jsonl(args.original_labeled)}
    revised = {row["id"]: row for row in read_jsonl(args.protected_revised_labeled)}
    failures = {row["id"]: row for row in read_jsonl(args.failures)}

    pairs = []
    transitions = Counter()
    edit_counts = Counter()
    edit_success = Counter()
    edit_failure = Counter()
    success_by_split = Counter()
    success_by_failure_type = Counter()
    success_by_pattern = Counter()
    remaining_by_failure_type = Counter()
    remaining_by_pattern = Counter()
    edited_failed = 0
    edited_passed = 0

    for item_id, orig in sorted(original.items()):
        rev = revised.get(item_id)
        if not rev:
            continue

        before = "pass" if orig.get("passed") else "fail"
        after = "pass" if rev.get("passed") else "fail"
        transitions[f"{before}_to_{after}"] += 1

        edits = rev.get("revision_edits") or []
        if orig.get("passed") and edits:
            edited_passed += 1
        if not orig.get("passed") and edits:
            edited_failed += 1
            for edit in edits:
                edit_counts[edit] += 1
                if rev.get("passed"):
                    edit_success[edit] += 1
                else:
                    edit_failure[edit] += 1

        failure = failures.get(item_id, orig)
        failure_type = failure.get("failure_type") or orig.get("failure_type") or "unknown"
        pattern = failure.get("error_pattern") or "unknown"

        if not orig.get("passed") and rev.get("passed"):
            split = split_from_id(orig)
            success_by_split[split] += 1
            success_by_failure_type[failure_type] += 1
            success_by_pattern[pattern] += 1

            chosen = clean_code(rev.get("generated_code", ""))
            rejected = clean_code(orig.get("generated_code", ""))
            if not chosen or not rejected:
                continue

            pairs.append(
                {
                    "id": item_id,
                    "dataset": orig.get("dataset"),
                    "split": split,
                    "prompt": orig.get("prompt"),
                    "response_a": rejected,
                    "critique": build_critique(failure, rev),
                    "response_b": chosen,
                    "preference": "A < B",
                    "chosen": chosen,
                    "rejected": rejected,
                    "chosen_source": "protected_rule_revised_success_output",
                    "rejected_source": "qwen25_k1_failed_output",
                    "self_discovery_source": "protected_rule_revision_proxy",
                    "llm_critic_generated": False,
                    "failure_type": failure_type,
                    "error_pattern": pattern,
                    "cluster_id": failure.get("cluster_id"),
                    "cluster_name": failure.get("cluster_name"),
                    "verifier_error": failure.get("error") or orig.get("error"),
                    "revision_method": rev.get("revision_method"),
                    "revision_edits": edits,
                    "rubric_version": "auto_rubric_refined_coding_v1",
                }
            )
        elif not orig.get("passed") and not rev.get("passed"):
            remaining_by_failure_type[failure_type] += 1
            remaining_by_pattern[pattern] += 1

    orig_passed = transitions["pass_to_pass"] + transitions["pass_to_fail"]
    orig_failed = transitions["fail_to_pass"] + transitions["fail_to_fail"]
    protected_passed = transitions["pass_to_pass"] + transitions["fail_to_pass"]
    protected_failed = transitions["pass_to_fail"] + transitions["fail_to_fail"]

    edit_stats = {}
    for edit, count in sorted(edit_counts.items()):
        edit_stats[edit] = {
            "attempted_on_failed": count,
            "successful_repairs": edit_success[edit],
            "unsuccessful_repairs": edit_failure[edit],
            "success_rate": safe_rate(edit_success[edit], count),
        }

    metrics = {
        "source": {
            "original_labeled": str(args.original_labeled),
            "protected_revised_labeled": str(args.protected_revised_labeled),
            "failures": str(args.failures),
            "output_pairs": str(args.output),
        },
        "type": "verifier_grounded_self_play_proxy",
        "caveat": (
            "Critiques are derived from failure metadata and protected revision rules, not generated "
            "by an LLM critic. Use this as a bootstrap dataset for Method 2."
        ),
        "counts": {
            "total": len(original),
            "original_passed": orig_passed,
            "original_failed": orig_failed,
            "protected_passed": protected_passed,
            "protected_failed": protected_failed,
            "self_play_pairs": len(pairs),
            "failed_rows_with_edits": edited_failed,
            "passed_rows_with_edits": edited_passed,
            "untouched_failures": orig_failed - edited_failed,
        },
        "transitions": dict(transitions),
        "metrics": {
            "detection_coverage_on_failures": safe_rate(edited_failed, orig_failed),
            "repair_precision_given_edit": safe_rate(transitions["fail_to_pass"], edited_failed),
            "repair_recall_all_failures": safe_rate(transitions["fail_to_pass"], orig_failed),
            "pass_preservation_rate": safe_rate(transitions["pass_to_pass"], orig_passed),
            "harmful_edit_rate_on_initially_passed": safe_rate(transitions["pass_to_fail"], orig_passed),
        },
        "success_by_split": dict(success_by_split),
        "success_by_failure_type": dict(success_by_failure_type),
        "success_by_error_pattern": dict(success_by_pattern),
        "remaining_by_failure_type": dict(remaining_by_failure_type),
        "remaining_by_error_pattern": dict(remaining_by_pattern),
        "edit_stats": edit_stats,
        "analysis": {
            "proxy_self_discoverable": [
                "Formatting/syntax artifacts caused by duplicated function bodies",
                "Benchmark contract noise such as print examples after the solution",
                "Trailing prose mixed into code",
            ],
            "likely_external_signal_needed": [
                "Deep functional logic errors that still fail after cleanup",
                "Runtime errors requiring semantic API or type reasoning",
                "Timeouts/non-terminating algorithms",
            ],
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in pairs:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    md = [
        "# Self-Play Error Discovery Bootstrap",
        "",
        "## 定位",
        "",
        "这份产物对齐作业中的 Method 2：`response A -> 找出 A 的错误 -> 生成改进版 B -> 用 (A < B) 训练`。",
        "",
        "重要 caveat：当前 critique 来自 verifier 失败信息和 protected rule revision 的编辑类型，不是单独由 LLM critic 生成。因此它是可训练的 self-play proxy/bootstrap，后续可以替换为真正的模型自找错版本。",
        "",
        "## 核心指标",
        "",
        "| 指标 | 数值 |",
        "| --- | ---: |",
        f"| 原始失败样本 | {orig_failed} |",
        f"| 生成的 A<B pairs | {len(pairs)} |",
        f"| failed rows with edits | {edited_failed} |",
        f"| detection coverage on failures | {pct(metrics['metrics']['detection_coverage_on_failures'])} |",
        f"| repair precision given edit | {pct(metrics['metrics']['repair_precision_given_edit'])} |",
        f"| repair recall over all failures | {pct(metrics['metrics']['repair_recall_all_failures'])} |",
        f"| pass preservation rate | {pct(metrics['metrics']['pass_preservation_rate'])} |",
        f"| harmful edit rate on initially passed | {pct(metrics['metrics']['harmful_edit_rate_on_initially_passed'])} |",
        "",
        "## Transition Matrix",
        "",
        "| Transition | Count |",
        "| --- | ---: |",
    ]
    for key in ["pass_to_pass", "pass_to_fail", "fail_to_pass", "fail_to_fail"]:
        md.append(f"| {key} | {transitions[key]} |")

    md.extend(["", "## Successful Repairs By Split", "", "| Split | Count |", "| --- | ---: |"])
    for key, value in sorted(success_by_split.items()):
        md.append(f"| {key} | {value} |")

    md.extend(["", "## Edit Success", "", "| Edit | Attempts | Success | Success Rate |", "| --- | ---: | ---: | ---: |"])
    for edit, stats in edit_stats.items():
        md.append(
            f"| {edit} | {stats['attempted_on_failed']} | {stats['successful_repairs']} | {pct(stats['success_rate'])} |"
        )

    md.extend(
        [
            "",
            "## 哪些错误更容易自发现",
            "",
            "在当前 proxy 里，最容易被发现并修复的是重复函数体、代码块后多余执行样例、代码后夹杂说明文字等格式/语法类错误。这类错误不需要深层语义判断，外部 verifier 确认后可直接形成高置信度 preference pair。",
            "",
            "仍需要外部信号或更强 critic 的主要是逻辑错误、复杂 runtime 错误和 timeout。protected revision 后剩余失败样本仍以 `logic_wrong_output` 为主，说明真正的 Method 2 下一步应让模型显式解释语义错误，再生成 B。",
            "",
            "## 输出文件",
            "",
            f"- `{args.output}`",
            f"- `{args.metrics_output}`",
        ]
    )
    args.md_output.parent.mkdir(parents=True, exist_ok=True)
    args.md_output.write_text("\n".join(md) + "\n", encoding="utf-8")

    print(json.dumps({"pairs": len(pairs), "metrics": str(args.metrics_output), "md": str(args.md_output)}, indent=2))


if __name__ == "__main__":
    main()
