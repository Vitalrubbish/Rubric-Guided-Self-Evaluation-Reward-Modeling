#!/usr/bin/env python3
"""Build a human-readable error attribution review packet.

The packet is designed for VSCode Remote SSH: annotators can open the Markdown
file in preview mode and fill the JSONL template in a neighboring editor pane.
Verifier labels are used for sampling and metadata, but exact expected/got
values are hidden by default.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DETAILED_LABELS = [
    "syntax_parseability_truncation",
    "interface_name_signature_mismatch",
    "runtime_api_type_misuse",
    "timeout_or_complexity",
    "truncation_or_overgeneration",
    "output_type_or_container_shape",
    "numeric_formula_arithmetic_error",
    "sequence_collection_transformation_error",
    "predicate_branch_condition_error",
    "string_regex_pattern_logic",
    "edge_case_handling",
    "logic_other",
    "public_contract_unclear",
    "not_a_failure",
]

PILOT_LABELS = [
    "syntax_or_parse_error",
    "interface_contract_error",
    "runtime_exception_or_timeout",
    "truncation_or_overgeneration",
    "output_format_or_type_error",
    "sequence_or_state_transformation_error",
    "numeric_formula_or_counting_error",
    "predicate_condition_or_edge_case_error",
    "string_pattern_or_text_error",
    "unclear_other_or_not_failure",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def short_text(value: Any, limit: int) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 24].rstrip() + "\n... [truncated]"


def extract_task(prompt: str | None) -> str:
    if not prompt:
        return ""
    marker = "Return only valid Python code, with no Markdown fences and no explanation."
    text = prompt.split(marker, 1)[-1] if marker in prompt else prompt
    if "\nPython code:" in text:
        text = text.split("\nPython code:", 1)[0]
    if text.strip().startswith("Task:"):
        text = text.strip()[len("Task:") :].strip()
    return text.strip()


def verifier_summary(row: dict[str, Any], include_raw_error: bool = False) -> dict[str, Any]:
    diagnostics = row.get("safe_diagnostics") if isinstance(row.get("safe_diagnostics"), dict) else {}
    summary: dict[str, Any] = {
        "failure_type": row.get("failure_type"),
        "diagnostic_kind": diagnostics.get("diagnostic_kind"),
        "first_failure_kind": diagnostics.get("first_failure_kind"),
        "first_exception_type": diagnostics.get("first_exception_type"),
        "finish_reason": row.get("finish_reason"),
    }
    if include_raw_error:
        summary["raw_error"] = short_text(row.get("error"), 500)
    elif row.get("error"):
        error = str(row.get("error") or "")
        lowered = error.lower()
        if "expected" in lowered and "got" in lowered:
            summary["error_public_summary"] = "wrong output reported; exact expected/got hidden"
        elif "invalid syntax" in lowered or "indentationerror" in lowered:
            summary["error_public_summary"] = short_text(error, 180)
        elif "traceback" in lowered or "error:" in lowered or "exception" in lowered:
            summary["error_public_summary"] = short_text(error.splitlines()[-1], 180)
        elif error.startswith(">"):
            summary["error_public_summary"] = short_text(error, 120)
        else:
            summary["error_public_summary"] = short_text(error, 160)
    return {key: value for key, value in summary.items() if value not in (None, "", {})}


def response_id(row: dict[str, Any]) -> str:
    return str(row.get("response_id") or f"{row.get('id')}__sample{row.get('sample_id', 0)}")


def build_items(
    labeled_rows: list[dict[str, Any]],
    assignments: list[dict[str, Any]],
    samples_per_category: int,
    include_all_rare_threshold: int,
    seed: int,
    task_chars: int,
    code_chars: int,
    include_raw_error: bool,
) -> list[dict[str, Any]]:
    labels_by_id = {response_id(row): row for row in labeled_rows}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for assignment in assignments:
        rid = str(assignment.get("response_id"))
        row = labels_by_id.get(rid)
        if not row or bool(row.get("passed")):
            continue
        category = str(assignment.get("taxonomy_category_id") or "unknown")
        grouped[category].append(assignment)

    rng = random.Random(seed)
    selected: list[tuple[str, dict[str, Any], str]] = []
    for category in sorted(grouped):
        rows = sorted(grouped[category], key=lambda item: str(item.get("response_id")))
        if len(rows) <= include_all_rare_threshold:
            sample = rows
            reason = "rare_category_all"
        else:
            sample = rng.sample(rows, min(samples_per_category, len(rows)))
            sample = sorted(sample, key=lambda item: str(item.get("response_id")))
            reason = "category_stratified_sample"
        selected.extend((category, item, reason) for item in sample)

    items = []
    for index, (category, assignment, sampling_reason) in enumerate(selected, start=1):
        rid = str(assignment["response_id"])
        row = labels_by_id[rid]
        task = short_text(extract_task(row.get("prompt")), task_chars)
        generated = short_text(row.get("generated_code"), code_chars)
        extracted = short_text(row.get("extracted_code") or row.get("generated_code"), code_chars)
        current_label = str(assignment.get("taxonomy_category_id") or category)
        item = {
            "review_index": index,
            "response_id": rid,
            "id": row.get("id"),
            "sample_id": row.get("sample_id", 0),
            "dataset": row.get("dataset"),
            "split": row.get("split"),
            "difficulty": row.get("difficulty"),
            "io_mode": row.get("io_mode"),
            "failure_type": row.get("failure_type"),
            "error_pattern": assignment.get("error_pattern"),
            "current_label": current_label,
            "current_label_name": assignment.get("taxonomy_category_name"),
            "current_rubric_dimension": assignment.get("rubric_dimension"),
            "llm_summary": assignment.get("llm_summary"),
            "verifier_summary": verifier_summary(row, include_raw_error),
            "sampling_reason": sampling_reason,
            "task": task,
            "public_interface": row.get("interface_signatures") or row.get("interface_names") or [],
            "generated_code": generated,
            "extracted_code": extracted,
        }
        items.append(item)
    return items


def markdown_code_block(language: str, value: str) -> str:
    fence = "```"
    text = value or ""
    if "```" in text:
        fence = "````"
    return f"{fence}{language}\n{text}\n{fence}"


def annotation_stub(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "response_id": item["response_id"],
        "current_label": item["current_label"],
        "provisional_label": item.get("provisional_label"),
        "provisional_confidence": item.get("provisional_confidence"),
        "human_primary_label": None,
        "human_secondary_label": None,
        "confidence": None,
        "evidence": "",
        "notes": "",
    }


def render_packet(items: list[dict[str, Any]], labels: list[str]) -> str:
    counts = Counter(item["current_label"] for item in items)
    lines = [
        "# APPS Simple Error Attribution Review Packet",
        "",
        "This packet is for human review of failure attribution labels. Use the public task and visible code as evidence. Verifier labels are shown only as sanitized metadata.",
        "",
        "## Label Options",
        "",
        markdown_code_block("text", "\n".join(labels)),
        "",
        "## Sample Counts",
        "",
        "| Current label | Count |",
        "| --- | ---: |",
    ]
    for label, count in sorted(counts.items()):
        lines.append(f"| `{label}` | {count} |")
    lines.extend(["", "## Review Items", ""])

    for item in items:
        stub = annotation_stub(item)
        lines.extend(
            [
                f"### Item {item['review_index']:03d}: `{item['response_id']}`",
                "",
                f"- Current label: `{item['current_label']}`",
                f"- Provisional label: `{item.get('provisional_label') or ''}`",
                f"- Provisional confidence: `{item.get('provisional_confidence') or ''}`",
                f"- Sampling reason: `{item.get('sampling_reason')}`",
                f"- Current rubric dimension: `{item.get('current_rubric_dimension') or ''}`",
                f"- Failure type: `{item.get('failure_type')}`",
                f"- Error pattern: `{item.get('error_pattern')}`",
                f"- IO mode: `{item.get('io_mode')}`",
                "",
                "**Verifier summary**",
                "",
                markdown_code_block("json", json.dumps(item["verifier_summary"], ensure_ascii=False, indent=2)),
                "",
                "**Current LLM summary**",
                "",
                str(item.get("llm_summary") or ""),
                "",
                "**Provisional rationale**",
                "",
                str(item.get("provisional_reason") or ""),
                "",
                "**Task**",
                "",
                markdown_code_block("text", item["task"]),
                "",
                "**Public interface**",
                "",
                markdown_code_block("json", json.dumps(item["public_interface"], ensure_ascii=False, indent=2)),
                "",
                "**Generated response**",
                "",
                markdown_code_block("python", item["generated_code"]),
                "",
                "**Extracted code**",
                "",
                markdown_code_block("python", item["extracted_code"]),
                "",
                "**Annotation stub**",
                "",
                markdown_code_block("yaml", json.dumps(stub, ensure_ascii=False, indent=2)),
                "",
                "---",
                "",
            ]
        )
    return "\n".join(lines)


def render_schema(labels: list[str]) -> str:
    return "\n".join(
        [
            "# Error Attribution Annotation Schema",
            "",
            "Annotate each item with a primary human label and an optional short visible-code evidence note.",
            "",
            "## Fields",
            "",
            "- `provisional_label`: machine suggestion for reference only.",
            "- `human_primary_label`: one label from the controlled label set.",
            "- `human_secondary_label`: optional second label if two mechanisms are genuinely material.",
            "- `confidence`: `high`, `medium`, or `low`.",
            "- `evidence`: optional short explanation grounded in public task text and visible code.",
            "- `notes`: optional review notes.",
            "",
            "## Controlled Labels",
            "",
            markdown_code_block("text", "\n".join(labels)),
            "",
            "## Rules",
            "",
            "- Prefer deterministic labels for syntax, interface, runtime exception, timeout, and truncation cases.",
            "- Use logic labels only when the visible code evidence is clear.",
            "- Use the unclear/other label when the code is wrong but none of the available mechanisms fit.",
            "- Use the unclear/other label when the public task does not resolve the convention needed for judgment.",
            "- Do not copy exact expected/got values into `evidence`.",
            "",
        ]
    )


def render_html_index(items: list[dict[str, Any]]) -> str:
    rows = []
    for item in items:
        rows.append(
            "<tr>"
            f"<td>{item['review_index']:03d}</td>"
            f"<td><code>{item['response_id']}</code></td>"
            f"<td><code>{item['current_label']}</code></td>"
            f"<td>{item.get('failure_type') or ''}</td>"
            f"<td>{item.get('io_mode') or ''}</td>"
            f"<td>{item.get('sampling_reason') or ''}</td>"
            "</tr>"
        )
    return """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>APPS Simple Error Review Index</title>
  <style>
    body { font-family: system-ui, sans-serif; margin: 24px; line-height: 1.4; }
    table { border-collapse: collapse; width: 100%; }
    th, td { border: 1px solid #d0d7de; padding: 6px 8px; vertical-align: top; }
    th { background: #f6f8fa; text-align: left; }
    code { white-space: nowrap; }
  </style>
</head>
<body>
  <h1>APPS Simple Error Review Index</h1>
  <p>Use <code>review_packet.md</code> as the main VSCode review file. This HTML file is only a compact index.</p>
  <table>
    <thead><tr><th>#</th><th>Response</th><th>Current label</th><th>Failure type</th><th>IO mode</th><th>Sampling</th></tr></thead>
    <tbody>
""" + "\n".join(rows) + """
    </tbody>
  </table>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a human annotation review packet for APPS error attributions.")
    parser.add_argument("--labeled", type=Path, default=Path("data/responses/apps_train_simple_executable_qwen25_k1_t2048_full_labeled_nonlength.jsonl"))
    parser.add_argument("--assignments", type=Path, default=Path("data/analysis/apps_simple_phase1/apps_train_simple_qwen25_k1_t2048_taxonomy_refined_response_assignments.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/annotation/apps_simple_error_review_v1"))
    parser.add_argument("--samples-per-category", type=int, default=5)
    parser.add_argument("--include-all-rare-threshold", type=int, default=0)
    parser.add_argument("--label-set", choices=["pilot", "detailed"], default="pilot")
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--task-chars", type=int, default=4500)
    parser.add_argument("--code-chars", type=int, default=3500)
    parser.add_argument("--include-raw-error", action="store_true")
    args = parser.parse_args()

    labeled_rows = read_jsonl(args.labeled)
    assignments = read_jsonl(args.assignments)
    items = build_items(
        labeled_rows=labeled_rows,
        assignments=assignments,
        samples_per_category=args.samples_per_category,
        include_all_rare_threshold=args.include_all_rare_threshold,
        seed=args.seed,
        task_chars=args.task_chars,
        code_chars=args.code_chars,
        include_raw_error=args.include_raw_error,
    )

    labels = list(PILOT_LABELS if args.label_set == "pilot" else DETAILED_LABELS)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "review_items.jsonl", items)
    write_jsonl(output_dir / "annotation_template.jsonl", [annotation_stub(item) for item in items])
    (output_dir / "review_packet.md").write_text(render_packet(items, labels), encoding="utf-8")
    (output_dir / "annotation_schema.md").write_text(render_schema(labels), encoding="utf-8")
    (output_dir / "review_index.html").write_text(render_html_index(items), encoding="utf-8")

    summary = {
        "labeled": str(args.labeled),
        "assignments": str(args.assignments),
        "output_dir": str(output_dir),
        "seed": args.seed,
        "num_items": len(items),
        "samples_per_category": args.samples_per_category,
        "include_all_rare_threshold": args.include_all_rare_threshold,
        "label_set": args.label_set,
        "current_label_counts": dict(Counter(item["current_label"] for item in items)),
        "failure_type_counts": dict(Counter(str(item.get("failure_type")) for item in items)),
        "io_mode_counts": dict(Counter(str(item.get("io_mode")) for item in items)),
        "label_options": labels,
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
