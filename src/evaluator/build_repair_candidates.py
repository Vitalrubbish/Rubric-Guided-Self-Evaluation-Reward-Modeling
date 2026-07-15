#!/usr/bin/env python3
"""Build critic-filtered repair candidates for Method 1.

The output is a prompt set for the next generation step: repair failed train
responses using only public task text, visible code, and rubric-style guidance.
It does not include hidden verifier details or exact expected/got values.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_problem_ids(paths: list[Path]) -> set[str]:
    ids: set[str] = set()
    for path in paths:
        ids.update(str(row.get("id") or "") for row in read_jsonl(path))
    ids.discard("")
    return ids


def short_text(value: Any, limit: int) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 24].rstrip() + "\n... [truncated]"


def repair_prompt(row: dict[str, Any], probability: float) -> str:
    task = str(row.get("task") or "")
    code = str(row.get("extracted_code") or row.get("generated_code") or "")
    public_interface = row.get("public_interface") or []
    interface_text = "\n".join(str(item) for item in public_interface) if public_interface else "stdin/stdout program"
    return (
        "You are an expert Python programmer repairing a previous solution.\n"
        "The previous solution failed an external verifier, but you are not given hidden tests or exact expected/got values.\n"
        "Use only the public task, public interface, and visible code below.\n"
        "Before writing code, silently check the rubric failure modes: public interface, syntax/truncation, runtime safety, output shape, core numeric formula, sequence/state transformation, predicate/branch logic, string/text logic, and edge cases.\n"
        "Return only valid Python code, with no Markdown fences and no explanation.\n\n"
        f"Critic estimated pass probability of the previous solution: {probability:.4f}\n\n"
        "Public interface:\n"
        f"{interface_text}\n\n"
        "Task:\n"
        f"{task}\n\n"
        "Previous failed code:\n"
        f"{code}\n\n"
        "Repaired Python code:\n"
    )


def reason_for(row: dict[str, Any], probability: float, selected_threshold: float) -> str:
    if probability >= selected_threshold:
        return "critic_false_positive_train"
    if probability >= 0.45:
        return "critic_borderline_failed_train"
    if row.get("finish_reason") == "length":
        return "truncation_repair_train"
    if row.get("io_mode") == "stdin_stdout":
        return "stdin_stdout_failed_train"
    return "failed_train_coverage"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Method 1 repair candidate prompts from failed train rows.")
    parser.add_argument("--evaluator-rows", type=Path, default=Path("data/evaluator/apps_simple_method1_evaluator_training_rows_v1.jsonl"))
    parser.add_argument("--critic-predictions", type=Path, default=Path("data/evaluator/apps_simple_static_critic_baseline_v1_all_predictions.jsonl"))
    parser.add_argument("--processed-prompts", type=Path, default=Path("data/processed/apps_train_simple_executable_prompts_unified.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/repair/apps_simple_method1_repair_candidates_v1.jsonl"))
    parser.add_argument("--prompts-output", type=Path, default=Path("data/repair/apps_simple_method1_repair_prompts_v1.jsonl"))
    parser.add_argument("--summary-output", type=Path, default=Path("data/repair/apps_simple_method1_repair_candidates_v1_summary.json"))
    parser.add_argument("--max-candidates", type=int, default=400)
    parser.add_argument("--forbidden-ids", type=Path, action="append", default=[])
    parser.add_argument("--task-chars", type=int, default=8000)
    parser.add_argument("--code-chars", type=int, default=5000)
    args = parser.parse_args()

    rows_by_id = {str(row.get("response_id")): row for row in read_jsonl(args.evaluator_rows)}
    predictions = {str(row.get("response_id")): row for row in read_jsonl(args.critic_predictions)}
    processed_by_id = {str(row.get("id")): row for row in read_jsonl(args.processed_prompts)}
    forbidden_ids = load_problem_ids(args.forbidden_ids)
    candidates = []
    missing_prediction = 0
    forbidden_rows = 0
    for rid, row in rows_by_id.items():
        if row.get("split") != "train" or bool(row.get("passed")):
            continue
        if str(row.get("id") or "") in forbidden_ids:
            forbidden_rows += 1
            continue
        prediction = predictions.get(rid)
        if not prediction:
            missing_prediction += 1
            continue
        probability = float(prediction.get("critic_pass_probability") or 0.0)
        selected_threshold = float(prediction.get("selected_threshold") or 0.5)
        candidates.append((probability, selected_threshold, row, prediction))

    # Prioritize failed rows that the critic thinks are plausible, then retain
    # some truncation/stdin-stdout rows for repair coverage.
    candidates = sorted(
        candidates,
        key=lambda item: (
            -(item[0] >= item[1]),
            -item[0],
            str(item[2].get("response_id")),
        ),
    )
    selected = candidates[: args.max_candidates]

    output_rows = []
    prompt_rows = []
    for index, (probability, selected_threshold, row, prediction) in enumerate(selected, start=1):
        candidate_id = f"apps_simple_repair_v1_{index:04d}"
        processed = processed_by_id.get(str(row.get("id")), {})
        prompt_text = repair_prompt(
            {
                **row,
                "task": short_text(row.get("task"), args.task_chars),
                "extracted_code": short_text(row.get("extracted_code") or row.get("generated_code"), args.code_chars),
            },
            probability,
        )
        compact_row = {
            "candidate_id": candidate_id,
            "response_id": row.get("response_id"),
            "id": row.get("id"),
            "dataset": row.get("dataset"),
            "source_split": row.get("source_split"),
            "split": row.get("split"),
            "sample_id": row.get("sample_id"),
            "io_mode": row.get("io_mode"),
            "difficulty": row.get("difficulty"),
            "passed": bool(row.get("passed")),
            "critic_pass_probability": probability,
            "critic_selected_threshold": selected_threshold,
            "critic_predicted_pass": bool(prediction.get("critic_predicted_pass")),
            "selection_reason": reason_for(row, probability, selected_threshold),
            "finish_reason": row.get("finish_reason"),
            "generated_token_count": row.get("generated_token_count"),
            "public_interface": row.get("public_interface") or [],
            "task": short_text(row.get("task"), args.task_chars),
            "previous_code": short_text(row.get("extracted_code") or row.get("generated_code"), args.code_chars),
            "repair_prompt": prompt_text,
        }
        output_rows.append(compact_row)
        prompt_rows.append(
            {
                "id": row.get("id"),
                "response_id_prefix": candidate_id,
                "repair_candidate_id": candidate_id,
                "original_response_id": row.get("response_id"),
                "original_generated_code": row.get("generated_code"),
                "previous_code": compact_row["previous_code"],
                "task": compact_row["task"],
                "public_interface": compact_row["public_interface"],
                "original_failure_type": row.get("failure_type"),
                "dataset": row.get("dataset"),
                "split": row.get("split"),
                "prompt_mode": "method1_repair_v1",
                "prompt": prompt_text,
                "interface_names": processed.get("interface_names") or [],
                "interface_signatures": processed.get("interface_signatures") or [],
                "starter_code": processed.get("starter_code", ""),
                "input_output": processed.get("input_output"),
                "difficulty": processed.get("difficulty") or row.get("difficulty"),
                "io_mode": processed.get("io_mode") or row.get("io_mode"),
                "critic_pass_probability": probability,
                "critic_selected_threshold": selected_threshold,
                "critic_predicted_pass": bool(prediction.get("critic_predicted_pass")),
                "selection_reason": compact_row["selection_reason"],
            }
        )

    summary = {
        "num_failed_train_rows": len(candidates),
        "num_candidates": len(output_rows),
        "missing_prediction": missing_prediction,
        "forbidden_id_count": len(forbidden_ids),
        "forbidden_failed_rows_excluded": forbidden_rows,
        "selection_reason_counts": dict(Counter(row["selection_reason"] for row in output_rows)),
        "io_mode_counts": dict(Counter(str(row.get("io_mode")) for row in output_rows)),
        "finish_reason_counts": dict(Counter(str(row.get("finish_reason")) for row in output_rows)),
        "critic_predicted_pass_count": sum(bool(row.get("critic_predicted_pass")) for row in output_rows),
        "critic_probability": {
            "min": min((row["critic_pass_probability"] for row in output_rows), default=None),
            "max": max((row["critic_pass_probability"] for row in output_rows), default=None),
            "mean": (
                sum(row["critic_pass_probability"] for row in output_rows) / len(output_rows)
                if output_rows else None
            ),
        },
        "policy": {
            "source": "failed train rows only",
            "priority": "critic false positives and high-probability failed rows first",
            "private_verifier_details": "excluded",
            "held_out_ids": "excluded before candidate ranking and generation",
            "use": "generate repaired candidates; verify repairs; build pass-repair > failed-original preference pairs",
        },
    }
    write_jsonl(args.output, output_rows)
    write_jsonl(args.prompts_output, prompt_rows)
    write_json(args.summary_output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
