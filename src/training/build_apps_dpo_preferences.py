#!/usr/bin/env python3
"""Build leakage-safe APPS train-only DPO preference pairs.

The rejected completion is the model's failed answer. The chosen completion is
the canonical solution for the same public task. Verifier and human labels are
retained only as auditable metadata; private tests and diagnostics are never
written to the preference file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


PRIVATE_KEYS = {
    "canonical_verifier",
    "canonical_solutions",
    "input_output",
    "private_diagnostics",
    "safe_diagnostics",
    "test",
    "test_list",
    "test_setup_code",
}


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
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


def text_stats(values: list[str]) -> dict[str, float | int]:
    lengths = sorted(len(value) for value in values)
    if not lengths:
        return {"min": 0, "mean": 0.0, "p50": 0, "p95": 0, "max": 0}

    def percentile(fraction: float) -> int:
        index = min(len(lengths) - 1, round((len(lengths) - 1) * fraction))
        return lengths[index]

    return {
        "min": lengths[0],
        "mean": mean(lengths),
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "max": lengths[-1],
    }


def normalize_for_comparison(value: str) -> str:
    return "\n".join(line.rstrip() for line in value.strip().splitlines())


def build_pairs(
    evaluator_rows: Iterable[dict[str, Any]],
    source_by_id: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], Counter[str]]:
    pairs: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    seen_problem_ids: set[str] = set()

    for row in evaluator_rows:
        if row.get("split") != "train":
            skipped["non_train_split"] += 1
            continue
        if bool(row.get("passed")):
            skipped["already_passing"] += 1
            continue

        problem_id = str(row.get("id") or "")
        if not problem_id:
            skipped["missing_problem_id"] += 1
            continue
        if problem_id in seen_problem_ids:
            skipped["duplicate_train_problem"] += 1
            continue

        source = source_by_id.get(problem_id)
        if source is None:
            skipped["missing_source_problem"] += 1
            continue

        prompt = str(row.get("prompt") or source.get("prompt") or "").strip()
        chosen = str(source.get("canonical_solution") or "").strip()
        rejected = str(row.get("generated_code") or row.get("extracted_code") or "").strip()
        if not prompt:
            skipped["empty_prompt"] += 1
            continue
        if not chosen:
            skipped["empty_chosen"] += 1
            continue
        if not rejected:
            skipped["empty_rejected"] += 1
            continue
        if normalize_for_comparison(chosen) == normalize_for_comparison(rejected):
            skipped["identical_completions"] += 1
            continue

        pair = {
            "pair_id": f"{problem_id}__canonical_vs_failed_v1",
            "id": problem_id,
            "dataset": "apps",
            "source_split": row.get("source_split", source.get("split")),
            "eval_split": "train",
            "difficulty": row.get("difficulty", source.get("difficulty")),
            "io_mode": row.get("io_mode", source.get("io_mode")),
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "chosen_source": "verified_canonical_solution",
            "rejected_source": "qwen25_failed_response",
            "preference_source": "external_verifier_pass_over_fail",
            "failure_type": row.get("failure_type"),
            "finish_reason": row.get("finish_reason"),
            "error_attribution_label": row.get("error_attribution_label"),
            "error_attribution_source": row.get("error_attribution_source"),
            "error_attribution_confidence": row.get("error_attribution_confidence"),
            "human_error_label": row.get("human_error_label"),
            "human_error_confidence": row.get("human_error_confidence"),
        }
        leaked = sorted(PRIVATE_KEYS.intersection(pair))
        if leaked:
            raise AssertionError(f"private keys leaked into pair {problem_id}: {leaked}")
        pairs.append(pair)
        seen_problem_ids.add(problem_id)

    return pairs, skipped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluator-rows",
        type=Path,
        default=Path("data/evaluator/apps_simple_method1_evaluator_training_rows_v1.jsonl"),
    )
    parser.add_argument(
        "--source-prompts",
        type=Path,
        default=Path("data/processed/apps_train_simple_executable_prompts_unified.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/preferences/apps_simple_method1_train_canonical_dpo_v1.jsonl"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("data/preferences/apps_simple_method1_train_canonical_dpo_v1_summary.json"),
    )
    args = parser.parse_args()

    source_rows = list(read_jsonl(args.source_prompts))
    source_by_id = {str(row.get("id")): row for row in source_rows if row.get("id")}
    if len(source_by_id) != len(source_rows):
        raise ValueError("source prompt IDs are missing or duplicated")

    evaluator_rows = list(read_jsonl(args.evaluator_rows))
    pairs, skipped = build_pairs(evaluator_rows, source_by_id)
    if not pairs:
        raise RuntimeError("no DPO preference pairs were built")

    pair_ids = [str(row["pair_id"]) for row in pairs]
    problem_ids = [str(row["id"]) for row in pairs]
    if len(pair_ids) != len(set(pair_ids)):
        raise AssertionError("duplicate pair IDs found")
    if len(problem_ids) != len(set(problem_ids)):
        raise AssertionError("a problem appears in more than one pair")
    if any(row["eval_split"] != "train" for row in pairs):
        raise AssertionError("non-train row leaked into DPO pairs")

    write_jsonl(args.output, pairs)
    summary = {
        "evaluator_rows": str(args.evaluator_rows),
        "source_prompts": str(args.source_prompts),
        "output": str(args.output),
        "evaluator_rows_sha256": sha256_file(args.evaluator_rows),
        "source_prompts_sha256": sha256_file(args.source_prompts),
        "pair_count": len(pairs),
        "unique_problem_count": len(set(problem_ids)),
        "split_counts": dict(Counter(str(row["eval_split"]) for row in pairs)),
        "failure_type_counts": dict(Counter(str(row.get("failure_type")) for row in pairs)),
        "finish_reason_counts": dict(Counter(str(row.get("finish_reason")) for row in pairs)),
        "attribution_source_counts": dict(Counter(str(row.get("error_attribution_source")) for row in pairs)),
        "human_labeled_pair_count": sum(bool(row.get("human_error_label")) for row in pairs),
        "prompt_char_stats": text_stats([str(row["prompt"]) for row in pairs]),
        "chosen_char_stats": text_stats([str(row["chosen"]) for row in pairs]),
        "rejected_char_stats": text_stats([str(row["rejected"]) for row in pairs]),
        "skipped_counts": dict(skipped),
        "private_keys_written": [],
        "policy": {
            "training_split_only": True,
            "chosen": "same-problem verified canonical solution",
            "rejected": "same-problem failed Qwen2.5 response",
            "preference_target": "verifier pass over fail",
            "rubric_and_human_labels": "audit metadata only; never used as unverified correctness truth",
            "validation_and_test_excluded": True,
        },
    }
    summary["output_sha256"] = sha256_file(args.output)
    write_json(args.summary, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
