#!/usr/bin/env python3
"""Select first verified passing self-play candidate per task from multiple candidate files."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def split_from_id(row: dict) -> str:
    if row.get("split"):
        split = row["split"]
        return split if split.startswith("mbpp/") else f"{row.get('dataset')}/{split}"
    parts = row.get("id", "").split("/")
    if len(parts) >= 3 and parts[0] == "mbpp":
        return f"mbpp/{parts[1]}"
    return row.get("dataset") or "unknown"


def safe_rate(num: int, den: int) -> float:
    return round(num / den, 6) if den else 0.0


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-labeled", type=Path, required=True)
    parser.add_argument("--candidate-labeled", type=Path, nargs="+", required=True)
    parser.add_argument("--pairs-output", type=Path, required=True)
    parser.add_argument("--metrics-output", type=Path, required=True)
    parser.add_argument("--md-output", type=Path, required=True)
    args = parser.parse_args()

    original = {row["id"]: row for row in read_jsonl(args.original_labeled)}
    candidates_by_id: dict[str, list[dict]] = defaultdict(list)
    source_counts = Counter()
    failure_counts = Counter()
    passed_candidate_count = 0

    for source_index, path in enumerate(args.candidate_labeled):
        for row in read_jsonl(path):
            row = dict(row)
            row["candidate_source_file"] = str(path)
            row["candidate_source_index"] = source_index
            row["candidate_rank_in_source"] = len(candidates_by_id[row["id"]])
            candidates_by_id[row["id"]].append(row)
            source_counts[str(path)] += 1
            if row.get("passed"):
                passed_candidate_count += 1
            else:
                failure_counts[row.get("failure_type") or "unknown"] += 1

    selected_pairs = []
    selected_rows = []
    unrepaired = []
    for item_id, candidates in sorted(candidates_by_id.items()):
        orig = original.get(item_id)
        if not orig:
            continue
        passed = [row for row in candidates if row.get("passed")]
        if not passed:
            unrepaired.append(
                {
                    "id": item_id,
                    "num_candidates": len(candidates),
                    "failure_types": Counter(row.get("failure_type") or "unknown" for row in candidates),
                    "first_errors": [row.get("error") for row in candidates[:3]],
                }
            )
            continue
        chosen_row = passed[0]
        selected_rows.append(chosen_row)
        selected_pairs.append(
            {
                "id": item_id,
                "dataset": chosen_row.get("dataset"),
                "split": split_from_id(chosen_row),
                "prompt": chosen_row.get("prompt"),
                "response_a": chosen_row.get("response_a") or orig.get("generated_code"),
                "critique": chosen_row.get("critique"),
                "response_b": chosen_row.get("generated_code"),
                "preference": "A < B",
                "chosen": chosen_row.get("generated_code"),
                "rejected": chosen_row.get("response_a") or orig.get("generated_code"),
                "chosen_source": "llm_self_play_logic_multicandidate_revised_passed",
                "rejected_source": "qwen25_k1_logic_failed_output",
                "self_discovery_source": "llm_critic_multicandidate",
                "llm_critic_generated": True,
                "failure_type": orig.get("failure_type"),
                "source_error": orig.get("error"),
                "critic_text": chosen_row.get("critic_text"),
                "candidate_source_file": chosen_row.get("candidate_source_file"),
                "candidate_source_index": chosen_row.get("candidate_source_index"),
                "rubric_version": chosen_row.get("rubric_version"),
            }
        )

    attempted_tasks = len(candidates_by_id)
    total_candidates = sum(source_counts.values())
    repaired_tasks = len(selected_pairs)
    metrics = {
        "source": {
            "original_labeled": str(args.original_labeled),
            "candidate_labeled": [str(path) for path in args.candidate_labeled],
            "pairs_output": str(args.pairs_output),
        },
        "type": "llm_critic_multicandidate_self_play",
        "counts": {
            "attempted_tasks": attempted_tasks,
            "total_candidates": total_candidates,
            "passed_candidates": passed_candidate_count,
            "repaired_tasks": repaired_tasks,
            "preference_pairs": len(selected_pairs),
            "unrepaired_tasks": attempted_tasks - repaired_tasks,
        },
        "metrics": {
            "task_repair_rate": safe_rate(repaired_tasks, attempted_tasks),
            "candidate_pass_rate": safe_rate(passed_candidate_count, total_candidates),
        },
        "source_counts": dict(source_counts),
        "failed_candidate_types": dict(failure_counts),
        "selected_ids": [row["id"] for row in selected_pairs],
        "unrepaired_examples": unrepaired[:20],
        "gate": {
            "min_repaired_for_merge": 6,
            "passed": repaired_tasks >= 6,
        },
    }

    args.pairs_output.parent.mkdir(parents=True, exist_ok=True)
    with args.pairs_output.open("w", encoding="utf-8") as f:
        for row in selected_pairs:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    md = [
        "# Logic Multi-Candidate Self-Play",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Attempted tasks | {attempted_tasks} |",
        f"| Total candidates | {total_candidates} |",
        f"| Passed candidates | {passed_candidate_count} |",
        f"| Repaired tasks | {repaired_tasks} |",
        f"| Preference pairs | {len(selected_pairs)} |",
        f"| Task repair rate | {pct(metrics['metrics']['task_repair_rate'])} |",
        f"| Candidate pass rate | {pct(metrics['metrics']['candidate_pass_rate'])} |",
        f"| Gate passed | {metrics['gate']['passed']} |",
        "",
        "## Selected IDs",
        "",
    ]
    md.extend(f"- `{item_id}`" for item_id in metrics["selected_ids"])
    md.extend(
        [
            "",
            "## Interpretation",
            "",
            "This selects at most one passing candidate per original failed task. If the gate fails, do not merge these pairs into DPO; use the failures to improve the critic prompt or add stronger external feedback.",
            "",
            "## Outputs",
            "",
            f"- `{args.pairs_output}`",
            f"- `{args.metrics_output}`",
        ]
    )
    args.md_output.parent.mkdir(parents=True, exist_ok=True)
    args.md_output.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
