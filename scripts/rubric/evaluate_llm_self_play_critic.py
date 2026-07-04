#!/usr/bin/env python3
"""Evaluate LLM critic revisions and build successful A<B pairs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterable


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def safe_rate(num: int, den: int) -> float:
    return round(num / den, 6) if den else 0.0


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def split_from_id(row: dict) -> str:
    if row.get("split"):
        return f"{row.get('dataset')}/{row['split']}"
    parts = row.get("id", "").split("/")
    if len(parts) >= 3 and parts[0] == "mbpp":
        return f"mbpp/{parts[1]}"
    if row.get("dataset") == "humanevalplus":
        return "humanevalplus/test"
    return row.get("dataset") or "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-labeled", type=Path, required=True)
    parser.add_argument("--critic-labeled", type=Path, required=True)
    parser.add_argument("--pairs-output", type=Path, required=True)
    parser.add_argument("--metrics-output", type=Path, required=True)
    parser.add_argument("--md-output", type=Path, required=True)
    args = parser.parse_args()

    original = {row["id"]: row for row in read_jsonl(args.original_labeled)}
    revised_rows = list(read_jsonl(args.critic_labeled))
    pairs = []
    transitions = Counter()
    failure_types = Counter()
    repair_failure_types = Counter()
    parseable_critique = 0

    for row in revised_rows:
        orig = original.get(row["id"])
        if not orig:
            continue
        before = "pass" if orig.get("passed") else "fail"
        after = "pass" if row.get("passed") else "fail"
        transitions[f"{before}_to_{after}"] += 1
        failure_type = orig.get("failure_type") or "unknown"
        failure_types[failure_type] += 1
        if row.get("critique"):
            parseable_critique += 1
        if not orig.get("passed") and row.get("passed"):
            repair_failure_types[failure_type] += 1
            pairs.append(
                {
                    "id": row["id"],
                    "dataset": row.get("dataset"),
                    "split": split_from_id(row),
                    "prompt": row.get("prompt"),
                    "response_a": row.get("response_a") or orig.get("generated_code"),
                    "critique": row.get("critique"),
                    "response_b": row.get("generated_code"),
                    "preference": "A < B",
                    "chosen": row.get("generated_code"),
                    "rejected": row.get("response_a") or orig.get("generated_code"),
                    "chosen_source": "llm_self_play_revised_passed",
                    "rejected_source": "qwen25_k1_failed_output",
                    "self_discovery_source": "llm_critic",
                    "llm_critic_generated": True,
                    "failure_type": failure_type,
                    "source_error": orig.get("error"),
                    "critic_text": row.get("critic_text"),
                    "rubric_version": row.get("rubric_version"),
                }
            )

    total = len(revised_rows)
    repaired = transitions["fail_to_pass"]
    metrics = {
        "source": {
            "original_labeled": str(args.original_labeled),
            "critic_labeled": str(args.critic_labeled),
            "pairs_output": str(args.pairs_output),
        },
        "type": "llm_critic_self_play_mini_loop",
        "counts": {
            "attempted": total,
            "successful_repairs": repaired,
            "preference_pairs": len(pairs),
            "parseable_or_extracted_critiques": parseable_critique,
        },
        "transitions": dict(transitions),
        "metrics": {
            "repair_rate": safe_rate(repaired, total),
            "critique_extraction_rate": safe_rate(parseable_critique, total),
        },
        "failure_types": dict(failure_types),
        "repair_failure_types": dict(repair_failure_types),
        "status": "completed" if total else "empty",
        "caveat": "This is a small resource-aware run; scale the limit before using it as a headline result.",
    }

    args.pairs_output.parent.mkdir(parents=True, exist_ok=True)
    with args.pairs_output.open("w", encoding="utf-8") as f:
        for row in pairs:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    md = [
        "# LLM Self-Play Critic Mini-Loop",
        "",
        "## 定位",
        "",
        "这是 Method 2 的小规模真实 LLM critic 闭环：模型先对失败输出 A 写错误发现，再生成改进版 B，之后用外部 verifier 判断是否形成 `A < B` preference pair。",
        "",
        "## 指标",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Attempted | {total} |",
        f"| Successful repairs | {repaired} |",
        f"| Preference pairs | {len(pairs)} |",
        f"| Repair rate | {pct(metrics['metrics']['repair_rate'])} |",
        f"| Critique extraction rate | {pct(metrics['metrics']['critique_extraction_rate'])} |",
        "",
        "## Transitions",
        "",
        "| Transition | Count |",
        "| --- | ---: |",
    ]
    for key in sorted(transitions):
        md.append(f"| {key} | {transitions[key]} |")
    md.extend(
        [
            "",
            "## 输出文件",
            "",
            f"- `{args.critic_labeled}`",
            f"- `{args.pairs_output}`",
            f"- `{args.metrics_output}`",
            "",
            "## Caveat",
            "",
            "这是小样本资源探针，不应替代全量 Method 2 实验。它的作用是证明 pipeline 真实可跑，并给后续扩大样本量提供检查点。",
        ]
    )
    args.md_output.parent.mkdir(parents=True, exist_ok=True)
    args.md_output.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"attempted": total, "repaired": repaired, "pairs": len(pairs)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
