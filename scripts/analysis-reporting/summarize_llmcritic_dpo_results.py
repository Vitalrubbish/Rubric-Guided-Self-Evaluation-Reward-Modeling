#!/usr/bin/env python3
"""Summarize LLM-critic DPO training and validation artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def pct(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value * 100:.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pref-summary", type=Path, required=True)
    parser.add_argument("--llmcritic-metrics", type=Path, required=True)
    parser.add_argument("--train-metrics", type=Path, required=True)
    parser.add_argument("--validation-summary", type=Path, required=True)
    parser.add_argument("--protected-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--md-output", type=Path, required=True)
    args = parser.parse_args()

    pref = load_json(args.pref_summary)
    critic = load_json(args.llmcritic_metrics)
    train = load_json(args.train_metrics)
    val = load_json(args.validation_summary)
    protected = load_json(args.protected_summary)
    result = {
        "preference_data": pref,
        "llm_critic": critic,
        "dpo_training": train,
        "validation": val,
        "protected_validation": protected,
        "status": {
            "preference_data": "done" if pref else "missing",
            "llm_critic": "done" if critic else "missing",
            "dpo_training": "done" if train else "missing",
            "validation": "done" if val else "missing",
            "protected_validation": "done" if protected else "missing",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    md = [
        "# LLM-Critic DPO Results",
        "",
        "## Status",
        "",
        "| Artifact | Status |",
        "| --- | --- |",
    ]
    for key, value in result["status"].items():
        md.append(f"| {key} | {value} |")
    md.extend(["", "## Key Metrics", "", "| Metric | Value |", "| --- | ---: |"])
    if pref:
        md.append(f"| Preference pairs | {pref.get('total_pairs')} |")
    if critic:
        md.append(f"| LLM critic attempted | {critic['counts'].get('attempted')} |")
        md.append(f"| LLM critic repaired | {critic['counts'].get('successful_repairs')} |")
        md.append(f"| LLM critic repair rate | {pct(critic['metrics'].get('repair_rate'))} |")
    if train:
        md.append(f"| DPO steps | {train.get('steps')} |")
        md.append(f"| DPO mean loss | {train.get('mean_loss'):.4f} |")
        md.append(f"| DPO preference accuracy | {pct(train.get('preference_accuracy'))} |")
    if val:
        md.append(f"| Validation passed | {val.get('passed')}/{val.get('total')} |")
        md.append(f"| Validation pass@1 | {pct(val.get('pass_rate'))} |")
    if protected:
        md.append(f"| Protected validation passed | {protected.get('passed')}/{protected.get('total')} |")
        md.append(f"| Protected validation pass@1 | {pct(protected.get('pass_rate'))} |")
    md.extend(
        [
            "",
            "## Caveat",
            "",
            "This run uses only MBPP train preference pairs. Validation is untouched, but the run is still small; use it as an ablation rather than a final headline unless it is repeated or scaled.",
        ]
    )
    args.md_output.parent.mkdir(parents=True, exist_ok=True)
    args.md_output.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "md": str(args.md_output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
