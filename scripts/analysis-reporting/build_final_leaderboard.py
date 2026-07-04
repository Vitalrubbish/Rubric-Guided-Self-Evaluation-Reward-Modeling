#!/usr/bin/env python3
"""Build final method leaderboard and ablation tables from checked metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def row(name: str, split: str, passed: int, total: int, note: str, leakage: str = "no") -> dict:
    return {
        "method": name,
        "split": split,
        "passed": passed,
        "total": total,
        "pass_rate": round(passed / total, 6) if total else 0,
        "leakage": leakage,
        "note": note,
    }


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, default=Path("data/final/project_metrics_summary.json"))
    parser.add_argument(
        "--logic-k5-validation-summary",
        type=Path,
        default=Path("data/eval/dpo_lora_train_augmented_llmcritic54_logic_k5_mbpp_validation_summary.json"),
    )
    parser.add_argument(
        "--logic-k5-protected-summary",
        type=Path,
        default=Path("data/eval/dpo_lora_train_augmented_llmcritic54_logic_k5_mbpp_validation_protected_revised_summary.json"),
    )
    parser.add_argument("--json-output", type=Path, default=Path("data/final/final_method_leaderboard.json"))
    parser.add_argument("--md-output", type=Path, default=Path("docs/final_method_leaderboard.md"))
    args = parser.parse_args()

    metrics = load_json(args.metrics)
    protected = metrics["protected_rule_revision"]
    train_only = metrics["train_only_dpo"]["untouched_validation"]
    augmented = metrics["augmented_train_only_dpo"]["untouched_validation"]
    leaked_dpo = metrics["dpo_adapter_evaluation"]

    overall = [
        row("Original Qwen2.5-7B vLLM", "all", metrics["baseline"]["passed"], metrics["num_prompts"], "Initial k=1 generation baseline"),
        row("Unprotected rule revision", "all", metrics["revision_loop"]["revised_passed"], metrics["num_prompts"], "Ablation; modifies all rows and has pass->fail risk"),
        row(
            "Protected rule revision",
            "all",
            protected["full_baseline"]["passed"],
            metrics["num_prompts"],
            "Main current baseline; only revises failed rows",
        ),
    ]

    validation = [
        row("Base-HF", "mbpp/validation", train_only["base_hf"]["passed"], 90, "Transformers baseline"),
        row("Train-only DPO", "mbpp/validation", train_only["train_only_dpo_hf"]["passed"], 90, "No validation leakage"),
        row(
            "Train-only DPO + protected revision",
            "mbpp/validation",
            protected["train_only_dpo_validation"]["protected_revised_passed"],
            90,
            "No validation leakage; protected cascade",
        ),
        row("Augmented train-only DPO", "mbpp/validation", augmented["augmented_dpo_hf"]["passed"], 90, "No validation leakage"),
        row(
            "Augmented train-only DPO + protected revision",
            "mbpp/validation",
            protected["augmented_dpo_validation"]["protected_revised_passed"],
            90,
            "Previous DPO-related baseline; no validation leakage",
        ),
        row(
            "Full-failure DPO",
            "mbpp/validation",
            leaked_dpo["dpo_hf"]["passed"],
            90,
            "Sanity check only; trained with validation failures",
            leakage="yes",
        ),
        row(
            "Protected rule revision",
            "mbpp/validation",
            protected["full_baseline"]["by_split"]["mbpp_validation"]["passed"],
            90,
            "Best validation result",
        ),
    ]
    if args.logic_k5_validation_summary.exists():
        logic_k5_validation = load_json(args.logic_k5_validation_summary)
        validation.append(
            row(
                "LLMCritic54 + logic k=5 DPO",
                "mbpp/validation",
                logic_k5_validation["passed"],
                logic_k5_validation["total"],
                "No validation leakage; includes 7 verifier-selected logic self-play pairs",
            )
        )
    if args.logic_k5_protected_summary.exists():
        logic_k5_protected = load_json(args.logic_k5_protected_summary)
        validation.append(
            row(
                "LLMCritic54 + logic k=5 DPO + protected revision",
                "mbpp/validation",
                logic_k5_protected["passed"],
                logic_k5_protected["total"],
                "Best DPO-related method; no validation leakage",
            )
        )
    else:
        logic_k5_protected = None

    protected_ablation = {
        "overall": {
            "unprotected_passed": metrics["revision_loop"]["revised_passed"],
            "protected_passed": protected["full_baseline"]["passed"],
            "delta": protected["full_baseline"]["passed"] - metrics["revision_loop"]["revised_passed"],
            "unprotected_pass_to_fail": metrics["revision_loop"]["transitions"]["pass_to_fail"],
            "protected_pass_to_fail": protected["full_baseline"]["transitions"]["pass_to_fail"],
        },
        "augmented_dpo_validation": {
            "unprotected_passed": 53,
            "protected_passed": protected["augmented_dpo_validation"]["protected_revised_passed"],
            "delta": protected["augmented_dpo_validation"]["protected_revised_passed"] - 53,
        },
        "train_only_dpo_validation": {
            "unprotected_passed": 49,
            "protected_passed": protected["train_only_dpo_validation"]["protected_revised_passed"],
            "delta": protected["train_only_dpo_validation"]["protected_revised_passed"] - 49,
        },
    }
    if args.logic_k5_validation_summary.exists() and args.logic_k5_protected_summary.exists():
        logic_k5_validation = load_json(args.logic_k5_validation_summary)
        logic_k5_protected = load_json(args.logic_k5_protected_summary)
        protected_ablation["logic_k5_dpo_validation"] = {
            "unprotected_passed": logic_k5_validation["passed"],
            "protected_passed": logic_k5_protected["passed"],
            "delta": logic_k5_protected["passed"] - logic_k5_validation["passed"],
        }

    output = {
        "source_metrics": str(args.metrics),
        "overall_leaderboard": overall,
        "mbpp_validation_leaderboard": sorted(validation, key=lambda item: item["passed"], reverse=True),
        "protected_revision_ablation": protected_ablation,
        "main_takeaway": "Protected rule revision is the strongest current overall method; LLMCritic54 + logic k=5 DPO + protected revision is the strongest no-leakage DPO-related validation method.",
    }

    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.md_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    md = [
        "# Final Method Leaderboard",
        "",
        "## Overall",
        "",
        "| Method | Split | Passed | Total | pass@1 | Leakage | Note |",
        "| --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for item in output["overall_leaderboard"]:
        md.append(
            f"| {item['method']} | {item['split']} | {item['passed']} | {item['total']} | "
            f"{pct(item['pass_rate'])} | {item['leakage']} | {item['note']} |"
        )
    md.extend(
        [
            "",
            "## MBPP Validation",
            "",
            "| Method | Passed | Total | pass@1 | Leakage | Note |",
            "| --- | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for item in output["mbpp_validation_leaderboard"]:
        md.append(
            f"| {item['method']} | {item['passed']} | {item['total']} | "
            f"{pct(item['pass_rate'])} | {item['leakage']} | {item['note']} |"
        )
    md.extend(
        [
            "",
            "## Protected Revision Ablation",
            "",
            "| Setting | Unprotected | Protected | Delta | pass->fail change |",
            "| --- | ---: | ---: | ---: | ---: |",
            (
                f"| Overall | {protected_ablation['overall']['unprotected_passed']} | "
                f"{protected_ablation['overall']['protected_passed']} | "
                f"{protected_ablation['overall']['delta']:+d} | "
                f"{protected_ablation['overall']['unprotected_pass_to_fail']} -> "
                f"{protected_ablation['overall']['protected_pass_to_fail']} |"
            ),
            (
                f"| Augmented DPO validation | {protected_ablation['augmented_dpo_validation']['unprotected_passed']} | "
                f"{protected_ablation['augmented_dpo_validation']['protected_passed']} | "
                f"{protected_ablation['augmented_dpo_validation']['delta']:+d} | 1 -> 0 |"
            ),
            (
                f"| Train-only DPO validation | {protected_ablation['train_only_dpo_validation']['unprotected_passed']} | "
                f"{protected_ablation['train_only_dpo_validation']['protected_passed']} | "
                f"{protected_ablation['train_only_dpo_validation']['delta']:+d} | 1 -> 0 |"
            ),
        ]
    )
    if "logic_k5_dpo_validation" in protected_ablation:
        item = protected_ablation["logic_k5_dpo_validation"]
        md.append(
            f"| Logic k=5 DPO validation | {item['unprotected_passed']} | "
            f"{item['protected_passed']} | {item['delta']:+d} | 0 -> 0 |"
        )
    md.extend(["", "## Takeaway", "", output["main_takeaway"]])
    args.md_output.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"json_output": str(args.json_output), "md_output": str(args.md_output)}, indent=2))


if __name__ == "__main__":
    main()
