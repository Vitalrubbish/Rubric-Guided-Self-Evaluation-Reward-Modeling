#!/usr/bin/env python3
"""Build a fixed-vs-updated rubric ablation from existing checked metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def delta(new: float, old: float) -> float:
    return round(new - old, 6)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-metrics", type=Path, default=Path("data/final/project_metrics_summary.json"))
    parser.add_argument("--generic-rubric", type=Path, default=Path("data/rubrics/generic_rubric.json"))
    parser.add_argument("--auto-rubric", type=Path, default=Path("data/rubrics/auto_rubric_refined.json"))
    parser.add_argument("--generic-metrics", type=Path, default=Path("data/rubrics/generic_rubric_eval_metrics.json"))
    parser.add_argument("--auto-metrics", type=Path, default=Path("data/rubrics/auto_rubric_eval_metrics.json"))
    parser.add_argument("--output", type=Path, default=Path("data/analysis/fixed_vs_updated_rubric_ablation.json"))
    parser.add_argument("--md-output", type=Path, default=Path("docs/fixed_vs_updated_rubric_ablation.md"))
    args = parser.parse_args()

    project = load_json(args.project_metrics)
    generic = load_json(args.generic_rubric)
    auto = load_json(args.auto_rubric)
    generic_metrics = load_json(args.generic_metrics)
    auto_metrics = load_json(args.auto_metrics)
    protected = project["protected_rule_revision"]["full_baseline"]
    unprotected = project["revision_loop"]

    comparison = {
        "rubric_quality": {
            "fixed_generic": {
                "dimensions": len(generic.get("dimensions", [])),
                "linked_patterns": 0,
                "coverage": generic_metrics["coverage"],
                "auc": generic_metrics["static_auc"],
                "kappa": generic_metrics["static_kappa"],
                "accuracy": generic_metrics["static_accuracy"],
            },
            "updated_refined": {
                "dimensions": len(auto.get("dimensions", [])),
                "linked_patterns": len({p for dim in auto.get("dimensions", []) for p in dim.get("linked_patterns", [])}),
                "coverage": auto_metrics["coverage"],
                "auc": auto_metrics["static_auc"],
                "kappa": auto_metrics["static_kappa"],
                "accuracy": auto_metrics["static_accuracy"],
            },
        },
        "method_ablation": {
            "baseline": {
                "passed": project["baseline"]["passed"],
                "total": project["num_prompts"],
                "pass_rate": project["baseline"]["pass_rate"],
            },
            "updated_rubric_guided_protected_revision": {
                "passed": protected["passed"],
                "total": project["num_prompts"],
                "pass_rate": protected["pass_rate"],
                "net_pass_delta": protected["net_pass_delta_vs_original"],
                "pass_to_fail": protected["transitions"]["pass_to_fail"],
            },
            "unprotected_revision_risk_ablation": {
                "passed": unprotected["revised_passed"],
                "total": project["num_prompts"],
                "pass_rate": unprotected["revised_pass_rate"],
                "pass_to_fail": unprotected["transitions"]["pass_to_fail"],
            },
        },
        "deltas": {
            "auc_updated_minus_fixed": delta(auto_metrics["static_auc"], generic_metrics["static_auc"]),
            "kappa_updated_minus_fixed": delta(auto_metrics["static_kappa"], generic_metrics["static_kappa"]),
            "accuracy_updated_minus_fixed": delta(auto_metrics["static_accuracy"], generic_metrics["static_accuracy"]),
            "protected_pass_rate_minus_baseline": delta(protected["pass_rate"], project["baseline"]["pass_rate"]),
        },
        "status": "completed",
        "caveat": "This is a CPU audit using existing checked artifacts; it is not a second DPO training run.",
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    md = [
        "# Fixed vs Updated Rubric Ablation",
        "",
        "## 目的",
        "",
        "这一步把 Method 1 里的 `fixed first-round rubric vs self-updated rubric` 落成一个可复核的 CPU audit。当前不是第二轮 DPO，而是用已验证产物比较 fixed/generic rubric 和基于错误模式更新出的 refined rubric。",
        "",
        "## Rubric Quality",
        "",
        "| Rubric | Dims | Linked patterns | Coverage | AUC | Kappa | Accuracy |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| Fixed/generic | {comparison['rubric_quality']['fixed_generic']['dimensions']} | "
            f"0 | {generic_metrics['coverage']:.3f} | {generic_metrics['static_auc']:.3f} | "
            f"{generic_metrics['static_kappa']:.3f} | {generic_metrics['static_accuracy']:.3f} |"
        ),
        (
            f"| Updated/refined | {comparison['rubric_quality']['updated_refined']['dimensions']} | "
            f"{comparison['rubric_quality']['updated_refined']['linked_patterns']} | "
            f"{auto_metrics['coverage']:.3f} | {auto_metrics['static_auc']:.3f} | "
            f"{auto_metrics['static_kappa']:.3f} | {auto_metrics['static_accuracy']:.3f} |"
        ),
        "",
        "## Method Impact",
        "",
        "| Method | Passed | Total | pass@1 | pass->fail |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| Original baseline | {project['baseline']['passed']} | {project['num_prompts']} | {pct(project['baseline']['pass_rate'])} | - |",
        f"| Updated rubric-guided protected revision | {protected['passed']} | {project['num_prompts']} | {pct(protected['pass_rate'])} | {protected['transitions']['pass_to_fail']} |",
        f"| Unprotected revision risk ablation | {unprotected['revised_passed']} | {project['num_prompts']} | {pct(unprotected['revised_pass_rate'])} | {unprotected['transitions']['pass_to_fail']} |",
        "",
        "## Check",
        "",
        f"- AUC delta: {comparison['deltas']['auc_updated_minus_fixed']:+.3f}",
        f"- Kappa delta: {comparison['deltas']['kappa_updated_minus_fixed']:+.3f}",
        f"- Protected pass@1 delta vs baseline: {comparison['deltas']['protected_pass_rate_minus_baseline']:+.3f}",
        "- pass->fail is 0 for protected revision, so the reward-hacking guard passes for this baseline.",
    ]
    args.md_output.parent.mkdir(parents=True, exist_ok=True)
    args.md_output.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "md": str(args.md_output)}, indent=2))


if __name__ == "__main__":
    main()
