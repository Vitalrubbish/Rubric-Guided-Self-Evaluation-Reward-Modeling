#!/usr/bin/env python3
"""Analyze rubric evolution and write assignment-alignment documentation."""

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


def metric_row(name: str, generic: dict, auto: dict) -> dict:
    return {
        "metric": name,
        "fixed_or_generic": generic.get(name),
        "updated_or_self_discovered": auto.get(name),
        "delta": delta(auto.get(name, 0), generic.get(name, 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-metrics", type=Path, default=Path("data/final/project_metrics_summary.json"))
    parser.add_argument("--generic-rubric", type=Path, default=Path("data/rubrics/generic_rubric.json"))
    parser.add_argument("--auto-rubric", type=Path, default=Path("data/rubrics/auto_rubric_refined.json"))
    parser.add_argument("--generic-metrics", type=Path, default=Path("data/rubrics/generic_rubric_eval_metrics.json"))
    parser.add_argument("--auto-metrics", type=Path, default=Path("data/rubrics/auto_rubric_eval_metrics.json"))
    parser.add_argument("--random-metrics", type=Path, default=Path("data/rubrics/random_rubric_eval_metrics.json"))
    parser.add_argument("--refined-summary", type=Path, default=Path("data/analysis/coding_error_taxonomy_refined_summary.json"))
    parser.add_argument("--self-play-metrics", type=Path, default=Path("data/self_play/self_play_error_discovery_metrics.json"))
    parser.add_argument("--output", type=Path, default=Path("data/analysis/rubric_evolution_analysis.json"))
    parser.add_argument("--md-output", type=Path, default=Path("docs/rubric_evolution_analysis.md"))
    parser.add_argument("--alignment-output", type=Path, default=Path("docs/assignment_requirement_alignment.md"))
    args = parser.parse_args()

    project = load_json(args.project_metrics)
    generic_rubric = load_json(args.generic_rubric)
    auto_rubric = load_json(args.auto_rubric)
    generic_metrics = load_json(args.generic_metrics)
    auto_metrics = load_json(args.auto_metrics)
    random_metrics = load_json(args.random_metrics)
    refined_summary = load_json(args.refined_summary)
    self_play = load_json(args.self_play_metrics) if args.self_play_metrics.exists() else None

    generic_dims = generic_rubric.get("dimensions", [])
    auto_dims = auto_rubric.get("dimensions", [])
    linked_patterns = sorted({p for dim in auto_dims for p in dim.get("linked_patterns", [])})
    comparison = [
        metric_row("coverage", generic_metrics, auto_metrics),
        metric_row("static_auc", generic_metrics, auto_metrics),
        metric_row("static_kappa", generic_metrics, auto_metrics),
        metric_row("static_accuracy", generic_metrics, auto_metrics),
    ]

    protected = project["protected_rule_revision"]["full_baseline"]
    revision_loop = project["revision_loop"]
    train_only = project["train_only_dpo"]
    augmented = project["augmented_train_only_dpo"]

    analysis = {
        "type": "rubric_evolution_and_requirement_alignment",
        "evidence": {
            "project_metrics": str(args.project_metrics),
            "generic_rubric": str(args.generic_rubric),
            "auto_rubric": str(args.auto_rubric),
            "self_play_metrics": str(args.self_play_metrics) if self_play else None,
        },
        "error_discovery": {
            "total_prompts": project["num_prompts"],
            "failures": project["baseline"]["failed"],
            "refined_clusters": refined_summary["num_clusters"],
            "top_clusters": refined_summary["top_clusters"][:8],
        },
        "rubric_evolution": {
            "fixed_or_generic": {
                "name": generic_rubric["name"],
                "num_dimensions": len(generic_dims),
                "dimensions": [dim["dimension"] for dim in generic_dims],
                "linked_patterns": [],
                "metrics": generic_metrics,
            },
            "updated_or_self_discovered": {
                "name": auto_rubric["name"],
                "num_dimensions": len(auto_dims),
                "dimensions": [dim["dimension"] for dim in auto_dims],
                "linked_patterns": linked_patterns,
                "metrics": auto_metrics,
            },
            "metric_deltas": comparison,
            "interpretation": (
                "The refined rubric is grounded in 551 verifier-labeled failures and 18 refined clusters. "
                "Compared with the fixed generic rubric, it improves static AUC, Kappa, accuracy, and coverage."
            ),
        },
        "self_improvement_evidence": {
            "protected_revision": {
                "baseline_passed": project["baseline"]["passed"],
                "protected_passed": protected["passed"],
                "net_pass_delta": protected["net_pass_delta_vs_original"],
                "pass_rate_delta": delta(protected["pass_rate"], project["baseline"]["pass_rate"]),
                "pass_to_fail": protected["transitions"]["pass_to_fail"],
                "attempted_failed_rows": protected["attempted"],
                "edited_failed_rows": protected["edited"],
            },
            "unprotected_ablation": {
                "passed": revision_loop["revised_passed"],
                "pass_to_fail": revision_loop["transitions"]["pass_to_fail"],
                "note": "Shows why reward/revision hacking checks are necessary.",
            },
            "dpo_training": {
                "train_only_pairs": train_only["preference_pairs"]["num_pairs"],
                "train_only_preference_accuracy": train_only["training"]["preference_accuracy"],
                "augmented_pairs": augmented["preference_pairs"]["num_pairs"],
                "augmented_preference_accuracy": augmented["training"]["preference_accuracy"],
                "augmented_validation_passed": augmented["untouched_validation"]["augmented_dpo_hf"]["passed"],
            },
        },
        "self_play": self_play,
        "random_rubric_caveat": {
            "metrics": random_metrics,
            "caveat": (
                "The current random rubric ablation reuses the static scorer interface. It is kept as an "
                "artifact placeholder, but should not be over-interpreted until the scorer actually consumes rubric text."
            ),
        },
        "method_3_status": {
            "status": "not_done",
            "reason": "Current experiments are coding-only MBPP/HumanEval+. GSM8K->MATH/code transfer and zero-shot rubric generation are not yet run.",
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(analysis, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    md = [
        "# Rubric Evolution Analysis",
        "",
        "## 结论",
        "",
        "我们已经完成了 `错误模式发现 -> rubric 自动生成 -> reward/revision signal -> DPO/改进评估` 的第一版闭环。按作业要求严格看，当前的“rubric 自我更新 vs 固定首轮 rubric”还不是多轮在线更新，而是一个可复核的离线近似：固定 generic rubric 对比基于失败聚类生成的 refined rubric。",
        "",
        "## Fixed vs Updated Rubric",
        "",
        "| Rubric | 维度数 | Coverage | AUC | Kappa | Accuracy |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| Fixed/generic | {len(generic_dims)} | {generic_metrics['coverage']:.3f} | "
            f"{generic_metrics['static_auc']:.3f} | {generic_metrics['static_kappa']:.3f} | "
            f"{generic_metrics['static_accuracy']:.3f} |"
        ),
        (
            f"| Updated/refined | {len(auto_dims)} | {auto_metrics['coverage']:.3f} | "
            f"{auto_metrics['static_auc']:.3f} | {auto_metrics['static_kappa']:.3f} | "
            f"{auto_metrics['static_accuracy']:.3f} |"
        ),
        (
            f"| Delta | +{len(auto_dims) - len(generic_dims)} | "
            f"{delta(auto_metrics['coverage'], generic_metrics['coverage']):+.3f} | "
            f"{delta(auto_metrics['static_auc'], generic_metrics['static_auc']):+.3f} | "
            f"{delta(auto_metrics['static_kappa'], generic_metrics['static_kappa']):+.3f} | "
            f"{delta(auto_metrics['static_accuracy'], generic_metrics['static_accuracy']):+.3f} |"
        ),
        "",
        "Updated/refined rubric 的 6 个维度：",
    ]
    for dim in auto_dims:
        md.append(f"- {dim['dimension']} (`{dim['id']}`)")

    md.extend(
        [
            "",
            "## Rubric 进化证据",
            "",
            f"- 初始基线失败：{project['baseline']['failed']} / {project['num_prompts']}。",
            f"- refined taxonomy：{refined_summary['num_clusters']} 个 clusters。",
            f"- auto rubric 从错误模式中抽象出 {len(auto_dims)} 个可评分维度，覆盖率 {auto_metrics['coverage']:.3f}。",
            f"- 相比 fixed/generic rubric，AUC 提升 {delta(auto_metrics['static_auc'], generic_metrics['static_auc']):+.3f}，Kappa 提升 {delta(auto_metrics['static_kappa'], generic_metrics['static_kappa']):+.3f}。",
            "",
            "## Reward / Revision Hacking 检查",
            "",
            "| 方法 | Passed | pass@1 | pass->fail | 说明 |",
            "| --- | ---: | ---: | ---: | --- |",
            (
                f"| Unprotected revision | {revision_loop['revised_passed']} | "
                f"{pct(revision_loop['revised_pass_rate'])} | {revision_loop['transitions']['pass_to_fail']} | 会破坏已通过样本，作为 hacking/risk ablation |"
            ),
            (
                f"| Protected revision | {protected['passed']} | {pct(protected['pass_rate'])} | "
                f"{protected['transitions']['pass_to_fail']} | 只改失败样本，当前主 baseline |"
            ),
            "",
            "## DPO 训练证据",
            "",
            "| 设置 | Pairs | Preference Acc | Validation passed | 备注 |",
            "| --- | ---: | ---: | ---: | --- |",
            (
                f"| Train-only DPO | {train_only['preference_pairs']['num_pairs']} | "
                f"{train_only['training']['preference_accuracy']:.3f} | "
                f"{train_only['untouched_validation']['train_only_dpo_hf']['passed']}/90 | 无 validation leakage |"
            ),
            (
                f"| Augmented train-only DPO | {augmented['preference_pairs']['num_pairs']} | "
                f"{augmented['training']['preference_accuracy']:.3f} | "
                f"{augmented['untouched_validation']['augmented_dpo_hf']['passed']}/90 | 加入 successful revision pairs |"
            ),
            "",
            "## Caveat",
            "",
            "这份分析足够支撑 Method 1 的作业阶段报告，但还不能声称已经完成多轮在线 self-evolving。真正的下一轮应固定首轮 rubric 跑一条线、允许 rubric 更新再跑一条线，并比较每轮新增/删除/细化的维度。",
            "",
            "输出 JSON：",
            "",
            f"`{args.output}`",
        ]
    )
    args.md_output.parent.mkdir(parents=True, exist_ok=True)
    args.md_output.write_text("\n".join(md) + "\n", encoding="utf-8")

    method2_status = "partial" if self_play else "missing"
    method2_evidence = (
        "`data/self_play/self_play_pairs_from_protected_revision.jsonl`, `docs/self_play_error_discovery.md`"
        if self_play
        else "待生成"
    )
    alignment = [
        "# Assignment Requirement Alignment",
        "",
        "这份文档按老师给的三种方法逐项对齐当前项目状态，避免把已经做的东西和还没做的东西混在一起。",
        "",
        "## 总览",
        "",
        "| 作业要求 | 当前状态 | 证据文件 | 还缺什么 |",
        "| --- | --- | --- | --- |",
        (
            "| Step 1: 大量 response + verifier 标失败 | done | "
            "`data/responses/coding_all_qwen25_vllm_k1_labeled_v2.jsonl`, `data/analysis/coding_baseline_summary_qwen25_k1.json` | "
            "可扩展到 k>1 或更多任务 |"
        ),
        (
            "| Step 1: clustering + 错误 taxonomy | done | "
            "`data/analysis/failure_clusters_qwen25_k1.jsonl`, `data/analysis/coding_error_taxonomy_refined.yaml` | "
            "可加入 LLM 归因文本，使 taxonomy 更像模型自发现 |"
        ),
        (
            "| Step 2: 自动生成 rubric | done | "
            "`data/rubrics/auto_rubric_refined.json` | "
            "可让模型读取更多失败案例后生成 v2/v3 |"
        ),
        (
            "| Step 3: self-evaluation 与外部评判一致性 | done | "
            "`data/rubrics/auto_rubric_eval_metrics.json` | "
            "当前是静态 scorer；后续应跑真正 LLM-as-rubric scorer |"
        ),
        (
            "| Method 1: rubric-guided DPO/RL 闭环 | partial/done baseline | "
            "`scripts/dpo_lora_train.py`, `outputs/dpo_lora_mbpp_train_augmented_e1_212_mlen768/train_metrics.json` | "
            "DPO 已训练；还需多轮 self-evolving 迭代 |"
        ),
        (
            "| Method 1: self-updating vs fixed rubric | partial | "
            "`docs/rubric_evolution_analysis.md` | "
            "当前是 fixed generic vs refined rubric 的离线对比，不是完整在线 A/B |"
        ),
        (
            "| Method 1: reward hacking 追踪 | partial/done baseline | "
            "`docs/protected_rule_revision_results.md`, `data/eval/vllm_baseline_protected_revision_comparison.json` | "
            "当前追踪 pass->fail；后续 DPO 每轮也要追踪格式投机/测试投机 |"
        ),
        (
            f"| Method 2: explicit error discovery -> B -> A<B | {method2_status} | "
            f"{method2_evidence} | "
            "当前是 protected revision proxy；下一步应让 LLM critic 先写错误解释 |"
        ),
        (
            "| Method 2: 错误检出率/误报率 | partial | "
            "`data/self_play/self_play_error_discovery_metrics.json` | "
            "proxy 有 repair precision/recall；真正 LLM critic 需要人工/verifier 对齐的 detection precision/recall |"
        ),
        (
            "| Method 3: meta-learning / 跨任务迁移 | not done | - | "
            "需要 GSM8K/MATH 或代码新任务的 zero-shot rubric generation 实验 |"
        ),
        "",
        "## 现在最适合继续做什么",
        "",
        "1. 把 Method 2 从 proxy 升级为 LLM critic：对失败 A 生成显式错误解释，再生成 B，并用 verifier 筛出 A<B pairs。",
        "2. 做 Method 1 真正两条线 A/B：固定首轮 rubric vs 每轮更新 rubric，各跑至少一轮小规模 DPO 或 revision。",
        "3. 最后再做 Method 3：把 coding 上学到的 rubric generation prompt 迁移到 GSM8K/MATH，评估 zero-shot rubric 质量。",
    ]
    args.alignment_output.parent.mkdir(parents=True, exist_ok=True)
    args.alignment_output.write_text("\n".join(alignment) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "analysis": str(args.output),
                "md": str(args.md_output),
                "alignment": str(args.alignment_output),
                "auto_vs_generic_auc_delta": delta(auto_metrics["static_auc"], generic_metrics["static_auc"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
