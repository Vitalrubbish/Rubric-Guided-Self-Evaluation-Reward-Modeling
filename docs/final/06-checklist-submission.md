# Submission Readiness Checklist

日期：2026-07-04  
项目：Rubric-Guided Self-Evaluation and Reward Modeling

## 总结

当前项目已达到可提交状态。核心交付是一个 coding benchmark 上的 self-evolving/rubric-guided 实验闭环；同时已补充 GSM8K n=100 小规模真实闭环、GSM8K -> MATH safe-subset transfer，以及 full MATH verifier pressure test，用来对齐原始选题推荐 benchmark。

## 作业 3 对齐

| 要求 | 状态 | 证据 | 备注 |
| --- | --- | --- | --- |
| 生成 500+ responses 并用 verifier 标失败 | done | `data/responses/coding_all_qwen25_vllm_k1_labeled_v2.jsonl` | 1128 条 coding prompts，551 个失败 |
| 错误 clustering + taxonomy | done | `data/analysis/failure_clusters_qwen25_k1.jsonl`, `data/analysis/coding_error_taxonomy_refined.yaml` | 18 个 refined clusters |
| 自动生成 rubric | done | `data/rubrics/auto_rubric_refined.json` | 6 个维度 |
| rubric 覆盖率/区分度 | done | `data/rubrics/auto_rubric_eval_metrics.json` | AUC 0.801，Kappa 0.525 |
| 与 generic/random 对比 | done | `data/rubrics/generic_rubric_eval_metrics.json`, `data/rubrics/random_rubric_eval_metrics.json` | random 为静态 scorer ablation |
| 推荐 benchmark GSM8K 小闭环 | done appendix | `docs/gsm8k_alignment_results.md`, `data/responses/gsm8k_qwen25_k1_n100_labeled.jsonl`, `data/rubrics/gsm8k_auto_rubric_metrics_n100.json` | 100 条 GSM8K，exact verifier accuracy 72%，static rubric AUC 0.849 |

## 作业 4 Method 1 对齐

| 要求 | 状态 | 证据 | 备注 |
| --- | --- | --- | --- |
| Error-pattern -> rubric -> reward/DPO 闭环 | done | `scripts/dpo_lora_train.py`, 多个 `outputs/dpo_lora_*` | 已完成多轮 DPO ablation |
| rubric 是否进化 | done/proxy | `docs/rubric_evolution_analysis.md`, `docs/method1_fixed_updated_training_ablation.md` | fixed vs updated 是 proxy A/B |
| 性能是否提升 | done | `docs/final_method_leaderboard.md` | DPO-related 最好 protected validation 56/90 |
| reward hacking 追踪 | done | `docs/protected_rule_revision_results.md` | protected pass->fail 为 0 |

## 作业 4 Method 2 对齐

| 要求 | 状态 | 证据 | 备注 |
| --- | --- | --- | --- |
| A -> 找错 -> B -> A<B preference pairs | done | `data/self_play/llm_critic_pairs_mbpp_train_n54_v1.jsonl`, `data/self_play/llm_critic_pairs_mbpp_train_logic_n20_k5.jsonl` | syntax/format 54 条，logic 7 条 |
| 与标准 self-rewarding 区分 | done | `docs/logic_two_stage_failure_diagnosis.md`, `docs/logic_algorithm_sketch_n20_results.md` | 明确要求先找错再修复 |
| 找错能力追踪 | done | `data/self_play/llm_critic_metrics_*` | syntax/format 54/54；logic k=5 7/20 |
| 哪些错误能自发现 | done | `docs/logic_two_stage_failure_diagnosis.md` | logic 失败主要是 right_spec_wrong_algorithm |
| preference pair DPO 训练 | done | `docs/logic_k5_dpo_results.md` | 273-pair DPO，protected validation 56/90 |

## 作业 4 Method 3 对齐

| 要求 | 状态 | 证据 | 备注 |
| --- | --- | --- | --- |
| 多任务/迁移 self-evaluation | done | `docs/method3_meta_transfer_final.md`, `docs/meta_transfer_audit.md`, `docs/gsm8k_to_math_transfer_results.md` | MBPP -> HumanEval+，GSM8K -> MATH safe-subset，以及 full MATH verifier pressure test |
| zero-shot rubric quality | partial | `data/analysis/meta_transfer_audit.json` | 评估 refined coding rubric 的跨代码任务区分度 |
| 泛化性结论 | partial/done | `docs/method3_meta_transfer_final.md` | 可声称跨代码、verifier-safe 跨数学迁移，以及 full MATH n=100 压力测试；完整 MATH test split 仍是 future work |
| GSM8K -> MATH safe-subset transfer | done | `docs/gsm8k_to_math_transfer_results.md`, `data/rubrics/math_transfer_gsm8k_rubric_metrics_n100.json` | GSM8K-derived rubric zero-shot 到 MATH safe subset，AUC 0.883，Kappa 0.181 |
| full MATH verifier pressure test | done appendix | `docs/math_full_verifier_results.md`, `data/rubrics/math_full_gsm8k_rubric_metrics_n100.json` | all subjects / Level 1-5 n=100；full verifier gold 自检 100/100；GSM8K-derived AUC 0.873 |

## 最终核心数字

| 指标 | 数值 |
| --- | ---: |
| Baseline pass@1 | 577/1128 = 51.15% |
| Protected rule revision | 755/1128 = 66.93% |
| Auto rubric AUC | 0.801 |
| Auto rubric Kappa | 0.525 |
| Best no-leakage DPO-related validation | 56/90 = 62.22% |
| Best overall validation | 61/90 = 67.78% |
| GSM8K appendix accuracy | 72/100 = 72.00% |
| GSM8K static rubric AUC | 0.849 |
| GSM8K static rubric Kappa | 0.051 |
| MATH safe-subset accuracy | 83/100 = 83.00% |
| GSM8K -> MATH zero-shot AUC | 0.883 |
| GSM8K -> MATH zero-shot Kappa | 0.181 |
| MATH full-format pressure accuracy | 43/100 = 43.00% |
| Full MATH GSM8K-derived zero-shot AUC | 0.873 |
| Full MATH GSM8K-derived zero-shot Kappa | 0.123 |

## 必须诚实说明的 caveats

1. 本项目主 benchmark 是 MBPP + HumanEval+ coding tasks；GSM8K 已补 n=100 附录闭环，但不是主训练 benchmark。
2. Method 1 的 fixed-vs-updated 是 proxy A/B，不是完整在线双轨 RL。
3. Method 3 已包含 MBPP -> HumanEval+、GSM8K -> MATH safe-subset，以及 full MATH verifier n=100 pressure test；但还不是完整 MATH test split 上的 meta-learning。
4. DPO 训练完成且有提升证据，但最终仍不如 protected deterministic revision。
5. random rubric ablation 当前是 static scorer 接口，不是真 LLM 读取随机 rubric 后评分。
6. GSM8K self-evaluation 的 static AUC 较好，但 Kappa 很低，说明阈值化自评仍弱；upper-bound 指标使用 verifier failure pattern，不能当成部署时自评能力。

## 最终入口文档

- `docs/final_project_report.md`
- `docs/final_method_leaderboard.md`
- `docs/assignment_requirement_alignment.md`
- `docs/project_completion_action_plan.md`
- `docs/method1_fixed_updated_training_ablation.md`
- `docs/method1_post_gsm8k_audit.md`
- `docs/method2_final_audit.md`
- `docs/method3_meta_transfer_final.md`
- `docs/method3_post_gsm8k_audit.md`
- `docs/logic_k5_dpo_results.md`
- `docs/gsm8k_alignment_results.md`
- `docs/gsm8k_to_math_transfer_results.md`
- `docs/math_full_verifier_results.md`

## 判定

可提交：yes

建议答辩主线：

1. 错误模式自动发现 -> refined rubric。
2. refined rubric 的自评区分度优于 generic/random。
3. protected revision 是当前最强 self-improvement baseline。
4. self-play critic 能产生 preference pairs，logic pairs 对 DPO protected cascade 有小幅增益。
5. GSM8K n=100 作为推荐 benchmark 附录：72% exact accuracy，rubric static AUC 0.849，但 Kappa 低，说明“会排序”和“能稳定自判 pass/fail”不是同一件事。
6. GSM8K -> MATH safe-subset：MATH baseline 83/100，GSM8K-derived rubric zero-shot AUC 0.883；MATH-derived rubric Kappa 更高，说明 target failures 对阈值判定有帮助。
7. Full MATH verifier pressure test：all subjects / Level 1-5 n=100，baseline 43/100，GSM8K-derived AUC 0.873；说明 ranking signal 仍在，但 full MATH 的阈值自评和 final-answer discipline 更难。
8. 负结果同样重要：单纯增加解释/algorithm sketch 不足以提升 7B 模型的 logic repair。
