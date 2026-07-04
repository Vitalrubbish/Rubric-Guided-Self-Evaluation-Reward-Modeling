# Assignment Requirement Alignment

这份文档按老师给的三种方法逐项对齐当前项目状态，避免把已经做的东西和还没做的东西混在一起。

## 总览

| 作业要求 | 当前状态 | 证据文件 | 还缺什么 |
| --- | --- | --- | --- |
| Step 1: 大量 response + verifier 标失败 | done | `data/responses/coding_all_qwen25_vllm_k1_labeled_v2.jsonl`, `data/analysis/coding_baseline_summary_qwen25_k1.json`, `data/responses/gsm8k_qwen25_k1_n100_labeled.jsonl` | coding 主线 1128 条；GSM8K appendix 100 条 |
| Step 1: clustering + 错误 taxonomy | done | `data/analysis/failure_clusters_qwen25_k1.jsonl`, `data/analysis/coding_error_taxonomy_refined.yaml`, `data/analysis/gsm8k_error_taxonomy_qwen25_k1_n100.yaml` | GSM8K taxonomy 为启发式归因，coding taxonomy 更完整 |
| Step 2: 自动生成 rubric | done | `data/rubrics/auto_rubric_refined.json`, `data/rubrics/gsm8k_auto_rubric_n100.json` | 可让模型读取更多失败案例后生成 v2/v3 |
| Step 3: self-evaluation 与外部评判一致性 | done | `data/rubrics/auto_rubric_eval_metrics.json`, `data/rubrics/gsm8k_auto_rubric_metrics_n100.json` | coding AUC 0.801/Kappa 0.525；GSM8K static AUC 0.849/Kappa 0.051 |
| Method 1: rubric-guided DPO/RL 闭环 | partial/done baseline | `scripts/dpo_lora_train.py`, `outputs/dpo_lora_mbpp_train_augmented_e1_212_mlen768/train_metrics.json` | DPO 已训练；还需多轮 self-evolving 迭代 |
| Method 1: self-updating vs fixed rubric | partial | `docs/rubric_evolution_analysis.md` | 当前是 fixed generic vs refined rubric 的离线对比，不是完整在线 A/B |
| Method 1: reward hacking 追踪 | partial/done baseline | `docs/protected_rule_revision_results.md`, `data/eval/vllm_baseline_protected_revision_comparison.json` | 当前追踪 pass->fail；后续 DPO 每轮也要追踪格式投机/测试投机 |
| Method 2: explicit error discovery -> B -> A<B | done | `data/self_play/self_play_pairs_from_protected_revision.jsonl`, `data/self_play/llm_critic_pairs_mbpp_train_n54_v1.jsonl`, `data/self_play/llm_critic_pairs_mbpp_train_logic_n20_k5.jsonl`, `docs/logic_multicandidate_n20_k5_results.md`, `docs/logic_k5_dpo_results.md`, `docs/logic_spec_prompt_n20_results.md`, `docs/logic_two_stage_n20_results.md`, `docs/logic_two_stage_failure_diagnosis.md`, `docs/logic_algorithm_sketch_n20_results.md` | 已有 178 条 proxy pairs、54 条 syntax/format LLM critic pairs、7 条 logic k=5 critic pairs；最终 273-pair DPO raw 42/90，protected 56/90 |
| Method 2: 错误检出率/误报率 | done small-scale | `data/self_play/self_play_error_discovery_metrics.json`, `data/self_play/llm_critic_metrics_mbpp_train_n54_v1.json`, `data/self_play/llm_critic_metrics_mbpp_train_logic_n20_k5.json`, `data/self_play/llm_critic_metrics_mbpp_train_logic_n20_specfirst_v1.json`, `data/self_play/llm_critic_metrics_mbpp_train_logic_n20_twostage_v1.json`, `data/self_play/llm_critic_metrics_mbpp_train_logic_n20_algosketch_v1.json`, `data/analysis/two_stage_failure_diagnosis.jsonl` | syntax/format critic 54/54；logic 单候选 2/20，k=3 到 6/20，k=5 到 7/20；spec-first v1 为 1/20，two-stage v1 为 3/20，algorithm-sketch v1 为 2/20；失败诊断 17/17 为 `right_spec_wrong_algorithm` |
| Method 3: meta-learning / 跨任务迁移 | minimal done + appendix | `docs/method3_meta_transfer_final.md`, `docs/gsm8k_alignment_results.md` | MBPP -> HumanEval+ 是主迁移；GSM8K 是推荐 benchmark 小闭环，不是完整 GSM8K -> MATH meta-learning |

## 现在最适合继续做什么

1. 若还有算力，可在 GSM8K 上做 repair/self-play preference pairs，但这会是新增实验，不影响当前可提交性。
2. 若要进一步贴近 Method 3，可做 MATH 小样本 zero-shot rubric evaluation；当前已有 MBPP -> HumanEval+ 与 GSM8K appendix。
3. 答辩时主线保持 coding，自然补充 GSM8K appendix，避免把两条实验混成一个训练闭环。
