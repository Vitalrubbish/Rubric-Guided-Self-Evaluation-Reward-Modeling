# Project Completion Action Plan

日期：2026-07-03  
远程目录：`/data2/acm-group-3/Rubric-Guided-Self-Evaluation-Reward-Modeling`

## 总目标

把选题五 `Rubric-Guided Self-Evaluation and Reward Modeling` 做到可提交状态：作业 3 的错误发现、rubric 自动生成、自评一致性齐全；作业 4 的三种方法至少都有可复现实验、训练/评估证据、负结果解释和后续边界。

## 当前审计结论

| 模块 | 当前状态 | 最重要证据 | 缺口 |
| --- | --- | --- | --- |
| 作业3 Step 1 错误发现 | done | `data/analysis/coding_failures_qwen25_k1.jsonl`, `data/analysis/coding_error_taxonomy_refined.yaml` | 可在报告中解释 coding benchmark 替代 GSM8K/MT-Bench |
| 作业3 Step 2 rubric 生成 | done | `data/rubrics/auto_rubric_refined.json` | 需强调每维评分标准/正反例 |
| 作业3 Step 3 自评一致性 | done | `data/rubrics/auto_rubric_eval_metrics.json` | 当前是 static scorer，不是真 LLM-as-rubric |
| Method 1 rubric-guided DPO/RL | partial/done | 多个 DPO adapter、protected revision | fixed vs updated 主要是离线 audit，需要更清楚地写成 proxy A/B |
| Method 2 Self-Play Error Discovery | strong partial/done | 54 syntax/format LLM critic pairs、7 logic k=5 pairs、多个 DPO | 还缺把 7 条最可靠 logic pairs 合入 DPO 的最终训练消融 |
| Method 3 Meta-Learning | minimal | `docs/meta_transfer_audit.md` | 不是 GSM8K→MATH；只能作为最小跨代码任务迁移，需要报告中诚实标注 |

## 完成标准

项目达到可提交状态需满足：

1. 每个要求都有对应证据文件和表格。
2. 所有训练都只在 train split 上构建 preference data，validation/test 不泄漏。
3. 每个训练后必须有 MBPP validation 和 protected revision 评估。
4. 若训练没有提升，必须写清楚负结果和停止原因。
5. 最终报告能回答：
   - 模型发现了哪些错误模式？
   - 自动 rubric 是否比 generic/random 更能区分好坏？
   - DPO/self-play 是否提升？若没有，为什么？
   - rubric 更新是否有价值？
   - meta-transfer 做到了什么，没做到什么？

## 执行表

| Step | 内容 | 产物 | 验收条件 | 状态 |
| --- | --- | --- | --- | --- |
| 0 | 项目审计与缺口确认 | 本文档 | 明确最短补齐路径 | done |
| 1 | Method 2 最终训练消融：`266 + 7 logic k5` DPO | `outputs/dpo_lora_mbpp_train_augmented_llmcritic54_logic_k5_e1_mlen768` | DPO 训练完成，写 train_metrics | done |
| 2 | 评估 Step 1 adapter | validation/protected validation JSONL 和 summaries | 90 条 validation 全部验证；记录 pass@1 | done |
| 3 | 更新 Method 2 训练结论 | `docs/logic_k5_dpo_results.md`, 总报告 | 明确 logic pairs 是否带来增益 | done |
| 4 | Method 1 fixed-vs-updated proxy A/B 梳理 | `docs/method1_fixed_updated_training_ablation.md` | 把 existing DPO/revision 映射成可解释 A/B | done |
| 5 | Method 3 meta-transfer 梳理和 caveat | `docs/method3_meta_transfer_final.md` | 明确跨代码任务迁移指标和 GSM8K/MATH 缺口 | done |
| 6 | 最终报告/leaderboard 重建 | `docs/final_project_report.md`, `docs/final_method_leaderboard.md` | 所有新结果写入，表格一致 | done |
| 7 | 最终可提交检查 | `docs/submission_readiness_checklist.md` | 每个要求有证据、命令、结论 | done |

## Step 1 训练命令

```bash
CUDA_VISIBLE_DEVICES=1 /data2/acm-group-3/miniconda3/envs/rubric/bin/python scripts/dpo_lora_train.py \
  --model models/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28 \
  --data data/preferences/preference_pairs_qwen25_k1_mbpp_train_augmented_llmcritic54_logic_k5.jsonl \
  --output-dir outputs/dpo_lora_mbpp_train_augmented_llmcritic54_logic_k5_e1_mlen768 \
  --epochs 1 \
  --grad-accum 8 \
  --max-length 768
```

## Step 2 评估命令

```bash
CUDA_VISIBLE_DEVICES=1 /data2/acm-group-3/miniconda3/envs/rubric/bin/python scripts/generate_with_lora_adapter.py \
  --model models/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28 \
  --adapter outputs/dpo_lora_mbpp_train_augmented_llmcritic54_logic_k5_e1_mlen768 \
  --input data/processed/mbpp_validation_prompts.jsonl \
  --output data/responses/dpo_lora_train_augmented_llmcritic54_logic_k5_mbpp_validation.jsonl \
  --dataset mbpp \
  --split validation \
  --batch-size 4 \
  --max-input-length 1536 \
  --max-new-tokens 256

/data2/acm-group-3/miniconda3/envs/rubric/bin/python scripts/verify_mbpp_smoke.py \
  --input data/responses/dpo_lora_train_augmented_llmcritic54_logic_k5_mbpp_validation.jsonl \
  --output data/responses/dpo_lora_train_augmented_llmcritic54_logic_k5_mbpp_validation_labeled.jsonl

/data2/acm-group-3/miniconda3/envs/rubric/bin/python scripts/build_failure_artifacts.py \
  --input data/responses/dpo_lora_train_augmented_llmcritic54_logic_k5_mbpp_validation_labeled.jsonl \
  --failure-output data/eval/dpo_lora_train_augmented_llmcritic54_logic_k5_mbpp_validation_failures.jsonl \
  --summary-output data/eval/dpo_lora_train_augmented_llmcritic54_logic_k5_mbpp_validation_summary.json \
  --taxonomy-output data/eval/dpo_lora_train_augmented_llmcritic54_logic_k5_mbpp_validation_taxonomy.yaml

/data2/acm-group-3/miniconda3/envs/rubric/bin/python scripts/protected_revise_code_outputs.py \
  --labeled data/responses/dpo_lora_train_augmented_llmcritic54_logic_k5_mbpp_validation_labeled.jsonl \
  --output data/responses/dpo_lora_train_augmented_llmcritic54_logic_k5_mbpp_validation_protected_revised.jsonl

/data2/acm-group-3/miniconda3/envs/rubric/bin/python scripts/verify_mbpp_smoke.py \
  --input data/responses/dpo_lora_train_augmented_llmcritic54_logic_k5_mbpp_validation_protected_revised.jsonl \
  --output data/responses/dpo_lora_train_augmented_llmcritic54_logic_k5_mbpp_validation_protected_revised_labeled.jsonl

/data2/acm-group-3/miniconda3/envs/rubric/bin/python scripts/build_failure_artifacts.py \
  --input data/responses/dpo_lora_train_augmented_llmcritic54_logic_k5_mbpp_validation_protected_revised_labeled.jsonl \
  --failure-output data/eval/dpo_lora_train_augmented_llmcritic54_logic_k5_mbpp_validation_protected_revised_failures.jsonl \
  --summary-output data/eval/dpo_lora_train_augmented_llmcritic54_logic_k5_mbpp_validation_protected_revised_summary.json \
  --taxonomy-output data/eval/dpo_lora_train_augmented_llmcritic54_logic_k5_mbpp_validation_protected_revised_taxonomy.yaml
```

## Gate

- 若 logic-k5 DPO validation 或 protected validation 超过 LLMCritic54 DPO，则更新 leaderboard。
- 若持平或下降，则作为负结果：少量 logic self-play pairs 不足以提升 DPO，停止继续加低质量 logic pairs。
- 若 protected validation 达到或超过 protected rule revision baseline `61/90`，进入 MBPP test 评估；否则不跑 test。

## 修订记录

| 时间 | Step | 检查结果 | 方案是否修改 |
| --- | --- | --- | --- |
| 2026-07-03 17:00 | 0 | 作业3完整；Method2 强；Method1/3 需要最终梳理；最值得补跑的是 logic-k5 DPO | 先跑 273-pair DPO，再整理 Method1/3 |
| 2026-07-03 17:08 | 1 | logic-k5 DPO 完成：273 steps，skipped 0，mean loss 0.6468，preference accuracy 0.8059 | 进入 MBPP validation + protected revision 评估 |
| 2026-07-03 17:18 | 2/3 | raw validation 42/90，protected validation 56/90；比 LLMCritic54 protected 54/90 高 2，但低于 protected rule revision 61/90 | 记录为 DPO-related 最好无泄漏结果；不跑 MBPP test |
| 2026-07-03 17:24 | 4 | Method 1 fixed-vs-updated proxy A/B 已写入，包含 rubric quality、revision impact、DPO variants、reward hacking guard | 进入 Method 3 meta-transfer 整理 |
| 2026-07-03 17:27 | 5 | Method 3 最小跨代码迁移审计已整理：MBPP -> HumanEval+，并明确不是 GSM8K->MATH | 进入最终报告和 checklist 重建 |
| 2026-07-03 17:32 | 6/7 | final report、leaderboard、assignment alignment、submission checklist 已更新 | 项目达到可提交状态 |
