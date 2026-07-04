# LLM-Critic 54 + DPO Execution Plan

日期：2026-07-03  
远程目录：`/data2/acm-group-3/Rubric-Guided-Self-Evaluation-Reward-Modeling`

## 目标

把上一阶段 16 条真实 LLM critic 探针扩大到 MBPP train 中 54 条 proxy-success 样本，然后把 LLM critic preference pairs 合并进无验证集泄漏的训练数据，尝试一轮 DPO，并在 MBPP validation 上复核。

## 执行表

| Step | 内容 | 产物 | 验收条件 | 状态 |
| --- | --- | --- | --- | --- |
| 0 | 资源与产物盘点 | 本文档 | 记录 GPU 满载、已有 n=16、已有 DPO baseline | done |
| 1 | LLM critic 扩到 54 条 MBPP train | `data/self_play/llm_critic_mbpp_train_n54_v1.jsonl` | 生成 54 条 critic+revision | done |
| 2 | verifier + pair 构建 | `data/self_play/llm_critic_pairs_mbpp_train_n54_v1.jsonl` | 记录 attempted/repaired/pairs；失败则分析是否需清洗/重跑 | done |
| 3 | 合并偏好数据 | `data/preferences/preference_pairs_qwen25_k1_mbpp_train_augmented_llmcritic54.jsonl` | 只含 MBPP train；预期约 266 pairs | done |
| 4 | DPO 训练 | `outputs/dpo_lora_mbpp_train_augmented_llmcritic54_e1_mlen768/train_metrics.json` | 训练完成并写 metrics；若资源失败则记录阻塞 | done |
| 5 | MBPP validation 生成与 verifier | `data/eval/dpo_lora_train_augmented_llmcritic54_mbpp_validation_summary.json` | 90 条 validation 全部验证，记录 pass@1 | done |
| 6 | protected revision 级联评估 | `data/eval/dpo_lora_train_augmented_llmcritic54_mbpp_validation_protected_revised_summary.json` | 记录 protected 后 pass@1 与 pass->fail 风险 | done |
| 7 | 汇总报告与最终检查 | `docs/llmcritic54_dpo_results.md` | JSON 可读、行数对齐、报告更新 | done |

## Step 1/2 命令

```bash
CUDA_VISIBLE_DEVICES=1 /data2/acm-group-3/miniconda3/envs/rubric/bin/python scripts/llm_self_play_critic.py \
  --model models/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28 \
  --labeled data/responses/coding_all_qwen25_vllm_k1_labeled_v2.jsonl \
  --rubric data/rubrics/auto_rubric_refined.json \
  --prefer-protected-success data/responses/coding_all_qwen25_vllm_k1_protected_revised_labeled.jsonl \
  --dataset mbpp \
  --split train \
  --limit 54 \
  --max-new-tokens 384 \
  --output data/self_play/llm_critic_mbpp_train_n54_v1.jsonl

/data2/acm-group-3/miniconda3/envs/rubric/bin/python scripts/verify_mbpp_smoke.py \
  --input data/self_play/llm_critic_mbpp_train_n54_v1.jsonl \
  --output data/self_play/llm_critic_mbpp_train_n54_v1_labeled.jsonl

/data2/acm-group-3/miniconda3/envs/rubric/bin/python scripts/evaluate_llm_self_play_critic.py \
  --original-labeled data/responses/coding_all_qwen25_vllm_k1_labeled_v2.jsonl \
  --critic-labeled data/self_play/llm_critic_mbpp_train_n54_v1_labeled.jsonl \
  --pairs-output data/self_play/llm_critic_pairs_mbpp_train_n54_v1.jsonl \
  --metrics-output data/self_play/llm_critic_metrics_mbpp_train_n54_v1.json \
  --md-output docs/llm_self_play_critic_n54_results.md
```

## Step 3 命令

```bash
/data2/acm-group-3/miniconda3/envs/rubric/bin/python scripts/build_llmcritic_augmented_preferences.py \
  --base data/preferences/preference_pairs_qwen25_k1_mbpp_train_augmented.jsonl \
  --llm-critic data/self_play/llm_critic_pairs_mbpp_train_n54_v1.jsonl \
  --output data/preferences/preference_pairs_qwen25_k1_mbpp_train_augmented_llmcritic54.jsonl \
  --summary-output data/preferences/preference_pairs_qwen25_k1_mbpp_train_augmented_llmcritic54_summary.json \
  --md-output docs/llmcritic_augmented_preferences.md
```

## Step 4 命令

```bash
CUDA_VISIBLE_DEVICES=1 /data2/acm-group-3/miniconda3/envs/rubric/bin/python scripts/dpo_lora_train.py \
  --model models/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28 \
  --data data/preferences/preference_pairs_qwen25_k1_mbpp_train_augmented_llmcritic54.jsonl \
  --output-dir outputs/dpo_lora_mbpp_train_augmented_llmcritic54_e1_mlen768 \
  --epochs 1 \
  --grad-accum 8 \
  --max-length 768
```

## Step 5/6 命令

```bash
CUDA_VISIBLE_DEVICES=1 /data2/acm-group-3/miniconda3/envs/rubric/bin/python scripts/generate_with_lora_adapter.py \
  --model models/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28 \
  --adapter outputs/dpo_lora_mbpp_train_augmented_llmcritic54_e1_mlen768 \
  --input data/processed/mbpp_validation_prompts.jsonl \
  --output data/responses/dpo_lora_train_augmented_llmcritic54_mbpp_validation.jsonl \
  --dataset mbpp \
  --split validation \
  --batch-size 1 \
  --max-input-length 1536 \
  --max-new-tokens 256

/data2/acm-group-3/miniconda3/envs/rubric/bin/python scripts/verify_mbpp_smoke.py \
  --input data/responses/dpo_lora_train_augmented_llmcritic54_mbpp_validation.jsonl \
  --output data/responses/dpo_lora_train_augmented_llmcritic54_mbpp_validation_labeled.jsonl

/data2/acm-group-3/miniconda3/envs/rubric/bin/python scripts/build_failure_artifacts.py \
  --input data/responses/dpo_lora_train_augmented_llmcritic54_mbpp_validation_labeled.jsonl \
  --failure-output data/eval/dpo_lora_train_augmented_llmcritic54_mbpp_validation_failures.jsonl \
  --summary-output data/eval/dpo_lora_train_augmented_llmcritic54_mbpp_validation_summary.json \
  --taxonomy-output data/eval/dpo_lora_train_augmented_llmcritic54_mbpp_validation_taxonomy.yaml

/data2/acm-group-3/miniconda3/envs/rubric/bin/python scripts/protected_revise_code_outputs.py \
  --labeled data/responses/dpo_lora_train_augmented_llmcritic54_mbpp_validation_labeled.jsonl \
  --output data/responses/dpo_lora_train_augmented_llmcritic54_mbpp_validation_protected_revised.jsonl

/data2/acm-group-3/miniconda3/envs/rubric/bin/python scripts/verify_mbpp_smoke.py \
  --input data/responses/dpo_lora_train_augmented_llmcritic54_mbpp_validation_protected_revised.jsonl \
  --output data/responses/dpo_lora_train_augmented_llmcritic54_mbpp_validation_protected_revised_labeled.jsonl

/data2/acm-group-3/miniconda3/envs/rubric/bin/python scripts/build_failure_artifacts.py \
  --input data/responses/dpo_lora_train_augmented_llmcritic54_mbpp_validation_protected_revised_labeled.jsonl \
  --failure-output data/eval/dpo_lora_train_augmented_llmcritic54_mbpp_validation_protected_revised_failures.jsonl \
  --summary-output data/eval/dpo_lora_train_augmented_llmcritic54_mbpp_validation_protected_revised_summary.json \
  --taxonomy-output data/eval/dpo_lora_train_augmented_llmcritic54_mbpp_validation_protected_revised_taxonomy.yaml
```

## 修订记录

| 时间 | Step | 检查结果 | 方案是否修改 |
| --- | --- | --- | --- |
| 2026-07-03 13:00 | 0 | GPU 仍满载；但 n=16 真实 critic 已跑通，DPO 脚本可用 | 保持小规模单卡训练，若 DPO 失败则记录资源阻塞 |
| 2026-07-03 13:08 | 1/2 | n=54 真实 LLM critic 完成，verifier 54/54 passed，生成 54 条 A<B pairs | 不需要 parser 修订；继续合并偏好数据 |
| 2026-07-03 13:10 | 3 | 合并偏好数据 266 条：158 canonical、54 rule-revised、54 LLM critic；无 validation/test | 继续 DPO 训练 |
| 2026-07-03 13:18 | 4 | DPO 训练完成：266 steps，skipped 0，mean loss 0.6464，preference accuracy 0.7970 | 继续 MBPP validation 评估 |
| 2026-07-03 13:22 | 5 | validation 生成 batch-size 1 约 3 条/分钟，预计过慢 | 停止慢速进程，改 batch-size 4 重跑生成 |
| 2026-07-03 13:36 | 5/6 | validation 生成 90 条并完成 verifier；DPO 单独 43/90，protected 后 54/90 | LLM critic pairs 提升 DPO 单独表现，但 protected 后与旧 augmented DPO 持平 |
| 2026-07-03 13:37 | 7 | `docs/llmcritic54_dpo_results.md` 与 `data/final/llmcritic54_dpo_results_summary.json` 已生成 | 下一步不继续盲目加同类 syntax pairs，优先补逻辑错误 critic |
