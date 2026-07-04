# Augmented Train-Only DPO Plan

日期：2026-07-03  
远程目录：`/data2/acm-group-3/Rubric-Guided-Self-Evaluation-Reward-Modeling`

## 0. 目标

上一轮 train-only DPO 只使用 canonical solution 作为 chosen，结果在 untouched MBPP validation 上与 base-HF 持平：33/90。  
本轮测试一个更贴近 rubric-guided self-evolving 的训练策略：把 rule revision 成功修好的 train split 输出也作为 chosen，让模型学习“如何从自己的失败输出修正到可通过输出”，而不只模仿 canonical solution。

## 1. 当前盘点

MBPP train：

- 总数：374
- 原始失败：158
- rule revision 后 fail->pass：54
- rule revision 后 fail->fail：104
- pass->fail：1

GPU：

- GPU1 空显存约 42GB，可跑 Qwen2.5-7B + LoRA
- 所有 GPU 利用率仍高，训练/推理速度可能不稳定

执行状态：

- 状态：completed
- 检查结果：存在 54 条 train-only rule-revised successful outputs，可用于增强 preference pairs。
- 是否修改后续方案：否。进入 augmented pair 构造。

## 2. Step 1：构造 augmented train-only preference pairs

目的：保留原 158 条 canonical chosen pair，再加入 54 条 rule-revised-success chosen pair。

计划产物：

`data/preferences/preference_pairs_qwen25_k1_mbpp_train_augmented.jsonl`

预期行数：

- canonical pairs：158
- rule-revised-success pairs：54
- total：212

验收标准：

- 输出文件存在
- 行数为 212
- 全部为 `dataset=mbpp` 且 `split=train`
- `chosen_source=rule_revised_success_output` 的行数为 54

执行状态：

- 状态：completed
- 检查结果：`data/preferences/preference_pairs_qwen25_k1_mbpp_train_augmented.jsonl` 已生成，行数 212；其中 `canonical_solution` 158 条，`rule_revised_success_output` 54 条；全部为 `dataset=mbpp` 且 `split=train`；组合 key 无重复。
- 是否修改后续方案：否。继续训练 augmented DPO adapter。

## 3. Step 2：训练 augmented DPO adapter

目的：测试增强后的 train-only preference pairs 是否能改善 untouched validation。

计划产物：

`outputs/dpo_lora_mbpp_train_augmented_e1_212_mlen768`

计划训练：

- epochs：1
- max_length：768
- grad_accum：8

验收标准：

- adapter 文件存在
- train_metrics.json 存在
- steps > 0
- skipped 不接近总样本数

执行状态：

- 状态：completed
- 检查结果：`outputs/dpo_lora_mbpp_train_augmented_e1_212_mlen768` 已生成；`adapter_model.safetensors` 为 80,792,096 bytes；`train_metrics.json` 显示 num_pairs=212, steps=212, skipped=0, mean_loss=0.651045, preference_accuracy=0.768868。
- 是否修改后续方案：否。训练正常完成，继续 untouched MBPP validation 评测。

## 4. Step 3：untouched MBPP validation 评测

目的：用 augmented adapter 在未参与训练的 MBPP validation 90 条上评估。

计划产物：

- `data/responses/dpo_lora_train_augmented_mbpp_validation.jsonl`
- `data/responses/dpo_lora_train_augmented_mbpp_validation_labeled.jsonl`
- `data/eval/dpo_lora_train_augmented_mbpp_validation_summary.json`

Gate：

- 若 augmented DPO > train-only DPO 33/90，说明增强数据有效。
- 若 augmented DPO >= 原始 vLLM baseline 49/90，才考虑跑 MBPP test。
- 若 augmented DPO 仍不超过 33/90，则不跑 test，改做训练策略诊断。

执行状态：

- 状态：completed
- 检查结果：`data/responses/dpo_lora_train_augmented_mbpp_validation.jsonl` 90 行；`data/responses/dpo_lora_train_augmented_mbpp_validation_labeled.jsonl` 90 行；summary 显示 passed=37, failed=53, pass_rate=0.411111。相比 train-only DPO 33/90 净增 4；相比 base-HF 33/90 净增 4；低于原始 vLLM baseline 49/90。
- 是否修改后续方案：是。augmented DPO 有小幅改善，但没有达到 test gate，不跑 MBPP test；继续执行 DPO + rule revision 级联，检查是否能超过 49/90 或 60/90。

## 5. Step 4：DPO + rule revision 级联

目的：测试 augmented adapter 输出经过 rule revision 后是否超过原始 vLLM baseline 或单独 rule revision。

计划产物：

- `data/responses/dpo_lora_train_augmented_mbpp_validation_revised.jsonl`
- `data/responses/dpo_lora_train_augmented_mbpp_validation_revised_labeled.jsonl`
- `data/eval/dpo_train_augmented_validation_revision_comparison.json`

验收标准：

- revised 文件 90 行
- comparison 有 transition counts
- 记录是否超过：
  - train-only DPO + rule revision：49/90
  - 单独 rule revision：60/90

执行状态：

- 状态：completed
- 检查结果：revised 文件 90 行；comparison 显示 original_passed=37, revised_passed=53, net_pass_delta=16；transition 为 fail->pass 17、pass->fail 1、pass->pass 36、fail->fail 36。级联后 pass@1=58.89%，超过原始 vLLM baseline 49/90 和 train-only DPO+revision 49/90，但仍低于单独 rule revision baseline 60/90。
- 是否修改后续方案：是。augmented DPO + rule revision 是当前最好的 DPO 相关方法，但仍不是全局最优；下一阶段应做保护版 rule revision 或改 DPO 训练目标，而不是直接跑 test。

## 6. Step 5：最终报告更新

需要更新：

- `docs/final_project_report.md`
- `docs/train_only_dpo_results.md`
- `data/final/project_metrics_summary.json`

必须说明：

- augmented pairs 是否构造成功
- augmented DPO 是否优于 train-only DPO
- 是否触发 test gate
- DPO + rule revision 是否超过已有 baseline

执行状态：

- 状态：completed
- 检查结果：`docs/augmented_train_only_dpo_results.md`、`docs/final_project_report.md`、`data/final/project_metrics_summary.json` 已更新 augmented DPO 结果；本地 JSON 解析检查通过。
- 是否修改后续方案：是。MBPP test 未执行，因为 augmented DPO 单独未过 49/90 gate；级联虽超过 vLLM baseline，但仍低于单独 rule revision，优先优化策略而不是扩大测试。
