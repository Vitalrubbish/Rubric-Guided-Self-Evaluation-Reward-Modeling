# Train-Only DPO Execution Plan

日期：2026-07-03  
远程目录：`/data2/acm-group-3/Rubric-Guided-Self-Evaluation-Reward-Modeling`

## 0. 目标

昨天的 DPO adapter 使用了从全量失败样本构造的 preference pairs，其中包含 validation/test 失败样本。因此它能证明 DPO 流程和 adapter 有效，但不能作为严格 held-out 泛化结论。

本阶段目标是构造 train-only preference pairs，只用 MBPP train 失败样本训练 DPO adapter，再在 untouched MBPP validation 上评估。若 validation 表现足够好，再继续跑更大的 test 评测；若不够好，则停止扩展评测，先调整训练方案。

## 1. 当前盘点

已有全量 preference pairs：

`data/preferences/preference_pairs_qwen25_k1.jsonl`

按 split 分布：

| dataset/split | pairs |
| --- | ---: |
| mbpp/train | 158 |
| mbpp/validation | 41 |
| mbpp/test | 261 |
| humanevalplus/test | 91 |

GPU 状态：GPU1 空显存约 42GB，可以容纳 Qwen2.5-7B + LoRA；但所有 GPU 利用率都在 100%，训练/推理会受其他任务影响。

执行状态：

- 状态：completed
- 检查结果：远程数据、昨日 DPO adapter、base-HF validation 对照均存在；train-only pairs 可由现有全量 pairs 过滤得到。
- 是否修改后续方案：是。由于 full MBPP test 500 条 Transformers 推理较慢，本计划加入 validation gate：先看 untouched validation，再决定是否跑 test。

## 2. Step 1：生成 train-only preference pairs

目的：只保留 MBPP train split 的失败样本 preference pairs，避免 validation/test 泄漏。

计划产物：

`data/preferences/preference_pairs_qwen25_k1_mbpp_train_only.jsonl`

计划命令：

```bash
/data2/acm-group-3/miniconda3/envs/rubric/bin/python scripts/filter_preference_pairs.py \
  --input data/preferences/preference_pairs_qwen25_k1.jsonl \
  --output data/preferences/preference_pairs_qwen25_k1_mbpp_train_only.jsonl \
  --dataset mbpp \
  --split train
```

验收标准：

- 输出文件存在
- 行数为 158
- 所有行的 `dataset=mbpp` 且 `split=train`

执行状态：

- 状态：completed
- 检查结果：`data/preferences/preference_pairs_qwen25_k1_mbpp_train_only.jsonl` 已生成，行数 158；所有行均为 `dataset=mbpp` 且 `split=train`；ID 无重复。
- 是否修改后续方案：否。继续训练 train-only DPO adapter。

## 3. Step 2：训练 train-only DPO adapter

目的：用无 validation/test 泄漏的 preference pairs 训练一个新的 LoRA DPO adapter。

计划产物：

`outputs/dpo_lora_mbpp_train_only_e1_158_mlen768`

计划命令：

```bash
export CUDA_VISIBLE_DEVICES=1
/data2/acm-group-3/miniconda3/envs/rubric/bin/python scripts/dpo_lora_train.py \
  --model models/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28 \
  --data data/preferences/preference_pairs_qwen25_k1_mbpp_train_only.jsonl \
  --output-dir outputs/dpo_lora_mbpp_train_only_e1_158_mlen768 \
  --epochs 1 \
  --grad-accum 8 \
  --max-length 768
```

验收标准：

- `adapter_model.safetensors` 存在
- `train_metrics.json` 存在
- `steps > 0`
- `skipped` 不应接近总样本数

执行状态：

- 状态：completed
- 检查结果：`outputs/dpo_lora_mbpp_train_only_e1_158_mlen768` 已生成；`adapter_model.safetensors` 为 80,792,096 bytes；`train_metrics.json` 显示 num_pairs=158, steps=158, skipped=0, mean_loss=0.648870, preference_accuracy=0.696203。
- 是否修改后续方案：否。训练正常完成，继续 untouched MBPP validation 评测。

## 4. Step 3：untouched MBPP validation 评测

目的：用 train-only adapter 在未用于训练的 MBPP validation 90 条上生成答案并 verifier。

计划产物：

- `data/responses/dpo_lora_train_only_mbpp_validation.jsonl`
- `data/responses/dpo_lora_train_only_mbpp_validation_labeled.jsonl`
- `data/eval/dpo_lora_train_only_mbpp_validation_summary.json`

验收标准：

- 生成文件 90 行
- 标注文件 90 行
- summary 包含 passed/failed/pass_rate

判断 gate：

- 若 train-only DPO >= base-HF validation，并接近或超过原始 vLLM validation，则进入 Step 4。
- 若明显低于 base-HF 或输出格式崩坏，则不跑 test，先修改训练/解码方案。
- 若低于 rule revision baseline，这是可接受但必须如实报告；不影响继续做 adapter+rule 级联实验。

执行状态：

- 状态：completed
- 检查结果：`data/responses/dpo_lora_train_only_mbpp_validation.jsonl` 90 行；`data/responses/dpo_lora_train_only_mbpp_validation_labeled.jsonl` 90 行；summary 显示 passed=33, failed=57, pass_rate=0.366667。与 base-HF validation 持平 33/90；低于原始 vLLM validation 49/90；低于 rule revision validation 60/90。
- 是否修改后续方案：是。validation gate 未通过；不继续跑 MBPP test 500 条，避免浪费 GPU。下一步改为执行 DPO + rule revision 级联，分析规则修正能否补救 train-only DPO 输出。

## 5. Step 4：MBPP test 评测 gate

目的：若 validation 结果值得继续，则在 MBPP test 500 条上评估泛化。

计划产物：

- `data/responses/dpo_lora_train_only_mbpp_test.jsonl`
- `data/responses/dpo_lora_train_only_mbpp_test_labeled.jsonl`
- `data/eval/dpo_lora_train_only_mbpp_test_summary.json`

验收标准：

- 生成文件 500 行
- 标注文件 500 行
- summary 包含 pass_rate

执行状态：

- 状态：skipped
- 检查结果：未执行。原因是 Step 3 validation gate 未通过：train-only DPO validation pass@1 仅 36.67%，没有超过同后端 base-HF，也明显低于原始 vLLM baseline。
- 是否修改后续方案：是。跳过 test，优先调试训练策略与执行 Step 5 的 DPO + rule revision 级联。

## 6. Step 5：DPO + rule revision 级联

目的：测试 DPO adapter 输出是否还能通过已有 rule revision 进一步提升。

计划：

1. 对 train-only DPO validation 输出运行 `scripts/revise_code_outputs.py`
2. 用 verifier 重新标注 revised 输出
3. 与未 revision 的 train-only DPO validation 比较

计划产物：

- `data/responses/dpo_lora_train_only_mbpp_validation_revised.jsonl`
- `data/responses/dpo_lora_train_only_mbpp_validation_revised_labeled.jsonl`
- `data/eval/dpo_train_only_validation_revision_comparison.json`

验收标准：

- revised 文件 90 行
- comparison 显示 transition counts
- 若 pass->fail 明显多，后续需要加保护规则

执行状态：

- 状态：completed
- 检查结果：revised 文件 90 行；comparison 显示 original_passed=33, revised_passed=49, net_pass_delta=16；transition 为 fail->pass 17、pass->fail 1、pass->pass 32、fail->fail 40。级联后 pass@1=54.44%，追平原始 vLLM validation baseline 49/90，但仍低于单独 rule revision baseline 60/90。
- 是否修改后续方案：是。DPO + rule revision 有补救价值，但最佳当前系统仍是 rule revision baseline；下一步不应继续扩大 train-only DPO test，而应改训练策略。

## 7. Step 6：最终报告更新

需要更新：

- `docs/final_project_report.md`
- `docs/dpo_training_complete.md`
- `data/final/project_metrics_summary.json`

必须写清楚：

- 全量 DPO adapter：有泄漏，作为 pipeline sanity check
- train-only DPO adapter：无 validation/test 泄漏，作为更严谨结果
- validation gate 是否通过
- 是否继续跑 test，以及原因
- DPO + rule revision 是否有效

执行状态：

- 状态：completed
- 检查结果：`docs/train_only_dpo_results.md`、`docs/final_project_report.md`、`data/final/project_metrics_summary.json` 已更新 train-only DPO 和 DPO+rule revision 结果。
- 是否修改后续方案：是。下一阶段建议改为训练策略调试：增加 train-only 数据、使用 revised outputs 作为 chosen、降低 pass->fail 风险，并评估 vLLM LoRA 推理以降低 full test 成本。
