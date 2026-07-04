# DPO Adapter Evaluation Execution Plan

日期：2026-07-02  
远程目录：`/data2/acm-group-3/Rubric-Guided-Self-Evaluation-Reward-Modeling`

## 0. 当前状态

已经完成：

- 基线推理与 verifier 标注：`data/responses/coding_all_qwen25_vllm_k1_labeled_v2.jsonl`
- 错误 taxonomy 与自动 rubric
- preference pairs：`data/preferences/preference_pairs_qwen25_k1.jsonl`，共 551 对
- DPO LoRA 训练：`outputs/dpo_lora_coding_e1_551_mlen768`
- DPO 训练指标：steps 549，skipped 2，mean loss 0.4266，preference accuracy 0.8980

尚未完成：

- 加载 DPO adapter 重新生成答案
- 用同一个 verifier 评估 DPO adapter 的 pass@1
- 将 DPO 后结果与原始 Qwen、rubric-guided revision baseline 做表格对比

## 1. 执行原则

每一步必须满足两个检查：

1. 产物检查：目标文件是否存在、行数/字段是否符合预期。
2. 结果检查：指标是否可信；如果不可信，必须先修改后续方案，再继续执行。

如果某一步失败，不跳过；先记录失败原因，再修改本文档的后续步骤。

## 2. Step 1：接口盘点

目的：确认 verifier 输入格式与 adapter 推理输出格式。

检查项：

- `scripts/verify_mbpp_smoke.py` 是否读取 `generated_code`
- 输入 JSONL 是否需要保留 `test_list` / `test` / `entry_point`
- DPO adapter 是否有 `adapter_model.safetensors` 与 `train_metrics.json`

验收标准：

- verifier 不需要修改
- adapter 产物完整

执行状态：

- 状态：completed
- 检查结果：verifier 可直接复用；adapter 文件完整，包括 `adapter_model.safetensors` 和 `train_metrics.json`
- 是否修改后续方案：否。继续补 adapter 推理脚本。

## 3. Step 2：补充 adapter 推理脚本

目的：生成一个可复用脚本，既能跑 base model，也能跑 base model + LoRA adapter。

计划产物：

`scripts/generate_with_lora_adapter.py`

脚本要求：

- 输入：`data/processed/coding_prompts.jsonl`
- 可选过滤：`--dataset`、`--split`
- 可选 adapter：`--adapter outputs/dpo_lora_coding_e1_551_mlen768`
- 输出字段兼容 verifier：`id`、`dataset`、`prompt`、`generated_code`、测试字段
- 支持 `--limit`、`--batch-size`、`--max-new-tokens`、`--max-input-length`

验收标准：

- `python scripts/generate_with_lora_adapter.py --help` 成功
- 脚本语法检查成功
- 远程脚本已同步

执行状态：

- 状态：completed
- 检查结果：本地 `py_compile` 通过；远程 conda 环境执行 `--help` 成功；脚本已同步到远程 `scripts/generate_with_lora_adapter.py`
- 是否修改后续方案：否。继续执行 DPO adapter 小样本 smoke。

## 4. Step 3：DPO adapter 小样本 smoke

目的：先用很小样本确认 adapter 推理和 verifier 能闭环。

计划命令：

```bash
cd /data2/acm-group-3/Rubric-Guided-Self-Evaluation-Reward-Modeling
export CUDA_VISIBLE_DEVICES=1

/data2/acm-group-3/miniconda3/envs/rubric/bin/python scripts/generate_with_lora_adapter.py \
  --model models/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28 \
  --adapter outputs/dpo_lora_coding_e1_551_mlen768 \
  --input data/processed/coding_prompts.jsonl \
  --output data/responses/dpo_lora_smoke8.jsonl \
  --dataset mbpp \
  --split validation \
  --limit 8 \
  --batch-size 2 \
  --max-input-length 1536 \
  --max-new-tokens 256

/data2/acm-group-3/miniconda3/envs/rubric/bin/python scripts/verify_mbpp_smoke.py \
  --input data/responses/dpo_lora_smoke8.jsonl \
  --output data/responses/dpo_lora_smoke8_labeled.jsonl \
  --timeout 5
```

验收标准：

- 生成文件 8 行
- 标注文件 8 行
- verifier 无脚本异常

执行状态：

- 状态：completed
- 检查结果：`data/responses/dpo_lora_smoke8.jsonl` 8 行；`data/responses/dpo_lora_smoke8_labeled.jsonl` 8 行；verifier 结果 passed=7, failed=1
- 是否修改后续方案：否。adapter 推理、JSONL 输出、verifier 输入格式均正常。

## 5. Step 4：根据 smoke 结果调整方案

判断规则：

- 如果 smoke 通过率明显异常为 0，先检查 prompt 格式、adapter 加载、截断长度。
- 如果输出大量空字符串或重复 prompt，修改 generation 脚本 decode 逻辑。
- 如果只是个别样本失败，继续跑验证集。

执行状态：

- 状态：completed
- 检查结果：smoke 通过率不是 0，没有空输出或重复 prompt 崩坏；失败为正常样本级失败，不是系统性格式失败。
- 是否修改后续方案：否。继续跑 MBPP validation 90 条。统计脚本复用 `scripts/build_failure_artifacts.py`；base-vs-DPO 对比复用 `scripts/compare_revision_results.py`。

## 6. Step 5：验证集评测

目的：先用 MBPP validation 90 条做中等规模评测，避免一上来跑全量浪费时间。

计划产物：

- `data/responses/dpo_lora_mbpp_validation.jsonl`
- `data/responses/dpo_lora_mbpp_validation_labeled.jsonl`
- `data/eval/dpo_lora_mbpp_validation_summary.json`

验收标准：

- 生成文件 90 行
- 标注文件 90 行
- summary 包含 pass/fail/pass_rate 与 failure_types

说明：

当前 DPO preference pairs 是从全量失败样本构造的，其中包含 validation/test 失败样本。因此这一步更适合作为 adapter 效果 sanity check，不应当被包装成严格 held-out 泛化结论。

执行状态：

- 状态：completed
- 检查结果：`data/responses/dpo_lora_mbpp_validation.jsonl` 90 行；`data/responses/dpo_lora_mbpp_validation_labeled.jsonl` 90 行；summary 显示 passed=55, failed=35, pass_rate=0.611111
- 是否修改后续方案：轻微修改。仅与原始 vLLM baseline 比较不够公平，因此继续执行同后端 base-HF 对照；同时补充与 rule revision baseline 的比较，避免夸大 DPO 效果。

## 7. Step 6：对照评测

目的：避免把推理引擎差异误判成 DPO 效果。用同一个 Transformers 脚本在同一批 validation 上跑 base model，对比 base-HF vs DPO-HF。

计划产物：

- `data/responses/base_hf_mbpp_validation.jsonl`
- `data/responses/base_hf_mbpp_validation_labeled.jsonl`
- `data/eval/base_hf_mbpp_validation_summary.json`
- `data/eval/dpo_vs_base_hf_mbpp_validation_comparison.json`

验收标准：

- base 与 DPO 都有 90 条标注
- comparison 包含 base_pass_rate、dpo_pass_rate、delta、transition counts

执行状态：

- 状态：completed
- 检查结果：base-HF validation passed=33/90, pass_rate=0.366667；DPO-HF validation passed=55/90, pass_rate=0.611111，净增 22。与原始 vLLM validation 比较，DPO 为 55/90，vLLM baseline 为 49/90，净增 6。与 rule revision validation 比较，DPO 为 55/90，rule revision 为 60/90，低 5。
- 是否修改后续方案：是。最终文档必须同时报告三种比较：同后端 base-HF vs DPO-HF、原始 vLLM baseline vs DPO-HF、rule revision vs DPO-HF；并明确当前 validation 不应作为严格 held-out 泛化结论。

## 8. Step 7：最终文档更新

目的：把实际完成内容、失败/限制、下一步建议写回可交付文档。

需要更新：

- `docs/final_project_report.md`
- `docs/dpo_training_complete.md`
- `data/final/project_metrics_summary.json`

验收标准：

- 文档明确区分：
  - 原始 vLLM baseline
  - rubric-guided deterministic revision baseline
  - DPO LoRA training
  - DPO adapter generation/evaluation
- 不把 sanity check 写成严格 held-out 结论

执行状态：

- 状态：completed
- 检查结果：`docs/final_project_report.md`、`docs/dpo_training_complete.md`、`data/final/project_metrics_summary.json` 均已更新 DPO adapter validation 结果，并明确区分训练完成、adapter 评测完成、全量评测未完成。
- 是否修改后续方案：是。下一阶段应优先做 train-only preference split 与 untouched held-out evaluation，而不是继续在已泄漏的 validation 上堆指标。
