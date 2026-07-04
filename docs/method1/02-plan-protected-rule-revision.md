# Protected Rule Revision Plan

日期：2026-07-03  
远程目录：`/data2/acm-group-3/Rubric-Guided-Self-Evaluation-Reward-Modeling`

## 0. 目标

现有 deterministic rule revision 能显著提升 pass@1，但会改坏少量已通过样本：

| 输入 | before | after | fail->pass | pass->fail |
| --- | ---: | ---: | ---: | ---: |
| 全量 Qwen baseline | 577/1128 | 745/1128 | 178 | 10 |
| Train-only DPO validation | 33/90 | 49/90 | 17 | 1 |
| Augmented DPO validation | 37/90 | 53/90 | 17 | 1 |

本阶段目标是实现 protected rule revision：默认不修改已通过样本，只对 verifier 已判失败的输出做清洗，从而降低 pass->fail 风险。

## 1. Step 1：实现 protected revision 脚本

计划产物：

`scripts/protected_revise_code_outputs.py`

策略：

- 默认 `--only-failed`
- 对 `passed=true` 的样本不改 `generated_code`
- 对 `passed=false` 的样本复用原规则：
  - `truncate_duplicate_function_body`
  - `drop_trailing_prose`
  - `remove_print_examples`
- 输出 `revision_skipped_reason` 方便验收

验收标准：

- 本地 `py_compile` 通过
- 远程 `--help` 成功
- 脚本已同步到远程

执行状态：

- 状态：completed
- 检查结果：本地 `py_compile` 通过；远程 `--help` 成功；脚本已同步到 `scripts/protected_revise_code_outputs.py`。
- 是否修改后续方案：否。继续在 augmented DPO validation 上验证 pass->fail 是否归零。

## 2. Step 2：在 augmented DPO validation 上评估

目的：验证 protected revision 能否保留原有 fail->pass，同时消除 pass->fail。

输入：

`data/responses/dpo_lora_train_augmented_mbpp_validation_labeled.jsonl`

计划产物：

- `data/responses/dpo_lora_train_augmented_mbpp_validation_protected_revised.jsonl`
- `data/responses/dpo_lora_train_augmented_mbpp_validation_protected_revised_labeled.jsonl`
- `data/eval/dpo_train_augmented_validation_protected_revision_comparison.json`

验收标准：

- 输出 90 行
- comparison 中 `pass->fail = 0`
- pass@1 应 >= unprotected revision 的 53/90

执行状态：

- 状态：completed
- 检查结果：输出 90 行；attempted=53, edited=22, skipped_passed=37；verifier passed=54/90；comparison 中 fail->pass=17, pass->fail=0, pass->pass=37, fail->fail=36。protected revision 比 unprotected revision 的 53/90 高 1 题。
- 是否修改后续方案：否。结果达到预期，继续在 train-only DPO validation 上复验。

## 3. Step 3：在 train-only DPO validation 上复验

目的：确认 protected revision 不是只对 augmented adapter 有效。

输入：

`data/responses/dpo_lora_train_only_mbpp_validation_labeled.jsonl`

计划产物：

- `data/responses/dpo_lora_train_only_mbpp_validation_protected_revised.jsonl`
- `data/responses/dpo_lora_train_only_mbpp_validation_protected_revised_labeled.jsonl`
- `data/eval/dpo_train_only_validation_protected_revision_comparison.json`

验收标准：

- 输出 90 行
- comparison 中 `pass->fail = 0`
- pass@1 应 >= unprotected revision 的 49/90

执行状态：

- 状态：completed
- 检查结果：输出 90 行；attempted=57, edited=24, skipped_passed=33；verifier passed=50/90；comparison 中 fail->pass=17, pass->fail=0, pass->pass=33, fail->fail=40。protected revision 比 unprotected revision 的 49/90 高 1 题。
- 是否修改后续方案：否。两个 DPO validation 均达到预期，继续扩展到原始 vLLM 全量 baseline。

## 4. Step 4：若 validation 成功，扩展到原始 vLLM validation

目的：看 protected revision 是否也能提升原始 baseline，并减少原 rule revision 的 pass->fail。

注意：

原始 labeled 文件是全量 1128 条，需要先运行 protected revision 全量，再用 comparison 自动只比较 validation 交集。

计划产物：

- `data/responses/coding_all_qwen25_vllm_k1_protected_revised.jsonl`
- `data/responses/coding_all_qwen25_vllm_k1_protected_revised_labeled.jsonl`
- `data/eval/vllm_baseline_protected_revision_comparison.json`

验收标准：

- 输出 1128 行
- pass->fail 应显著低于 unprotected 的 10
- pass@1 应不低于 unprotected 全量 745/1128，或若略低需解释 tradeoff

执行状态：

- 状态：completed
- 检查结果：输出 1128 行；attempted=551, edited=305, skipped_passed=577；verifier passed=755/1128；comparison 中 fail->pass=178, pass->fail=0, pass->pass=577, fail->fail=373。相比 unprotected revision 745/1128 多 10 题，且 pass->fail 从 10 降为 0。
- 是否修改后续方案：是。protected revision 支配 unprotected revision，应作为新的默认 rule revision baseline。

## 5. Step 5：更新最终报告

需要更新：

- `docs/final_project_report.md`
- `data/final/project_metrics_summary.json`
- 新增 `docs/protected_rule_revision_results.md`

执行状态：

- 状态：completed
- 检查结果：`docs/protected_rule_revision_results.md`、`docs/final_project_report.md`、`data/final/project_metrics_summary.json` 已更新 protected revision 结果。
- 是否修改后续方案：是。后续不再使用 unprotected revision 作为主 baseline；只保留它作为 ablation。
