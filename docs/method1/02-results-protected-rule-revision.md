# Protected Rule Revision Results

日期：2026-07-03  
远程目录：`/data2/acm-group-3/Rubric-Guided-Self-Evaluation-Reward-Modeling`

## 目的

旧版 rule revision 会修改所有样本，因此虽然大幅提升 pass@1，但会把少量已通过样本改坏。保护版规则修正默认只修改 `passed=false` 的样本，保留所有已通过输出。

## 脚本

`scripts/protected_revise_code_outputs.py`

默认策略：

- `--only-failed`
- `passed=true`：不修改 `generated_code`
- `passed=false`：复用 deterministic cleanup 规则
- 输出 `revision_skipped_reason`

## Augmented DPO Validation

输入：

`data/responses/dpo_lora_train_augmented_mbpp_validation_labeled.jsonl`

结果：

| 指标 | 数值 |
| --- | ---: |
| before | 37/90 |
| protected after | 54/90 |
| unprotected after | 53/90 |
| fail->pass | 17 |
| pass->fail | 0 |
| skipped passed rows | 37 |

## Train-Only DPO Validation

输入：

`data/responses/dpo_lora_train_only_mbpp_validation_labeled.jsonl`

结果：

| 指标 | 数值 |
| --- | ---: |
| before | 33/90 |
| protected after | 50/90 |
| unprotected after | 49/90 |
| fail->pass | 17 |
| pass->fail | 0 |
| skipped passed rows | 33 |

## Full Baseline

输入：

`data/responses/coding_all_qwen25_vllm_k1_labeled_v2.jsonl`

结果：

| 方法 | 通过数 | 总数 | pass@1 | pass->fail |
| --- | ---: | ---: | ---: | ---: |
| 原始 Qwen vLLM | 577 | 1128 | 51.15% | - |
| unprotected rule revision | 745 | 1128 | 66.05% | 10 |
| protected rule revision | 755 | 1128 | 66.93% | 0 |

按 split：

| split | passed | total | pass@1 |
| --- | ---: | ---: | ---: |
| MBPP train | 270 | 374 | 72.19% |
| MBPP validation | 61 | 90 | 67.78% |
| MBPP test | 320 | 500 | 64.00% |
| HumanEval+ test | 104 | 164 | 63.41% |

## Decision

Protected rule revision 支配 unprotected rule revision：它保留全部 fail->pass 收益，同时消除 pass->fail，并在全量结果上多通过 10 个样本。因此后续主 baseline 应切换到 protected rule revision。
