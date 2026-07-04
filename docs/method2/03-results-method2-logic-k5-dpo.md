# Logic k=5 Self-Play DPO Results

日期：2026-07-03  
远程目录：`/data2/acm-group-3/Rubric-Guided-Self-Evaluation-Reward-Modeling`

## 目的

前面 Method 2 已经得到：

- syntax/format LLM critic pairs: 54 条，训练后 raw validation 43/90，protected 后 54/90。
- logic k=5 verifier-selected self-play pairs: 7 条。

本轮把 7 条最可靠 logic self-play pairs 合入 266 条 LLM-critic augmented train-only preference data，形成 273 条训练集，检查显式 logic error discovery 是否能进一步提升 DPO。

## Preference Data

文件：

`data/preferences/preference_pairs_qwen25_k1_mbpp_train_augmented_llmcritic54_logic_k5.jsonl`

行数：273

组成：

| Source | Count |
| --- | ---: |
| canonical_solution | 158 |
| rule_revised_success_output | 54 |
| llm_self_play_revised_passed | 54 |
| llm_self_play_logic_multicandidate_revised_passed | 7 |
| total | 273 |

## DPO 训练

输出：

`outputs/dpo_lora_mbpp_train_augmented_llmcritic54_logic_k5_e1_mlen768`

训练指标：

| Metric | Value |
| --- | ---: |
| Preference pairs | 273 |
| Steps | 273 |
| Skipped | 0 |
| Mean loss | 0.6468 |
| Preference accuracy | 80.59% |

## MBPP Validation

| Method | Raw validation | Protected validation |
| --- | ---: | ---: |
| LLMCritic54 DPO | 43/90 | 54/90 |
| LLMCritic54 + logic k=5 DPO | 42/90 | 56/90 |
| Protected rule revision baseline | - | 61/90 |

Protected revision transition for logic k=5 DPO:

| Transition | Count |
| --- | ---: |
| pass->pass | 42 |
| fail->pass | 14 |
| fail->fail | 34 |
| pass->fail | 0 |

## 结论

logic k=5 pairs 没有提升 raw DPO：`43/90 -> 42/90`。

但与 protected revision 级联后，logic k=5 DPO 从旧 LLMCritic54 DPO 的 `54/90` 提升到 `56/90`，成为当前最好的无 validation 泄漏 DPO-related validation 结果。不过它仍低于单独 protected rule revision 的 `61/90`，因此不进入 MBPP test。

## 决策

1. 记录为 Method 2 的最终 DPO 消融：少量 verifier-selected logic pairs 对 protected cascade 有小幅增益。
2. 不继续把 spec-first/two-stage/algorithm-sketch 的低质 pairs 合并训练。
3. 若继续追求 logic 修复率，需要 verifier-feedback repair 或更强外部信号；否则保留本结果作为最终 self-play DPO 证据。

## 证据文件

- `outputs/dpo_lora_mbpp_train_augmented_llmcritic54_logic_k5_e1_mlen768/train_metrics.json`
- `data/eval/dpo_lora_train_augmented_llmcritic54_logic_k5_mbpp_validation_summary.json`
- `data/eval/dpo_lora_train_augmented_llmcritic54_logic_k5_mbpp_validation_protected_revised_summary.json`
- `data/eval/dpo_lora_train_augmented_llmcritic54_logic_k5_validation_protected_revision_comparison.json`
