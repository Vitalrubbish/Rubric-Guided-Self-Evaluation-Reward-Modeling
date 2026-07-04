# GSM8K Alignment Results

日期：2026-07-03  
目的：补齐原始选题推荐 benchmark 中的 GSM8K 小规模真实闭环。主项目仍以 MBPP + HumanEval+ coding benchmark 为核心，本实验作为 cross-benchmark alignment appendix。

## 实验设置

| 项目 | 设置 |
| --- | --- |
| 数据 | GSM8K original test split |
| 样本数 | 100 |
| 模型 | Qwen2.5-7B-Instruct |
| 生成方式 | Transformers + chat template, greedy decoding |
| verifier | exact final-answer match after normalizing `#### <answer>` |
| 输出文件 | `data/responses/gsm8k_qwen25_k1_n100.jsonl` |
| 标注文件 | `data/responses/gsm8k_qwen25_k1_n100_labeled.jsonl` |

## Exact Verifier 结果

| 指标 | 数值 |
| --- | ---: |
| total | 100 |
| passed | 72 |
| failed | 28 |
| accuracy | 72.00% |

## 错误模式 Taxonomy

| error pattern | count | interpretation |
| --- | ---: | --- |
| final_format_violation | 15 | 没有按 `#### <answer>` 给出清晰最终答案，或者 final 标记不可用 |
| arithmetic_or_algebra_slip | 9 | 推理框架基本存在，但计算/代数步骤导致最终数值错误 |
| wrong_problem_model | 2 | 对题目数量关系建模错误 |
| reasoning_truncation | 1 | 推理没有完整收束到可靠答案 |
| ambiguous_final_answer | 1 | 最终答案不够明确，需要 verifier fallback |

相关文件：

- `data/analysis/gsm8k_failures_qwen25_k1_n100.jsonl`
- `data/analysis/gsm8k_failure_summary_qwen25_k1_n100.json`
- `data/analysis/gsm8k_error_taxonomy_qwen25_k1_n100.yaml`

## Failure-Derived Rubric

rubric 从 GSM8K failure taxonomy 生成，包含 4 个维度：

1. Problem modeling
2. Calculation accuracy
3. Stepwise reasoning completeness
4. Final answer format

相关文件：

- `data/rubrics/gsm8k_auto_rubric_n100.json`
- `data/rubrics/gsm8k_auto_rubric_scores_n100.jsonl`
- `data/rubrics/gsm8k_auto_rubric_metrics_n100.json`

## Self-Evaluation 指标

这里区分两个口径：

| 口径 | AUC | Accuracy@4 | Cohen's Kappa@4 | mean pass score | mean fail score |
| --- | ---: | ---: | ---: | ---: | ---: |
| static self-eval | 0.849 | 0.730 | 0.051 | 4.486 | 4.098 |
| verifier-informed upper bound | 1.000 | 1.000 | 1.000 | 5.000 | 3.491 |

解释：

- `static self-eval` 只看模型输出文本，不读取 gold answer 或 verifier 失败标签；这是更接近可部署 self-evaluation 的结果。
- `upper bound` 使用 exact verifier 归因后的 failure pattern，因此只能作为上界，不应当当成真实自评能力。
- static AUC 较高，说明 rubric 能排序区分好坏；Kappa 很低，说明固定阈值仍不稳定，尤其容易把“推理形式完整但最终答案错误”的样本评高。

## 对项目要求的意义

这个补充实验把项目从“只在 coding benchmark 上完成闭环”推进到“coding 主线 + GSM8K 推荐 benchmark 小规模闭环”：

- 有真实 GSM8K response generation。
- 有 exact-answer external verifier。
- 有 failure clustering/taxonomy。
- 有 failure-derived rubric。
- 有 self-evaluation vs external verifier 的 AUC/Kappa。

仍需诚实说明：这不是 GSM8K 上的多轮 DPO/RL self-evolving 训练；它是为了满足原始推荐 benchmark 的最小真实闭环和跨 benchmark 诊断。
