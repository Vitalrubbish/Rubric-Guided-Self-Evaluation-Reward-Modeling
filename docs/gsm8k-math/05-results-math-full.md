# Full MATH Verifier Pressure Test

日期：2026-07-04

## 目的

在已经完成的 `GSM8K -> MATH safe subset` 最小迁移实验之外，继续扩展 MATH verifier，使其支持更接近完整 MATH benchmark 的答案格式，并在 all subjects / Level 1-5 的 n=100 子集上做压力测试。

这不是替代 safe subset 的主结论。safe subset 仍是 Method 3 的最小完成版；本实验用于说明 full MATH 扩展的可行性、难度来源和下一步优化方向。

## Verifier 能力

`scripts/verify_math_full.py` 当前支持：

- `\boxed{...}` 和 `#### <answer>` final answer 抽取。
- 数字、分数、小数和千位逗号等价。
- `\frac{}`, `\sqrt{}`, `\pi`、幂次等 LaTeX 表达式到 SymPy 的等价比较。
- 集合、tuple、多答案、`\pm` / `±`。
- 区间和区间并集，包括 `\infty` / `∞` 和 `\cup` / `∪`。

Gold verifier 自检：

| Check | Passed | Total | Accuracy |
| --- | ---: | ---: | ---: |
| full MATH gold solution extraction/equivalence | 100 | 100 | 100.00% |
| safe subset regression with full verifier | 83 | 100 | 83.00% |

## Full MATH 子集

数据源：`EleutherAI/hendrycks_math` test parquet mirror。

| Subject | Total |
| --- | ---: |
| Algebra | 15 |
| Prealgebra | 15 |
| Counting & Probability | 14 |
| Geometry | 14 |
| Intermediate Algebra | 14 |
| Number Theory | 14 |
| Precalculus | 14 |

| Level | Total |
| --- | ---: |
| Level 1 | 21 |
| Level 2 | 21 |
| Level 3 | 21 |
| Level 4 | 21 |
| Level 5 | 16 |

| Gold answer type | Total |
| --- | ---: |
| numeric | 66 |
| fraction | 15 |
| symbolic | 10 |
| radical_or_pi | 6 |
| interval | 2 |
| multi_answer | 1 |

## Base Model Result

| Group | Passed | Total | Accuracy |
| --- | ---: | ---: | ---: |
| Overall | 43 | 100 | 43.00% |

By subject:

| Subject | Passed | Total | Accuracy |
| --- | ---: | ---: | ---: |
| Algebra | 10 | 15 | 66.67% |
| Prealgebra | 10 | 15 | 66.67% |
| Number Theory | 7 | 14 | 50.00% |
| Counting & Probability | 6 | 14 | 42.86% |
| Geometry | 6 | 14 | 42.86% |
| Intermediate Algebra | 3 | 14 | 21.43% |
| Precalculus | 1 | 14 | 7.14% |

By level:

| Level | Passed | Total | Accuracy |
| --- | ---: | ---: | ---: |
| Level 1 | 18 | 21 | 85.71% |
| Level 2 | 11 | 21 | 52.38% |
| Level 3 | 7 | 21 | 33.33% |
| Level 4 | 4 | 21 | 19.05% |
| Level 5 | 3 | 16 | 18.75% |

Failure taxonomy:

| Error pattern | Count |
| --- | ---: |
| ambiguous_final_answer | 38 |
| numeric_or_algebra_error | 14 |
| symbolic_equivalence_error | 4 |
| missing_final_answer | 1 |

Interpretation: full MATH sharply lowers accuracy relative to the safe subset, mainly because high-level subjects require longer symbolic reasoning and because many responses do not end with a clean `####` answer line under the current max-new-token budget.

## Rubric Transfer Metrics

| Rubric | Uses full MATH failures? | Pattern coverage | Static AUC | Accuracy@4 | Kappa@4 |
| --- | --- | ---: | ---: | ---: | ---: |
| GSM8K-derived rubric | no | 0.684 | 0.873 | 0.510 | 0.123 |
| Generic math rubric | no | 1.000 | 0.879 | 0.820 | 0.651 |
| MATH-derived rubric | yes | 1.000 | 0.879 | 0.820 | 0.651 |

Takeaway:

1. GSM8K-derived rubric still transfers ranking signal to a broader MATH subset: AUC `0.873`.
2. Thresholded pass/fail self-evaluation is much weaker on full MATH: Kappa `0.123`.
3. Generic/MATH-derived rubrics score better at thresholding because they explicitly include answer-format and symbolic-equivalence dimensions matching full MATH failures.
4. The dominant next improvement is not more DPO immediately; it is a stronger generation/evaluation protocol for final-answer formatting and longer symbolic derivations.

## Evidence Files

- `scripts/prepare_math_full_prompts.py`
- `scripts/verify_math_full.py`
- `data/processed/math_full_prompts_n100.jsonl`
- `data/eval/math_full_gold_verifier_check.json`
- `data/eval/math_full_qwen25_n100_summary.json`
- `data/eval/math_full_qwen25_n100_breakdown.json`
- `data/responses/math_full_qwen25_n100.jsonl`
- `data/responses/math_full_qwen25_n100_labeled.jsonl`
- `data/analysis/math_full_error_taxonomy_qwen25_n100.yaml`
- `data/rubrics/math_full_gsm8k_rubric_metrics_n100.json`
- `data/rubrics/math_full_generic_rubric_metrics_n100.json`
- `data/rubrics/math_full_derived_rubric_metrics_n100.json`

## Remaining Caveat

This is a 100-sample pressure test, not a full MATH benchmark run. The verifier is broader than the safe-subset verifier, but it is still not a complete symbolic mathematics theorem prover. Scaling to all MATH test examples should add more adversarial gold self-checks and manual audits for finite intervals, coordinate tuples, equivalent set-builder notation, and unusual LaTeX macros.
