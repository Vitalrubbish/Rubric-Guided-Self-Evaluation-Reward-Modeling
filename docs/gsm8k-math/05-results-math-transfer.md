# GSM8K -> MATH Transfer Results

日期：2026-07-04

## 目的

本实验补齐 Method 3 的最小跨数学 benchmark 迁移：用 GSM8K failure-derived rubric 直接 zero-shot 评估 MATH responses，再与 generic math rubric 和 MATH-derived rubric 对比。

## 数据

| 项目 | 设置 |
| --- | --- |
| Source benchmark | GSM8K n=100 appendix |
| Target benchmark | MATH safe subset |
| Data source | `EleutherAI/hendrycks_math` parquet mirror |
| Split | test |
| Subjects | Algebra 50, Prealgebra 50 |
| Levels | Level 1: 23, Level 2: 31, Level 3: 46 |
| Filter | verifier-safe answers: simple numeric/rational/short expression |

Gold verifier 自检：

| Check | Passed | Total | Accuracy |
| --- | ---: | ---: | ---: |
| gold solution answer extraction/equivalence | 100 | 100 | 100.00% |

## Base Model on MATH

| Group | Passed | Total | Accuracy |
| --- | ---: | ---: | ---: |
| Overall | 83 | 100 | 83.00% |
| Algebra | 44 | 50 | 88.00% |
| Prealgebra | 39 | 50 | 78.00% |
| Level 1 | 21 | 23 | 91.30% |
| Level 2 | 25 | 31 | 80.65% |
| Level 3 | 37 | 46 | 80.43% |

Failure taxonomy:

| Error pattern | Count |
| --- | ---: |
| symbolic_or_arithmetic_error | 9 |
| ambiguous_final_answer | 8 |

## Rubric Transfer Comparison

| Rubric | Uses MATH failures to define rubric? | Pattern coverage | Static AUC | Accuracy@4 | Kappa@4 |
| --- | --- | ---: | ---: | ---: | ---: |
| GSM8K-derived rubric | no | 0.471 | 0.883 | 0.850 | 0.181 |
| Generic math rubric | no | 1.000 | 0.883 | 0.850 | 0.181 |
| MATH-derived rubric | yes | 1.000 | 0.883 | 0.910 | 0.596 |

Interpretation:

- GSM8K-derived rubric transfers useful ranking signal to MATH: static AUC `0.883`.
- Thresholded pass/fail self-evaluation is still weak zero-shot: Kappa `0.181`.
- MATH-derived rubric improves threshold behavior: Kappa `0.596`, because it includes target-specific failure patterns such as `symbolic_or_arithmetic_error`.
- Generic math rubric ties GSM8K-derived static metrics in this first subset, meaning the current static scorer is dominated by general reasoning/format/calculation signals rather than nuanced source-rubric wording.

## Method 3 Conclusion

This is now stronger than the previous MBPP -> HumanEval+ only audit:

1. MBPP -> HumanEval+ still provides cross-code benchmark transfer evidence.
2. GSM8K -> MATH now provides cross-math benchmark transfer evidence.
3. The result supports a limited claim: failure-derived math rubrics can carry ranking signal from GSM8K-style arithmetic reasoning to a controlled MATH subset.
4. It does not prove full meta-learning of rubric generation, because MATH-derived rubric is generated after seeing MATH failures and because the MATH subset is verifier-safe rather than full MATH.

## Evidence Files

- `data/processed/math_transfer_prompts_n100.jsonl`
- `data/eval/math_transfer_gold_verifier_check.json`
- `data/responses/math_transfer_qwen25_n100.jsonl`
- `data/responses/math_transfer_qwen25_n100_labeled.jsonl`
- `data/eval/math_transfer_qwen25_n100_summary.json`
- `data/analysis/math_transfer_failures_qwen25_n100.jsonl`
- `data/analysis/math_transfer_error_taxonomy_qwen25_n100.yaml`
- `data/rubrics/math_transfer_gsm8k_rubric_metrics_n100.json`
- `data/rubrics/math_transfer_generic_rubric_metrics_n100.json`
- `data/rubrics/math_transfer_derived_rubric_metrics_n100.json`

## Full MATH Verifier Extension

在 safe subset 完成后，我们继续做了 full MATH verifier pressure test。这个实验扩展到 all subjects / Level 1-5，并支持集合、区间、根式、多答案和 LaTeX 等价。

Verifier gate：

| Check | Passed | Total | Accuracy |
| --- | ---: | ---: | ---: |
| full MATH gold verifier self-check | 100 | 100 | 100.00% |
| safe subset regression with full verifier | 83 | 100 | 83.00% |

Full MATH n=100 result：

| Setting | Passed | Total | Accuracy |
| --- | ---: | ---: | ---: |
| Qwen2.5-7B-Instruct on full MATH pressure subset | 43 | 100 | 43.00% |

By level:

| Level | Passed | Total | Accuracy |
| --- | ---: | ---: | ---: |
| Level 1 | 18 | 21 | 85.71% |
| Level 2 | 11 | 21 | 52.38% |
| Level 3 | 7 | 21 | 33.33% |
| Level 4 | 4 | 21 | 19.05% |
| Level 5 | 3 | 16 | 18.75% |

Rubric transfer on full MATH pressure subset:

| Rubric | Uses full MATH failures? | Static AUC | Accuracy@4 | Kappa@4 |
| --- | --- | ---: | ---: | ---: |
| GSM8K-derived rubric | no | 0.873 | 0.510 | 0.123 |
| Generic math rubric | no | 0.879 | 0.820 | 0.651 |
| MATH-derived rubric | yes | 0.879 | 0.820 | 0.651 |

Interpretation:

- Safe subset remains the clean Method 3 minimal completion result.
- Full MATH pressure subset shows the expected difficulty jump: accuracy drops from `83%` to `43%`.
- GSM8K-derived rubric keeps ranking signal on full MATH (`AUC 0.873`), but thresholded pass/fail self-evaluation is weak (`Kappa 0.123`).
- The main next improvement is better final-answer discipline and longer symbolic derivations, because `ambiguous_final_answer` accounts for 38/57 full-subset failures.

Detailed report:

- `docs/math_full_verifier_results.md`

## Remaining Caveat

The full verifier extension is a 100-sample pressure test, not a full MATH benchmark run. It improves coverage of complex answer formats, but scaling to all MATH test examples still needs more adversarial verifier checks and manual audits.
