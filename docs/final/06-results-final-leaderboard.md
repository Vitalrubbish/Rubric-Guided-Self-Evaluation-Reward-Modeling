# Final Method Leaderboard

## Overall

| Method | Split | Passed | Total | pass@1 | Leakage | Note |
| --- | --- | ---: | ---: | ---: | --- | --- |
| Original Qwen2.5-7B vLLM | all | 577 | 1128 | 51.15% | no | Initial k=1 generation baseline |
| Unprotected rule revision | all | 745 | 1128 | 66.05% | no | Ablation; modifies all rows and has pass->fail risk |
| Protected rule revision | all | 755 | 1128 | 66.93% | no | Main current baseline; only revises failed rows |

## MBPP Validation

| Method | Passed | Total | pass@1 | Leakage | Note |
| --- | ---: | ---: | ---: | --- | --- |
| Protected rule revision | 61 | 90 | 67.78% | no | Best validation result |
| LLMCritic54 + logic k=5 DPO + protected revision | 56 | 90 | 62.22% | no | Best DPO-related method; no validation leakage |
| Full-failure DPO | 55 | 90 | 61.11% | yes | Sanity check only; trained with validation failures |
| Augmented train-only DPO + protected revision | 54 | 90 | 60.00% | no | Previous DPO-related baseline; no validation leakage |
| Train-only DPO + protected revision | 50 | 90 | 55.56% | no | No validation leakage; protected cascade |
| LLMCritic54 + logic k=5 DPO | 42 | 90 | 46.67% | no | No validation leakage; includes 7 verifier-selected logic self-play pairs |
| Augmented train-only DPO | 37 | 90 | 41.11% | no | No validation leakage |
| Base-HF | 33 | 90 | 36.67% | no | Transformers baseline |
| Train-only DPO | 33 | 90 | 36.67% | no | No validation leakage |

## Protected Revision Ablation

| Setting | Unprotected | Protected | Delta | pass->fail change |
| --- | ---: | ---: | ---: | ---: |
| Overall | 745 | 755 | +10 | 10 -> 0 |
| Augmented DPO validation | 53 | 54 | +1 | 1 -> 0 |
| Train-only DPO validation | 49 | 50 | +1 | 1 -> 0 |
| Logic k=5 DPO validation | 42 | 56 | +14 | 0 -> 0 |

## GSM8K Supplemental Benchmark

| Method | Split | Passed | Total | Accuracy | Leakage | Note |
| --- | --- | ---: | ---: | ---: | --- | --- |
| Qwen2.5-7B-Instruct | GSM8K test n=100 | 72 | 100 | 72.00% | no | Exact final-answer verifier; appendix benchmark |

| Rubric metric | Value | Note |
| --- | ---: | --- |
| failure pattern coverage | 1.000 | Failure-derived taxonomy covers all 28 failures |
| static self-eval AUC | 0.849 | Uses response text only |
| static Cohen's Kappa@4 | 0.051 | Thresholded pass/fail self-eval remains weak |
| verifier-informed upper-bound AUC | 1.000 | Uses external failure patterns; not deployable self-eval |

## GSM8K -> MATH Transfer

| Method | Split | Passed | Total | Accuracy | Leakage | Note |
| --- | --- | ---: | ---: | ---: | --- | --- |
| Qwen2.5-7B-Instruct | MATH safe subset n=100 | 83 | 100 | 83.00% | no | Algebra/Prealgebra Level 1-3, verifier-safe answers |
| Qwen2.5-7B-Instruct | MATH full-format pressure n=100 | 43 | 100 | 43.00% | no | All subjects, Level 1-5, full verifier with sets/intervals/radicals/multi-answer |

| Rubric | Uses MATH failures? | Static AUC | Accuracy@4 | Kappa@4 |
| --- | --- | ---: | ---: | ---: |
| GSM8K-derived rubric | no | 0.883 | 0.850 | 0.181 |
| Generic math rubric | no | 0.883 | 0.850 | 0.181 |
| MATH-derived rubric | yes | 0.883 | 0.910 | 0.596 |

Full-format pressure test:

| Rubric | Uses full MATH failures? | Static AUC | Accuracy@4 | Kappa@4 |
| --- | --- | ---: | ---: | ---: |
| GSM8K-derived rubric | no | 0.873 | 0.510 | 0.123 |
| Generic math rubric | no | 0.879 | 0.820 | 0.651 |
| MATH-derived rubric | yes | 0.879 | 0.820 | 0.651 |

## Takeaway

Protected rule revision is the strongest current overall coding method; LLMCritic54 + logic k=5 DPO + protected revision is the strongest no-leakage DPO-related validation method. GSM8K n=100, GSM8K -> MATH safe-subset transfer, and the full MATH verifier pressure test are included as supplemental alignment benchmarks, not as the main training benchmark.
