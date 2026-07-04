# Method 3 Meta-Transfer Final Note

日期：2026-07-04  
远程目录：`/data2/acm-group-3/Rubric-Guided-Self-Evaluation-Reward-Modeling`

## 目标

Method 3 要求观察模型是否学会了“如何生成好的 rubric”的 meta-skill，并在新任务上评估 zero-shot rubric generation 质量。

本项目实际完成的是最小可落地版本：

- 在 MBPP + HumanEval+ coding benchmark 上生成错误 taxonomy 和 refined coding rubric。
- 检查 refined rubric 在不同 split / 不同代码任务集合上的区分度。
- 用 HumanEval+ 作为未参与 MBPP train preference construction 的跨代码任务迁移目标。

追加 GSM8K -> MATH safe subset 后，当前 Method 3 包含两层迁移证据：

- MBPP -> HumanEval+：跨代码 benchmark 迁移。
- GSM8K -> MATH safe subset：跨数学 benchmark 的 rubric zero-shot 迁移。

safe subset 是 Method 3 的最小完成版。之后又补了 full MATH verifier n=100 pressure test，用来覆盖集合、区间、根式、多答案和 LaTeX 等价；但这仍不是 full-scale MATH meta-learning，因为还没有跑完整 MATH test split。

## Meta-Transfer Audit

Auto rubric by group:

| Group | N | Passed | AUC | Kappa | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| MBPP train | 374 | 216 | 0.798 | 0.512 | 0.778 |
| MBPP validation | 90 | 49 | 0.795 | 0.438 | 0.733 |
| MBPP test | 500 | 239 | 0.785 | 0.498 | 0.744 |
| HumanEval+ test | 164 | 73 | 0.846 | 0.644 | 0.817 |

Generic rubric by group:

| Group | N | Passed | AUC | Kappa | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| MBPP train | 374 | 216 | 0.653 | 0.280 | 0.615 |
| MBPP validation | 90 | 49 | 0.688 | 0.359 | 0.667 |
| MBPP test | 500 | 239 | 0.621 | 0.247 | 0.632 |
| HumanEval+ test | 164 | 73 | 0.824 | 0.629 | 0.811 |

## 结论

1. Refined auto rubric 在 MBPP train/validation/test 上都保持较高 AUC，说明不是只记住训练 split。
2. 在 HumanEval+ test 上，auto rubric 仍有 AUC `0.846`、Kappa `0.644`，说明它对另一个代码 benchmark 也有迁移区分度。
3. Auto rubric 在 MBPP 上明显优于 generic rubric，尤其是 MBPP test：AUC `0.785` vs `0.621`。
4. HumanEval+ 上 generic rubric 也较强，说明代码任务的 correctness/syntax/interface 通用维度本身已经有效；auto rubric 的优势更明显体现在 MBPP 族任务。

## Caveat

这不是完整 Method 3：

- 已补 GSM8K-derived rubric -> MATH safe subset 的 zero-shot evaluation，并补做 full MATH verifier n=100 pressure test；但还没有在完整 MATH test split 上评估。
- 没有验证跨领域 rubric generation。
- 没有证明模型学会了通用的“生成 rubric 的 meta-skill”。

可以在最终报告中表述为：

> We include a minimal meta-transfer audit across coding benchmarks, a GSM8K -> MATH safe-subset transfer audit, and a full-MATH-format verifier pressure test. The refined coding rubric generalizes from MBPP to HumanEval+, and the GSM8K-derived math rubric transfers ranking signal to both MATH safe-subset responses and a broader all-subject Level 1-5 MATH pressure subset. Full-scale MATH meta-learning remains future work.

## GSM8K -> MATH Safe-Subset Addendum

| Rubric | Uses MATH failures? | Static AUC | Accuracy@4 | Kappa@4 |
| --- | --- | ---: | ---: | ---: |
| GSM8K-derived rubric | no | 0.883 | 0.850 | 0.181 |
| Generic math rubric | no | 0.883 | 0.850 | 0.181 |
| MATH-derived rubric | yes | 0.883 | 0.910 | 0.596 |

MATH target subset:

- 100 test problems from `EleutherAI/hendrycks_math`.
- Algebra 50 / Prealgebra 50.
- Level 1-3 only.
- Gold verifier self-check: 100/100.
- Base Qwen2.5-7B accuracy: 83/100.

Conclusion: GSM8K-derived rubric carries useful ranking signal to MATH safe subset, but target-adapted MATH-derived rubric is better for thresholded pass/fail judgment.

## Full MATH Verifier Pressure Test

| Setting | Passed | Total | Accuracy |
| --- | ---: | ---: | ---: |
| Qwen2.5-7B-Instruct on all-subject Level 1-5 MATH pressure subset | 43 | 100 | 43.00% |

| Rubric | Uses full MATH failures? | Static AUC | Accuracy@4 | Kappa@4 |
| --- | --- | ---: | ---: | ---: |
| GSM8K-derived rubric | no | 0.873 | 0.510 | 0.123 |
| Generic math rubric | no | 0.879 | 0.820 | 0.651 |
| MATH-derived rubric | yes | 0.879 | 0.820 | 0.651 |

Full subset summary:

- Subjects: Algebra, Prealgebra, Counting & Probability, Geometry, Intermediate Algebra, Number Theory, Precalculus.
- Levels: Level 1-5, including 21 Level 4 and 16 Level 5 problems.
- Gold verifier self-check: 100/100.
- Safe subset regression with full verifier: 83/100.
- Dominant failure: `ambiguous_final_answer` 38/57 failures.

Interpretation: full MATH confirms the expected difficulty jump. GSM8K-derived rubric still ranks responses reasonably well, but thresholded pass/fail self-evaluation needs target-specific answer-format and symbolic-equivalence dimensions.

## 下一步若继续扩展

GSM8K -> MATH safe subset 和 full verifier n=100 pressure test 已经补完。若时间允许，真正的 full-scale MATH 扩展应继续做：

1. 用更强 prompt / 更长 max tokens 降低 `ambiguous_final_answer`。
2. 扩到更大的 full MATH sample，再做 manual verifier audit。
3. 用 GSM8K-trained adapter 或 repair loop 评估 MATH accuracy 是否提升。
4. 做 MATH held-out 上的 rubric generation meta-evaluation。

当前交付版本不跑完整 MATH test split，是因为完整 MATH final-answer verifier 清洗成本明显高于 GSM8K；临时覆盖所有答案格式容易引入错误标签，反而降低报告可信度。

## 证据文件

- `docs/meta_transfer_audit.md`
- `data/analysis/meta_transfer_audit.json`
- `scripts/build_meta_transfer_audit.py`
- `docs/gsm8k_alignment_results.md`
- `docs/gsm8k_to_math_transfer_results.md`
- `docs/math_full_verifier_results.md`
