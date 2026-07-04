# Method 3 Post-GSM8K Audit

日期：2026-07-03

## 检查问题

GSM8K n=100 appendix 以及 GSM8K -> MATH safe-subset transfer 完成后，Method 3 是否已经“完全”等价于 full GSM8K -> MATH meta-learning？

## 结论

仍然不是 full MATH meta-learning，但已经完成了最小跨数学 benchmark transfer。当前 Method 3 的准确定位是：

1. 主 Method 3：MBPP -> HumanEval+ 的最小跨代码迁移审计。
2. 数学补充：GSM8K n=100 推荐 benchmark 小闭环。
3. 跨数学迁移：GSM8K-derived rubric -> MATH safe subset zero-shot evaluation。
4. 未完成部分：full MATH 全 subject/高难度/复杂答案格式的完整跨领域 meta-learning。

## 已完成的迁移证据

| Group | N | Auto rubric AUC | Kappa |
| --- | ---: | ---: | ---: |
| MBPP train | 374 | 0.798 | 0.512 |
| MBPP validation | 90 | 0.795 | 0.438 |
| MBPP test | 500 | 0.785 | 0.498 |
| HumanEval+ test | 164 | 0.846 | 0.644 |

解释：coding refined rubric 在 HumanEval+ 上仍保持较强区分度，说明它不是只贴合 MBPP train。

## GSM8K Appendix 带来的补强

| 指标 | 数值 |
| --- | ---: |
| GSM8K exact accuracy | 72/100 |
| failure pattern coverage | 1.000 |
| static self-eval AUC | 0.849 |
| static Kappa@4 | 0.051 |

解释：GSM8K 证明 pipeline 能搬到数学推荐 benchmark 上，但 static Kappa 很低，说明 math self-evaluation 还没有达到稳定 pass/fail 判断。

## GSM8K -> MATH Safe-Subset 结果

| Rubric | Uses MATH failures? | Static AUC | Kappa@4 |
| --- | --- | ---: | ---: |
| GSM8K-derived | no | 0.883 | 0.181 |
| Generic math | no | 0.883 | 0.181 |
| MATH-derived | yes | 0.883 | 0.596 |

MATH safe subset baseline：83/100。这个结果说明 GSM8K-derived rubric 已经有 zero-shot ranking signal，但 target-adapted rubric 更适合 pass/fail 阈值判断。

## 为什么不继续扩到 full MATH

MATH 的答案格式包含 LaTeX、等价表达式、分数、根式、区间等，可靠 verifier 需要额外清洗。若只做弱字符串匹配，标签噪声会很大，反而损害 Method 3 结论。因此 MATH 应作为后续扩展，而不是在当前提交版里仓促加入。

## 最终表述建议

> We provide a minimal coding meta-transfer audit from MBPP to HumanEval+, a GSM8K n=100 appendix, and a GSM8K-derived rubric zero-shot evaluation on a verifier-safe MATH subset. Full MATH meta-learning remains future work because reliable symbolic answer verification requires additional normalization.
