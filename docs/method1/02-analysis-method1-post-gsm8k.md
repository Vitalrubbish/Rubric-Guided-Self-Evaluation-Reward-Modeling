# Method 1 Post-GSM8K Audit

日期：2026-07-03

## 检查问题

GSM8K n=100 appendix 完成后，Method 1 是否还能从 proxy A/B 继续推进到完整 fixed-vs-updated online RL？

## 当前结论

不继续新开训练。当前最稳妥的表述是：

> Method 1 is implemented as a reproducible proxy A/B on the coding benchmark, with a supplemental GSM8K diagnostic loop. It is not a full online two-track RL experiment.

## 为什么不继续跑

| 选项 | 需要新增工作 | 预期收益 | 当前判断 |
| --- | --- | --- | --- |
| 保持 proxy A/B | 更新文档和 caveat | 清晰、可提交、证据充分 | done |
| 完整 fixed vs updated DPO 双轨 | 重新构建两套 preference pairs，训练两个 adapters，重新评估 validation/test | 更贴近原始 Method 1，但耗时且可能不改变结论 | future work |
| GSM8K 上跑 DPO/RL | 先构造 math preference pairs，再训练/评估 | 跨 benchmark 更完整 | future work |

## 已有证据是否足够

足够支撑“updated rubric 更有用”的结论：

- fixed/generic rubric: AUC 0.660, Kappa 0.316。
- updated/refined rubric: AUC 0.801, Kappa 0.525。
- protected revision: 577/1128 -> 755/1128，pass->fail 为 0。
- no-leakage DPO-related protected validation: train-only 50/90 -> logic-k5 updated signal 56/90。

## 与 GSM8K appendix 的关系

GSM8K n=100 证明 pipeline 能迁移到推荐数学 benchmark：

- exact-answer accuracy 72/100。
- failure taxonomy 覆盖 28/28 failures。
- static self-eval AUC 0.849。
- static Kappa 0.051，提示 math self-evaluation 的阈值化判断仍弱。

它增强了项目对原始推荐 benchmark 的对齐，但不把 Method 1 升级成完整 GSM8K RL。
