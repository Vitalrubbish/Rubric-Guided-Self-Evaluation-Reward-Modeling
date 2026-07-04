# 项目工作总结

日期：2026-07-04  
项目：Rubric-Guided Self-Evaluation and Reward Modeling

本阶段完成了一个以 MBPP + HumanEval+ coding benchmark 为主线、GSM8K/MATH 为补充的 rubric-guided self-evaluation 与 self-improvement 实验闭环。主线共有 1128 条 coding 样本，补充实验包括 GSM8K、MATH safe subset 和 full MATH pressure subset。

首先，我完成了基线推理和 verifier 标注。使用 Qwen2.5-7B-Instruct 生成代码任务回答，并用测试用例判断通过/失败。原始结果为 577/1128，pass@1 为 51.15%。随后基于 551 个失败样本做错误模式发现和聚类，形成 refined taxonomy，共 18 个错误簇，主要包括逻辑输出错误、语法不可解析、重复函数体、运行时名称错误等。

其次，我基于错误 taxonomy 自动生成了 6 维 coding rubric，覆盖功能正确性、语法可解析性、接口契约、运行时安全、复杂度/终止性和输出整洁度。静态自评中，自动 rubric 的 AUC 为 0.801，Kappa 为 0.525，accuracy 为 0.765，说明它能区分通过和失败样本，可作为 reward/revision signal。

第三，我完成了 self-improvement baseline。根据高频错误类型设计规则化修复，并加入 protected revision：只修改 verifier 判失败的样本，已经通过的样本保持不动。最终 protected rule revision 达到 755/1128，pass@1 为 66.93%，比原始基线净增 178 个通过样本，且 pass->fail 为 0，是目前最强整体方法。

第四，我完成了多组 DPO 实验。已构造 preference pairs，并运行 full-failure、train-only、augmented train-only、LLMCritic54 + logic k=5 等 DPO。最佳无泄漏 DPO-related validation 为 56/90，但仍低于 protected revision 的 61/90。因此当前结论是：DPO 闭环已完成，但现阶段效果弱于 protected deterministic revision。

第五，我补做了 Method 2 self-play error discovery。流程是模型先找错，再写改进版，形成 A<B preference pairs。syntax/format 类错误修复稳定；logic 类更难，logic k=5 中 7/20 有效。这说明模型有一定自我找错能力，但复杂逻辑修复仍需要更强 prompt 或外部 verifier。

第六，我补充了 GSM8K 和 MATH 迁移实验。GSM8K n=100 上模型得到 72/100，GSM8K rubric 的 static AUC 为 0.849，但 Kappa 只有 0.051，说明排序能力存在但阈值化自评较弱。Method 3 方面，coding 侧完成 MBPP -> HumanEval+ 迁移，auto rubric 在 HumanEval+ 上 AUC 0.846、Kappa 0.644；数学侧完成 GSM8K -> MATH safe subset，模型准确率 83/100，GSM8K-derived rubric zero-shot AUC 0.883、Kappa 0.181，MATH-derived rubric Kappa 提升到 0.596。

最后，我扩展了 full MATH verifier，使其支持集合、区间、区间并集、根式、`\pi`、多答案、LaTeX 等价和千位逗号。full MATH gold 自检 100/100，safe subset 回归仍为 83/100。随后构建 all subjects / Level 1-5 的 MATH pressure subset n=100，模型准确率为 43/100。GSM8K-derived rubric 在 full MATH 上 AUC 0.873、Kappa 0.123，说明 ranking signal 仍存在，但 full MATH 的符号推理和 final-answer 格式更难。

目前项目已达到可提交状态。最终报告、leaderboard、Method 3 总结、submission checklist、GSM8K -> MATH 和 full MATH 结果页都已更新；机器可读指标也已写入 `data/final/`。需要诚实说明的是：DPO 已完成但不是当前最强方法，full MATH 只完成 n=100 pressure test，完整 MATH test split 仍是后续工作。
