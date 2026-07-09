# Phase 2 数据集扩展建议

Date: 2026-07-09

## 结论

更大的数据集是合理的后续扩展，但不应该作为 Phase 2 的第一步。

Phase 1 应冻结为：

```text
MBPP hidden-tests k=3/k=5 -> verifier -> safe failures -> LLM attribution -> clustering -> consolidated taxonomy -> rubric-operational refined taxonomy
```

Phase 2 的第一步应是使用 Phase 1 refined taxonomy 自动生成 rubric，并用 LLM rubric judge 做 held-out 评估。数据集扩展应放在 rubric baseline 之后，作为 Phase 2.5 或 robustness extension。

## 为什么可以在后续考虑扩展

Phase 1 已经解决了几个关键工程问题：

- prompt 不含 assert；
- verifier 与模型输入隔离；
- safe/private artifacts 分离；
- response_id 级对齐；
- LLM attribution + automatic clustering 跑通；
- k=3 与 k=5 的错误分布大体稳定。

这说明 pipeline 具备迁移基础。但在没有完成 taxonomy -> rubric -> rubric judge 之前，换大数据集不能解决 Phase 2 的核心问题。

## 为什么不能一步全量换大数据集

当前仍有几个约束：

- verifier 仍是轻量 multiprocessing，不是完整 sandbox；
- 聚类仍有 task/sample 重复偏置；
- taxonomy 已有自动 consolidated + refined 版本，但 rubric judge 尚未完成；
- 大数据集通常需要处理 stdin/stdout、多个文件、依赖包、复杂超时和安全执行。

因此，扩展应先做 adapter + smoke，再做中等规模，再考虑全量。

## 候选数据集

| 候选 | 角色 | 优点 | 风险 | 建议 |
| --- | --- | --- | --- | --- |
| HumanEval+ / EvalPlus | 跨基准验证 | 本地已有 `data/raw/humanevalplus_test.jsonl`，工程成本低，测试更严格 | 题目数小，不是真正扩大规模 | 第一优先级，作为 pipeline 迁移 smoke |
| BigCodeBench | 更真实代码任务 | 任务更接近实际函数调用和复杂指令，官方说明有 1140 software-engineering-oriented tasks | 需要适配其 prompt/evaluator/依赖策略 | 第二优先级，适合 Phase 2 主实验 |
| LiveCodeBench | 更新、更少污染 | 官方持续更新，release_v6 有 1055 code-generation problems | 在线风格和执行框架更复杂 | 适合做时间切片泛化评估 |
| APPS | 真正大规模 | 10,000 problems，覆盖从简单到竞赛题 | stdin/stdout、sandbox、时间限制、失败归因都更复杂 | 不建议立刻全量上，先抽样 500-1000 |

## 推荐路线

### Step 1：HumanEval+ 迁移 smoke

目标不是扩大规模，而是确认 Phase 1 pipeline 能跨数据格式工作。

输出目录建议：

```text
data/responses/phase2_humanevalplus_*
data/analysis/phase2_humanevalplus_*
```

验收标准：

- prompt 不泄漏 hidden tests；
- verifier 通过；
- safe failure artifacts 不含 private fields；
- attribution/clustering 可跑完；
- taxonomy 与 MBPP 结果可比较。

### Step 2：BigCodeBench 小规模试跑

先选 200-300 个任务，k=3。

需要新增：

- `prepare_bigcodebench_prompts.py`
- `verify_bigcodebench.py` 或复用官方 evaluator 输出适配器；
- `build_failure_artifacts.py` 的 dataset adapter；
- 与 Phase 1 一致的 safe/private diagnostics 分离。

验收标准：

- 至少 500 个失败 response 可用于归因；
- verifier 失败类型能映射到 syntax/runtime/logic/timeout/interface；
- 聚类最大簇不超过 25%；
- unique task count 与 response count 都被报告。

### Step 3：APPS 抽样，不直接全量

如果目标转向 reward model / DPO 训练，再抽样 APPS。

建议先做：

```text
APPS introductory/interview subset, 500-1000 tasks, k=3
```

不要一开始全量 10,000，因为执行安全、运行时间和归因成本都会明显上升。

## 当前决策

合理的下一步是：

1. 冻结 Phase 1 文档和 refined taxonomy。
2. 基于 refined taxonomy 自动生成 rubric。
3. 用 held-out validation/test responses 跑 LLM rubric judge。
4. 在 rubric baseline 成立后，用 HumanEval+ 做最小跨基准迁移。
5. 再设计 BigCodeBench adapter，先跑 200-300 task。
6. 在 BigCodeBench smoke 稳定后，再决定是否上 APPS 抽样。

不建议现在直接把 Phase 1 改成“大数据集版本”。Phase 1 应作为稳定 baseline；数据扩展应在 rubric baseline 之后推进。
