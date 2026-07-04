# Logic Two-Stage Failure Diagnosis Plan

日期：2026-07-03  
远程目录：`/data2/acm-group-3/Rubric-Guided-Self-Evaluation-Reward-Modeling`

## 背景

two-stage v1 在同一批 20 个 MBPP train `logic_error` 样本上得到：

| 指标 | 数值 |
| --- | ---: |
| Attempted | 20 |
| Passed | 3 |
| Failed | 17 |
| Syntax error after revision | 0 |
| Logic error after revision | 17 |

结论是：两阶段已经解决格式退化，但语义修复仍不足。因此下一步不能直接扩大 k 或训练 DPO，而是先诊断 17 个失败样本到底卡在哪里。

## 目标

为每个 two-stage 失败样本生成可查阅诊断：

1. 抽取原题、可见测试、Stage 1 spec、Stage 2 code、verifier error。
2. 判断失败主因属于哪一类。
3. 汇总各类失败数量和代表样本。
4. 根据主因分布修订下一轮 prompt/实验方案。

## 分类标准

| 分类 | 含义 | 后续动作 |
| --- | --- | --- |
| `wrong_spec` | Stage 1 对题意/测试隐含规格归纳错或漏掉关键边界 | 强化 Stage 1：逐条测试含义、counterexample thinking、边界枚举 |
| `right_spec_wrong_algorithm` | Stage 1 规格大体正确，但 Stage 2 算法/实现没有满足规格 | 强化 Stage 2：先写 algorithm sketch，再写 code |
| `signature_or_interface` | 函数名、参数、返回类型、导入或接口与测试不匹配 | 强化硬约束和接口提取 |
| `insufficient_tests` | 可见测试不足以让模型自我归纳正确规则，需要外部/检索 hint | 考虑 retrieval 相似题或 oracle hint，并在报告中标明不再是纯 self-discovery |
| `unknown` | 诊断证据不足或多原因混杂 | 抽样人工复核 |

## Gate

诊断完成后按主因分布决定下一步：

- 若 `right_spec_wrong_algorithm` 占失败样本 `>= 50%`：优先做 Stage 2 algorithm-sketch prompt。
- 若 `wrong_spec` 占失败样本 `>= 50%`：优先做 Stage 1 counterexample/spec prompt。
- 若 `signature_or_interface` 占失败样本 `>= 20%`：先加接口提取和签名检查。
- 若 `insufficient_tests` 占失败样本 `>= 30%`：进入 retrieval/oracle-hint 路线，并在方法描述中单独标注。
- 若没有单一主因：先做一个小型混合 prompt，但继续保持 n=20 gate。

## 执行表

| Step | 内容 | 产物 | 验收条件 | 状态 |
| --- | --- | --- | --- | --- |
| 0 | 确认 two-stage raw/labeled 输入完整 | `data/self_play/llm_critic_mbpp_train_logic_n20_twostage_v1*.jsonl` | raw=20 行、labeled=20 行、fail=17 | done |
| 1 | 写诊断计划文档 | 本文档 | 分类标准和 gate 明确 | done |
| 2 | 实现诊断脚本 | `scripts/diagnose_two_stage_failures.py` | 本地 `py_compile` 通过 | done |
| 3 | 运行诊断脚本 | `data/analysis/two_stage_failure_diagnosis.jsonl`, `docs/logic_two_stage_failure_diagnosis.md` | 17 个失败样本全部有分类和证据 | done |
| 4 | 检查诊断质量 | 本文档 | 分类覆盖率 100%，分布表与 JSONL 一致 | done |
| 5 | 按 gate 修订下一步方案 | 本文档、总报告、对齐表 | 明确下一轮是 Stage 1、Stage 2、interface 还是 retrieval | done |
| 6 | 同步远程 | 远程同名文件 | 远程文件存在且 metrics/报告一致 | done |

## 诊断命令

```bash
python3 scripts/diagnose_two_stage_failures.py \
  --labeled data/self_play/llm_critic_mbpp_train_logic_n20_twostage_v1_labeled.jsonl \
  --output-jsonl data/analysis/two_stage_failure_diagnosis.jsonl \
  --output-md docs/logic_two_stage_failure_diagnosis.md
```

远程同步后可用同一命令在服务器复现。

## 检查清单

1. 输入文件是否都是 20 行。
2. 诊断 JSONL 是否正好 17 行。
3. 每行是否包含 `id`、`diagnosis`、`confidence`、`evidence`、`recommended_fix`。
4. Markdown 报告是否有分类分布、代表样本和下一步建议。
5. 若主因分布触发 gate，是否已修改后续方案。

## 修订记录

| 时间 | Step | 检查结果 | 方案是否修改 |
| --- | --- | --- | --- |
| 2026-07-03 16:08 | 0/1 | raw/labeled 均为 20 行，失败 17 个且全是 logic_error | 进入失败诊断，不继续 k=3/DPO |
| 2026-07-03 16:15 | 2 | `scripts/diagnose_two_stage_failures.py` 本地 `py_compile` 通过；无 LLM smoke 前 3 条可生成诊断 | 同步远程并用 Qwen 做 17 条完整归因 |
| 2026-07-03 16:20 | 3 | Qwen 诊断完成：17/17 均有分类、证据、recommended fix | 进入质量检查 |
| 2026-07-03 16:22 | 4 | JSONL 为 17 行；分类分布 `right_spec_wrong_algorithm=17`；置信度 high=4、medium=13；关键字段无缺失 | 主因高度集中，触发 Stage 2 algorithm-sketch gate |
| 2026-07-03 16:25 | 5 | 下一步从 Stage 1/spec 改为 Stage 2：先生成 algorithm sketch 和 test simulation，再写 code | 不进入 retrieval/oracle；不优先改 Stage 1 |
| 2026-07-03 16:28 | 6 | 远程检查通过：脚本、JSONL、诊断报告、总报告、对齐表均存在；诊断分布仍为 `right_spec_wrong_algorithm=17` | 本轮 goal 完成 |

## 诊断结果

| Diagnosis | Count |
| --- | ---: |
| `right_spec_wrong_algorithm` | 17 |

| Confidence | Count |
| --- | ---: |
| `high` | 4 |
| `medium` | 13 |

代表性证据：

- `mbpp/train/603`: Stage 1 能描述 ludic number，但 Stage 2 过滤逻辑输出空列表。
- `mbpp/train/609`: Stage 2 返回中位数 `20`，而测试期望 periodic function minimum `15`。
- `mbpp/train/610`: 规格方向正确，但 Stage 2 删除了错误下标，actual `[1, 1, 2, 4, 4, 5, 1]` vs expected `[1, 1, 3, 4, 4, 5, 1]`。
- `mbpp/train/659`: 重复元素识别正确但顺序错，actual `[20, 30, 60, -20]` vs expected `[20, 30, -20, 60]`。

## Gate 决策

`right_spec_wrong_algorithm` 占 `17/17 = 100%`，超过 `>= 50%` 的 Stage 2 gate。

因此下一轮不优先做：

- Stage 1 counterexample/spec prompt。
- interface/signature 修复。
- retrieval/oracle hint。
- DPO 训练。

下一轮应做：

1. Stage 2a：基于 Stage 1 spec 先写 algorithm sketch。
2. Stage 2b：让模型手工模拟至少 2 条可见测试，写出 expected vs predicted。
3. Stage 2c：再生成 revised code。
4. verifier 后仍使用同样 gate：单候选若 `<4/20` 停止，`>=4/20` 扩 k=3，`>=8/20` 才合并 DPO。
