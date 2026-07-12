# Phase 2 标注资产与训练就绪检查

日期：2026-07-11

## 结论

项目中确实已有大量“标注数据”，但目前保留的 active training signal 是代码 verifier 自动标签和 v5-lite failures 生成的 verifier-gated rubric signal，不是人工 rubric 标注。因此这些数据可以训练或评估 verifier-supervised pass/fail critic，也可以构造代码 preference；不能用于报告 human-GT Kappa，也不能冒充 8 个 rubric 维度的人工 1--5 分。

GPU 可用性需要在实际启动训练前重新检查。模型、Python 环境和磁盘均已就绪。

## 标注资产盘点

### 1. 外部 verifier 标签

主要文件：

```text
data/responses/phase1_mbpp_hidden_qwen25_k3_labeled.jsonl
```

共 2892 条，字段包含 `passed`、`failure_type`、`safe_diagnostics`、隐藏测试和 private diagnostics。任何 judge/critic 输入只能使用公开任务、公开接口和提交代码，禁止把隐藏测试、`passed`、诊断或 expected/actual 放进输入 prompt。

| Split | Rows | Tasks | Pass | Fail | Mixed tasks |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 1122 | 374 | 603 | 519 | 67 |
| validation | 270 | 90 | 135 | 135 | 14 |
| test | 1500 | 500 | 732 | 768 | 99 |

三个 benchmark split 的 task overlap 为 0，response id 重复为 0。train 标签接近平衡，可以作为辅助 pass/fail critic 的监督数据。失败类型为 `logic_error/runtime_error/syntax_error/timeout`，但它们仍不能可靠映射成全部 8 个 rubric 维度的精确人类分数。

另有 K=5 扩展文件：

```text
data/responses/phase1_5_mbpp_hidden_qwen25_k5_labeled.jsonl
```

共 4820 条，同样是 verifier 标签，不是人标。

### 2. Archived human-review assets

```text
data/hitl/rubric_human_review_queue_blind_v1.jsonl
data/hitl/rubric_annotations_v1.sqlite3
```

这些文件是旧人工审核尝试的归档资产，不再是 active Phase 2 pipeline。SQLite annotations 数量为 0、annotator 数量为 0。当前训练 baseline/signal 不依赖人工审核 UI。

### 3. Archived private verifier key

```text
data/hitl/rubric_human_review_key_private_v1.jsonl
```

该文件保存旧 queue 对应的 verifier label 和旧 judge 结果，只能作为归档核验材料，不能称为 human GT。

### 4. Current judge score files

`data/rubrics/phase2/*judge_scores*.jsonl` 是模型/rubric judge 结果，不是人工标注。当前 active 用法是：

- v3 scores: pre-RL self-evaluation baseline;
- v5-lite failures scores: verifier-gated teacher/scaffold for reward construction.

它们都不能作为 human GT。

## 训练条件检查

- GPU availability must be checked at run time before starting new jobs.
- Qwen2.5-7B-Instruct 权重完整。
- 环境可导入 `torch 2.11.0+cu130`、`transformers 5.12.1`、`peft 0.19.1`。
- `/data2` 尚余约 15TB。
- `outputs/` 中有多个已完成 LoRA adapter，但没有 `trainer_state.json` 或 `checkpoint-*`，它们不是可直接 resume 的中断训练。

## Go/No-Go 决策

当前结论为 **Go for verifier-gated reward construction, No-Go for human-GT claims**。

现在可以安全继续的工作：

1. 使用 v3 作为 pre-RL self-evaluation baseline。
2. 使用 v5-lite failures 构造低噪声 reward/preference 信号。
3. 训练后评估时关闭 execution gate，检验模型自身 rubric-based self-evaluation 是否相对 v3 提升。
4. 若未来重新引入人工标注，必须作为独立 human-GT track 命名和报告。
