# Phase 2: Rubric Judge 高分塌缩诊断与 v3 修复

日期：2026-07-11

## 1. 结论

问题仍然存在于当前已经完成的 Qwen judge 结果中，而且程度很严重。1770 条 MBPP validation/test 回答中，外部 verifier 判错的 903 条回答有 75.75% 仍得到至少 4 分，60.91% 至少 4.5 分，51.50% 得到满分 5；其中 51.16% 是所有 8 个维度都为 5。错误回答的分数中位数和正确回答一样，都是 5。

| Verifier label | N | Mean | Median | Score >= 4 | Score >= 4.5 | Score = 5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Correct | 867 | 4.735 | 5 | 90.08% | 79.82% | 67.47% |
| Incorrect | 903 | 4.452 | 5 | 75.75% | 60.91% | 51.50% |

原始二分类结果的 Cohen's Kappa 为 0.153、accuracy 为 57.29%、AUC 为 0.604。validation 上训练的 logistic calibrator 将总体 Kappa 提高到 0.290，但仍错误接受 527 条 verifier-fail 回答。只把 overall threshold 提到 5，validation/test 的错误高分率仍分别为 55.56%/50.78%，因此阈值调整无法解决关键维度本身被误判为 5 的问题。

这里应称为 **Cohen's Kappa（一致性系数）**，不是卡方。Kappa 的理论上限是 1，1 表示逐条完全一致，不应把“必须达到 1”作为现实验收条件；但当前 0.15--0.29 确实不够可靠。

## 2. 根因

1. 旧 prompt 不强制执行具体输入 trace，模型可凭代码外观直接给 5。
2. “语法正确、接口正确”被当成功能正确的证据。
3. 不相关维度也常被打 5，等权均值掩盖单个致命语义错误。
4. 边界错误被误当作轻微瑕疵而给 4；实际上只要产生错误输出，就是功能错误。
5. 同一模型评自己的输出存在自我偏好；没有 human GT 时，prompt 无法可靠校准。

## 3. v3 已实施修改

- 统一 1--5 分锚点：任何具体错误值、类型/形状错误、异常、不终止或公开契约违例，受影响语义维度最高只能为 2；未发现反例但仍无法证明正确时只能为 3，且 3 不通过。
- 强制 ordinary、boundary、adversarial 三类公开规格探针。缺少任一类时 overall 最高为 3；任何不一致探针把相关语义维度压到最多 2。
- primary overall 改成“适用语义维度的最低分”。结构维度的 5 分不能抬高语义错误；均值只保留为质量诊断。
- 输出契约、算法语义、接口、语法和运行时设为 always-applicable，模型不能通过标成 N/A 隐藏关键错误。
- 加入 7 个 validation-only 正反例和 bad-rubric pattern，采用 `contrastive_fewshot`；正式默认只评 test，避免 few-shot task 泄漏。
- 标注平台展示统一分数锚点、维度角色和必标维度，并由服务端再次强制必标。
- 每次评测自动报告 verifier-fail 回答中 `score>=4`、`score>=4.5`、`score=5` 和“所有适用维度=5”的比例。

关键文件：

```text
data/rubrics/phase2/judge_guidance_score_collapse_v3.json
data/rubrics/phase2/mbpp_hidden_llm_rubric_hitl_v3.json
src/rubric/evaluate_llm_rubric_judge.py
scripts/run_phase2_hitl_v3_judge.sh
tests/test_score_collapse_fix.py
```

## 4. 当前使用方式

v3 已完成 full test run，但它不是 solved judge，而是后续 RL 的 pre-RL self-evaluation baseline。

Full test metrics:

| Metric | Value |
| --- | ---: |
| AUC | 0.618196 |
| Accuracy | 0.586667 |
| Cohen's Kappa | 0.184811 |
| Overacceptance | 0.701823 |
| False rejection | 0.110656 |

当前保留策略：

1. v3 作为训练前 self-evaluation baseline。
2. v5-lite failures 作为 verifier-gated reward/preference 信号来源。
3. 训练后评估必须关闭 execution gate，并与 v3 baseline 对比。
4. 不再使用旧 human-dev/human-test HITL 流程作为 active Phase 2 pipeline。
5. Coding reward 仍采用 verifier-first 边界：verifier failure 不应被 judge-only 高分覆盖。

若训练后的无 gate self-evaluation 仍接近 v3 水平，说明模型没有真正内化 rubric；v5-lite failures 的提升只能报告为外部执行证据带来的 teacher/scaffold 效果。
