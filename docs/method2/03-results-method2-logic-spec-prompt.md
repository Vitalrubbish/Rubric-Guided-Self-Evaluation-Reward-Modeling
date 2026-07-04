# LLM Self-Play Critic Mini-Loop

## 定位

这是 Method 2 的小规模真实 LLM critic 闭环：模型先对失败输出 A 写错误发现，再生成改进版 B，之后用外部 verifier 判断是否形成 `A < B` preference pair。

## 指标

| Metric | Value |
| --- | ---: |
| Attempted | 20 |
| Successful repairs | 1 |
| Preference pairs | 1 |
| Repair rate | 5.00% |
| Critique extraction rate | 100.00% |

## Transitions

| Transition | Count |
| --- | ---: |
| fail_to_fail | 19 |
| fail_to_pass | 1 |

## 对比

| 方法 | 修复任务数 | 备注 |
| --- | ---: | --- |
| default prompt, single candidate | 2/20 | 旧单候选基线 |
| default prompt, k=3 | 6/20 | 多候选 + verifier |
| default prompt, k=5 | 7/20 | 边际收益下降 |
| spec-first prompt v1, single candidate | 1/20 | 本轮结果 |

## 结论

spec-first v1 没有通过最低有效 gate。虽然 critique 提取率是 100%，但只有 1 个 revised code 通过 verifier；失败样本主要仍是 logic assertion failed，并出现了 3 个 syntax error。因此本轮不合并进 DPO 训练集。

下一步应避免继续扩大这个单轮长 JSON prompt，改做两阶段流程：先单独生成断言解释和 inferred spec，再用这个 spec 生成严格多行 Python 代码，并在 verifier 前增加轻量 syntax repair。

## 输出文件

- `data/self_play/llm_critic_mbpp_train_logic_n20_specfirst_v1_labeled.jsonl`
- `data/self_play/llm_critic_pairs_mbpp_train_logic_n20_specfirst_v1.jsonl`
- `data/self_play/llm_critic_metrics_mbpp_train_logic_n20_specfirst_v1.json`

## Caveat

这是小样本资源探针，不应替代全量 Method 2 实验。它的作用是证明 pipeline 真实可跑，并给后续扩大样本量提供检查点。
