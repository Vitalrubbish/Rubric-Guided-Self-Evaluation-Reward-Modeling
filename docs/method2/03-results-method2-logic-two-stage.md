# LLM Self-Play Critic Mini-Loop

## 定位

这是 Method 2 的小规模真实 LLM critic 闭环：模型先对失败输出 A 写错误发现，再生成改进版 B，之后用外部 verifier 判断是否形成 `A < B` preference pair。

## 指标

| Metric | Value |
| --- | ---: |
| Attempted | 20 |
| Successful repairs | 3 |
| Preference pairs | 3 |
| Repair rate | 15.00% |
| Critique extraction rate | 100.00% |

## Transitions

| Transition | Count |
| --- | ---: |
| fail_to_fail | 17 |
| fail_to_pass | 3 |

## 对比

| 方法 | 修复任务数 | Repair rate | 备注 |
| --- | ---: | ---: | --- |
| default prompt, single candidate | 2/20 | 10.00% | 旧单候选基线 |
| default prompt, k=3 | 6/20 | 30.00% | 多候选 + verifier |
| default prompt, k=5 | 7/20 | 35.00% | 边际收益下降 |
| spec-first prompt v1, single candidate | 1/20 | 5.00% | 长 JSON 解释引入格式退化 |
| two-stage spec-code v1, single candidate | 3/20 | 15.00% | 本轮结果 |

## 通过样本

- `mbpp/train/648`
- `mbpp/train/650`
- `mbpp/train/661`

## 失败分布

| Failure after revision | Count |
| --- | ---: |
| logic_error | 17 |
| syntax_error | 0 |
| passed | 3 |

## 结论

two-stage v1 比 spec-first v1 更干净：最终 20 条代码都能 compile，没有残留 syntax error。这说明“把规格归纳和代码生成拆开”确实修复了单轮长 JSON prompt 的格式退化问题。

但 verifier 只确认 3/20 成功，低于预设的 4/20 最低有效 gate。因此本轮不继续 two-stage k=3，不合并进 DPO。下一步应先诊断 17 个失败样本：到底是 Stage 1 规格错、Stage 2 算法实现错，还是可见测试不足以支持模型自我归纳。

## 输出文件

- `data/self_play/llm_critic_mbpp_train_logic_n20_twostage_v1_labeled.jsonl`
- `data/self_play/llm_critic_pairs_mbpp_train_logic_n20_twostage_v1.jsonl`
- `data/self_play/llm_critic_metrics_mbpp_train_logic_n20_twostage_v1.json`

## Caveat

这是小样本资源探针，不应替代全量 Method 2 实验。它的作用是证明 pipeline 真实可跑，并给后续扩大样本量提供检查点。
