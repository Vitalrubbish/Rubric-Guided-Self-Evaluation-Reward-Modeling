# LLM Self-Play Critic Mini-Loop

## 定位

这是 Method 2 的小规模真实 LLM critic 闭环：模型先对失败输出 A 写错误发现，再生成改进版 B，之后用外部 verifier 判断是否形成 `A < B` preference pair。

## 指标

| Metric | Value |
| --- | ---: |
| Attempted | 16 |
| Successful repairs | 16 |
| Preference pairs | 16 |
| Repair rate | 100.00% |
| Critique extraction rate | 100.00% |

## Transitions

| Transition | Count |
| --- | ---: |
| fail_to_pass | 16 |

## 输出文件

- `data/self_play/llm_critic_mbpp_train_n16_v2_labeled.jsonl`
- `data/self_play/llm_critic_pairs_mbpp_train_n16_v2.jsonl`
- `data/self_play/llm_critic_metrics_mbpp_train_n16_v2.json`

## Caveat

这是小样本资源探针，不应替代全量 Method 2 实验。它的作用是证明 pipeline 真实可跑，并给后续扩大样本量提供检查点。
