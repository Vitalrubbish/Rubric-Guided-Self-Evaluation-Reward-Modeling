# LLM Self-Play Critic Mini-Loop

## 定位

这是 Method 2 的小规模真实 LLM critic 闭环：模型先对失败输出 A 写错误发现，再生成改进版 B，之后用外部 verifier 判断是否形成 `A < B` preference pair。

## 指标

| Metric | Value |
| --- | ---: |
| Attempted | 20 |
| Successful repairs | 2 |
| Preference pairs | 2 |
| Repair rate | 10.00% |
| Critique extraction rate | 100.00% |

## Transitions

| Transition | Count |
| --- | ---: |
| fail_to_fail | 18 |
| fail_to_pass | 2 |

## 对比

| 方法 | 修复任务数 | Repair rate | Syntax error after revision |
| --- | ---: | ---: | ---: |
| default prompt, single candidate | 2/20 | 10.00% | - |
| default prompt, k=3 | 6/20 | 30.00% | - |
| default prompt, k=5 | 7/20 | 35.00% | - |
| spec-first prompt v1, single candidate | 1/20 | 5.00% | 3 |
| two-stage spec-code v1, single candidate | 3/20 | 15.00% | 0 |
| algorithm-sketch v1, single candidate | 2/20 | 10.00% | 0 |

## 通过样本

- `mbpp/train/648`
- `mbpp/train/650`

## 与 two-stage v1 的差异

| 对比项 | Task ids |
| --- | --- |
| two-stage v1 成功 | `mbpp/train/648`, `mbpp/train/650`, `mbpp/train/661` |
| algorithm-sketch v1 成功 | `mbpp/train/648`, `mbpp/train/650` |
| algorithm-sketch 新增成功 | none |
| algorithm-sketch 丢失成功 | `mbpp/train/661` |

## 结论

algorithm-sketch v1 没有通过最低有效 gate。它保持了代码格式稳定，最终 20 条代码均可 compile，但 verifier 只确认 2/20 成功，低于 two-stage v1 的 3/20，也没有新增任何成功 task。

因此本轮不继续 k=3，不合并进 DPO。下一步不应继续堆“解释/草图”式 prompt，而应回到 verifier-selected multi-candidate，或明确引入执行反馈/外部 hint。

## 输出文件

- `data/self_play/llm_critic_mbpp_train_logic_n20_algosketch_v1_labeled.jsonl`
- `data/self_play/llm_critic_pairs_mbpp_train_logic_n20_algosketch_v1.jsonl`
- `data/self_play/llm_critic_metrics_mbpp_train_logic_n20_algosketch_v1.json`

## Caveat

这是小样本资源探针，不应替代全量 Method 2 实验。它的作用是证明 pipeline 真实可跑，并给后续扩大样本量提供检查点。
