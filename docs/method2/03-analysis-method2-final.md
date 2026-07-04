# Method 2 Final Audit

日期：2026-07-03

## 检查问题

Method 2 要求：

1. 模型生成 response A。
2. 模型显式找出 A 的错误。
3. 生成改进版 B。
4. 用 `(A < B)` preference pairs 训练。
5. 追踪找错能力，分析哪些错误能自发现、哪些需要外部信号。

## 已完成证据

| 模块 | 结果 | 证据 |
| --- | ---: | --- |
| verifier-grounded proxy pairs | 178 pairs | `data/self_play/self_play_pairs_from_protected_revision.jsonl` |
| proxy repair precision given edit | 58.36% | `data/self_play/self_play_error_discovery_metrics.json` |
| proxy repair recall over all failures | 32.30% | `data/self_play/self_play_error_discovery_metrics.json` |
| pass preservation | 100.00% | `data/self_play/self_play_error_discovery_metrics.json` |
| syntax/format LLM critic repairs | 54/54 | `data/self_play/llm_critic_metrics_mbpp_train_n54_v1.json` |
| logic single/default | 2/20 | `data/self_play/llm_critic_metrics_mbpp_train_logic_n20_v1.json` |
| logic k=3 | 6/20 | `data/self_play/llm_critic_metrics_mbpp_train_logic_n20_k3.json` |
| logic k=5 | 7/20 | `data/self_play/llm_critic_metrics_mbpp_train_logic_n20_k5.json` |
| spec-first prompt | 1/20 | `data/self_play/llm_critic_metrics_mbpp_train_logic_n20_specfirst_v1.json` |
| two-stage spec/code | 3/20 | `data/self_play/llm_critic_metrics_mbpp_train_logic_n20_twostage_v1.json` |
| algorithm-sketch prompt | 2/20 | `data/self_play/llm_critic_metrics_mbpp_train_logic_n20_algosketch_v1.json` |
| final preference training | 273 pairs | `outputs/dpo_lora_mbpp_train_augmented_llmcritic54_logic_k5_e1_mlen768/train_metrics.json` |
| best no-leakage DPO-related validation | 56/90 protected | `data/eval/dpo_lora_train_augmented_llmcritic54_logic_k5_mbpp_validation_protected_revised_summary.json` |

## 关键结论

### 哪些错误模型能自发现

模型和 proxy pipeline 最擅长发现/修复：

- duplicated function body after return。
- Markdown/prose/print examples 混入代码。
- 简单 syntax/format contract violation。

这类错误表现为“形式上明显坏”，即使 critic 能力不强，也能通过显式找错和 verifier 形成高置信 preference pairs。

### 哪些错误需要外部信号

logic repair 明显更难：

- k=5 从 20 个 logic tasks 中只修好 7 个。
- two-stage diagnosis 显示 17/17 失败是 `right_spec_wrong_algorithm`：模型能大致说对规格，但写不出正确算法。
- spec-first 和 algorithm-sketch prompt 没有提升，反而低于默认 k=5 verifier selection。

## DPO 后的实际收益

| Method | Raw validation | Protected validation |
| --- | ---: | ---: |
| LLMCritic54 DPO | 43/90 | 54/90 |
| LLMCritic54 + logic k=5 DPO | 42/90 | 56/90 |
| Protected rule revision baseline | - | 61/90 |

结论：

- logic k=5 pairs 没有提高 raw DPO。
- 与 protected revision 级联后提高 `54/90 -> 56/90`，说明少量高置信 logic pairs 对 protected cascade 有小幅贡献。
- 仍未超过 protected rule revision baseline `61/90`，所以不进入 MBPP test，也不继续把低质量 logic pairs 合入训练。

## 停止继续 Method 2 训练的理由

1. 已经完成 `(A < B)` preference pair 构建和 DPO 训练闭环。
2. syntax/format 自发现强，logic 自发现弱，这个差异已经被多组 prompt 实验证实。
3. 继续堆 k 或 prompt 会增加算力成本，但前面 spec-first/two-stage/algorithm-sketch 已显示边际收益不足。
4. 更合理的下一步不是继续当前 7B self-play，而是引入 verifier-feedback repair、更强 critic 或执行轨迹级反馈。

## 与 GSM8K Appendix 的关系

GSM8K n=100 appendix 目前只完成错误发现和 rubric/self-eval，没有继续做 math self-play DPO。它可以作为未来 Method 2 math extension 的数据来源：28 个失败样本可以构造 first-pass repair prompts，但这属于新增研究实验，不影响当前 Method 2 已完成状态。
