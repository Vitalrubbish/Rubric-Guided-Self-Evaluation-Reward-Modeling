# Train-Only DPO Results

日期：2026-07-03  
远程目录：`/data2/acm-group-3/Rubric-Guided-Self-Evaluation-Reward-Modeling`

## 目的

昨天的 DPO adapter 使用了全量失败样本，包含 validation/test 失败样本。本轮重新构造只来自 MBPP train 的 preference pairs，用来检查无 validation 泄漏时 DPO 是否仍能泛化。

## Train-Only Preference Pairs

文件：

`data/preferences/preference_pairs_qwen25_k1_mbpp_train_only.jsonl`

验收：

- 行数：158
- 全部为 `dataset=mbpp`
- 全部为 `split=train`
- ID 无重复

## DPO 训练

输出：

`outputs/dpo_lora_mbpp_train_only_e1_158_mlen768`

训练指标：

```json
{
  "num_pairs": 158,
  "epochs": 1,
  "steps": 158,
  "skipped": 0,
  "mean_loss": 0.6488701542721519,
  "preference_accuracy": 0.6962025316455697
}
```

## Untouched MBPP Validation

| 方法 | 通过数 | 总数 | pass@1 |
| --- | ---: | ---: | ---: |
| Base Qwen2.5-7B HF | 33 | 90 | 36.67% |
| Train-only DPO HF | 33 | 90 | 36.67% |
| Train-only DPO + rule revision | 49 | 90 | 54.44% |
| 原始 vLLM baseline | 49 | 90 | 54.44% |
| 单独 rule revision baseline | 60 | 90 | 66.67% |

## Checks

Train-only DPO vs base-HF：

- base-HF passed：33
- train-only DPO passed：33
- net delta：0
- fail->pass：3
- pass->fail：3

Train-only DPO + rule revision：

- before revision：33/90
- after revision：49/90
- net delta：+16
- fail->pass：17
- pass->fail：1
- edited responses：25

## Decision

Validation gate 未通过：train-only DPO 本身没有超过 base-HF，也低于原始 vLLM baseline。因此不继续跑 MBPP test 500 条。

后续应先改训练策略，而不是扩大评测：

- 增加无泄漏训练数据
- 加入 rule-revised successful outputs 作为 chosen
- 降低 learning rate 或增加 epoch 并做 validation early stopping
- 设计保护版 rule revision，降低 pass->fail
