# Augmented Train-Only DPO Results

日期：2026-07-03  
远程目录：`/data2/acm-group-3/Rubric-Guided-Self-Evaluation-Reward-Modeling`

## 目的

上一轮 train-only DPO 只用 canonical solution 作为 chosen，在 untouched MBPP validation 上没有超过 base-HF。  
本轮加入 rule-revised successful outputs 作为额外 chosen，测试模型是否能学习“从失败输出到修正输出”的模式。

## Augmented Preference Pairs

文件：

`data/preferences/preference_pairs_qwen25_k1_mbpp_train_augmented.jsonl`

构成：

| chosen source | 数量 |
| --- | ---: |
| canonical_solution | 158 |
| rule_revised_success_output | 54 |
| total | 212 |

验收：

- 全部为 `dataset=mbpp`
- 全部为 `split=train`
- 组合 key 无重复

## DPO 训练

输出：

`outputs/dpo_lora_mbpp_train_augmented_e1_212_mlen768`

训练指标：

```json
{
  "num_pairs": 212,
  "epochs": 1,
  "steps": 212,
  "skipped": 0,
  "mean_loss": 0.6510447376179245,
  "preference_accuracy": 0.7688679245283019
}
```

## MBPP Validation

| 方法 | 通过数 | 总数 | pass@1 |
| --- | ---: | ---: | ---: |
| Base-HF | 33 | 90 | 36.67% |
| Train-only DPO | 33 | 90 | 36.67% |
| Augmented DPO | 37 | 90 | 41.11% |
| Train-only DPO + rule revision | 49 | 90 | 54.44% |
| Augmented DPO + rule revision | 53 | 90 | 58.89% |
| 原始 vLLM baseline | 49 | 90 | 54.44% |
| 单独 rule revision baseline | 60 | 90 | 66.67% |

## Checks

Augmented DPO vs train-only DPO：

- train-only DPO passed：33
- augmented DPO passed：37
- net delta：+4
- fail->pass：9
- pass->fail：5

Augmented DPO + rule revision：

- before revision：37/90
- after revision：53/90
- net delta：+16
- fail->pass：17
- pass->fail：1
- edited responses：23

Augmented DPO + rule revision vs baselines：

- vs original vLLM：+4
- vs train-only DPO + rule revision：+4
- vs single rule revision：-7

## Decision

Augmented pairs 有帮助，但 DPO 单独没有达到 49/90 gate，因此不跑 MBPP test 500 条。  
Augmented DPO + rule revision 是目前最好的 DPO 相关方法，但仍低于单独 rule revision。下一步应优先做保护版 rule revision 或更稳的 DPO 训练目标。
