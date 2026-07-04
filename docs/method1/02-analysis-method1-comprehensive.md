# Method 1 Fixed vs Updated Rubric Training Ablation

日期：2026-07-03  
远程目录：`/data2/acm-group-3/Rubric-Guided-Self-Evaluation-Reward-Modeling`

## 目的

老师要求 Method 1 比较：

- 固定首轮 rubric
- 允许 rubric 根据错误模式自我更新

当前项目没有跑两条完全独立的在线 RL 轨道。我们采用一个可复现 proxy A/B：

1. `fixed/generic rubric`：不使用错误模式更新，只用通用 correctness/syntax/interface 维度做 baseline。
2. `updated/refined rubric`：基于 551 个失败样本聚类出的 18 类错误模式生成 6 维 refined rubric，并用于 protected revision、偏好数据扩充、DPO 训练和 reward-hacking guard。

这不是完整在线 RL A/B，但能回答：错误模式驱动的 rubric 更新是否带来更强 reward signal 和更好的训练/修复表现。

## Rubric Quality A/B

| Rubric | Dims | Linked error patterns | Coverage | AUC | Kappa | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fixed/generic | 3 | 0 | 0.000 | 0.660 | 0.316 | 0.655 |
| Updated/refined | 6 | 16 | 1.000 | 0.801 | 0.525 | 0.765 |

结论：updated/refined rubric 在 AUC、Kappa、Accuracy 上均明显高于 fixed/generic rubric，说明错误模式归纳确实提高了自评 reward signal 的区分度。

## Revision Impact

| Method | Passed | Total | pass@1 | pass->fail |
| --- | ---: | ---: | ---: | ---: |
| Original baseline | 577 | 1128 | 51.15% | - |
| Unprotected rule revision | 745 | 1128 | 66.05% | 10 |
| Updated rubric-guided protected revision | 755 | 1128 | 66.93% | 0 |

结论：updated rubric-guided protected revision 带来 `+178` fail->pass，且 protected guard 将 pass->fail 降到 0，是当前最强整体方法。

## DPO / Training Proxy A/B

| Training variant | Preference pairs | Added signal | Raw validation | Protected validation |
| --- | ---: | --- | ---: | ---: |
| Base-HF | - | none | 33/90 | - |
| Train-only DPO | 158 | canonical solution only | 33/90 | 50/90 |
| Augmented DPO | 212 | + rubric-guided successful revisions | 37/90 | 54/90 |
| LLMCritic54 DPO | 266 | + explicit error discovery pairs | 43/90 | 54/90 |
| LLMCritic54 + logic k=5 DPO | 273 | + verifier-selected logic self-play pairs | 42/90 | 56/90 |

解释：

- Train-only DPO 是较接近 fixed setting 的训练 baseline：只用 canonical pairs，没有利用 rubric 更新出的修复经验。
- Augmented / LLMCritic / logic-k5 DPO 是 updated setting 的逐步增强：把 refined rubric 引导的 successful revision 和 self-play repaired outputs 作为额外 reward signal。
- updated setting 在 raw validation 上从 `33/90` 提升到最高 `43/90`，protected cascade 后从 `50/90` 提升到最高 `56/90`。
- 但所有 DPO-related 方法仍低于单独 protected rule revision 的 `61/90`，说明当前 LoRA DPO 还没有完全学会 protected revision 的规则化收益。

## Reward Hacking / Safety Check

| Setting | Before protected | After protected | pass->fail |
| --- | ---: | ---: | ---: |
| Full baseline revision | 745/1128 | 755/1128 | 10 -> 0 |
| Train-only DPO validation revision | 49/90 | 50/90 | 1 -> 0 |
| Augmented DPO validation revision | 53/90 | 54/90 | 1 -> 0 |
| Logic k=5 DPO validation revision | 42/90 | 56/90 | 0 -> 0 |

结论：protected revision guard 是当前 reward-hacking 风险控制的主要机制。它只修改 verifier 失败样本，不改已通过样本，因此 pass->fail 被压到 0。

## Method 1 结论

1. 错误模式驱动的 updated rubric 明显优于 fixed/generic rubric：AUC `0.660 -> 0.801`，Kappa `0.316 -> 0.525`。
2. updated rubric 能作为 revision/reward signal，protected revision 将全量 pass@1 从 `51.15%` 提升到 `66.93%`。
3. DPO 训练能吸收部分 updated signal：无泄漏 DPO-related protected validation 最好达到 `56/90`。
4. 但 DPO 仍未超过 protected rule revision `61/90`，说明小规模 LoRA DPO 对规则化修复收益学习不足。
5. 本实验是 proxy A/B，不是完整在线 RL 双轨；最终报告需要诚实说明这一点。

## Post-GSM8K Audit

2026-07-03 追加 GSM8K n=100 推荐 benchmark appendix 后，Method 1 的状态没有被夸大为完整在线 RL：

- GSM8K appendix 完成的是 generation -> exact verifier -> failure taxonomy -> rubric -> self-eval metrics。
- 它增强了“推荐 benchmark 也有真实闭环”的证据，但没有在 GSM8K 上继续跑 DPO/RL。
- 因此 Method 1 仍应表述为 coding 主线上的 proxy A/B，而不是 fixed-vs-updated online RL 双轨。
- 继续升级的最小下一步是：在同一 train split 上分别用 fixed/generic rubric 与 updated/refined rubric 构建两套 preference pairs，训练两个 adapter，并在同一 held-out validation/test 上比较；当前算力和收益比不如保留为 future work。

停止继续训练的理由：已有 updated rubric 在自评区分度、protected revision、DPO-related protected validation 上均优于 fixed/generic/canonical-only baseline；额外双轨训练主要会增加成本，不会改变当前最核心结论。

## 证据文件

- `docs/fixed_vs_updated_rubric_ablation.md`
- `data/analysis/fixed_vs_updated_rubric_ablation.json`
- `data/rubrics/generic_rubric_eval_metrics.json`
- `data/rubrics/auto_rubric_eval_metrics.json`
- `docs/logic_k5_dpo_results.md`
- `docs/final_method_leaderboard.md`
- `docs/gsm8k_alignment_results.md`
