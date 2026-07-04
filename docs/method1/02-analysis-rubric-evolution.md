# Rubric Evolution Analysis

## 结论

我们已经完成了 `错误模式发现 -> rubric 自动生成 -> reward/revision signal -> DPO/改进评估` 的第一版闭环。按作业要求严格看，当前的“rubric 自我更新 vs 固定首轮 rubric”还不是多轮在线更新，而是一个可复核的离线近似：固定 generic rubric 对比基于失败聚类生成的 refined rubric。

## Fixed vs Updated Rubric

| Rubric | 维度数 | Coverage | AUC | Kappa | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed/generic | 3 | 0.000 | 0.660 | 0.316 | 0.655 |
| Updated/refined | 6 | 1.000 | 0.801 | 0.525 | 0.765 |
| Delta | +3 | +1.000 | +0.142 | +0.209 | +0.110 |

Updated/refined rubric 的 6 个维度：
- Functional Correctness and Edge-Case Coverage (`functional_correctness`)
- Syntax Validity and Parseability (`syntax_parseability`)
- Interface and Test Contract Compliance (`interface_contract_compliance`)
- Runtime Dependency and API Safety (`runtime_dependency_safety`)
- Termination and Complexity Control (`termination_complexity`)
- Output Cleanliness and Single-Solution Formatting (`output_format_cleanliness`)

## Rubric 进化证据

- 初始基线失败：551 / 1128。
- refined taxonomy：18 个 clusters。
- auto rubric 从错误模式中抽象出 6 个可评分维度，覆盖率 1.000。
- 相比 fixed/generic rubric，AUC 提升 +0.142，Kappa 提升 +0.209。

## Reward / Revision Hacking 检查

| 方法 | Passed | pass@1 | pass->fail | 说明 |
| --- | ---: | ---: | ---: | --- |
| Unprotected revision | 745 | 66.05% | 10 | 会破坏已通过样本，作为 hacking/risk ablation |
| Protected revision | 755 | 66.93% | 0 | 只改失败样本，当前主 baseline |

## DPO 训练证据

| 设置 | Pairs | Preference Acc | Validation passed | 备注 |
| --- | ---: | ---: | ---: | --- |
| Train-only DPO | 158 | 0.696 | 33/90 | 无 validation leakage |
| Augmented train-only DPO | 212 | 0.769 | 37/90 | 加入 successful revision pairs |

## Caveat

这份分析足够支撑 Method 1 的作业阶段报告，但还不能声称已经完成多轮在线 self-evolving。真正的下一轮应固定首轮 rubric 跑一条线、允许 rubric 更新再跑一条线，并比较每轮新增/删除/细化的维度。

输出 JSON：

`data/analysis/rubric_evolution_analysis.json`
