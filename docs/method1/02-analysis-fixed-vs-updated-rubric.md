# Fixed vs Updated Rubric Ablation

## 目的

这一步把 Method 1 里的 `fixed first-round rubric vs self-updated rubric` 落成一个可复核的 CPU audit。当前不是第二轮 DPO，而是用已验证产物比较 fixed/generic rubric 和基于错误模式更新出的 refined rubric。

## Rubric Quality

| Rubric | Dims | Linked patterns | Coverage | AUC | Kappa | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fixed/generic | 3 | 0 | 0.000 | 0.660 | 0.316 | 0.655 |
| Updated/refined | 6 | 16 | 1.000 | 0.801 | 0.525 | 0.765 |

## Method Impact

| Method | Passed | Total | pass@1 | pass->fail |
| --- | ---: | ---: | ---: | ---: |
| Original baseline | 577 | 1128 | 51.15% | - |
| Updated rubric-guided protected revision | 755 | 1128 | 66.93% | 0 |
| Unprotected revision risk ablation | 745 | 1128 | 66.05% | 10 |

## Check

- AUC delta: +0.142
- Kappa delta: +0.209
- Protected pass@1 delta vs baseline: +0.158
- pass->fail is 0 for protected revision, so the reward-hacking guard passes for this baseline.
