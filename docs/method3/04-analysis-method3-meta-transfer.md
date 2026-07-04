# Minimal Meta-Transfer Audit

## 目的

这一步只做 Method 3 的最小可落地产物：检查当前 coding rubric 在 MBPP train/validation/test 与 HumanEval+ 上是否仍有区分度。它不是完整的 GSM8K -> MATH meta-learning。

## Auto Rubric By Group

| Group | N | Passed | AUC | Kappa | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| humanevalplus | 164 | 73 | 0.846 | 0.644 | 0.817 |
| humanevalplus/test | 164 | 73 | 0.846 | 0.644 | 0.817 |
| mbpp | 964 | 504 | 0.791 | 0.501 | 0.756 |
| mbpp/test | 500 | 239 | 0.785 | 0.498 | 0.744 |
| mbpp/train | 374 | 216 | 0.798 | 0.512 | 0.778 |
| mbpp/validation | 90 | 49 | 0.795 | 0.438 | 0.733 |

## Generic Rubric By Group

| Group | N | Passed | AUC | Kappa | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| humanevalplus | 164 | 73 | 0.824 | 0.629 | 0.811 |
| humanevalplus/test | 164 | 73 | 0.824 | 0.629 | 0.811 |
| mbpp | 964 | 504 | 0.640 | 0.273 | 0.629 |
| mbpp/test | 500 | 239 | 0.621 | 0.247 | 0.632 |
| mbpp/train | 374 | 216 | 0.653 | 0.280 | 0.615 |
| mbpp/validation | 90 | 49 | 0.688 | 0.359 | 0.667 |

## Check

如果 auto rubric 在 HumanEval+ 上仍明显高于 generic rubric，说明它至少有一定跨代码数据集迁移能力。GSM8K n=100 已作为推荐 benchmark appendix 补充；完整 Method 3 仍需要进一步做 MATH held-out 或真正的 GSM8K -> MATH 跨领域 meta-learning。

输出 JSON：

`data/analysis/meta_transfer_audit.json`
