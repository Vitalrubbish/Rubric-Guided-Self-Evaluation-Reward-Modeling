# GSM8K -> MATH Transfer Execution Plan

日期：2026-07-04

## 目标

补齐 Method 3 中“GSM8K -> MATH”的最小真实迁移实验：

1. 使用已经完成的 GSM8K failure-derived rubric 作为 source rubric。
2. 在 MATH 小样本上生成模型回答并用独立 verifier 标注 pass/fail。
3. 不使用 MATH failures 生成 rubric，先做 GSM8K rubric zero-shot self-evaluation。
4. 再用 MATH failures 生成 MATH-derived rubric，作为 adapted comparison / upper diagnostic。
5. 比较 GSM8K-derived、MATH-derived、generic math rubric 的 AUC/Kappa，判断 rubric 是否有跨 benchmark 迁移能力。

## 数据选择

MATH 全集答案形式复杂，包含 LaTeX、根式、集合、区间、多答案和等价表达式。第一版使用更可靠的小样本：

| 维度 | 设置 |
| --- | --- |
| 数据源 | `EleutherAI/hendrycks_math` parquet mirror |
| split | test |
| subjects | Algebra, Prealgebra |
| levels | Level 1-3 |
| answer filter | simple numeric / rational / decimal / short symbolic expression |
| target size | n=50 smoke，若 verifier 稳定扩到 n=100 |

## 执行步骤与 Gate

| Step | 内容 | 产物 | Gate | 状态 |
| --- | --- | --- | --- | --- |
| M0 | 写方案与准备数据源 | 本文档，parquet files | algebra/prealgebra parquet 可读 | done |
| M1 | 构建 MATH prompts | `data/processed/math_transfer_prompts_n100.jsonl` | gold answer 抽取率 100%，人工可读样本正常 | done |
| M2 | verifier 自检 | `data/eval/math_transfer_gold_verifier_check.json` | gold solution 自验 accuracy = 100% | done |
| M3 | 生成 MATH responses | `data/responses/math_transfer_qwen25_n100.jsonl` | 100 条真实模型输出 | done |
| M4 | verifier 标注与失败 taxonomy | labeled JSONL + taxonomy | pass/fail、failure patterns 完整 | done |
| M5 | GSM8K-derived rubric zero-shot | metrics JSON | 输出 AUC/Kappa；未使用 MATH failure labels 修 rubric | done |
| M6 | MATH-derived/generic comparison | metrics JSON + report | 与 GSM8K-derived 对比完整 | done |
| M7 | 更新 Method 3/final docs | final docs | 明确完成项和 remaining caveat | done |
| M8 | 扩展 full MATH verifier | `scripts/verify_math_full.py` | 支持集合、区间、根式、多答案、LaTeX 等价；safe subset regression 不下降 | done |
| M9 | 构建 all-subject Level 1-5 子集 | `data/processed/math_full_prompts_n100.jsonl` | all subjects 覆盖，Level 4-5 覆盖，gold verifier 100/100 | done |
| M10 | full MATH n=100 生成与标注 | `data/responses/math_full_qwen25_n100_labeled.jsonl` | 100 条真实模型输出，full verifier summary 完整 | done |
| M11 | full MATH rubric transfer metrics | `data/rubrics/math_full_*_metrics_n100.json` | GSM8K/generic/MATH-derived 三组指标齐全 | done |
| M12 | 更新 full verifier 报告 | `docs/math_full_verifier_results.md` | safe subset 和 pressure test 定位清楚 | done |

## 关键实验矩阵

| Rubric | 是否看 MATH failures | 用途 |
| --- | --- | --- |
| GSM8K-derived rubric | no | 真正 zero-shot transfer |
| Generic math rubric | no | 通用数学评分 baseline |
| MATH-derived rubric | yes | adapted comparison，不作为 zero-shot 证据 |
| MATH-derived upper bound | yes + verifier pattern | 诊断上界，不代表部署时自评 |

## 风险控制

1. MATH verifier 是最大风险。若 gold solution 自验不能达到 100%，停止并修 verifier。
2. 若 MATH 输出大量非 `####` final answer，先改 prompt，不直接放宽 verifier。
3. 若 static Kappa 低但 AUC 高，应解释为“排序能力存在，阈值化 pass/fail 判断仍弱”。
4. 若 MATH-derived rubric 明显高于 GSM8K-derived，只能说明 target-adapted rubric 更贴合 MATH，不能声称 zero-shot 成功。

## 复现命令草案

```bash
python scripts/prepare_math_transfer_prompts.py \
  --parquet-root data/raw/hendrycks_math_parquet \
  --output data/processed/math_transfer_prompts_n100.jsonl \
  --limit 100

python scripts/verify_math_transfer.py \
  --input data/processed/math_transfer_prompts_n100.jsonl \
  --text-field gold_solution \
  --output data/eval/math_transfer_gold_verifier_check.jsonl \
  --summary-output data/eval/math_transfer_gold_verifier_check.json

CUDA_VISIBLE_DEVICES=1 python scripts/generate_math_transfer_responses.py \
  --model models/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28 \
  --input data/processed/math_transfer_prompts_n100.jsonl \
  --output data/responses/math_transfer_qwen25_n50.jsonl \
  --limit 50
```

## 修订记录

| 时间 | Step | 检查结果 | 后续调整 |
| --- | --- | --- | --- |
| 2026-07-03 21:00 | M0 | 本地可下载 HF parquet；远端可读 `pyarrow/pandas/sympy` | 先传 algebra/prealgebra test parquet 到远端，再构建 safe subset |
| 2026-07-03 21:10 | M1/M2 | 构建 100 条 balanced subset：Algebra 50 / Prealgebra 50；gold verifier 100/100 | 进入模型生成 |
| 2026-07-03 21:28 | M3/M4 | MATH n=100 生成完成；verifier 83/100，Algebra 44/50，Prealgebra 39/50 | 进入 GSM8K-derived rubric zero-shot 和 comparison |
| 2026-07-03 21:31 | M5/M6 | GSM8K-derived AUC 0.883/Kappa 0.181；generic AUC 0.883/Kappa 0.181；MATH-derived AUC 0.883/Kappa 0.596 | 写入 transfer results，并更新 Method 3/final docs |
| 2026-07-03 21:38 | M7 | `final_project_report`、`method3_meta_transfer_final`、`submission_readiness_checklist`、final JSON 均已加入 GSM8K -> MATH safe-subset transfer | 进入最终一致性审计 |
| 2026-07-04 00:40 | M8 | full verifier 新增千位逗号、区间并集、Unicode infinity、多答案分隔处理；safe subset regression 仍为 83/100 | 进入 full MATH pressure subset |
| 2026-07-04 00:45 | M9 | full subset n=100：7 个 subjects，Level 1-5，gold verifier 100/100 | 进入模型生成 |
| 2026-07-04 01:05 | M10/M11 | full MATH pressure subset 43/100；GSM8K-derived AUC 0.873/Kappa 0.123；generic/MATH-derived AUC 0.879/Kappa 0.651 | 更新文档；后续若继续，应优先改 prompt/max tokens 降低 `ambiguous_final_answer` |
