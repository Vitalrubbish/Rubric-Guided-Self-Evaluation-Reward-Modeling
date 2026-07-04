# Full Alignment Execution Plan

日期：2026-07-03

## 目标

现有项目已经完成 coding benchmark 主线：MBPP + HumanEval+ 上的错误发现、rubric 生成、自评指标、protected revision、self-play preference pairs、DPO ablation 和最小跨代码迁移。为了更贴近原始选题说明，本轮新增一个 GSM8K 小规模真实闭环，补上老师推荐 benchmark 中“数学 verifier + 错误模式发现 + rubric + self-evaluation”的证据。

## 当前差距

| 项目 | 现在状态 | 本轮处理 |
| --- | --- | --- |
| 主 benchmark | coding，MBPP + HumanEval+ | 保留为主线，因为已有完整 verifier 与训练证据 |
| 推荐 benchmark | 已补 GSM8K n=100 小闭环；未跑 MT-Bench/MATH 完整实验 | 将 GSM8K 作为推荐 benchmark appendix 写入最终报告 |
| Method 1 | fixed vs updated 是 proxy A/B | 继续诚实标注；若 GSM8K 闭环完成，补入跨 benchmark evidence |
| Method 2 | self-play 在 coding 上完成，logic repair 偏弱 | 不重复训练，先把 GSM8K 作为诊断补强 |
| Method 3 | MBPP -> HumanEval+ 最小跨代码迁移 | 补充“不是完整跨领域 meta-learning”的说明；若 GSM8K 有指标，作为未来跨领域起点 |

## 执行步骤与 Gate

| Step | 内容 | 产物 | 检查标准 | 状态 |
| --- | --- | --- | --- | --- |
| A0 | 远端/本地状态审计 | 本文档 | 确认 GPU 高占用、远端外网不可用、本机可下载 GSM8K | done |
| A1 | 下载并传入 GSM8K 原始数据 | `data/raw/gsm8k_train.jsonl`, `data/raw/gsm8k_test.jsonl` | train/test JSONL 可读，字段含 question/answer | done |
| A2 | 生成 GSM8K prompts | `data/processed/gsm8k_*_prompts.jsonl` | gold answer 可从 `####` 抽取 | done |
| A3 | 小规模 base model 推理 | `data/responses/gsm8k_qwen25_k1_n100.jsonl` | 100 条真实模型输出 | done |
| A4 | exact verifier 标注 | `data/responses/gsm8k_qwen25_k1_n100_labeled.jsonl` | accuracy 72/100，passed/failed 计数完整 | done |
| A5 | 错误模式发现与 taxonomy | `data/analysis/gsm8k_failures_qwen25_k1_n100.jsonl`, `data/analysis/gsm8k_error_taxonomy_qwen25_k1_n100.yaml` | 每个失败有 error_pattern | done |
| A6 | GSM8K rubric 与自评指标 | `data/rubrics/gsm8k_auto_rubric_n100.json`, metrics JSON | static AUC/Kappa/coverage 已输出 | done |
| A7 | 更新最终报告和 readiness checklist | docs/final* 与 checklist | caveat 已更新为“已补小规模 GSM8K，非完整多轮训练” | done |

## 复现命令

```bash
python scripts/prepare_gsm8k_prompts.py --train-limit 200 --test-limit 100

CUDA_VISIBLE_DEVICES=1 python scripts/generate_gsm8k_responses.py \
  --model models/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28 \
  --input data/processed/gsm8k_test_prompts.jsonl \
  --output data/responses/gsm8k_qwen25_k1_n100.jsonl \
  --limit 100 \
  --batch-size 1 \
  --max-new-tokens 192

python scripts/verify_gsm8k.py \
  --input data/responses/gsm8k_qwen25_k1_n100.jsonl \
  --output data/responses/gsm8k_qwen25_k1_n100_labeled.jsonl \
  --summary-output data/eval/gsm8k_qwen25_k1_n100_summary.json

python scripts/build_gsm8k_failure_artifacts.py \
  --input data/responses/gsm8k_qwen25_k1_n100_labeled.jsonl \
  --failure-output data/analysis/gsm8k_failures_qwen25_k1_n100.jsonl \
  --summary-output data/analysis/gsm8k_failure_summary_qwen25_k1_n100.json \
  --taxonomy-output data/analysis/gsm8k_error_taxonomy_qwen25_k1_n100.yaml

python scripts/generate_gsm8k_rubric.py \
  --taxonomy data/analysis/gsm8k_error_taxonomy_qwen25_k1_n100.yaml \
  --output data/rubrics/gsm8k_auto_rubric_n100.json

python scripts/evaluate_gsm8k_rubric_static.py \
  --labeled data/responses/gsm8k_qwen25_k1_n100_labeled.jsonl \
  --failures data/analysis/gsm8k_failures_qwen25_k1_n100.jsonl \
  --rubric data/rubrics/gsm8k_auto_rubric_n100.json \
  --scores-output data/rubrics/gsm8k_auto_rubric_scores_n100.jsonl \
  --metrics-output data/rubrics/gsm8k_auto_rubric_metrics_n100.json
```

## 修订记录

| 时间 | Step | 检查结果 | 后续调整 |
| --- | --- | --- | --- |
| 2026-07-03 20:05 | A0 | 远端 Hugging Face/GitHub 超时，本机可访问 GitHub raw；GPU 全高占用但部分卡仍有显存余量 | 数据从本机下载后传远端；推理先跑 n=20，成功后扩到 n=100 |
| 2026-07-03 20:10 | A1/A2 | 本机下载 GSM8K train 7473/test 1319；生成 200 train / 100 test prompts，首条 gold answer 为 18 | 同步脚本和数据到远端，先跑远端 dry check |
| 2026-07-03 20:32 | A3/A4/A5/A6 | GSM8K n=100 完成：72/100；失败模式为 final_format_violation 15、arithmetic_or_algebra_slip 9、wrong_problem_model 2、reasoning_truncation 1、ambiguous_final_answer 1；static AUC 0.849、Kappa 0.051 | 将结果写入 `docs/gsm8k_alignment_results.md`，更新最终报告/checklist |
| 2026-07-03 20:40 | A7 | final report、leaderboard、alignment、readiness checklist 均已写入 GSM8K appendix；脚本编译和产物行数检查通过 | 本轮 full alignment 结束；下一步只剩新增研究实验，不是补缺口 |
