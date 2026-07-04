# Logic Spec-First Prompt Execution Plan

日期：2026-07-03  
远程目录：`/data2/acm-group-3/Rubric-Guided-Self-Evaluation-Reward-Modeling`

## 目标

前一轮同一批 20 个 MBPP train `logic_error` 样本上：

| 方法 | 修复任务数 |
| --- | ---: |
| 单候选 default prompt | 2/20 |
| default prompt, k=3 | 6/20 |
| default prompt, k=5 | 7/20 |

k=5 相比 k=3 只多修复 1 个任务，说明继续堆同 prompt 候选的边际收益变低。本轮改 prompt：让模型先逐条解释失败断言的输入输出含义，归纳函数规格，再写 revised code。

## Gate

本轮先做单候选、同批 `n=20` 可比实验：

- 若 repaired tasks `< 4`：说明 spec-first 单候选没有稳定超过旧单候选，后续要换成 stronger oracle hint 或检索相似题。
- 若 repaired tasks `>= 4`：说明 prompt 有效，继续扩成 spec-first k=3。
- 若 repaired tasks `>= 6`：单候选已接近旧 k=3，优先扩大样本而不是继续 default k。
- 若 repaired tasks `>= 8`：达到完整 DPO gate，合并 preference data 并跑一轮 DPO + validation。

## 执行表

| Step | 内容 | 产物 | 验收条件 | 状态 |
| --- | --- | --- | --- | --- |
| 0 | 盘点 default prompt 结果 | 本文档 | 明确 2/20、6/20、7/20 三个对照 | done |
| 1 | 修改 critic prompt，新增 `spec_first` 模式 | `scripts/llm_self_play_critic.py` | 本地 `py_compile` 通过，旧 `default` 模式保留 | done |
| 2 | 同步脚本到远程 | 远程 `scripts/llm_self_play_critic.py` | 远程 help 能看到 `--prompt-mode` | done |
| 3 | 运行 spec-first 单候选 n=20 | `data/self_play/llm_critic_mbpp_train_logic_n20_specfirst_v1.jsonl` | 20 行输出，记录 `prompt_mode=spec_first` | done |
| 4 | verifier 标注 revised code | `data/self_play/llm_critic_mbpp_train_logic_n20_specfirst_v1_labeled.jsonl` | 20 行，含 passed/failure_type | done |
| 5 | 生成 metrics、pairs、结果报告 | `docs/logic_spec_prompt_n20_results.md` | metrics 中 attempted=20，pairs 与 successful_repairs 一致 | done |
| 6 | gate 检查并修订后续计划 | 本文档 | 明确是否 DPO、是否扩大到 k=3 | done |
| 7 | 更新总报告和 assignment alignment | `docs/final_project_report.md`, `docs/assignment_requirement_alignment.md` | 对比表加入 spec-first 结果 | done |

## Step 3 命令

```bash
CUDA_VISIBLE_DEVICES=1 /data2/acm-group-3/miniconda3/envs/rubric/bin/python scripts/llm_self_play_critic.py \
  --model models/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28 \
  --labeled data/responses/coding_all_qwen25_vllm_k1_labeled_v2.jsonl \
  --rubric data/rubrics/auto_rubric_refined.json \
  --dataset mbpp \
  --split train \
  --failure-type logic_error \
  --limit 20 \
  --prompt-mode spec_first \
  --temperature 0.0 \
  --max-new-tokens 768 \
  --output data/self_play/llm_critic_mbpp_train_logic_n20_specfirst_v1.jsonl
```

## Step 4/5 命令

```bash
/data2/acm-group-3/miniconda3/envs/rubric/bin/python scripts/verify_mbpp_smoke.py \
  --input data/self_play/llm_critic_mbpp_train_logic_n20_specfirst_v1.jsonl \
  --output data/self_play/llm_critic_mbpp_train_logic_n20_specfirst_v1_labeled.jsonl

/data2/acm-group-3/miniconda3/envs/rubric/bin/python scripts/evaluate_llm_self_play_critic.py \
  --original-labeled data/responses/coding_all_qwen25_vllm_k1_labeled_v2.jsonl \
  --critic-labeled data/self_play/llm_critic_mbpp_train_logic_n20_specfirst_v1_labeled.jsonl \
  --pairs-output data/self_play/llm_critic_pairs_mbpp_train_logic_n20_specfirst_v1.jsonl \
  --metrics-output data/self_play/llm_critic_metrics_mbpp_train_logic_n20_specfirst_v1.json \
  --md-output docs/logic_spec_prompt_n20_results.md
```

## 修订记录

| 时间 | Step | 检查结果 | 方案是否修改 |
| --- | --- | --- | --- |
| 2026-07-03 15:05 | 0/1 | 本地新增 `spec_first` prompt mode，`py_compile` 通过 | 保留 default prompt 做可复现对照，先跑单候选 n=20 |
| 2026-07-03 15:10 | 2 | 远程 `--help` 显示 `--prompt-mode {default,spec_first}`，远程 `py_compile` 通过 | 可以正式生成 |
| 2026-07-03 15:18 | 3 | 生成 20/20；本地检查 `prompt_mode=spec_first` 为 20 条 | 进入 verifier，不用模型自评替代测试 |
| 2026-07-03 15:19 | 4/5 | verifier passed=1、failed=19；metrics 为 attempted=20、successful_repairs=1、pairs=1 | 未达到 `<4` 的最低有效 gate，跳过 DPO 和 spec-first k=3 |
| 2026-07-03 15:23 | 6/7 | 失败样本主要仍是 logic assertion failed，另有 3 条 syntax error；唯一修复成功为 `mbpp/train/677` | 后续从单轮长 JSON prompt 改为两阶段：先生成规格，再用规格+测试生成代码；同时加强代码格式约束 |

## 本轮结果

| 指标 | 数值 |
| --- | ---: |
| Attempted tasks | 20 |
| Successful repairs | 1 |
| Preference pairs | 1 |
| Repair rate | 5.00% |
| Critique extraction rate | 100.00% |
| DPO gate | failed |

唯一成功修复的 task：

- `mbpp/train/677`

失败分布：

| Failure after revision | Count |
| --- | ---: |
| logic_error | 16 |
| syntax_error | 3 |
| passed | 1 |

## 结论与方案修订

spec-first v1 没有超过旧单候选 default prompt 的 2/20，反而降到 1/20。它能让模型生成可解析 critique，但长 JSON 输出没有稳定转化为正确代码，还引入了少量代码格式退化。

因此本轮不做 DPO，也不扩成 spec-first k=3。下一步更合理的路线是：

1. 两阶段 prompt：第一阶段只输出 assertion analysis 和 inferred spec；第二阶段只基于 spec、原题和测试输出代码。
2. 给 code generation 阶段加硬格式约束：多行 Python、保留函数签名、不要把循环/条件压成一行。
3. 加一个轻量 compile/syntax repair pass，避免本来可能可测的候选死在格式问题上。
4. 若两阶段单候选达到至少 4/20，再做 k=3；若达到 8/20，再合并 preference data 并跑 DPO。
