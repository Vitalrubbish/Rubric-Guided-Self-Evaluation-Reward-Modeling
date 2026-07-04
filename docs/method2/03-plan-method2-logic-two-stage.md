# Logic Two-Stage Self-Play Critic Execution Plan

日期：2026-07-03  
远程目录：`/data2/acm-group-3/Rubric-Guided-Self-Evaluation-Reward-Modeling`

## 背景

同一批 20 个 MBPP train `logic_error` 样本上，已有结果：

| 方法 | 修复任务数 | 结论 |
| --- | ---: | --- |
| default prompt, single candidate | 2/20 | 旧单候选基线 |
| default prompt, k=3 | 6/20 | 多候选 + verifier 有效 |
| default prompt, k=5 | 7/20 | 边际收益下降 |
| spec-first prompt v1, single candidate | 1/20 | 单轮长 JSON 解释没有转化成正确代码，还引入 syntax error |

因此本轮改为两阶段流程，而不是继续把解释和代码塞进同一次输出。

## 本轮目标

做一个真实可跑的 Method 2 改进版：

1. Stage 1：模型只做错误发现、断言解释、规格归纳，不写代码。
2. Stage 2：模型只基于 Stage 1 的规格和原始测试写 revised code。
3. Syntax repair：若 revised code 本身不能 compile，则再让模型只修语法/格式，不改变算法意图。
4. 用同一个 verifier 判断 `A -> B` 是否真的从 fail 变 pass。
5. 每个 checkpoint 都记录是否达到 gate，并据此修改后续方案。

## Gate

本轮先做单候选、同批 `n=20` 可比实验：

- 若 repaired tasks `< 4`：两阶段单候选仍未明显超过旧 default 单候选，停止本路线，改用 retrieval/oracle hint 或选更适合的任务子集。
- 若 repaired tasks `>= 4`：两阶段 prompt 有效，继续做 two-stage k=3。
- 若 repaired tasks `>= 6`：单候选已达到旧 default k=3，优先扩大到更多 logic samples。
- 若 repaired tasks `>= 8`：达到完整 DPO gate，合并 preference data 并跑一轮 DPO + MBPP validation。

## 执行表

| Step | 内容 | 产物 | 验收条件 | 状态 |
| --- | --- | --- | --- | --- |
| 0 | 盘点历史结果，确定两阶段设计 | 本文档 | 明确 2/20、6/20、7/20、1/20 四个对照 | done |
| 1 | 新建两阶段 critic 脚本 | `scripts/llm_two_stage_self_play_critic.py` | 本地 `py_compile` 通过；输出字段兼容 verifier | done |
| 2 | 同步脚本和计划文档到远程 | 远程同名文件 | 远程 `--help` 正常、远程 `py_compile` 通过 | done |
| 3 | 运行 two-stage 单候选 n=20 | `data/self_play/llm_critic_mbpp_train_logic_n20_twostage_v1.jsonl` | 20 行输出；每行含 `spec_text`、`code_text`、`generated_code` | done |
| 4 | verifier 标注 revised code | `data/self_play/llm_critic_mbpp_train_logic_n20_twostage_v1_labeled.jsonl` | 20 行，含 passed/failure_type | done |
| 5 | 生成 metrics、pairs、结果报告 | `docs/logic_two_stage_n20_results.md` | attempted=20；successful_repairs 与 pairs 一致 | done |
| 6 | 产物检查和 gate 决策 | 本文档 | 明确是否继续 k=3、是否 DPO、是否修订方案 | done |
| 7 | 如 gate 通过：继续 two-stage k=3 或 DPO | 后续文档/数据 | 只在 gate 通过时执行 | skipped |
| 8 | 更新总报告和 assignment alignment | `docs/final_project_report.md`, `docs/assignment_requirement_alignment.md` | 写入 two-stage 结果和修订后的下一步 | done |

## Step 3 命令

```bash
CUDA_VISIBLE_DEVICES=1 /data2/acm-group-3/miniconda3/envs/rubric/bin/python scripts/llm_two_stage_self_play_critic.py \
  --model models/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28 \
  --labeled data/responses/coding_all_qwen25_vllm_k1_labeled_v2.jsonl \
  --rubric data/rubrics/auto_rubric_refined.json \
  --dataset mbpp \
  --split train \
  --failure-type logic_error \
  --limit 20 \
  --temperature 0.0 \
  --spec-max-new-tokens 640 \
  --code-max-new-tokens 640 \
  --repair-max-new-tokens 384 \
  --syntax-repair-attempts 1 \
  --output data/self_play/llm_critic_mbpp_train_logic_n20_twostage_v1.jsonl
```

## Step 4/5 命令

```bash
/data2/acm-group-3/miniconda3/envs/rubric/bin/python scripts/verify_mbpp_smoke.py \
  --input data/self_play/llm_critic_mbpp_train_logic_n20_twostage_v1.jsonl \
  --output data/self_play/llm_critic_mbpp_train_logic_n20_twostage_v1_labeled.jsonl

/data2/acm-group-3/miniconda3/envs/rubric/bin/python scripts/evaluate_llm_self_play_critic.py \
  --original-labeled data/responses/coding_all_qwen25_vllm_k1_labeled_v2.jsonl \
  --critic-labeled data/self_play/llm_critic_mbpp_train_logic_n20_twostage_v1_labeled.jsonl \
  --pairs-output data/self_play/llm_critic_pairs_mbpp_train_logic_n20_twostage_v1.jsonl \
  --metrics-output data/self_play/llm_critic_metrics_mbpp_train_logic_n20_twostage_v1.json \
  --md-output docs/logic_two_stage_n20_results.md
```

## 检查清单

每个阶段完成后必须检查：

1. 脚本是否能编译：`python -m py_compile scripts/llm_two_stage_self_play_critic.py`
2. 远程参数是否可见：`--help`
3. 生成 JSONL 是否 20 行。
4. 每行是否有 `spec_text`、`code_text`、`generated_code`。
5. verifier 是否实际跑完 20 行。
6. metrics 中 `attempted=20` 且 `successful_repairs=preference_pairs`。
7. 若 repaired `<4`，不做 k=3 和 DPO，立即修订后续方案。
8. 若 repaired `>=4`，继续 k=3；若 `>=8`，继续 DPO。

## 修订记录

| 时间 | Step | 检查结果 | 方案是否修改 |
| --- | --- | --- | --- |
| 2026-07-03 15:35 | 0 | 已确认 spec-first v1 为 1/20，低于 default 单候选 2/20 | 改为两阶段流程，先单候选 n=20，不直接 DPO |
| 2026-07-03 15:42 | 1 | 新建 `scripts/llm_two_stage_self_play_critic.py`，本地 `py_compile` 通过；最终仍写 `generated_code`，兼容现有 verifier | 继续同步远程并跑 n=20 |
| 2026-07-03 15:44 | 2 | 远程 `--help` 可见 two-stage 参数，远程 `py_compile` 通过 | 可以正式运行 Step 3 |
| 2026-07-03 15:58 | 3 | 生成 20/20；每行含 `spec_text`、`code_text`、`generated_code`；`prompt_mode=two_stage_spec_code`；最终 `compile_error_after_repair=0/20` | 进入 verifier |
| 2026-07-03 15:59 | 4/5 | verifier passed=3、failed=17；metrics 为 attempted=20、successful_repairs=3、pairs=3 | 未达到 4/20 gate，不做 k=3 和 DPO |
| 2026-07-03 16:01 | 6 | 失败分布为 logic_error 17、passed 3；syntax_error 0 | 两阶段解决了格式退化，但语义修复不足；后续改为成功/失败对照诊断 + targeted hint/retrieval |
| 2026-07-03 16:05 | 8 | 已更新 `docs/final_project_report.md` 和 `docs/assignment_requirement_alignment.md` | 当前 goal 到这里完成；下一轮若继续，应先做失败诊断脚本 |

## 本轮结果

| 指标 | 数值 |
| --- | ---: |
| Attempted tasks | 20 |
| Successful repairs | 3 |
| Preference pairs | 3 |
| Repair rate | 15.00% |
| Critique extraction rate | 100.00% |
| Final compile errors | 0 |
| DPO gate | failed |

成功修复的 task：

- `mbpp/train/648`
- `mbpp/train/650`
- `mbpp/train/661`

失败分布：

| Failure after revision | Count |
| --- | ---: |
| logic_error | 17 |
| passed | 3 |
| syntax_error | 0 |

## Gate 决策

two-stage v1 从 spec-first v1 的 1/20 提升到 3/20，并完全消除了 spec-first 中出现的 syntax error。但它仍低于预设的 4/20 最低有效 gate。

因此：

1. 不继续 two-stage k=3。
2. 不把这 3 条 pair 合并进 DPO 训练集。
3. 暂时保留 default k=5 的 7 条 logic pairs 作为当前最可靠的 semantic self-play 增量。
4. 下一步先做诊断：对比 3 个成功样本和 17 个失败样本，确认 Stage 1 是规格归纳错、Stage 2 是算法实现错，还是任务本身需要更多 oracle/test hint。

## 修订后的后续方案

下一轮不直接扩大采样，而是做 targeted improvement：

1. 新增诊断脚本，抽取每个失败样本的 Stage 1 spec、Stage 2 code、verifier error、visible tests。
2. 人工/LLM 归因 17 个失败：`wrong_spec`、`right_spec_wrong_algorithm`、`signature_or_interface`、`insufficient_tests`。
3. 若多数是 `wrong_spec`：Stage 1 加入更强的测试逐行归纳和 counterexample thinking。
4. 若多数是 `right_spec_wrong_algorithm`：Stage 2 加入 algorithm sketch，然后再写 code。
5. 若多数是 `insufficient_tests`：考虑 retrieval 相似 MBPP 题或用 canonical solution 生成弱 oracle hint，但需要在报告中明确这不是纯 self-discovery。
