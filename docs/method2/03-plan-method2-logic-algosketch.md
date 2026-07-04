# Logic Algorithm-Sketch Self-Play Execution Plan

日期：2026-07-03  
远程目录：`/data2/acm-group-3/Rubric-Guided-Self-Evaluation-Reward-Modeling`

## 背景

同一批 20 个 MBPP train `logic_error` 样本上，已有结果：

| 方法 | 修复任务数 | 结论 |
| --- | ---: | --- |
| default prompt, single candidate | 2/20 | 旧单候选基线 |
| default prompt, k=3 | 6/20 | 多候选 + verifier 有效 |
| default prompt, k=5 | 7/20 | 边际收益下降 |
| spec-first prompt v1, single candidate | 1/20 | 长 JSON 解释没有转化成正确代码 |
| two-stage spec-code v1, single candidate | 3/20 | 消除 syntax error，但语义修复不足 |

two-stage 失败诊断显示：17 个失败样本全部是 `right_spec_wrong_algorithm`。因此本轮不继续改 Stage 1，也不进入 retrieval/oracle hint，而是改 Stage 2：先生成算法草图并模拟可见测试，再写 revised code。

## 本轮目标

做一个三阶段 Method 2 改进版：

1. Stage 1：生成 assertion analysis 和 inferred spec。
2. Stage 2a：基于 spec 生成 algorithm sketch，并手工模拟至少 2 条 visible tests。
3. Stage 2b：基于 spec + algorithm sketch + test simulation 生成 revised code。
4. Syntax repair：只在 revised code 不能 compile 时修语法/格式。
5. verifier 判断是否形成真实 `A < B` pair。

## Gate

本轮仍固定同一批 `n=20` 样本：

- 若 repaired tasks `< 4`：algorithm-sketch 没有明显超过 two-stage v1，停止本路线，回到 multi-candidate 或更强外部 hint。
- 若 repaired tasks `>= 4`：algorithm-sketch 有效，继续做 k=3 多候选。
- 若 repaired tasks `>= 6`：单候选达到旧 default k=3 水平，优先扩大到更多 logic samples。
- 若 repaired tasks `>= 8`：达到完整 DPO gate，合并 preference data 并跑 DPO + MBPP validation。

## 执行表

| Step | 内容 | 产物 | 验收条件 | 状态 |
| --- | --- | --- | --- | --- |
| 0 | 盘点 two-stage 诊断结果 | 本文档 | `right_spec_wrong_algorithm=17/17` 明确 | done |
| 1 | 写 algorithm-sketch 执行计划 | 本文档 | gate、检查点、命令明确 | done |
| 2 | 实现三阶段 critic 脚本 | `scripts/llm_algorithm_sketch_self_play_critic.py` | 本地 `py_compile` 通过；输出兼容 verifier | done |
| 3 | 同步脚本和文档到远程 | 远程同名文件 | 远程 `--help` 和 `py_compile` 通过 | done |
| 4 | 运行 n=20 algorithm-sketch 生成 | `data/self_play/llm_critic_mbpp_train_logic_n20_algosketch_v1.jsonl` | 20 行；每行含 `spec_text`、`algorithm_text`、`generated_code` | done |
| 5 | verifier 标注 revised code | `data/self_play/llm_critic_mbpp_train_logic_n20_algosketch_v1_labeled.jsonl` | 20 行，含 passed/failure_type | done |
| 6 | 生成 metrics、pairs、报告 | `docs/logic_algorithm_sketch_n20_results.md` | attempted=20；successful_repairs 与 pairs 一致 | done |
| 7 | gate 检查并修订后续方案 | 本文档 | 明确是否 k=3、是否 DPO、是否停止 | done |
| 8 | 更新总报告和 assignment alignment | `docs/final_project_report.md`, `docs/assignment_requirement_alignment.md` | 写入 algosketch 结果和下一步 | done |
| 9 | 同步远程并最终检查 | 远程文件 | 远程产物存在且指标一致 | done |

## Step 4 命令

```bash
CUDA_VISIBLE_DEVICES=1 /data2/acm-group-3/miniconda3/envs/rubric/bin/python scripts/llm_algorithm_sketch_self_play_critic.py \
  --model models/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28 \
  --labeled data/responses/coding_all_qwen25_vllm_k1_labeled_v2.jsonl \
  --rubric data/rubrics/auto_rubric_refined.json \
  --dataset mbpp \
  --split train \
  --failure-type logic_error \
  --limit 20 \
  --temperature 0.0 \
  --spec-max-new-tokens 640 \
  --algorithm-max-new-tokens 768 \
  --code-max-new-tokens 640 \
  --repair-max-new-tokens 384 \
  --syntax-repair-attempts 1 \
  --output data/self_play/llm_critic_mbpp_train_logic_n20_algosketch_v1.jsonl
```

## Step 5/6 命令

```bash
/data2/acm-group-3/miniconda3/envs/rubric/bin/python scripts/verify_mbpp_smoke.py \
  --input data/self_play/llm_critic_mbpp_train_logic_n20_algosketch_v1.jsonl \
  --output data/self_play/llm_critic_mbpp_train_logic_n20_algosketch_v1_labeled.jsonl

/data2/acm-group-3/miniconda3/envs/rubric/bin/python scripts/evaluate_llm_self_play_critic.py \
  --original-labeled data/responses/coding_all_qwen25_vllm_k1_labeled_v2.jsonl \
  --critic-labeled data/self_play/llm_critic_mbpp_train_logic_n20_algosketch_v1_labeled.jsonl \
  --pairs-output data/self_play/llm_critic_pairs_mbpp_train_logic_n20_algosketch_v1.jsonl \
  --metrics-output data/self_play/llm_critic_metrics_mbpp_train_logic_n20_algosketch_v1.json \
  --md-output docs/logic_algorithm_sketch_n20_results.md
```

## 检查清单

每个阶段完成后必须检查：

1. 脚本能否本地/远程 `py_compile`。
2. 远程 `--help` 是否有 `--algorithm-max-new-tokens`。
3. 生成 JSONL 是否 20 行。
4. 每行是否有 `spec_text`、`algorithm_text`、`generated_code`。
5. `compile_error_after_repair` 是否为 0 或记录清楚。
6. verifier 是否跑完 20 行。
7. metrics 中 `attempted=20` 且 `successful_repairs=preference_pairs`。
8. gate 是否触发 k=3、DPO 或停止。

## 修订记录

| 时间 | Step | 检查结果 | 方案是否修改 |
| --- | --- | --- | --- |
| 2026-07-03 16:30 | 0/1 | two-stage 失败诊断为 `right_spec_wrong_algorithm=17/17` | 本轮只改 Stage 2，先做单候选 n=20，不直接 DPO |
| 2026-07-03 16:34 | 2 | 新建 `scripts/llm_algorithm_sketch_self_play_critic.py`，本地 `py_compile` 通过；输出保留 `generated_code` 并新增 `algorithm_text` | 继续同步远程 |
| 2026-07-03 16:36 | 3 | 远程 `py_compile` 通过，`--help` 可见 `--algorithm-max-new-tokens` | 可以启动 n=20 生成 |
| 2026-07-03 16:51 | 4 | 生成 20/20；每行含 `spec_text`、`algorithm_text`、`generated_code`；`compile_error_after_repair=0/20` | 进入 verifier |
| 2026-07-03 16:52 | 5/6 | verifier passed=2、failed=18；metrics 为 attempted=20、successful_repairs=2、pairs=2 | 未达到 4/20 gate，不做 k=3/DPO |
| 2026-07-03 16:54 | 7 | 成功题为 `mbpp/train/648`、`mbpp/train/650`；相比 two-stage v1 没有新增成功题，且丢失 `mbpp/train/661` | 停止 algorithm-sketch 路线，回到 verifier-selected multi-candidate 或更强外部信号 |
| 2026-07-03 16:58 | 8/9 | 总报告、对齐表、脚本、JSONL、metrics、pairs 均已同步远程；远程指标仍为 20/2/2 | 本轮 goal 完成 |

## 本轮结果

| 指标 | 数值 |
| --- | ---: |
| Attempted tasks | 20 |
| Successful repairs | 2 |
| Preference pairs | 2 |
| Repair rate | 10.00% |
| Critique extraction rate | 100.00% |
| Final compile errors | 0 |
| DPO gate | failed |

成功修复的 task：

- `mbpp/train/648`
- `mbpp/train/650`

与 two-stage v1 对比：

| 对比项 | Task ids |
| --- | --- |
| two-stage v1 成功 | `mbpp/train/648`, `mbpp/train/650`, `mbpp/train/661` |
| algosketch v1 成功 | `mbpp/train/648`, `mbpp/train/650` |
| 共同成功 | `mbpp/train/648`, `mbpp/train/650` |
| algosketch 新增成功 | none |
| algosketch 丢失成功 | `mbpp/train/661` |

失败分布：

| Failure after revision | Count |
| --- | ---: |
| logic_error | 18 |
| passed | 2 |
| syntax_error | 0 |

## Gate 决策

algorithm-sketch v1 只有 2/20，低于 two-stage v1 的 3/20，也低于继续实验的 4/20 gate。因此：

1. 不继续 algorithm-sketch k=3。
2. 不合并这 2 条 pair 进 DPO。
3. 暂时保留 default k=5 的 7 条 logic pairs 作为当前最可靠的 semantic self-play 增量。
4. 下一步不再继续加“解释/草图”式 prompt；更合理的是回到 verifier-selected multi-candidate，或引入真实执行反馈/外部 hint，但需在报告中明确外部信号来源。

## 修订后的后续方案

下一轮建议二选一：

1. Resource-safe 路线：使用当前最可靠的 `default k=5` 7 条 logic pairs，只做报告整合和小规模 DPO ablation，不继续增加低质 logic pairs。
2. Accuracy 路线：做 verifier-feedback repair。模型先生成候选，运行 verifier 得到 actual/expected mismatch，再把 mismatch 作为第二轮输入修复；这会引入外部执行反馈，需要在 Method 2 中明确标注。
