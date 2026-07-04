# Logic Error Critic Execution Plan

日期：2026-07-03  
远程目录：`/data2/acm-group-3/Rubric-Guided-Self-Evaluation-Reward-Modeling`

## 目标

上一阶段 LLM critic 在 syntax/format 类错误上很强：54/54 修复成功，但 DPO + protected 后仍停在 54/90。当前瓶颈是 `logic_wrong_output`。本轮只针对 MBPP train 的 `logic_error` 做真实 LLM critic，检验模型能否显式发现语义错误并生成可通过 verifier 的 B。

## 当前盘点

| 项目 | 数量/状态 |
| --- | ---: |
| MBPP train 原始失败 | 158 |
| MBPP train logic_error | 75 |
| protected revision 后仍失败的 logic_error | 74 |
| GPU 状态 | A800 仍满载，但前几轮单卡小实验可跑 |

## Gate

先跑 n=20 逻辑错误样本：

- 如果修复成功数 `< 4`：不进入 DPO，写失败模式分析，下一步改 prompt 或引入 external solution hints。
- 如果修复成功数 `>= 4`：生成 logic preference pairs，并合并进上一轮 266 pairs。
- 如果修复成功数 `>= 8`：尝试一轮 logic-augmented DPO，并在 MBPP validation 上评估。

## 执行表

| Step | 内容 | 产物 | 验收条件 | 状态 |
| --- | --- | --- | --- | --- |
| 0 | 盘点 logic 样本和 GPU | 本文档 | 记录 75 个 train logic errors、74 个 protected remaining | done |
| 1 | 改 critic 脚本支持 `--failure-type logic_error` | `scripts/llm_self_play_critic.py` | py_compile 通过 | done |
| 2 | 跑 n=20 logic critic | `data/self_play/llm_critic_mbpp_train_logic_n20_v1.jsonl` | 生成 20 条 critic+revision | done |
| 3 | verifier + pair 构建 | `data/self_play/llm_critic_pairs_mbpp_train_logic_n20_v1.jsonl` | 记录 attempted/repaired/pairs | done |
| 4 | 根据 gate 修订方案 | 本文档 | 决定是否合并 DPO | done |
| 5 | 若 gate 通过：合并偏好数据 | `data/preferences/preference_pairs_qwen25_k1_mbpp_train_augmented_llmcritic54_logic20.jsonl` | 只含 MBPP train，无 validation/test | skipped |
| 6 | 若 gate 通过：DPO + validation | `docs/logic_critic_dpo_results.md` | 训练、validation、protected validation 均有指标 | skipped |
| 7 | 更新总报告与对齐文档 | `docs/final_project_report.md`, `docs/assignment_requirement_alignment.md` | 新结论写入并最终断言通过 | in_progress |

## Step 2/3 命令

```bash
CUDA_VISIBLE_DEVICES=1 /data2/acm-group-3/miniconda3/envs/rubric/bin/python scripts/llm_self_play_critic.py \
  --model models/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28 \
  --labeled data/responses/coding_all_qwen25_vllm_k1_labeled_v2.jsonl \
  --rubric data/rubrics/auto_rubric_refined.json \
  --dataset mbpp \
  --split train \
  --failure-type logic_error \
  --limit 20 \
  --max-new-tokens 512 \
  --output data/self_play/llm_critic_mbpp_train_logic_n20_v1.jsonl

/data2/acm-group-3/miniconda3/envs/rubric/bin/python scripts/verify_mbpp_smoke.py \
  --input data/self_play/llm_critic_mbpp_train_logic_n20_v1.jsonl \
  --output data/self_play/llm_critic_mbpp_train_logic_n20_v1_labeled.jsonl

/data2/acm-group-3/miniconda3/envs/rubric/bin/python scripts/evaluate_llm_self_play_critic.py \
  --original-labeled data/responses/coding_all_qwen25_vllm_k1_labeled_v2.jsonl \
  --critic-labeled data/self_play/llm_critic_mbpp_train_logic_n20_v1_labeled.jsonl \
  --pairs-output data/self_play/llm_critic_pairs_mbpp_train_logic_n20_v1.jsonl \
  --metrics-output data/self_play/llm_critic_metrics_mbpp_train_logic_n20_v1.json \
  --md-output docs/logic_critic_n20_results.md
```

## 修订记录

| 时间 | Step | 检查结果 | 方案是否修改 |
| --- | --- | --- | --- |
| 2026-07-03 13:40 | 0 | MBPP train 有 75 个 logic errors，protected 后仍剩 74 个；确认为当前瓶颈 | 本轮聚焦 `failure_type=logic_error`，先用 n=20 gate 控制风险 |
| 2026-07-03 13:47 | 1 | `llm_self_play_critic.py` 已支持 `--failure-type`，并为 logic error 加入语义修复提示；py_compile 通过 | 继续 n=20 critic |
| 2026-07-03 13:52 | 2/3 | n=20 logic critic 完成，verifier 2/20 passed，生成 2 条 A<B pairs，repair rate 10% | 未达到 gate `<4`，不合并 DPO |
| 2026-07-03 13:54 | 4 | 失败样本多为“critique 方向部分正确但算法仍错”；少量格式/JSON 截断 | 后续改为多候选 + verifier 筛选，或加入 canonical hint/失败单测反馈 |

## 本轮结果

| 指标 | 数值 |
| --- | ---: |
| Attempted logic errors | 20 |
| Successful repairs | 2 |
| Preference pairs | 2 |
| Repair rate | 10.00% |
| Gate for DPO | failed |

成功样本：

- `mbpp/train/612`
- `mbpp/train/648`

主要失败模式：

- 模型能指出局部错误，但生成的算法仍不满足测试，例如 ludic numbers、median、dynamic programming 等。
- 少量输出出现 JSON 截断或单行 Python 导致 runtime/syntax failure。
- 仅给少量 visible tests 不足以让模型稳定恢复完整语义。

后续方案：

1. 对 logic errors 使用 `k>1` 多候选 critic/revision，并用 verifier 自动筛选 successful B。
2. 在 prompt 中加入失败断言的输入输出解释，让模型先归纳规格再写代码。
3. 对仍失败的 semantic tasks，可加入 canonical solution hint 生成“诊断式 preference pair”，但要在报告中标注其不再是纯 self-discovery。
