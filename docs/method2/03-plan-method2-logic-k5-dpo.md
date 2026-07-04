# Logic k=5 Multi-Candidate Execution Plan

日期：2026-07-03  
远程目录：`/data2/acm-group-3/Rubric-Guided-Self-Evaluation-Reward-Modeling`

## 目标

上一轮在同一批 20 个 MBPP train `logic_error` 样本上做 `k=3` 多候选 self-play，任务级修复从单候选 2/20 提升到 6/20。本轮继续追加两个随机 seed，把候选数扩到 `k=5`，检验 verifier 筛选能否把 repaired tasks 推过 DPO gate。

## Gate

本轮固定同一批 `n=20` 样本：

- 若 repaired tasks `< 8`：不跑完整 DPO，只记录 k=5 结果，后续改 prompt 或继续 k=8。
- 若 repaired tasks `>= 8`：合并 preference data，并跑一轮 train-only DPO + MBPP validation。

## 执行表

| Step | 内容 | 产物 | 验收条件 | 状态 |
| --- | --- | --- | --- | --- |
| 0 | 盘点 k=3 产物 | 本文档 | k=3 为 6/20，合并数据为 272 pairs | done |
| 1 | 生成 seed 404/505 两组候选 | `data/self_play/llm_critic_mbpp_train_logic_n20_seed404.jsonl`, `seed505.jsonl` | 每组 20 行 | done |
| 2 | verifier seed 404/505 | `*_labeled.jsonl` | 每组 20 行，记录 passed | done |
| 3 | 合并 5 组候选并筛选 | `data/self_play/llm_critic_pairs_mbpp_train_logic_n20_k5.jsonl` | 每题最多 1 条 pair，metrics 写入 | done |
| 4 | gate 检查与方案修订 | 本文档 | 决定是否 DPO | done |
| 5 | 若 gate 通过：合并偏好数据 | `data/preferences/preference_pairs_qwen25_k1_mbpp_train_augmented_llmcritic54_logic_k5.jsonl` | 只含 MBPP train，无 validation/test | done |
| 6 | 若 gate 通过：DPO + validation | `docs/logic_k5_dpo_results.md` | 训练、validation、protected validation 均有指标 | skipped |
| 7 | 更新总报告和对齐文档 | `docs/final_project_report.md`, `docs/assignment_requirement_alignment.md` | 新结论写入，最终断言通过 | in_progress |

## Step 1 命令

```bash
for seed in 404 505; do
  CUDA_VISIBLE_DEVICES=1 /data2/acm-group-3/miniconda3/envs/rubric/bin/python scripts/llm_self_play_critic.py \
    --model models/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28 \
    --labeled data/responses/coding_all_qwen25_vllm_k1_labeled_v2.jsonl \
    --rubric data/rubrics/auto_rubric_refined.json \
    --dataset mbpp \
    --split train \
    --failure-type logic_error \
    --limit 20 \
    --temperature 0.7 \
    --top-p 0.95 \
    --seed "$seed" \
    --max-new-tokens 512 \
    --output "data/self_play/llm_critic_mbpp_train_logic_n20_seed${seed}.jsonl"
done
```

## Step 2/3 命令

```bash
for seed in 404 505; do
  /data2/acm-group-3/miniconda3/envs/rubric/bin/python scripts/verify_mbpp_smoke.py \
    --input "data/self_play/llm_critic_mbpp_train_logic_n20_seed${seed}.jsonl" \
    --output "data/self_play/llm_critic_mbpp_train_logic_n20_seed${seed}_labeled.jsonl"
done

/data2/acm-group-3/miniconda3/envs/rubric/bin/python scripts/select_multicandidate_self_play.py \
  --original-labeled data/responses/coding_all_qwen25_vllm_k1_labeled_v2.jsonl \
  --candidate-labeled \
    data/self_play/llm_critic_mbpp_train_logic_n20_seed101_labeled.jsonl \
    data/self_play/llm_critic_mbpp_train_logic_n20_seed202_labeled.jsonl \
    data/self_play/llm_critic_mbpp_train_logic_n20_seed303_labeled.jsonl \
    data/self_play/llm_critic_mbpp_train_logic_n20_seed404_labeled.jsonl \
    data/self_play/llm_critic_mbpp_train_logic_n20_seed505_labeled.jsonl \
  --pairs-output data/self_play/llm_critic_pairs_mbpp_train_logic_n20_k5.jsonl \
  --metrics-output data/self_play/llm_critic_metrics_mbpp_train_logic_n20_k5.json \
  --md-output docs/logic_multicandidate_n20_k5_results.md
```

## 修订记录

| 时间 | Step | 检查结果 | 方案是否修改 |
| --- | --- | --- | --- |
| 2026-07-03 14:20 | 0 | k=3 已完成：20 tasks、60 candidates、6 repaired tasks、272 merged train pairs；GPU 仍满载但可跑小批生成 | 扩展到 k=5，只新增 seed 404/505，避免重复生成已有 3 组 |
| 2026-07-03 14:27 | 1/2 | seed 404/505 均生成 20 条；verifier 分别为 2/20、2/20 | 继续合并五组候选 |
| 2026-07-03 14:28 | 3/4 | k=5 总候选 100 条，passed candidates 13，repaired tasks 7/20 | 未达到完整 DPO gate 8/20，因此不训练 |
| 2026-07-03 14:29 | 5 | 合并 train-only preference data 得到 273 条，其中 logic k=5 pairs 7 条 | 记录数据，下一步应改 prompt 或 k=8，而不是直接 DPO |

## 本轮结果

| 指标 | 数值 |
| --- | ---: |
| Attempted tasks | 20 |
| Total candidates | 100 |
| Passed candidates | 13 |
| Repaired tasks | 7 |
| Preference pairs | 7 |
| Task repair rate | 35.00% |
| Candidate pass rate | 13.00% |
| DPO gate | failed |

k=5 通过的 task ids：

- `mbpp/train/610`
- `mbpp/train/612`
- `mbpp/train/622`
- `mbpp/train/631`
- `mbpp/train/648`
- `mbpp/train/661`
- `mbpp/train/670`

合并后的 preference 文件：

- `data/preferences/preference_pairs_qwen25_k1_mbpp_train_augmented_llmcritic54_logic_k5.jsonl`
- `data/preferences/preference_pairs_qwen25_k1_mbpp_train_augmented_llmcritic54_logic_k5_summary.json`

结论：

- k=5 相比 k=3 只从 6/20 提升到 7/20，边际收益变小。
- 未达到完整 DPO gate 8/20，本轮跳过 DPO。
- 下一步更适合改 prompt：让模型先解释每个失败断言的输入输出含义，再写 revised code；或者做 k=8 但应预期成本较高。
