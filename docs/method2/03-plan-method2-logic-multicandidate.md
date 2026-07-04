# Logic Multi-Candidate Self-Play Execution Plan

日期：2026-07-03  
远程目录：`/data2/acm-group-3/Rubric-Guided-Self-Evaluation-Reward-Modeling`

## 目标

上一轮单候选 logic critic 只有 2/20 修复成功。本轮在同一批 MBPP train logic errors 上做 `k=3` 多候选 self-play，用 verifier 自动筛选每题第一个通过的 B，检验多候选是否显著提高语义错误修复率。

## Gate

本轮固定 `n=20, k=3`：

- 若 repaired tasks `< 6`：不合并 DPO，记录失败模式，下一步改 prompt/加入更强失败断言解释。
- 若 repaired tasks `>= 6`：合并到 preference data，但先不训练或只做小 DPO probe。
- 若 repaired tasks `>= 8`：可以进入完整 train-only DPO + validation。

## 执行表

| Step | 内容 | 产物 | 验收条件 | 状态 |
| --- | --- | --- | --- | --- |
| 0 | 盘点上一轮 logic n=20 | 本文档 | 确认单候选 2/20，样本 ids 固定 | done |
| 1 | 新增多候选筛选脚本 | `scripts/select_multicandidate_self_play.py` | py_compile 通过 | done |
| 2 | 生成 seed 101/202/303 三组候选 | `data/self_play/llm_critic_mbpp_train_logic_n20_seed*.jsonl` | 每组 20 行 | done |
| 3 | verifier 三组候选 | `*_labeled.jsonl` | 每组 20 行，记录 passed | done |
| 4 | 多候选筛选 | `data/self_play/llm_critic_pairs_mbpp_train_logic_n20_k3.jsonl` | 每题最多 1 条 pair，metrics 写入 | done |
| 5 | gate 检查与方案修订 | 本文档 | 明确是否 DPO | done |
| 6 | 更新总报告和对齐文档 | `docs/final_project_report.md`, `docs/assignment_requirement_alignment.md` | 新结论写入，最终断言通过 | in_progress |

## Step 2 命令

```bash
for seed in 101 202 303; do
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

## Step 3/4 命令

```bash
for seed in 101 202 303; do
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
  --pairs-output data/self_play/llm_critic_pairs_mbpp_train_logic_n20_k3.jsonl \
  --metrics-output data/self_play/llm_critic_metrics_mbpp_train_logic_n20_k3.json \
  --md-output docs/logic_multicandidate_n20_k3_results.md
```

## 修订记录

| 时间 | Step | 检查结果 | 方案是否修改 |
| --- | --- | --- | --- |
| 2026-07-03 14:00 | 0 | 单候选 logic critic 为 2/20；GPU 仍高负载但可跑小批生成 | 改成 k=3 多候选 + verifier 筛选，gate 设为 repaired tasks >= 6 |
| 2026-07-03 14:07 | 1/2 | 新筛选脚本通过 py_compile；seed 101/202/303 三组各生成 20 条 | 继续 verifier |
| 2026-07-03 14:09 | 3/4 | 三组候选 verifier 分别为 4/20、2/20、3/20；多候选任务级 repaired 6/20，生成 6 条 pairs | 刚好达到合并 gate，但未达到完整 DPO gate 8/20 |
| 2026-07-03 14:10 | 5 | k=3 比单候选从 2/20 提升到 6/20，证明 verifier 筛选有效 | 合并 preference data；本轮不跑完整 DPO，只建议小 probe |

## 本轮结果

| 指标 | 数值 |
| --- | ---: |
| Attempted tasks | 20 |
| Total candidates | 60 |
| Passed candidates | 9 |
| Repaired tasks | 6 |
| Preference pairs | 6 |
| Task repair rate | 30.00% |
| Candidate pass rate | 15.00% |

通过的 task ids：

- `mbpp/train/612`
- `mbpp/train/622`
- `mbpp/train/631`
- `mbpp/train/648`
- `mbpp/train/661`
- `mbpp/train/670`

结论：

- k=3 多候选 + verifier 筛选显著优于单候选 logic critic，从 2/20 提升到 6/20。
- 但仍未达到完整 DPO gate 8/20，说明 logic 错误需要更强规格归纳或更多候选。
- 本轮只合并 preference data，不做完整 DPO 训练；下一步可以做 k=5 或加入失败断言解释。
