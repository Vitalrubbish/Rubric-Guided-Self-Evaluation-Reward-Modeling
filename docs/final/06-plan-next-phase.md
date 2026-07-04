# Next Phase Execution Plan

日期：2026-07-03  
远程目录：`/data2/acm-group-3/Rubric-Guided-Self-Evaluation-Reward-Modeling`

## 目标

把当前项目从“阶段性闭环”推进到更贴近作业要求的下一阶段：

1. Method 2：从 protected revision proxy 升级到真实 LLM critic 小规模闭环。
2. Method 1：补 fixed rubric vs updated rubric 的可复核 A/B audit。
3. Method 3：先做最小跨代码数据集迁移 audit，再决定是否扩到 GSM8K/MATH。

每一步都必须有产物、检查条件和修订记录。不能把“脚本写好了”当成“实验完成了”。

## 当前资源盘点

| 项目 | 结果 | 对方案的影响 |
| --- | --- | --- |
| GPU | 8 张 A800 当前 util 均为 100%，显存已有较大占用 | GPU 实验先小样本，跑不动则记录资源阻塞 |
| Python 环境 | `torch/transformers/vllm/peft/sklearn` 可用 | 可以跑 LLM critic、DPO、评估脚本 |
| 已有数据 | MBPP + HumanEval+ 共 1128 条，baseline/protected revision/DPO 产物齐全 | 下一阶段优先复用已有数据，避免重复下载 |

## 执行表

| Step | 内容 | 产物 | 验收条件 | 状态 |
| --- | --- | --- | --- | --- |
| 0 | 资源与文件盘点 | 本文档 | GPU/依赖/数据状态写入本文档 | done |
| 1 | Method 2 LLM critic mini-loop | `data/self_play/llm_critic_*`, `docs/llm_self_play_critic_results.md` | 生成 N 条 critic+revision，verifier 完成，metrics 写入 | done |
| 2 | Step 1 check + 修订计划 | 本文档修订记录 | 明确 repaired/pairs 数量；如 GPU 阻塞，记录复跑命令 | done |
| 3 | Method 1 fixed-vs-updated audit | `data/analysis/fixed_vs_updated_rubric_ablation.json`, `docs/fixed_vs_updated_rubric_ablation.md` | AUC/Kappa/pass@1/pass->fail 对比齐全 | done |
| 4 | Step 3 check + 修订计划 | 本文档修订记录 | 判断是否需要真正第二轮 DPO | done |
| 5 | Method 3 minimal transfer audit | `data/analysis/meta_transfer_audit.json`, `docs/meta_transfer_audit.md` | 分 split/dataset 指标齐全，caveat 明确 | done |
| 6 | 总报告更新与最终复核 | `docs/final_project_report.md` | 新产物链接齐全，所有 JSON 可读，脚本语法通过 | done |

## Step 1 命令

小样本真实 LLM critic 闭环，优先选择 MBPP train 里 protected revision 已能修好的失败样本，提高探针成功率：

```bash
CUDA_VISIBLE_DEVICES=1 /data2/acm-group-3/miniconda3/envs/rubric/bin/python scripts/llm_self_play_critic.py \
  --model models/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28 \
  --labeled data/responses/coding_all_qwen25_vllm_k1_labeled_v2.jsonl \
  --rubric data/rubrics/auto_rubric_refined.json \
  --prefer-protected-success data/responses/coding_all_qwen25_vllm_k1_protected_revised_labeled.jsonl \
  --dataset mbpp \
  --split train \
  --limit 4 \
  --output data/self_play/llm_critic_mbpp_train_n4.jsonl

/data2/acm-group-3/miniconda3/envs/rubric/bin/python scripts/verify_mbpp_smoke.py \
  --input data/self_play/llm_critic_mbpp_train_n4.jsonl \
  --output data/self_play/llm_critic_mbpp_train_n4_labeled.jsonl

/data2/acm-group-3/miniconda3/envs/rubric/bin/python scripts/evaluate_llm_self_play_critic.py \
  --original-labeled data/responses/coding_all_qwen25_vllm_k1_labeled_v2.jsonl \
  --critic-labeled data/self_play/llm_critic_mbpp_train_n4_labeled.jsonl \
  --pairs-output data/self_play/llm_critic_pairs_mbpp_train_n4.jsonl \
  --metrics-output data/self_play/llm_critic_metrics_mbpp_train_n4.json \
  --md-output docs/llm_self_play_critic_results.md
```

### Step 1b 扩大到 n=16

n=4 探针已通过，下一步扩大到 16 条，产物作为正式小样本记录：

```bash
CUDA_VISIBLE_DEVICES=1 /data2/acm-group-3/miniconda3/envs/rubric/bin/python scripts/llm_self_play_critic.py \
  --model models/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28 \
  --labeled data/responses/coding_all_qwen25_vllm_k1_labeled_v2.jsonl \
  --rubric data/rubrics/auto_rubric_refined.json \
  --prefer-protected-success data/responses/coding_all_qwen25_vllm_k1_protected_revised_labeled.jsonl \
  --dataset mbpp \
  --split train \
  --limit 16 \
  --output data/self_play/llm_critic_mbpp_train_n16.jsonl

/data2/acm-group-3/miniconda3/envs/rubric/bin/python scripts/verify_mbpp_smoke.py \
  --input data/self_play/llm_critic_mbpp_train_n16.jsonl \
  --output data/self_play/llm_critic_mbpp_train_n16_labeled.jsonl

/data2/acm-group-3/miniconda3/envs/rubric/bin/python scripts/evaluate_llm_self_play_critic.py \
  --original-labeled data/responses/coding_all_qwen25_vllm_k1_labeled_v2.jsonl \
  --critic-labeled data/self_play/llm_critic_mbpp_train_n16_labeled.jsonl \
  --pairs-output data/self_play/llm_critic_pairs_mbpp_train_n16.jsonl \
  --metrics-output data/self_play/llm_critic_metrics_mbpp_train_n16.json \
  --md-output docs/llm_self_play_critic_results.md
```

### Step 1c parser 修订后重跑 n=16 v2

n=16 首轮发现 4 条失败来自 JSON 字符串里保留字面量 `\n`，不是模型语义修复失败。因此修订 `llm_self_play_critic.py` 的 `clean_code()`，把双重转义换行恢复成真实换行，然后重跑 v2：

```bash
CUDA_VISIBLE_DEVICES=1 /data2/acm-group-3/miniconda3/envs/rubric/bin/python scripts/llm_self_play_critic.py \
  --model models/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28 \
  --labeled data/responses/coding_all_qwen25_vllm_k1_labeled_v2.jsonl \
  --rubric data/rubrics/auto_rubric_refined.json \
  --prefer-protected-success data/responses/coding_all_qwen25_vllm_k1_protected_revised_labeled.jsonl \
  --dataset mbpp \
  --split train \
  --limit 16 \
  --output data/self_play/llm_critic_mbpp_train_n16_v2.jsonl

/data2/acm-group-3/miniconda3/envs/rubric/bin/python scripts/verify_mbpp_smoke.py \
  --input data/self_play/llm_critic_mbpp_train_n16_v2.jsonl \
  --output data/self_play/llm_critic_mbpp_train_n16_v2_labeled.jsonl

/data2/acm-group-3/miniconda3/envs/rubric/bin/python scripts/evaluate_llm_self_play_critic.py \
  --original-labeled data/responses/coding_all_qwen25_vllm_k1_labeled_v2.jsonl \
  --critic-labeled data/self_play/llm_critic_mbpp_train_n16_v2_labeled.jsonl \
  --pairs-output data/self_play/llm_critic_pairs_mbpp_train_n16_v2.jsonl \
  --metrics-output data/self_play/llm_critic_metrics_mbpp_train_n16_v2.json \
  --md-output docs/llm_self_play_critic_results.md
```

## Step 3 命令

```bash
/data2/acm-group-3/miniconda3/envs/rubric/bin/python scripts/build_fixed_vs_updated_ablation.py
```

## Step 5 命令

```bash
/data2/acm-group-3/miniconda3/envs/rubric/bin/python scripts/build_meta_transfer_audit.py
```

## 修订记录

| 时间 | Step | 检查结果 | 方案是否修改 |
| --- | --- | --- | --- |
| 2026-07-03 12:30 | 0 | GPU 满载，依赖可用，数据齐全 | 修改为 CPU-first + GPU 小样本探针 |
| 2026-07-03 12:40 | 1 | n=4 真实 LLM critic 跑通，verifier 4/4 passed，生成 4 条 A<B pairs | 将 Method 2 从 n=4 探针扩大到 n=16 小样本实验 |
| 2026-07-03 12:41 | 3/5 | fixed-vs-updated audit 与 minimal transfer audit 均生成 JSON/Markdown，指标完整 | 暂不追加第二轮 DPO；先完成 n=16 Method 2 后再决定 |
| 2026-07-03 12:46 | 1 | n=16 首轮 verifier 12/16 passed；4 个失败均因字面量 `\n` 未转成真实换行 | 修订 `clean_code()`，新增 n=16 v2 重跑 |
| 2026-07-03 12:50 | 1/2 | n=16 v2 verifier 16/16 passed，生成 16 条真实 LLM critic A<B pairs | Method 2 小样本闭环完成；下一轮再考虑扩到 54 条 train proxy-success 样本 |
| 2026-07-03 12:55 | 6 | JSON 断言通过：LLM critic 16/16、AUC delta +0.141652、protected pass->fail 0、HumanEval+ auto AUC 0.846；总报告和对齐文档已更新 | 本轮 goal 完成，下一阶段是扩大 LLM critic 样本并合并 DPO |
