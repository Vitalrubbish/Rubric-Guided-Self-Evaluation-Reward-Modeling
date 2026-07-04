# DPO LoRA Training Complete

日期：2026-07-02  
远程目录：`/data2/acm-group-3/Rubric-Guided-Self-Evaluation-Reward-Modeling`

## 训练说明

本次跑的是实际 DPO LoRA 训练，不是规则修正或 verifier 测试。

训练数据：

`data/preferences/preference_pairs_qwen25_k1.jsonl`

训练样本数：551 对 chosen/rejected preference pairs。

输出目录：

`outputs/dpo_lora_coding_e1_551_mlen768`

日志：

`logs/dpo_lora_coding_e1_551_mlen768_20260702_224226.log`

## 命令

```bash
cd /data2/acm-group-3/Rubric-Guided-Self-Evaluation-Reward-Modeling
export XDG_CACHE_HOME=/data2/acm-group-3/cache
export HF_HOME=/data2/acm-group-3/cache/huggingface
export TRANSFORMERS_CACHE=/data2/acm-group-3/cache/huggingface
export TMPDIR=/data2/acm-group-3/cache/tmp
export CUDA_VISIBLE_DEVICES=1

/data2/acm-group-3/miniconda3/envs/rubric/bin/python scripts/dpo_lora_train.py \
  --model models/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28 \
  --data data/preferences/preference_pairs_qwen25_k1.jsonl \
  --output-dir outputs/dpo_lora_coding_e1_551_mlen768 \
  --epochs 1 \
  --grad-accum 8 \
  --max-length 768
```

## 结果

```json
{
  "data": "data/preferences/preference_pairs_qwen25_k1.jsonl",
  "num_pairs": 551,
  "epochs": 1,
  "lr": 5e-06,
  "beta": 0.1,
  "max_length": 768,
  "steps": 549,
  "skipped": 2,
  "mean_loss": 0.4265603654371585,
  "preference_accuracy": 0.8979963570127505
}
```

产物校验：

- `adapter_model.safetensors`: 80,792,096 bytes
- `adapter_config.json`
- `train_metrics.json`
- tokenizer/chat template files

## 边界

这一步完成了 DPO adapter 训练本身。随后已补充 MBPP validation 上的 adapter 推理评测；全量 1128 条 adapter 评测尚未运行。

## Adapter Validation 评测

已完成 MBPP validation 90 条上的 adapter 推理评测：

| 方法 | 推理后端 | 通过数 | 总数 | pass@1 |
| --- | --- | ---: | ---: | ---: |
| Base Qwen2.5-7B | Transformers | 33 | 90 | 36.67% |
| DPO LoRA adapter | Transformers + PEFT | 55 | 90 | 61.11% |
| 原始基线 | vLLM | 49 | 90 | 54.44% |
| Rubric-guided rule revision | verifier 后处理 | 60 | 90 | 66.67% |

同后端比较下，DPO adapter 比 base model 净增 22 题；与原始 vLLM validation baseline 比净增 6 题；但仍低于 rule revision baseline 5 题。

评测文件：

- `scripts/generate_with_lora_adapter.py`
- `data/responses/dpo_lora_mbpp_validation_labeled.jsonl`
- `data/eval/dpo_lora_mbpp_validation_summary.json`
- `data/eval/dpo_vs_base_hf_mbpp_validation_comparison.json`
- `data/eval/dpo_vs_vllm_baseline_mbpp_validation_comparison.json`
- `data/eval/dpo_vs_rule_revision_mbpp_validation_comparison.json`

限制：当前 DPO preference pairs 来自全量失败样本，包含 validation 失败样本，因此该结果是 adapter effectiveness check，不是严格 held-out 泛化结论。
