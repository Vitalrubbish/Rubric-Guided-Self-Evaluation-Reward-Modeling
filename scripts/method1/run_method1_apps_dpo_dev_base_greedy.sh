#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT_DIR"

PYTHON=${PYTHON:-/data2/acm-group-3/miniconda3/envs/rubric/bin/python}
MODEL=${MODEL:-models/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28}
GPU=${GPU:-2}
INPUT=${INPUT:-data/processed/apps_simple_method1_dpo_dev_v2_prompts.jsonl}
RESPONSES=${RESPONSES:-data/responses/apps_simple_method1_dpo_dev_v2_base_greedy.jsonl}
LABELED=${LABELED:-data/responses/apps_simple_method1_dpo_dev_v2_base_greedy_labeled.jsonl}

export PATH="$(dirname "$PYTHON"):$PATH"
export CUDA_VISIBLE_DEVICES="$GPU"
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/data2/acm-group-3/cache}
export HF_HOME=${HF_HOME:-/data2/acm-group-3/cache/huggingface}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-/data2/acm-group-3/cache/huggingface}
export TMPDIR=${TMPDIR:-/data2/acm-group-3/cache/tmp}

"$PYTHON" src/generation/vllm_smoke_generate.py \
  --model "$MODEL" \
  --input "$INPUT" \
  --output "$RESPONSES" \
  --limit 160 \
  --k 1 \
  --temperature 0 \
  --top-p 1 \
  --repetition-penalty "${REPETITION_PENALTY:-1.0}" \
  --max-tokens 2048 \
  --max-model-len 8192 \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.35}" \
  --prompt-batch-size "${PROMPT_BATCH_SIZE:-32}" \
  --seed 42

"$PYTHON" -m src.verification.verify_mbpp_smoke \
  --input "$RESPONSES" \
  --output "$LABELED" \
  --timeout "${VERIFY_TIMEOUT:-10}" \
  --workers "${VERIFY_WORKERS:-8}" \
  --process-start-method spawn

echo "Frozen greedy DPO-dev base complete:"
echo "  responses=$RESPONSES"
echo "  labeled=$LABELED"
