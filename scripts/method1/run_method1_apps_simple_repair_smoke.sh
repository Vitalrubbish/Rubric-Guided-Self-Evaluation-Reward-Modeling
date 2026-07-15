#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT_DIR"

PYTHON=${PYTHON:-/data2/acm-group-3/miniconda3/envs/rubric/bin/python}
CONDA_BIN=${CONDA_BIN:-/data2/acm-group-3/miniconda3/envs/rubric/bin}
MODEL=${MODEL:-models/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28}
GPU=${GPU:-2}

INPUT=${INPUT:-data/repair/apps_simple_method1_repair_prompts_v1.jsonl}
OUTPUT=${OUTPUT:-data/repair/apps_simple_method1_repair_smoke20_responses.jsonl}
LABELED=${LABELED:-data/repair/apps_simple_method1_repair_smoke20_labeled.jsonl}
LIMIT=${LIMIT:-20}
K=${K:-1}
TEMPERATURE=${TEMPERATURE:-0.2}
TOP_P=${TOP_P:-0.9}

MAX_MODEL_LEN=${MAX_MODEL_LEN:-12288}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.35}
MAX_TOKENS=${MAX_TOKENS:-2048}
PROMPT_BATCH_SIZE=${PROMPT_BATCH_SIZE:-20}
VERIFY_WORKERS=${VERIFY_WORKERS:-4}
VERIFY_TIMEOUT=${VERIFY_TIMEOUT:-8}

export PATH="$CONDA_BIN:$PATH"
export CUDA_VISIBLE_DEVICES="$GPU"
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/tmp/rubric-cache}
export HF_HOME=${HF_HOME:-/tmp/rubric-cache/huggingface}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-/tmp/rubric-cache/huggingface}
export TMPDIR=${TMPDIR:-/tmp/rubric-tmp}

"$PYTHON" src/generation/vllm_smoke_generate.py \
  --model "$MODEL" \
  --input "$INPUT" \
  --output "$OUTPUT" \
  --limit "$LIMIT" \
  --k "$K" \
  --temperature "$TEMPERATURE" \
  --top-p "$TOP_P" \
  --max-tokens "$MAX_TOKENS" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --prompt-batch-size "$PROMPT_BATCH_SIZE"

"$PYTHON" src/verification/verify_mbpp_smoke.py \
  --input "$OUTPUT" \
  --output "$LABELED" \
  --timeout "$VERIFY_TIMEOUT" \
  --workers "$VERIFY_WORKERS" \
  --process-start-method spawn

echo "Repair smoke complete:"
echo "  responses=$OUTPUT"
echo "  labeled=$LABELED"
