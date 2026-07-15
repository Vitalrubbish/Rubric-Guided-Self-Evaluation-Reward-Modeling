#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT_DIR"

PYTHON=${PYTHON:-/data2/acm-group-3/miniconda3/envs/rubric/bin/python}
MODEL=${MODEL:-models/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28}
GPU=${GPU:-2}
PAIRS=${PAIRS:-data/preferences/apps_simple_method1_self_repair_semantic_fenced_canary_dpo_v2.jsonl}
DEV_PROMPTS=${DEV_PROMPTS:-data/processed/apps_simple_method1_dpo_dev_v2_prompts.jsonl}
BASE_RESPONSES=${BASE_RESPONSES:-data/responses/apps_simple_method1_dpo_dev_v2_base_greedy.jsonl}

: "${ADAPTER:?set ADAPTER to the LoRA adapter or checkpoint directory}"
: "${DEV_RESPONSES:?set DEV_RESPONSES to a new raw response path}"
: "${PAIRED_BASE_LABELED:?set PAIRED_BASE_LABELED to a new paired base label path}"
: "${DEV_LABELED:?set DEV_LABELED to a new paired candidate label path}"
: "${PAIRED_VERIFY_MANIFEST:?set PAIRED_VERIFY_MANIFEST to a new manifest path}"
: "${DEV_SUMMARY:?set DEV_SUMMARY to a new comparison summary path}"
: "${DEV_REPORT:?set DEV_REPORT to a new report path}"

export PATH="$(dirname "$PYTHON"):$PATH"
export CUDA_VISIBLE_DEVICES="$GPU"
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/data2/acm-group-3/cache}
export HF_HOME=${HF_HOME:-/data2/acm-group-3/cache/huggingface}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-/data2/acm-group-3/cache/huggingface}
export TMPDIR=${TMPDIR:-/data2/acm-group-3/cache/tmp}

"$PYTHON" -m src.generation.vllm_lora_generate \
  --model "$MODEL" \
  --adapter "$ADAPTER" \
  --input "$DEV_PROMPTS" \
  --output "$DEV_RESPONSES" \
  --temperature 0.0 \
  --top-p 1.0 \
  --repetition-penalty "${REPETITION_PENALTY:-1.0}" \
  --max-tokens 2048 \
  --max-model-len 8192 \
  --gpu-memory-utilization "${EVAL_GPU_MEMORY_UTILIZATION:-0.35}" \
  --prompt-batch-size "${EVAL_PROMPT_BATCH_SIZE:-32}" \
  --max-lora-rank 16

"$PYTHON" -m src.verification.verify_paired_apps_dpo_dev \
  --base-input "$BASE_RESPONSES" \
  --candidate-input "$DEV_RESPONSES" \
  --base-output "$PAIRED_BASE_LABELED" \
  --candidate-output "$DEV_LABELED" \
  --manifest "$PAIRED_VERIFY_MANIFEST" \
  --timeout "${VERIFY_TIMEOUT:-30}" \
  --workers "${VERIFY_WORKERS:-4}" \
  --process-start-method spawn

"$PYTHON" src/analysis-reporting/compare_apps_dpo_dev.py \
  --base-labeled "$PAIRED_BASE_LABELED" \
  --candidate-labeled "$DEV_LABELED" \
  --training-preferences "$PAIRS" \
  --output "$DEV_SUMMARY" \
  --report "$DEV_REPORT" \
  --require-paired-verification
