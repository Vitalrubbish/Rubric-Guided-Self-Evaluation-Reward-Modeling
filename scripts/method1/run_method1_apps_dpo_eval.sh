#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT_DIR"

PYTHON=${PYTHON:-/data2/acm-group-3/miniconda3/envs/rubric/bin/python}
export PATH="$(dirname "$PYTHON"):$PATH"
MODEL=${MODEL:-models/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28}
ADAPTER=${ADAPTER:-outputs/apps_simple_method1_dpo_lora_v1}
PROMPTS=${PROMPTS:-data/processed/apps_simple_method1_internal_eval_prompts_v1.jsonl}
RESPONSES=${RESPONSES:-data/responses/apps_simple_method1_dpo_lora_v1_internal_eval.jsonl}
LABELED=${LABELED:-data/responses/apps_simple_method1_dpo_lora_v1_internal_eval_labeled.jsonl}
SUMMARY=${SUMMARY:-data/eval/apps_simple_method1_dpo_lora_v1_internal_eval_summary.json}
REPORT=${REPORT:-docs/method1/07-apps-dpo-heldout-results.md}

export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/data2/acm-group-3/cache}
export HF_HOME=${HF_HOME:-/data2/acm-group-3/cache/huggingface}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-/data2/acm-group-3/cache/huggingface}
export TMPDIR=${TMPDIR:-/data2/acm-group-3/cache/tmp}

"$PYTHON" src/data-prep/build_apps_internal_eval_prompts.py

"$PYTHON" -m src.generation.vllm_lora_generate \
  --model "$MODEL" \
  --adapter "$ADAPTER" \
  --input "$PROMPTS" \
  --output "$RESPONSES" \
  --temperature 0.0 \
  --top-p 1.0 \
  --max-tokens 2048 \
  --max-model-len 8192 \
  --gpu-memory-utilization "${EVAL_GPU_MEMORY_UTILIZATION:-0.70}" \
  --prompt-batch-size "${EVAL_PROMPT_BATCH_SIZE:-64}" \
  --max-lora-rank "${LORA_R:-16}"

"$PYTHON" -m src.verification.verify_mbpp_smoke \
  --input "$RESPONSES" \
  --output "$LABELED" \
  --timeout "${VERIFY_TIMEOUT:-10}" \
  --workers "${VERIFY_WORKERS:-8}" \
  --process-start-method spawn

"$PYTHON" src/analysis-reporting/compare_apps_dpo_eval.py \
  --dpo-labeled "$LABELED" \
  --output "$SUMMARY" \
  --report "$REPORT"
