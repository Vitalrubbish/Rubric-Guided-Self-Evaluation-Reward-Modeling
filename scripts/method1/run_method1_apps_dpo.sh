#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT_DIR"

PYTHON=${PYTHON:-/data2/acm-group-3/miniconda3/envs/rubric/bin/python}
MODEL=${MODEL:-models/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28}
DATA=${DATA:-data/preferences/apps_simple_method1_train_canonical_dpo_v1.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/apps_simple_method1_dpo_lora_v1}

export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/data2/acm-group-3/cache}
export HF_HOME=${HF_HOME:-/data2/acm-group-3/cache/huggingface}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-/data2/acm-group-3/cache/huggingface}
export TMPDIR=${TMPDIR:-/data2/acm-group-3/cache/tmp}
mkdir -p "$XDG_CACHE_HOME" "$HF_HOME" "$TMPDIR" logs

if [[ "${SKIP_PREPARE:-0}" != "1" ]]; then
  "$PYTHON" -m src.training.build_apps_dpo_preferences
fi

if [[ -s "$OUTPUT_DIR/adapter_model.safetensors" && "${ALLOW_OVERWRITE:-0}" != "1" ]]; then
  echo "Refusing to overwrite completed adapter: $OUTPUT_DIR/adapter_model.safetensors" >&2
  exit 3
fi

ARGS=(
  "$PYTHON" -m src.training.train_dpo_lora
  --model "$MODEL"
  --data "$DATA"
  --output-dir "$OUTPUT_DIR"
  --epochs "${EPOCHS:-1}"
  --max-length "${MAX_LENGTH:-4096}"
  --min-completion-tokens "${MIN_COMPLETION_TOKENS:-128}"
  --prompt-format "${PROMPT_FORMAT:-raw}"
  --per-device-train-batch-size "${BATCH_SIZE:-1}"
  --gradient-accumulation-steps "${GRAD_ACCUM:-8}"
  --learning-rate "${LEARNING_RATE:-5e-6}"
  --beta "${BETA:-0.1}"
  --warmup-ratio "${WARMUP_RATIO:-0.03}"
  --logging-steps "${LOGGING_STEPS:-5}"
  --save-steps "${SAVE_STEPS:-100}"
  --save-total-limit "${SAVE_TOTAL_LIMIT:-2}"
  --lora-r "${LORA_R:-16}"
  --lora-alpha "${LORA_ALPHA:-32}"
  --seed "${SEED:-42}"
)
if [[ -n "${MAX_PAIRS:-}" ]]; then
  ARGS+=(--max-pairs "$MAX_PAIRS")
fi
if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
  ARGS+=(--resume-from-checkpoint "$RESUME_FROM_CHECKPOINT")
fi

exec "${ARGS[@]}"
