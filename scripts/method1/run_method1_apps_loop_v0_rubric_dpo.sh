#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT_DIR"

PYTHON=${PYTHON:-/data2/acm-group-3/miniconda3/envs/rubric/bin/python}
MODEL=${MODEL:-models/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28}
GPU=${GPU:-2}

PAIRS=${PAIRS:-data/preferences/apps_simple_method1_loop_v0_rubric_dpo_pairs.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/apps_simple_method1_loop_v0_rubric_dpo_lora}

export PATH="$(dirname "$PYTHON"):$PATH"
export CUDA_VISIBLE_DEVICES="$GPU"
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/tmp/rubric-cache}
export HF_HOME=${HF_HOME:-/tmp/rubric-cache/huggingface}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-/tmp/rubric-cache/huggingface}
export TMPDIR=${TMPDIR:-/tmp/rubric-tmp}

ARGS=(
  "$PYTHON" -m src.training.train_dpo_lora
  --model "$MODEL"
  --data "$PAIRS"
  --output-dir "$OUTPUT_DIR"
  --epochs "${EPOCHS:-1}"
  --max-length "${MAX_LENGTH:-3072}"
  --min-completion-tokens "${MIN_COMPLETION_TOKENS:-128}"
  --prompt-format raw
  --per-device-train-batch-size "${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS:-8}"
  --learning-rate "${LEARNING_RATE:-8e-7}"
  --beta "${BETA:-0.2}"
  --ld-alpha "${LD_ALPHA:-1.0}"
  --warmup-ratio "${WARMUP_RATIO:-0.03}"
  --weight-decay "${WEIGHT_DECAY:-0.0}"
  --lora-r "${LORA_R:-16}"
  --lora-alpha "${LORA_ALPHA:-32}"
  --lora-dropout "${LORA_DROPOUT:-0.05}"
  --logging-steps "${LOGGING_STEPS:-5}"
  --save-steps "${SAVE_STEPS:-50}"
  --save-total-limit "${SAVE_TOTAL_LIMIT:-3}"
  --seed "${SEED:-42}"
)

if [[ -n "${MAX_PAIRS:-}" ]]; then
  ARGS+=(--max-pairs "$MAX_PAIRS")
fi

IFS=',' read -r -a LOSS_TYPE_VALUES <<< "${LOSS_TYPES:-sigmoid}"
for value in "${LOSS_TYPE_VALUES[@]}"; do
  ARGS+=(--loss-type "$value")
done

if [[ -n "${LOSS_WEIGHTS:-}" ]]; then
  IFS=',' read -r -a LOSS_WEIGHT_VALUES <<< "$LOSS_WEIGHTS"
  for value in "${LOSS_WEIGHT_VALUES[@]}"; do
    ARGS+=(--loss-weight "$value")
  done
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  ARGS+=(--dry-run)
fi

echo "+ ${ARGS[*]}"
"${ARGS[@]}"
