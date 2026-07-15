#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

PYTHON=${PYTHON:-/data2/acm-group-3/miniconda3/envs/rubric/bin/python}
MODEL=${MODEL:-models/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28}
GPU=${GPU:-2}

DATA=${DATA:-data/sft/method2_self_play_critic_repair_v0.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/method2_self_play_critic_repair_sft_lora_v0}

export PATH="$(dirname "$PYTHON"):$PATH"
export CUDA_VISIBLE_DEVICES="$GPU"
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/tmp/rubric-cache}
export HF_HOME=${HF_HOME:-/tmp/rubric-cache/huggingface}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-/tmp/rubric-cache/huggingface}
export TMPDIR=${TMPDIR:-/tmp/rubric-tmp}

"$PYTHON" -m src.training.train_causallm_sft_lora \
  --model "$MODEL" \
  --data "$DATA" \
  --output-dir "$OUTPUT_DIR" \
  --max-length "${MAX_LENGTH:-4096}" \
  --prompt-format raw \
  --epochs "${EPOCHS:-1}" \
  --per-device-train-batch-size "${PER_DEVICE_TRAIN_BATCH_SIZE:-1}" \
  --per-device-eval-batch-size "${PER_DEVICE_EVAL_BATCH_SIZE:-1}" \
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS:-8}" \
  --learning-rate "${LEARNING_RATE:-5e-7}" \
  --warmup-ratio "${WARMUP_RATIO:-0.03}" \
  --weight-decay "${WEIGHT_DECAY:-0.0}" \
  --lora-r "${LORA_R:-16}" \
  --lora-alpha "${LORA_ALPHA:-32}" \
  --lora-dropout "${LORA_DROPOUT:-0.05}" \
  --logging-steps "${LOGGING_STEPS:-5}" \
  --eval-steps "${EVAL_STEPS:-10}" \
  --save-steps "${SAVE_STEPS:-10}" \
  --save-total-limit "${SAVE_TOTAL_LIMIT:-2}" \
  --seed "${SEED:-42}"

echo "Method2 self-play critic/repair SFT complete:"
echo "  adapter: $OUTPUT_DIR"
