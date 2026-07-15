#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT_DIR"

PYTHON=${PYTHON:-/data2/acm-group-3/miniconda3/envs/rubric/bin/python}
MODEL=${MODEL:-models/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28}
GPU=${GPU:-2}
PAIRS=${PAIRS:-data/preferences/apps_simple_method1_self_repair_dpo_v2.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/apps_simple_method1_dpo_v2_canary_lora}
DEV_PROMPTS=${DEV_PROMPTS:-data/processed/apps_simple_method1_dpo_dev_v2_prompts.jsonl}
BASE_RESPONSES=${BASE_RESPONSES:-data/responses/apps_simple_method1_dpo_dev_v2_base_greedy.jsonl}
PAIRED_BASE_LABELED=${PAIRED_BASE_LABELED:-data/responses/apps_simple_method1_dpo_dev_v2_base_greedy_paired_labeled.jsonl}
DEV_RESPONSES=${DEV_RESPONSES:-data/responses/apps_simple_method1_dpo_v2_canary_dev.jsonl}
DEV_LABELED=${DEV_LABELED:-data/responses/apps_simple_method1_dpo_v2_canary_dev_labeled.jsonl}
DEV_SUMMARY=${DEV_SUMMARY:-data/eval/apps_simple_method1_dpo_v2_canary_dev_summary.json}
DEV_REPORT=${DEV_REPORT:-docs/method1/10-apps-dpo-v2-canary-results.md}
PAIRED_VERIFY_MANIFEST=${PAIRED_VERIFY_MANIFEST:-data/eval/apps_simple_method1_dpo_v2_canary_paired_verification_manifest.json}

export PATH="$(dirname "$PYTHON"):$PATH"
export CUDA_VISIBLE_DEVICES="$GPU"
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/data2/acm-group-3/cache}
export HF_HOME=${HF_HOME:-/data2/acm-group-3/cache/huggingface}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-/data2/acm-group-3/cache/huggingface}
export TMPDIR=${TMPDIR:-/data2/acm-group-3/cache/tmp}

TRAIN_ARGS=(
  "$PYTHON" -m src.training.train_dpo_lora
  --model "$MODEL" \
  --data "$PAIRS" \
  --output-dir "$OUTPUT_DIR" \
  --epochs "${EPOCHS:-1}" \
  --max-pairs "${MAX_PAIRS:-400}" \
  --max-length "${MAX_LENGTH:-3072}" \
  --prompt-format raw \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS:-8}" \
  --learning-rate "${LEARNING_RATE:-1e-6}" \
  --beta "${BETA:-0.2}" \
  --ld-alpha "${LD_ALPHA:-1.0}" \
  --warmup-ratio 0.03 \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --logging-steps 5 \
  --save-steps "${SAVE_STEPS:-10}" \
  --save-total-limit 4 \
  --seed 42
)

IFS=',' read -r -a LOSS_TYPE_VALUES <<< "${LOSS_TYPES:-sigmoid}"
for value in "${LOSS_TYPE_VALUES[@]}"; do
  TRAIN_ARGS+=(--loss-type "$value")
done
if [[ -n "${LOSS_WEIGHTS:-}" ]]; then
  IFS=',' read -r -a LOSS_WEIGHT_VALUES <<< "$LOSS_WEIGHTS"
  for value in "${LOSS_WEIGHT_VALUES[@]}"; do
    TRAIN_ARGS+=(--loss-weight "$value")
  done
fi

"${TRAIN_ARGS[@]}"

"$PYTHON" -m src.generation.vllm_lora_generate \
  --model "$MODEL" \
  --adapter "$OUTPUT_DIR" \
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

"$PYTHON" - "$DEV_SUMMARY" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text())
if not summary.get("canary_passed"):
    raise SystemExit("DPO-v2 canary gate failed; do not run final held-out")
print("DPO-v2 canary gate passed")
PY
