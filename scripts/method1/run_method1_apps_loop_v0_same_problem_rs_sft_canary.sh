#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT_DIR"

PYTHON=${PYTHON:-/data2/acm-group-3/miniconda3/envs/rubric/bin/python}
MODEL=${MODEL:-models/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28}
GPU=${GPU:-2}

DATA=${DATA:-data/sft/apps_simple_method1_loop_v0_same_problem_rs_sft.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/apps_simple_method1_loop_v0_same_problem_rs_sft_lora}
DEV_PROMPTS=${DEV_PROMPTS:-data/processed/apps_simple_method1_dpo_dev_v2_prompts.jsonl}
BASE_RESPONSES=${BASE_RESPONSES:-data/responses/apps_simple_method1_dpo_dev_v2_base_greedy.jsonl}
PAIRED_BASE_LABELED=${PAIRED_BASE_LABELED:-data/responses/apps_simple_method1_loop_v0_same_problem_rs_sft_base_greedy_paired_labeled.jsonl}
DEV_RESPONSES=${DEV_RESPONSES:-data/responses/apps_simple_method1_loop_v0_same_problem_rs_sft_dev.jsonl}
DEV_LABELED=${DEV_LABELED:-data/responses/apps_simple_method1_loop_v0_same_problem_rs_sft_dev_labeled.jsonl}
DEV_SUMMARY=${DEV_SUMMARY:-data/eval/apps_simple_method1_loop_v0_same_problem_rs_sft_dev_summary.json}
DEV_REPORT=${DEV_REPORT:-docs/method1/46-apps-loop-v0-same-problem-rs-sft-canary-results.md}
PAIRED_VERIFY_MANIFEST=${PAIRED_VERIFY_MANIFEST:-data/eval/apps_simple_method1_loop_v0_same_problem_rs_sft_paired_verification_manifest.json}

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
  --max-length "${MAX_LENGTH:-3072}" \
  --prompt-format raw \
  --epochs "${EPOCHS:-1}" \
  --per-device-train-batch-size "${PER_DEVICE_TRAIN_BATCH_SIZE:-1}" \
  --per-device-eval-batch-size "${PER_DEVICE_EVAL_BATCH_SIZE:-1}" \
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS:-8}" \
  --learning-rate "${LEARNING_RATE:-2e-6}" \
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
  --training-preferences "$DATA" \
  --output "$DEV_SUMMARY" \
  --report "$DEV_REPORT" \
  --require-paired-verification

"$PYTHON" - "$DEV_SUMMARY" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text())
if not summary.get("canary_passed"):
    raise SystemExit("RS-SFT canary gate failed; inspect summary before continuing")
print("RS-SFT canary gate passed")
PY
