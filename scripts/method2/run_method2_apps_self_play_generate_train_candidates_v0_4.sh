#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT_DIR"

PYTHON=${PYTHON:-/data2/acm-group-3/miniconda3/envs/rubric/bin/python}
MODEL=${MODEL:-models/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28}
GPU=${GPU:-1}

ADAPTER=${ADAPTER:-outputs/method2_apps_self_play_critic_repair_sft_lora_v0_3_no_end_marker}
INPUT=${INPUT:-data/sft/method2_apps_self_play_critic_repair_v0_3_no_end_marker.jsonl}
TRAIN_INPUT=${TRAIN_INPUT:-data/self_play/method2_apps_self_play_v0_4_train_input.jsonl}
TRAIN_INPUT_SUMMARY=${TRAIN_INPUT_SUMMARY:-data/self_play/method2_apps_self_play_v0_4_train_input_summary.json}
GENERATIONS=${GENERATIONS:-data/self_play/method2_apps_self_play_v0_4_train_candidates_generations.jsonl}
EXTRACTED=${EXTRACTED:-data/self_play/method2_apps_self_play_v0_4_train_candidates_extracted.jsonl}
EXTRACT_SUMMARY=${EXTRACT_SUMMARY:-data/self_play/method2_apps_self_play_v0_4_train_candidates_extract_summary.json}
LABELED=${LABELED:-data/self_play/method2_apps_self_play_v0_4_train_candidates_labeled.jsonl}
CANDIDATE_SUMMARY=${CANDIDATE_SUMMARY:-data/self_play/method2_apps_self_play_v0_4_train_candidates_summary.json}

export PATH="$(dirname "$PYTHON"):$PATH"
export CUDA_VISIBLE_DEVICES="$GPU"
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/tmp/rubric-cache}
export HF_HOME=${HF_HOME:-/tmp/rubric-cache/huggingface}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-/tmp/rubric-cache/huggingface}
export TMPDIR=${TMPDIR:-/tmp/rubric-tmp}

SELECT_ARGS=(
  "$PYTHON" src/data-prep/select_prompts_by_metadata.py
  --input "$INPUT"
  --output "$TRAIN_INPUT"
  --summary-output "$TRAIN_INPUT_SUMMARY"
  --split train
  --min-rows "${MIN_TRAIN_ROWS:-300}"
)
if [[ -n "${TRAIN_LIMIT:-}" ]]; then
  SELECT_ARGS+=(--limit "$TRAIN_LIMIT")
fi
"${SELECT_ARGS[@]}"

"$PYTHON" -m src.generation.vllm_lora_generate \
  --model "$MODEL" \
  --adapter "$ADAPTER" \
  --input "$TRAIN_INPUT" \
  --output "$GENERATIONS" \
  --n "${K:-5}" \
  --temperature "${TEMPERATURE:-0.7}" \
  --top-p "${TOP_P:-0.95}" \
  --repetition-penalty "${REPETITION_PENALTY:-1.0}" \
  --max-tokens "${MAX_TOKENS:-768}" \
  --max-model-len "${MAX_MODEL_LEN:-8192}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.25}" \
  --prompt-batch-size "${PROMPT_BATCH_SIZE:-8}" \
  --max-lora-rank "${MAX_LORA_RANK:-16}" \
  --stop $'\nPublic task prompt:' \
  --stop $'\nPrevious failed code:'

"$PYTHON" src/self_play/extract_method2_revised_code.py \
  --input "$GENERATIONS" \
  --output "$EXTRACTED" \
  --summary-output "$EXTRACT_SUMMARY"

"$PYTHON" src/verification/verify_mbpp_smoke.py \
  --input "$EXTRACTED" \
  --output "$LABELED" \
  --timeout "${VERIFY_TIMEOUT:-30}" \
  --workers "${VERIFY_WORKERS:-4}" \
  --process-start-method spawn

"$PYTHON" src/self_play/summarize_method2_repair_gate.py \
  --labeled "$LABELED" \
  --output "$CANDIDATE_SUMMARY" \
  --min-pass-rate 0.0 \
  --max-syntax-rate 1.0

echo "Method2 APPS v0.4 train candidate generation complete:"
echo "  train_input=$TRAIN_INPUT"
echo "  generations=$GENERATIONS"
echo "  extracted=$EXTRACTED"
echo "  labeled=$LABELED"
echo "  summary=$CANDIDATE_SUMMARY"
