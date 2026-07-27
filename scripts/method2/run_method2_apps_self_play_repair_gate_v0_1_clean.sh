#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT_DIR"

PYTHON=${PYTHON:-/data2/acm-group-3/miniconda3/envs/rubric/bin/python}
MODEL=${MODEL:-models/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28}
GPU=${GPU:-2}

ADAPTER=${ADAPTER:-outputs/method2_apps_self_play_critic_repair_sft_lora_v0_1_clean}
INPUT=${INPUT:-data/sft/method2_apps_self_play_critic_repair_v0_1_clean.jsonl}
VALIDATION_INPUT=${VALIDATION_INPUT:-data/self_play/method2_apps_self_play_v0_1_clean_validation_input.jsonl}
VALIDATION_INPUT_SUMMARY=${VALIDATION_INPUT_SUMMARY:-data/self_play/method2_apps_self_play_v0_1_clean_validation_input_summary.json}
GENERATIONS=${GENERATIONS:-data/self_play/method2_apps_self_play_v0_1_clean_validation_generations.jsonl}
EXTRACTED=${EXTRACTED:-data/self_play/method2_apps_self_play_v0_1_clean_validation_extracted.jsonl}
EXTRACT_SUMMARY=${EXTRACT_SUMMARY:-data/self_play/method2_apps_self_play_v0_1_clean_validation_extract_summary.json}
LABELED=${LABELED:-data/self_play/method2_apps_self_play_v0_1_clean_validation_labeled.jsonl}
SUMMARY=${SUMMARY:-data/self_play/method2_apps_self_play_v0_1_clean_validation_repair_gate_summary.json}

export PATH="$(dirname "$PYTHON"):$PATH"
export CUDA_VISIBLE_DEVICES="$GPU"
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/tmp/rubric-cache}
export HF_HOME=${HF_HOME:-/tmp/rubric-cache/huggingface}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-/tmp/rubric-cache/huggingface}
export TMPDIR=${TMPDIR:-/tmp/rubric-tmp}

"$PYTHON" src/data-prep/select_prompts_by_metadata.py \
  --input "$INPUT" \
  --output "$VALIDATION_INPUT" \
  --summary-output "$VALIDATION_INPUT_SUMMARY" \
  --split validation \
  --min-rows "${MIN_VALIDATION_ROWS:-38}"

GEN_ARGS=(
  "$PYTHON" -m src.generation.vllm_lora_generate
  --model "$MODEL"
  --adapter "$ADAPTER"
  --input "$VALIDATION_INPUT"
  --output "$GENERATIONS"
  --temperature "${TEMPERATURE:-0.0}"
  --top-p "${TOP_P:-1.0}"
  --repetition-penalty "${REPETITION_PENALTY:-1.0}"
  --max-tokens "${MAX_TOKENS:-768}"
  --max-model-len "${MAX_MODEL_LEN:-8192}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.25}"
  --prompt-batch-size "${PROMPT_BATCH_SIZE:-16}"
  --max-lora-rank "${MAX_LORA_RANK:-16}"
)

for stop_var in STOP_SEQUENCE STOP_SEQUENCE_2 STOP_SEQUENCE_3 STOP_SEQUENCE_4 STOP_SEQUENCE_5; do
  stop_value="${!stop_var:-}"
  if [[ -n "$stop_value" ]]; then
    GEN_ARGS+=(--stop "$stop_value")
  fi
done

if [[ -n "${QUANTIZATION:-}" ]]; then
  GEN_ARGS+=(--quantization "$QUANTIZATION")
fi
if [[ -n "${LOAD_FORMAT:-}" ]]; then
  GEN_ARGS+=(--load-format "$LOAD_FORMAT")
fi

"${GEN_ARGS[@]}"

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
  --output "$SUMMARY" \
  --min-pass-rate "${MIN_PASS_RATE:-0.20}" \
  --max-syntax-rate "${MAX_SYNTAX_RATE:-0.30}"

echo "Method2 APPS repair gate complete:"
echo "  validation_input=$VALIDATION_INPUT"
echo "  generations=$GENERATIONS"
echo "  extracted=$EXTRACTED"
echo "  extract_summary=$EXTRACT_SUMMARY"
echo "  labeled=$LABELED"
echo "  summary=$SUMMARY"
