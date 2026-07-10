#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

PYTHON=${PYTHON:-/data2/acm-group-3/miniconda3/envs/rubric/bin/python}
CONDA_BIN=${CONDA_BIN:-/data2/acm-group-3/miniconda3/envs/rubric/bin}
MODEL=${MODEL:-models/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28}
GPU=${GPU:-2}
DETERMINISTIC_ONLY=${DETERMINISTIC_ONLY:-0}
REUSE_RAW_OUTPUT=${REUSE_RAW_OUTPUT:-}
CALIBRATED_PREDICTION=${CALIBRATED_PREDICTION:-0}
CALIBRATED_PASS_THRESHOLD=${CALIBRATED_PASS_THRESHOLD:-}
SKIP_EXISTING=${SKIP_EXISTING:-0}
DRY_RUN=${DRY_RUN:-0}

SPLITS=${SPLITS:-validation,test}
LIMIT=${LIMIT:-}
OFFSET=${OFFSET:-0}
TAG=${TAG:-validation_test}

MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.25}
TEMPERATURE=${TEMPERATURE:-0.0}
TOP_P=${TOP_P:-1.0}
MAX_TOKENS=${MAX_TOKENS:-768}
BATCH_SIZE=${BATCH_SIZE:-16}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-16}
MAX_RESPONSE_CHARS=${MAX_RESPONSE_CHARS:-1800}
MAX_CODE_CHARS=${MAX_CODE_CHARS:-2600}

export PATH="$CONDA_BIN:$PATH"
export CUDA_VISIBLE_DEVICES="$GPU"
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/tmp/rubric-cache}
export HF_HOME=${HF_HOME:-/tmp/rubric-cache/huggingface}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-/tmp/rubric-cache/huggingface}
export TMPDIR=${TMPDIR:-/tmp/rubric-tmp}

RUBRIC_DIR="data/rubrics/phase2"
LABELED="data/responses/phase1_mbpp_hidden_qwen25_k3_labeled.jsonl"
RUBRIC="${RUBRIC_DIR}/mbpp_hidden_llm_rubric_from_refined_taxonomy.json"
SCORES_OUTPUT="${RUBRIC_DIR}/mbpp_hidden_llm_judge_scores_${TAG}.jsonl"
METRICS_OUTPUT="${RUBRIC_DIR}/mbpp_hidden_llm_judge_metrics_${TAG}.json"
AUDIT_OUTPUT="${RUBRIC_DIR}/mbpp_hidden_llm_judge_audit_${TAG}.json"
RAW_OUTPUT="${RUBRIC_DIR}/mbpp_hidden_llm_judge_raw_${TAG}.jsonl"

mkdir -p "$RUBRIC_DIR" "$TMPDIR"

run_cmd() {
  echo "+ $*"
  if [[ "$DRY_RUN" != "1" ]]; then
    "$@"
  fi
}

if [[ "$SKIP_EXISTING" == "1" && -s "$SCORES_OUTPUT" && -s "$METRICS_OUTPUT" && -s "$AUDIT_OUTPUT" ]]; then
  echo "skip: judge outputs already exist for tag=$TAG"
  exit 0
fi

COMMON_ARGS=(
  "$PYTHON" src/rubric/evaluate_llm_rubric_judge.py
  --labeled "$LABELED"
  --rubric "$RUBRIC"
  --scores-output "$SCORES_OUTPUT"
  --metrics-output "$METRICS_OUTPUT"
  --audit-output "$AUDIT_OUTPUT"
  --raw-output "$RAW_OUTPUT"
  --splits "$SPLITS"
  --offset "$OFFSET"
  --max-response-chars "$MAX_RESPONSE_CHARS"
  --max-code-chars "$MAX_CODE_CHARS"
)

if [[ -n "$LIMIT" ]]; then
  COMMON_ARGS+=(--limit "$LIMIT")
fi

if [[ "$CALIBRATED_PREDICTION" == "1" ]]; then
  COMMON_ARGS+=(--calibrated-prediction)
fi

if [[ -n "$CALIBRATED_PASS_THRESHOLD" ]]; then
  COMMON_ARGS+=(--calibrated-pass-threshold "$CALIBRATED_PASS_THRESHOLD")
fi

if [[ -n "$REUSE_RAW_OUTPUT" ]]; then
  run_cmd "${COMMON_ARGS[@]}" --reuse-raw-output "$REUSE_RAW_OUTPUT" --model "$MODEL"
elif [[ "$DETERMINISTIC_ONLY" == "1" ]]; then
  run_cmd "${COMMON_ARGS[@]}" --deterministic-only
else
  run_cmd "${COMMON_ARGS[@]}" \
    --model "$MODEL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --temperature "$TEMPERATURE" \
    --top-p "$TOP_P" \
    --max-tokens "$MAX_TOKENS" \
    --batch-size "$BATCH_SIZE" \
    --max-num-seqs "$MAX_NUM_SEQS"
fi

echo "Phase 2 LLM judge evaluation complete."
echo "Scores: $SCORES_OUTPUT"
echo "Metrics: $METRICS_OUTPUT"
echo "Audit: $AUDIT_OUTPUT"
