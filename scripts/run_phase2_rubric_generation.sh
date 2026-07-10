#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

PYTHON=${PYTHON:-/data2/acm-group-3/miniconda3/envs/rubric/bin/python}
CONDA_BIN=${CONDA_BIN:-/data2/acm-group-3/miniconda3/envs/rubric/bin}
MODEL=${MODEL:-models/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28}
GPU=${GPU:-2}
DETERMINISTIC_ONLY=${DETERMINISTIC_ONLY:-0}
SKIP_EXISTING=${SKIP_EXISTING:-0}
DRY_RUN=${DRY_RUN:-0}

MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.30}
TEMPERATURE=${TEMPERATURE:-0.2}
TOP_P=${TOP_P:-0.95}
MAX_TOKENS=${MAX_TOKENS:-4096}

export PATH="$CONDA_BIN:$PATH"
export CUDA_VISIBLE_DEVICES="$GPU"
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/tmp/rubric-cache}
export HF_HOME=${HF_HOME:-/tmp/rubric-cache/huggingface}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-/tmp/rubric-cache/huggingface}
export TMPDIR=${TMPDIR:-/tmp/rubric-tmp}

PHASE_DIR="data/analysis/phase1"
RUBRIC_DIR="data/rubrics/phase2"
TRAIN_BASE="mbpp_hidden_train_qwen25_k3"

TAXONOMY="${PHASE_DIR}/${TRAIN_BASE}_taxonomy_refined_for_rubric.yaml"
SOURCE_AUDIT="${PHASE_DIR}/${TRAIN_BASE}_taxonomy_refined_for_rubric_audit.json"
OUTPUT="${RUBRIC_DIR}/mbpp_hidden_llm_rubric_from_refined_taxonomy.json"
AUDIT_OUTPUT="${RUBRIC_DIR}/mbpp_hidden_llm_rubric_from_refined_taxonomy_audit.json"
RAW_OUTPUT="${RUBRIC_DIR}/mbpp_hidden_llm_rubric_from_refined_taxonomy_raw_response.txt"

mkdir -p "$RUBRIC_DIR" "$TMPDIR"

run_cmd() {
  echo "+ $*"
  if [[ "$DRY_RUN" != "1" ]]; then
    "$@"
  fi
}

if [[ "$SKIP_EXISTING" == "1" && -s "$OUTPUT" && -s "$AUDIT_OUTPUT" ]]; then
  echo "skip: rubric generation outputs already exist"
  exit 0
fi

if [[ "$DETERMINISTIC_ONLY" == "1" ]]; then
  run_cmd "$PYTHON" src/rubric/generate_llm_rubric_from_taxonomy.py \
    --taxonomy "$TAXONOMY" \
    --source-audit "$SOURCE_AUDIT" \
    --output "$OUTPUT" \
    --audit-output "$AUDIT_OUTPUT" \
    --deterministic-only
else
  run_cmd "$PYTHON" src/rubric/generate_llm_rubric_from_taxonomy.py \
    --taxonomy "$TAXONOMY" \
    --source-audit "$SOURCE_AUDIT" \
    --output "$OUTPUT" \
    --audit-output "$AUDIT_OUTPUT" \
    --raw-llm-output "$RAW_OUTPUT" \
    --model "$MODEL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --temperature "$TEMPERATURE" \
    --top-p "$TOP_P" \
    --max-tokens "$MAX_TOKENS"
fi

echo "Phase 2 rubric generation complete."
echo "Rubric: $OUTPUT"
echo "Audit: $AUDIT_OUTPUT"
