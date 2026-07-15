#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT_DIR"

PYTHON=${PYTHON:-/data2/acm-group-3/miniconda3/envs/rubric/bin/python}
CONDA_BIN=${CONDA_BIN:-/data2/acm-group-3/miniconda3/envs/rubric/bin}
MODEL=${MODEL:-models/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28}
GPU=${GPU:-2}

LABELED=${LABELED:-data/evaluator/apps_simple_method1_evaluator_training_rows_v1.jsonl}
RUBRIC_PATH=${RUBRIC_PATH:-data/rubrics/apps_simple_phase2/apps_train_simple_llm_rubric_from_refined_taxonomy.json}
GUIDANCE_PATH=${GUIDANCE_PATH:-configs/rubric_judge/apps_simple_no_gate_guidance.json}
OUTPUT_DIR=${OUTPUT_DIR:-data/rubrics/apps_simple_method1}
TAG=${TAG:-apps_simple_no_gate_baseline_eval_v1}
SPLITS=${SPLITS:-validation,test}
LIMIT=${LIMIT:-}
OFFSET=${OFFSET:-0}

MAX_MODEL_LEN=${MAX_MODEL_LEN:-16384}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.35}
MAX_TOKENS=${MAX_TOKENS:-1400}
BATCH_SIZE=${BATCH_SIZE:-8}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-8}

export PATH="$CONDA_BIN:$PATH"
export CUDA_VISIBLE_DEVICES="$GPU"
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/tmp/rubric-cache}
export HF_HOME=${HF_HOME:-/tmp/rubric-cache/huggingface}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-/tmp/rubric-cache/huggingface}
export TMPDIR=${TMPDIR:-/tmp/rubric-tmp}

ARGS=(
  "$PYTHON" -m src.rubric.evaluate_llm_rubric_judge
  --labeled "$LABELED"
  --rubric "$RUBRIC_PATH"
  --judge-guidance "$GUIDANCE_PATH"
  --prompt-profile strict
  --strict-prediction
  --use-chat-template
  --scores-output "$OUTPUT_DIR/${TAG}_scores.jsonl"
  --metrics-output "$OUTPUT_DIR/${TAG}_metrics.json"
  --audit-output "$OUTPUT_DIR/${TAG}_audit.json"
  --raw-output "$OUTPUT_DIR/${TAG}_raw.jsonl"
  --splits "$SPLITS"
  --offset "$OFFSET"
)

if [[ -n "$LIMIT" ]]; then
  ARGS+=(--limit "$LIMIT")
fi

ARGS+=(
  --model "$MODEL"
  --max-model-len "$MAX_MODEL_LEN"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --temperature 0.0
  --top-p 1.0
  --max-tokens "$MAX_TOKENS"
  --batch-size "$BATCH_SIZE"
  --max-num-seqs "$MAX_NUM_SEQS"
)

echo "+ ${ARGS[*]}"
"${ARGS[@]}"

echo "APPS simple no-gate baseline complete: tag=$TAG"
