#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

PYTHON=${PYTHON:-/data2/acm-group-3/miniconda3/envs/rubric/bin/python}
CONDA_BIN=${CONDA_BIN:-/data2/acm-group-3/miniconda3/envs/rubric/bin}
MODEL=${MODEL:-models/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28}
GPU=${GPU:-2}
PROFILE=${PROFILE:-strict_fewshot}
TAG=${TAG:-hitl_v3_test}
RUBRIC_PATH=${RUBRIC_PATH:-data/rubrics/phase2/mbpp_hidden_llm_rubric_hitl_v3.json}
GUIDANCE_PATH=${GUIDANCE_PATH:-data/rubrics/phase2/judge_guidance_score_collapse_v3.json}
EXAMPLES_PATH=${EXAMPLES_PATH:-data/rubrics/phase2/judge_fewshot_examples_validation_v1.json}
SPLITS=${SPLITS:-test}
LIMIT=${LIMIT:-}
OFFSET=${OFFSET:-0}
DETERMINISTIC_ONLY=${DETERMINISTIC_ONLY:-0}
REUSE_RAW_OUTPUT=${REUSE_RAW_OUTPUT:-}
REQUIRE_TEST_PROBES=${REQUIRE_TEST_PROBES:-0}
EXECUTION_GATE=${EXECUTION_GATE:-none}
DRY_RUN=${DRY_RUN:-0}

MAX_MODEL_LEN=${MAX_MODEL_LEN:-12288}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.25}
MAX_TOKENS=${MAX_TOKENS:-1200}
BATCH_SIZE=${BATCH_SIZE:-8}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-8}
MAX_FEW_SHOT_EXAMPLES=${MAX_FEW_SHOT_EXAMPLES:-7}

export PATH="$CONDA_BIN:$PATH"
export CUDA_VISIBLE_DEVICES="$GPU"
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/tmp/rubric-cache}
export HF_HOME=${HF_HOME:-/tmp/rubric-cache/huggingface}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-/tmp/rubric-cache/huggingface}
export TMPDIR=${TMPDIR:-/tmp/rubric-tmp}

RUBRIC_DIR=data/rubrics/phase2
ARGS=(
  "$PYTHON" -m src.rubric.evaluate_llm_rubric_judge
  --labeled data/responses/phase1_mbpp_hidden_qwen25_k3_labeled.jsonl
  --rubric "$RUBRIC_PATH"
  --judge-guidance "$GUIDANCE_PATH"
  --few-shot-examples "$EXAMPLES_PATH"
  --prompt-profile "$PROFILE"
  --strict-prediction
  --use-chat-template
  --max-few-shot-examples "$MAX_FEW_SHOT_EXAMPLES"
  --scores-output "$RUBRIC_DIR/mbpp_hidden_llm_judge_scores_${TAG}.jsonl"
  --metrics-output "$RUBRIC_DIR/mbpp_hidden_llm_judge_metrics_${TAG}.json"
  --audit-output "$RUBRIC_DIR/mbpp_hidden_llm_judge_audit_${TAG}.json"
  --raw-output "$RUBRIC_DIR/mbpp_hidden_llm_judge_raw_${TAG}.jsonl"
  --splits "$SPLITS"
  --offset "$OFFSET"
)

if [[ -n "$LIMIT" ]]; then
  ARGS+=(--limit "$LIMIT")
fi

if [[ "$REQUIRE_TEST_PROBES" == "1" ]]; then
  ARGS+=(--require-test-probes)
fi

if [[ "$EXECUTION_GATE" != "none" ]]; then
  ARGS+=(--execution-gate "$EXECUTION_GATE")
fi

if [[ -n "$REUSE_RAW_OUTPUT" ]]; then
  ARGS+=(--reuse-raw-output "$REUSE_RAW_OUTPUT" --model "$MODEL")
elif [[ "$DETERMINISTIC_ONLY" == "1" ]]; then
  ARGS+=(--deterministic-only)
else
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
fi

echo "+ ${ARGS[*]}"
if [[ "$DRY_RUN" != "1" ]]; then
  "${ARGS[@]}"
fi

echo "HITL judge run complete: tag=$TAG profile=$PROFILE"
