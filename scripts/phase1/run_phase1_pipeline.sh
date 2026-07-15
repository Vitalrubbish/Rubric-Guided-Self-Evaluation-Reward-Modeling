#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT_DIR"

PYTHON=${PYTHON:-/data2/acm-group-3/miniconda3/envs/rubric/bin/python}
CONDA_BIN=${CONDA_BIN:-/data2/acm-group-3/miniconda3/envs/rubric/bin}
MODEL=${MODEL:-models/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28}
GPU=${GPU:-2}
K=${K:-3}
LIMIT=${LIMIT:-964}
SKIP_EXISTING=${SKIP_EXISTING:-0}
DRY_RUN=${DRY_RUN:-0}
RUN_REPORTS=${RUN_REPORTS:-1}
RUN_STATIC_BASELINE=${RUN_STATIC_BASELINE:-1}

MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
GEN_GPU_MEMORY_UTILIZATION=${GEN_GPU_MEMORY_UTILIZATION:-0.30}
TAXONOMY_GPU_MEMORY_UTILIZATION=${TAXONOMY_GPU_MEMORY_UTILIZATION:-0.35}
REFINE_GPU_MEMORY_UTILIZATION=${REFINE_GPU_MEMORY_UTILIZATION:-0.30}
PROMPT_BATCH_SIZE=${PROMPT_BATCH_SIZE:-128}

export PATH="$CONDA_BIN:$PATH"
export CUDA_VISIBLE_DEVICES="$GPU"
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/tmp/rubric-cache}
export HF_HOME=${HF_HOME:-/tmp/rubric-cache/huggingface}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-/tmp/rubric-cache/huggingface}
export TMPDIR=${TMPDIR:-/tmp/rubric-tmp}

BASE="mbpp_hidden_qwen25_k${K}"
TRAIN_BASE="mbpp_hidden_train_qwen25_k${K}"
PHASE_DIR="data/analysis/phase1"
RUBRIC_DIR="data/rubrics/phase1"

PROMPTS="data/processed/coding_prompts.jsonl"
RESPONSES="data/responses/${BASE}.jsonl"
LABELED="data/responses/phase1_${BASE}_labeled.jsonl"
TASK_METRICS="${PHASE_DIR}/${BASE}_task_metrics.json"
FAILURES="${PHASE_DIR}/${BASE}_failures_safe.jsonl"
FAILURE_SUMMARY="${PHASE_DIR}/${BASE}_summary.json"
INITIAL_TAXONOMY="${PHASE_DIR}/${BASE}_taxonomy_initial_safe.yaml"

TRAIN_FAILURES="${PHASE_DIR}/${TRAIN_BASE}_failures_safe.jsonl"
LLM_SUMMARIES="${PHASE_DIR}/${TRAIN_BASE}_failures_with_safe_llm_summaries.jsonl"
DISCOVERED_CLUSTERS="${PHASE_DIR}/${TRAIN_BASE}_discovered_clusters_safe.jsonl"
DISCOVERED_TAXONOMY="${PHASE_DIR}/${TRAIN_BASE}_discovered_taxonomy_safe.yaml"
DISCOVERED_SUMMARY="${PHASE_DIR}/${TRAIN_BASE}_discovered_taxonomy_summary_safe.json"

CONSOLIDATED_TAXONOMY="${PHASE_DIR}/${TRAIN_BASE}_taxonomy_consolidated.yaml"
CONSOLIDATED_AUDIT="${PHASE_DIR}/${TRAIN_BASE}_taxonomy_consolidated_audit.json"
CONSOLIDATED_MAPPING="${PHASE_DIR}/${TRAIN_BASE}_taxonomy_consolidated_cluster_mapping.jsonl"
CONSOLIDATED_ASSIGNMENTS="${PHASE_DIR}/${TRAIN_BASE}_taxonomy_consolidated_response_assignments.jsonl"
CONSOLIDATION_RAW="${PHASE_DIR}/${TRAIN_BASE}_taxonomy_consolidation_raw_response.txt"

REFINED_TAXONOMY="${PHASE_DIR}/${TRAIN_BASE}_taxonomy_refined_for_rubric.yaml"
REFINED_AUDIT="${PHASE_DIR}/${TRAIN_BASE}_taxonomy_refined_for_rubric_audit.json"
REFINED_ASSIGNMENTS="${PHASE_DIR}/${TRAIN_BASE}_taxonomy_refined_response_assignments.jsonl"
REFINEMENT_RAW="${PHASE_DIR}/${TRAIN_BASE}_taxonomy_refinement_raw_response.txt"

AUTO_RUBRIC="${RUBRIC_DIR}/mbpp_hidden_auto_rubric_refined.json"
GENERIC_RUBRIC="${RUBRIC_DIR}/mbpp_hidden_generic_rubric.json"
RANDOM_RUBRIC="${RUBRIC_DIR}/mbpp_hidden_random_rubric_ablation.json"

mkdir -p data/processed data/responses "$PHASE_DIR" "$RUBRIC_DIR" "$TMPDIR"

run_cmd() {
  echo "+ $*"
  if [[ "$DRY_RUN" != "1" ]]; then
    "$@"
  fi
}

run_if_needed() {
  local output="$1"
  shift
  if [[ "$SKIP_EXISTING" == "1" && -s "$output" ]]; then
    echo "skip: $output exists"
    return
  fi
  run_cmd "$@"
}

filter_train_failures() {
  if [[ "$SKIP_EXISTING" == "1" && -s "$TRAIN_FAILURES" ]]; then
    echo "skip: $TRAIN_FAILURES exists"
    return
  fi
  echo "+ filter train failures: $FAILURES -> $TRAIN_FAILURES"
  if [[ "$DRY_RUN" == "1" ]]; then
    return
  fi
  "$PYTHON" - "$FAILURES" "$TRAIN_FAILURES" <<'PY'
import json
import sys
from pathlib import Path

source = Path(sys.argv[1])
target = Path(sys.argv[2])
target.parent.mkdir(parents=True, exist_ok=True)
count = 0
with source.open("r", encoding="utf-8") as f, target.open("w", encoding="utf-8") as out:
    for line in f:
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("split") == "train":
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
print(f"wrote {count} train failures to {target}")
PY
}

run_if_needed "$PROMPTS" \
  "$PYTHON" src/data-prep/prepare_coding_prompts.py \
  --raw-dir data/raw \
  --output "$PROMPTS" \
  --dataset mbpp \
  --mbpp-prompt-mode hidden_tests

run_if_needed "$RESPONSES" \
  "$PYTHON" src/generation/vllm_smoke_generate.py \
  --model "$MODEL" \
  --input "$PROMPTS" \
  --output "$RESPONSES" \
  --limit "$LIMIT" \
  --k "$K" \
  --temperature 0.7 \
  --top-p 0.9 \
  --max-tokens 512 \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GEN_GPU_MEMORY_UTILIZATION" \
  --prompt-batch-size "$PROMPT_BATCH_SIZE"

run_if_needed "$LABELED" \
  "$PYTHON" src/verification/verify_mbpp_smoke.py \
  --input "$RESPONSES" \
  --output "$LABELED" \
  --timeout 5

if [[ "$RUN_REPORTS" == "1" ]]; then
  run_if_needed "$TASK_METRICS" \
    "$PYTHON" src/analysis-reporting/compute_coding_task_metrics.py \
    --input "$LABELED" \
    --output "$TASK_METRICS" \
    --k "$K"
else
  echo "skip: response/task metrics disabled by RUN_REPORTS=$RUN_REPORTS"
fi

run_if_needed "$FAILURES" \
  "$PYTHON" src/error-analysis/build_failure_artifacts.py \
  --input "$LABELED" \
  --failure-output "$FAILURES" \
  --summary-output "$FAILURE_SUMMARY" \
  --taxonomy-output "$INITIAL_TAXONOMY"

filter_train_failures

run_if_needed "$DISCOVERED_TAXONOMY" \
  "$PYTHON" src/error-analysis/discover_error_taxonomy.py \
  --failures "$TRAIN_FAILURES" \
  --stage1-output "$LLM_SUMMARIES" \
  --assignments-output "$DISCOVERED_CLUSTERS" \
  --taxonomy-output "$DISCOVERED_TAXONOMY" \
  --summary-output "$DISCOVERED_SUMMARY" \
  --summarize-model "$MODEL" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$TAXONOMY_GPU_MEMORY_UTILIZATION" \
  --temperature 0.0 \
  --max-tokens 128 \
  --batch-size 64 \
  --min-cluster-size 8 \
  --min-samples 3 \
  --max-cluster-ratio 0.25

run_if_needed "$CONSOLIDATED_TAXONOMY" \
  "$PYTHON" src/error-analysis/consolidate_taxonomy.py \
  --taxonomy "$DISCOVERED_TAXONOMY" \
  --raw-assignments "$DISCOVERED_CLUSTERS" \
  --output "$CONSOLIDATED_TAXONOMY" \
  --audit-output "$CONSOLIDATED_AUDIT" \
  --cluster-mapping-output "$CONSOLIDATED_MAPPING" \
  --response-assignments-output "$CONSOLIDATED_ASSIGNMENTS" \
  --raw-llm-output "$CONSOLIDATION_RAW" \
  --model "$MODEL" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$REFINE_GPU_MEMORY_UTILIZATION" \
  --temperature 0.0 \
  --max-tokens 2048 \
  --min-categories 6 \
  --max-categories 8

run_if_needed "$REFINED_TAXONOMY" \
  "$PYTHON" src/error-analysis/refine_taxonomy_for_rubric.py \
  --taxonomy "$CONSOLIDATED_TAXONOMY" \
  --assignments "$CONSOLIDATED_ASSIGNMENTS" \
  --failures "$TRAIN_FAILURES" \
  --output "$REFINED_TAXONOMY" \
  --audit-output "$REFINED_AUDIT" \
  --response-assignments-output "$REFINED_ASSIGNMENTS" \
  --raw-llm-output "$REFINEMENT_RAW" \
  --model "$MODEL" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$REFINE_GPU_MEMORY_UTILIZATION" \
  --temperature 0.4 \
  --max-tokens 2048 \
  --max-num-seqs 6 \
  --candidates-per-category 3 \
  --revision-candidates 2 \
  --targeted-repair-candidates 2 \
  --max-examples-per-category 5

if [[ "$RUN_STATIC_BASELINE" == "1" ]]; then
  run_if_needed "$AUTO_RUBRIC" \
    "$PYTHON" src/rubric/generate_auto_rubric.py \
    --taxonomy "$REFINED_TAXONOMY" \
    --output "$AUTO_RUBRIC" \
    --generic-output "$GENERIC_RUBRIC" \
    --random-output "$RANDOM_RUBRIC"

  run_if_needed "${RUBRIC_DIR}/mbpp_hidden_auto_rubric_eval_metrics.json" \
    "$PYTHON" src/rubric/evaluate_rubric_static.py \
    --labeled "$LABELED" \
    --failures "$FAILURES" \
    --rubric "$AUTO_RUBRIC" \
    --scores-output "${RUBRIC_DIR}/mbpp_hidden_auto_rubric_scores_static.jsonl" \
    --metrics-output "${RUBRIC_DIR}/mbpp_hidden_auto_rubric_eval_metrics.json"

  run_if_needed "${RUBRIC_DIR}/mbpp_hidden_generic_rubric_eval_metrics.json" \
    "$PYTHON" src/rubric/evaluate_rubric_static.py \
    --labeled "$LABELED" \
    --failures "$FAILURES" \
    --rubric "$GENERIC_RUBRIC" \
    --scores-output "${RUBRIC_DIR}/mbpp_hidden_generic_rubric_scores_static.jsonl" \
    --metrics-output "${RUBRIC_DIR}/mbpp_hidden_generic_rubric_eval_metrics.json"

  run_if_needed "${RUBRIC_DIR}/mbpp_hidden_random_rubric_eval_metrics.json" \
    "$PYTHON" src/rubric/evaluate_rubric_static.py \
    --labeled "$LABELED" \
    --failures "$FAILURES" \
    --rubric "$RANDOM_RUBRIC" \
    --scores-output "${RUBRIC_DIR}/mbpp_hidden_random_rubric_scores_static.jsonl" \
    --metrics-output "${RUBRIC_DIR}/mbpp_hidden_random_rubric_eval_metrics.json"
else
  echo "skip: static rubric baseline disabled by RUN_STATIC_BASELINE=$RUN_STATIC_BASELINE"
fi

echo "Phase 1 pipeline complete."
echo "Refined taxonomy: $REFINED_TAXONOMY"
echo "Refinement audit: $REFINED_AUDIT"
