#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT_DIR"

PYTHON=${PYTHON:-/data2/acm-group-3/miniconda3/envs/rubric/bin/python}
MODEL=${MODEL:-models/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28}
GPU=${GPU:-2}
LIMIT=${LIMIT:-400}
INPUT=${INPUT:-data/repair/apps_simple_method1_repair_prompts_v2.jsonl}
CANDIDATES=${CANDIDATES:-data/repair/apps_simple_method1_repair_candidates_v2.jsonl}
CANDIDATE_SUMMARY=${CANDIDATE_SUMMARY:-data/repair/apps_simple_method1_repair_candidates_v2_summary.json}
RESPONSES=${RESPONSES:-data/repair/apps_simple_method1_two_stage_repair_v2_responses.jsonl}
LABELED=${LABELED:-data/repair/apps_simple_method1_two_stage_repair_v2_labeled.jsonl}
AUDIT=${AUDIT:-data/repair/apps_simple_method1_two_stage_repair_v2_audit.json}
PAIRS=${PAIRS:-data/preferences/apps_simple_method1_self_repair_dpo_v2.jsonl}
PAIR_SUMMARY=${PAIR_SUMMARY:-data/preferences/apps_simple_method1_self_repair_dpo_v2_summary.json}
K=${K:-1}
MIN_SUCCESSFUL_TASKS=${MIN_SUCCESSFUL_TASKS:-1}

export PATH="$(dirname "$PYTHON"):$PATH"
export CUDA_VISIBLE_DEVICES="$GPU"
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/data2/acm-group-3/cache}
export HF_HOME=${HF_HOME:-/data2/acm-group-3/cache/huggingface}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-/data2/acm-group-3/cache/huggingface}
export TMPDIR=${TMPDIR:-/data2/acm-group-3/cache/tmp}

BUILD_ARGS=(
  "$PYTHON" src/evaluator/build_repair_candidates.py
  --output "$CANDIDATES"
  --prompts-output "$INPUT"
  --summary-output "$CANDIDATE_SUMMARY"
  --max-candidates "${MAX_CANDIDATES:-$LIMIT}"
)
if [[ "${EXCLUDE_EVAL_IDS:-0}" == "1" ]]; then
  BUILD_ARGS+=(
    --forbidden-ids data/processed/apps_simple_method1_dpo_dev_v2_prompts.jsonl
    --forbidden-ids data/processed/apps_simple_method1_internal_eval_prompts_v1.jsonl
  )
fi
"${BUILD_ARGS[@]}"

"$PYTHON" src/generation/vllm_two_stage_repair.py \
  --model "$MODEL" \
  --input "$INPUT" \
  --output "$RESPONSES" \
  --limit "$LIMIT" \
  --k "$K" \
  --spec-max-tokens "${SPEC_MAX_TOKENS:-640}" \
  --repair-max-tokens "${REPAIR_MAX_TOKENS:-2048}" \
  --repair-temperature "${REPAIR_TEMPERATURE:-0.2}" \
  --repetition-penalty "${REPETITION_PENALTY:-1.05}" \
  --prompt-format "${PROMPT_FORMAT:-chat}" \
  --max-model-len "${MAX_MODEL_LEN:-12288}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.35}" \
  --prompt-batch-size "${PROMPT_BATCH_SIZE:-20}"

"$PYTHON" src/verification/verify_mbpp_smoke.py \
  --input "$RESPONSES" \
  --output "$LABELED" \
  --timeout "${VERIFY_TIMEOUT:-8}" \
  --workers "${VERIFY_WORKERS:-8}" \
  --process-start-method spawn

"$PYTHON" src/analysis-reporting/analyze_apps_repair_pool.py \
  --input "$LABELED" \
  --forbidden-ids data/processed/apps_simple_method1_dpo_dev_v2_prompts.jsonl \
  --forbidden-ids data/processed/apps_simple_method1_internal_eval_prompts_v1.jsonl \
  --expected-rows "$((LIMIT * K))" \
  --expected-k "$K" \
  --min-eligible-successful-tasks "$MIN_SUCCESSFUL_TASKS" \
  --fail-on-gate \
  --output "$AUDIT"

"$PYTHON" -m src.training.build_apps_dpo_v2_preferences \
  --repair-labeled "$LABELED" \
  --repair-labeled data/repair/apps_simple_method1_repair_v1_labeled.jsonl \
  --forbidden-ids data/processed/apps_simple_method1_dpo_dev_v2_prompts.jsonl \
  --forbidden-ids data/processed/apps_simple_method1_internal_eval_prompts_v1.jsonl \
  --output "$PAIRS" \
  --summary-output "$PAIR_SUMMARY"

echo "Two-stage APPS repair v2 complete:"
echo "  candidates=$CANDIDATES"
echo "  candidate_summary=$CANDIDATE_SUMMARY"
echo "  responses=$RESPONSES"
echo "  labeled=$LABELED"
echo "  audit=$AUDIT"
echo "  pairs=$PAIRS"
