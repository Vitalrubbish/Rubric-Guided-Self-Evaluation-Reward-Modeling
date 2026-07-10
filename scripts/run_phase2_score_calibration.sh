#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

PYTHON=${PYTHON:-/data2/acm-group-3/miniconda3/envs/rubric/bin/python}
TAG=${TAG:-validation_logistic}
SOURCE_TAG=${SOURCE_TAG:-validation_test_calibrated_t475}
CALIBRATION_SPLIT=${CALIBRATION_SPLIT:-validation}
THRESHOLD=${THRESHOLD:-}

RUBRIC_DIR="data/rubrics/phase2"
SCORES="${RUBRIC_DIR}/mbpp_hidden_llm_judge_scores_${SOURCE_TAG}.jsonl"
SCORES_OUTPUT="${RUBRIC_DIR}/mbpp_hidden_llm_judge_scores_${TAG}.jsonl"
METRICS_OUTPUT="${RUBRIC_DIR}/mbpp_hidden_llm_judge_metrics_${TAG}.json"
AUDIT_OUTPUT="${RUBRIC_DIR}/mbpp_hidden_llm_judge_audit_${TAG}.json"

ARGS=(
  "$PYTHON" src/rubric/calibrate_llm_judge_scores.py
  --scores "$SCORES"
  --scores-output "$SCORES_OUTPUT"
  --metrics-output "$METRICS_OUTPUT"
  --audit-output "$AUDIT_OUTPUT"
  --calibration-split "$CALIBRATION_SPLIT"
)

if [[ -n "$THRESHOLD" ]]; then
  ARGS+=(--threshold "$THRESHOLD")
fi

echo "+ ${ARGS[*]}"
"${ARGS[@]}"

echo "Phase 2 judge score calibration complete."
echo "Scores: $SCORES_OUTPUT"
echo "Metrics: $METRICS_OUTPUT"
echo "Audit: $AUDIT_OUTPUT"
