#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT_DIR"

PYTHON=${PYTHON:-/data2/acm-group-3/miniconda3/envs/rubric/bin/python}

EVALUATOR_ROWS=${EVALUATOR_ROWS:-data/evaluator/apps_simple_method1_evaluator_training_rows_v1.jsonl}
REPAIR_PAIRS=${REPAIR_PAIRS:-data/preferences/apps_simple_method1_all_train_failures_k5_dpo_v2.jsonl}
OUTPUT=${OUTPUT:-data/sft/apps_simple_method1_generative_self_evaluator_v1_4_data_repair.jsonl}
SPLIT_OUTPUT_DIR=${SPLIT_OUTPUT_DIR:-data/sft/apps_simple_method1_generative_self_evaluator_v1_4_data_repair}
SUMMARY_OUTPUT=${SUMMARY_OUTPUT:-data/sft/apps_simple_method1_generative_self_evaluator_v1_4_data_repair_summary.json}

export PATH="$(dirname "$PYTHON"):$PATH"

"$PYTHON" -m src.evaluator.build_generative_evaluator_sft_data \
  --evaluator-rows "$EVALUATOR_ROWS" \
  --repair-pairs "$REPAIR_PAIRS" \
  --output "$OUTPUT" \
  --split-output-dir "$SPLIT_OUTPUT_DIR" \
  --summary-output "$SUMMARY_OUTPUT" \
  --task-chars "${TASK_CHARS:-4500}" \
  --code-chars "${CODE_CHARS:-4500}" \
  --answer-first-judge \
  --evidence-aware-judge \
  --calibrated-extra-content-policy \
  --add-hard-case-records \
  --train-task-types ${TRAIN_TASK_TYPES:-judge_single judge_single_hard_positive judge_single_hard_negative} \
  --heldout-task-types ${HELDOUT_TASK_TYPES:-judge_single} \
  --train-pass-repeat "${TRAIN_PASS_REPEAT:-1.0}" \
  --train-fail-repeat "${TRAIN_FAIL_REPEAT:-1.0}" \
  --train-primary-error-repeat "logic_other_or_unknown=${LOGIC_OTHER_REPEAT:-0.0}" \
  --train-primary-error-repeat "unclear_other_or_not_failure=${UNCLEAR_OTHER_REPEAT:-0.0}" \
  --sampling-seed "${SAMPLING_SEED:-42}"

echo "Generative evaluator SFT v1.4 data-repair data complete: $OUTPUT"
