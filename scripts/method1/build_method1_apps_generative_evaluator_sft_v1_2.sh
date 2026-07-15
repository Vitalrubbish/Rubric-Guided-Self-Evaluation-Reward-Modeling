#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT_DIR"

PYTHON=${PYTHON:-/data2/acm-group-3/miniconda3/envs/rubric/bin/python}

EVALUATOR_ROWS=${EVALUATOR_ROWS:-data/evaluator/apps_simple_method1_evaluator_training_rows_v1.jsonl}
REPAIR_PAIRS=${REPAIR_PAIRS:-data/preferences/apps_simple_method1_all_train_failures_k5_dpo_v2.jsonl}
OUTPUT=${OUTPUT:-data/sft/apps_simple_method1_generative_self_evaluator_v1_2_answer_first.jsonl}
SPLIT_OUTPUT_DIR=${SPLIT_OUTPUT_DIR:-data/sft/apps_simple_method1_generative_self_evaluator_v1_2_answer_first}
SUMMARY_OUTPUT=${SUMMARY_OUTPUT:-data/sft/apps_simple_method1_generative_self_evaluator_v1_2_answer_first_summary.json}

export PATH="$(dirname "$PYTHON"):$PATH"

"$PYTHON" -m src.evaluator.build_generative_evaluator_sft_data \
  --evaluator-rows "$EVALUATOR_ROWS" \
  --repair-pairs "$REPAIR_PAIRS" \
  --output "$OUTPUT" \
  --split-output-dir "$SPLIT_OUTPUT_DIR" \
  --summary-output "$SUMMARY_OUTPUT" \
  --task-chars "${TASK_CHARS:-4500}" \
  --code-chars "${CODE_CHARS:-4500}" \
  --answer-first-judge

echo "Generative evaluator SFT v1.2 answer-first data complete: $OUTPUT"
