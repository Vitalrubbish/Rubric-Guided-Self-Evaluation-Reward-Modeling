#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT_DIR"

PYTHON=${PYTHON:-/data2/acm-group-3/miniconda3/envs/rubric/bin/python}

REPAIR_ROWS=${REPAIR_ROWS:-data/repair/apps_simple_method1_repair_all_train_failures_k5_v1_labeled.jsonl}
EVALUATOR_ROWS=${EVALUATOR_ROWS:-data/evaluator/apps_simple_method1_evaluator_training_rows_v1.jsonl}
STRICT_VERIFIER_PAIRS=${STRICT_VERIFIER_PAIRS:-data/preferences/apps_simple_method1_all_train_failures_k5_dpo_v2.jsonl}

RUBRIC_OUTPUT=${RUBRIC_OUTPUT:-data/rubrics/apps_simple_method1_loop_v0_rubric.json}
SCORES_OUTPUT=${SCORES_OUTPUT:-data/rubrics/apps_simple_method1_loop_v0_rubric_scores.jsonl}
PAIRS_OUTPUT=${PAIRS_OUTPUT:-data/preferences/apps_simple_method1_loop_v0_rubric_dpo_pairs.jsonl}
SUMMARY_OUTPUT=${SUMMARY_OUTPUT:-data/preferences/apps_simple_method1_loop_v0_rubric_dpo_pairs_summary.json}

export PATH="$(dirname "$PYTHON"):$PATH"

"$PYTHON" -m src.rubric.build_apps_method1_loop_v0 \
  --repair-rows "$REPAIR_ROWS" \
  --evaluator-rows "$EVALUATOR_ROWS" \
  --strict-verifier-pairs "$STRICT_VERIFIER_PAIRS" \
  --forbidden-ids data/processed/apps_simple_method1_dpo_dev_v2_prompts.jsonl \
  --forbidden-ids data/processed/apps_simple_method1_internal_eval_prompts_v1.jsonl \
  --rubric-output "$RUBRIC_OUTPUT" \
  --scores-output "$SCORES_OUTPUT" \
  --pairs-output "$PAIRS_OUTPUT" \
  --summary-output "$SUMMARY_OUTPUT" \
  --max-contrast-pairs-per-problem "${MAX_CONTRAST_PAIRS_PER_PROBLEM:-2}" \
  --max-weak-rubric-pairs "${MAX_WEAK_RUBRIC_PAIRS:-120}" \
  --max-weak-pairs-per-problem "${MAX_WEAK_PAIRS_PER_PROBLEM:-1}" \
  --min-weak-rubric-margin "${MIN_WEAK_RUBRIC_MARGIN:-1.0}" \
  --max-completion-chars "${MAX_COMPLETION_CHARS:-6000}" \
  --max-length-ratio "${MAX_LENGTH_RATIO:-8.0}"

echo "Method1 loop-v0 rubric data complete:"
echo "  rubric: $RUBRIC_OUTPUT"
echo "  scores: $SCORES_OUTPUT"
echo "  pairs: $PAIRS_OUTPUT"
echo "  summary: $SUMMARY_OUTPUT"
