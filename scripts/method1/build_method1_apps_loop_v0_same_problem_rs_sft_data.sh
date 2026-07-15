#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT_DIR"

PYTHON=${PYTHON:-/data2/acm-group-3/miniconda3/envs/rubric/bin/python}
PAIRS=${PAIRS:-data/preferences/apps_simple_method1_loop_v0_same_problem_only_dpo_pairs.jsonl}
OUTPUT=${OUTPUT:-data/sft/apps_simple_method1_loop_v0_same_problem_rs_sft.jsonl}
SUMMARY_OUTPUT=${SUMMARY_OUTPUT:-data/sft/apps_simple_method1_loop_v0_same_problem_rs_sft_summary.json}

export PATH="$(dirname "$PYTHON"):$PATH"

"$PYTHON" -m src.training.build_apps_loop_v0_rs_sft \
  --pairs "$PAIRS" \
  --output "$OUTPUT" \
  --summary-output "$SUMMARY_OUTPUT" \
  --validation-percent "${VALIDATION_PERCENT:-10}" \
  --deduplicate-problem-chosen

echo "Method1 loop-v0 same-problem RS-SFT data complete: $OUTPUT"
