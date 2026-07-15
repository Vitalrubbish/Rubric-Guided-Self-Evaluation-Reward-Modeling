#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT_DIR"

PYTHON=${PYTHON:-/data2/acm-group-3/miniconda3/envs/rubric/bin/python}

INPUT=${INPUT:-data/preferences/apps_simple_method1_loop_v0_same_problem_only_dpo_pairs.jsonl}
OUTPUT=${OUTPUT:-data/preferences/apps_simple_method1_loop_v0_same_problem_balanced_dpo_pairs.jsonl}
SUMMARY_OUTPUT=${SUMMARY_OUTPUT:-data/preferences/apps_simple_method1_loop_v0_same_problem_balanced_dpo_pairs_summary.json}

export PATH="$(dirname "$PYTHON"):$PATH"

"$PYTHON" src/training/filter_apps_loop_v0_dpo_pairs.py \
  --input "$INPUT" \
  --output "$OUTPUT" \
  --summary-output "$SUMMARY_OUTPUT" \
  --max-completion-chars "${MAX_COMPLETION_CHARS:-6000}" \
  --max-char-ratio "${MAX_CHAR_RATIO:-3.0}" \
  --max-whitespace-token-ratio "${MAX_WHITESPACE_TOKEN_RATIO:-3.0}" \
  --max-line-ratio "${MAX_LINE_RATIO:-8.0}" \
  --max-pairs-per-problem "${MAX_PAIRS_PER_PROBLEM:-2}" \
  --max-pairs-per-rejected-failure-type "${MAX_PAIRS_PER_REJECTED_FAILURE_TYPE:-40}" \
  --max-pairs "${MAX_PAIRS:-160}"

echo "Method1 loop-v0 same-problem balanced DPO data complete:"
echo "  pairs: $OUTPUT"
echo "  summary: $SUMMARY_OUTPUT"
