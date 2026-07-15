#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT_DIR"

PYTHON=${PYTHON:-/data2/acm-group-3/miniconda3/envs/rubric/bin/python}

REPAIR_SFT=${REPAIR_SFT:-data/sft/apps_simple_method1_loop_v0_same_problem_rs_sft.jsonl}
PRESERVATION_ROWS=${PRESERVATION_ROWS:-data/responses/apps_train_simple_executable_qwen25_k1_t2048_full_labeled_nonlength.jsonl}
OUTPUT=${OUTPUT:-data/sft/apps_simple_method1_loop_v0_mixed_strict_rs_sft_v1_5.jsonl}
SUMMARY_OUTPUT=${SUMMARY_OUTPUT:-data/sft/apps_simple_method1_loop_v0_mixed_strict_rs_sft_v1_5_summary.json}

export PATH="$(dirname "$PYTHON"):$PATH"

"$PYTHON" src/training/build_apps_loop_v0_mixed_rs_sft.py \
  --repair-sft "$REPAIR_SFT" \
  --preservation-rows "$PRESERVATION_ROWS" \
  --forbidden-ids data/processed/apps_simple_method1_dpo_dev_v2_prompts.jsonl \
  --forbidden-ids data/processed/apps_simple_method1_internal_eval_prompts_v1.jsonl \
  --output "$OUTPUT" \
  --summary-output "$SUMMARY_OUTPUT" \
  --max-preservation-rows "${MAX_PRESERVATION_ROWS:-712}" \
  --validation-percent "${VALIDATION_PERCENT:-10}" \
  --strict-interface-filter

echo "Method1 loop-v0 mixed strict RS-SFT v1.5 data complete:"
echo "  data: $OUTPUT"
echo "  summary: $SUMMARY_OUTPUT"
