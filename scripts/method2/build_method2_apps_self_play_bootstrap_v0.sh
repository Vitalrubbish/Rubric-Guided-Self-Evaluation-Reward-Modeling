#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT_DIR"

PYTHON=${PYTHON:-/data2/acm-group-3/miniconda3/envs/rubric/bin/python}

export PATH="$(dirname "$PYTHON"):$PATH"

"$PYTHON" src/self_play/build_method2_bootstrap_data.py \
  --apps-repair-pairs "${APPS_REPAIR_PAIRS:-data/preferences/apps_simple_method1_repair_all_train_failures_k5_v1_pairs.jsonl}" \
  --no-include-proxy \
  --sft-output "${SFT_OUTPUT:-data/sft/method2_apps_self_play_critic_repair_v0.jsonl}" \
  --dpo-output "${DPO_OUTPUT:-data/preferences/method2_apps_self_play_critic_repair_pairs_v0.jsonl}" \
  --summary-output "${SUMMARY_OUTPUT:-data/self_play/method2_apps_self_play_bootstrap_v0_summary.json}" \
  --validation-percent "${VALIDATION_PERCENT:-10}"

echo "Method2 APPS self-play bootstrap v0 complete"
