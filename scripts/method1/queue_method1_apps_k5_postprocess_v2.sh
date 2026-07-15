#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT_DIR"

PYTHON=${PYTHON:-/data2/acm-group-3/miniconda3/envs/rubric/bin/python}
WAIT_PID=${WAIT_PID:-}
POLL_SECONDS=${POLL_SECONDS:-30}
LABELED=${LABELED:-data/repair/apps_simple_method1_repair_all_train_failures_k5_v1_labeled.jsonl}
AUDIT=${AUDIT:-data/repair/apps_simple_method1_repair_all_train_failures_k5_v1_audit.json}
PAIRS=${PAIRS:-data/preferences/apps_simple_method1_all_train_failures_k5_dpo_v2.jsonl}
PAIR_SUMMARY=${PAIR_SUMMARY:-data/preferences/apps_simple_method1_all_train_failures_k5_dpo_v2_summary.json}
MIN_PAIRS=${MIN_PAIRS:-100}

if [[ -n "$WAIT_PID" ]]; then
  while kill -0 "$WAIT_PID" 2>/dev/null; do
    sleep "$POLL_SECONDS"
  done
fi

"$PYTHON" src/analysis-reporting/analyze_apps_repair_pool.py \
  --input "$LABELED" \
  --forbidden-ids data/processed/apps_simple_method1_dpo_dev_v2_prompts.jsonl \
  --forbidden-ids data/processed/apps_simple_method1_internal_eval_prompts_v1.jsonl \
  --expected-rows 6010 \
  --expected-k 5 \
  --fail-on-gate \
  --output "$AUDIT"

"$PYTHON" -m src.training.build_apps_dpo_v2_preferences \
  --repair-labeled "$LABELED" \
  --forbidden-ids data/processed/apps_simple_method1_dpo_dev_v2_prompts.jsonl \
  --forbidden-ids data/processed/apps_simple_method1_internal_eval_prompts_v1.jsonl \
  --max-pairs-per-problem 1 \
  --output "$PAIRS" \
  --summary-output "$PAIR_SUMMARY"

"$PYTHON" - "$PAIR_SUMMARY" "$MIN_PAIRS" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
minimum = int(sys.argv[2])
pair_count = int(summary.get("pair_count") or 0)
if pair_count < minimum:
    raise SystemExit(f"strict k5 pair gate failed: {pair_count} < {minimum}")
print(f"strict k5 pair gate passed: {pair_count} >= {minimum}")
PY

echo "K5 APPS repair post-processing complete:"
echo "  audit=$AUDIT"
echo "  pairs=$PAIRS"
echo "  pair_summary=$PAIR_SUMMARY"
