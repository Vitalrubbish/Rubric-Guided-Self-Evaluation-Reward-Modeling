#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT_DIR"

WAIT_PID=${WAIT_PID:-}
POLL_SECONDS=${POLL_SECONDS:-30}
if [[ -n "$WAIT_PID" ]]; then
  while kill -0 "$WAIT_PID" 2>/dev/null; do
    sleep "$POLL_SECONDS"
  done
fi

export LIMIT=${LIMIT:-20}
export K=${K:-3}
export MIN_SUCCESSFUL_TASKS=${MIN_SUCCESSFUL_TASKS:-8}
export RESPONSES=${RESPONSES:-data/repair/apps_simple_method1_two_stage_repair_smoke20_k3_v2_responses.jsonl}
export LABELED=${LABELED:-data/repair/apps_simple_method1_two_stage_repair_smoke20_k3_v2_labeled.jsonl}
export AUDIT=${AUDIT:-data/repair/apps_simple_method1_two_stage_repair_smoke20_k3_v2_audit.json}
export PAIRS=${PAIRS:-data/preferences/apps_simple_method1_two_stage_smoke20_k3_combined_dpo_v2.jsonl}
export PAIR_SUMMARY=${PAIR_SUMMARY:-data/preferences/apps_simple_method1_two_stage_smoke20_k3_combined_dpo_v2_summary.json}
export GPU=${GPU:-2}
export PROMPT_BATCH_SIZE=${PROMPT_BATCH_SIZE:-10}

exec bash scripts/method1/run_method1_apps_two_stage_repair_v2.sh
