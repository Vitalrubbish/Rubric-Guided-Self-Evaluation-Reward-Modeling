#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT_DIR"

export PAIRS=${PAIRS:-data/preferences/apps_simple_method1_self_repair_canary_dpo_v2.jsonl}
export OUTPUT_DIR=${OUTPUT_DIR:-outputs/apps_simple_method1_dpo_v2_canary_lora_v1}
export DEV_RESPONSES=${DEV_RESPONSES:-data/responses/apps_simple_method1_dpo_v2_canary_v1_dev.jsonl}
export DEV_LABELED=${DEV_LABELED:-data/responses/apps_simple_method1_dpo_v2_canary_v1_dev_labeled.jsonl}
export DEV_SUMMARY=${DEV_SUMMARY:-data/eval/apps_simple_method1_dpo_v2_canary_v1_dev_summary.json}
export DEV_REPORT=${DEV_REPORT:-docs/method1/10-apps-dpo-v2-canary-v1-results.md}
export GPU=${GPU:-2}
export MAX_PAIRS=${MAX_PAIRS:-400}

exec bash scripts/method1/run_method1_apps_dpo_v2_canary.sh
