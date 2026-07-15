#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT_DIR"

PAIRS=data/preferences/apps_simple_method1_self_repair_semantic_fenced_canary_dpo_v2.jsonl
EXPECTED_SHA256=385fe020847ad530bbea213f61c9ce31dfd2263222771d04c4e788ba0eadf507
ACTUAL_SHA256=$(sha256sum "$PAIRS" | awk '{print $1}')
if [[ "$ACTUAL_SHA256" != "$EXPECTED_SHA256" ]]; then
  echo "semantic canary dataset hash mismatch: $ACTUAL_SHA256" >&2
  exit 1
fi

export PAIRS
export OUTPUT_DIR=outputs/apps_simple_method1_dpo_v2_ld_alpha0_canary_lora_v3
export DEV_RESPONSES=data/responses/apps_simple_method1_dpo_v2_ld_alpha0_canary_v3_dev.jsonl
export PAIRED_BASE_LABELED=data/responses/apps_simple_method1_dpo_dev_v2_base_greedy_ld_alpha0_canary_v3_paired30_labeled.jsonl
export DEV_LABELED=data/responses/apps_simple_method1_dpo_v2_ld_alpha0_canary_v3_dev_paired30_labeled.jsonl
export PAIRED_VERIFY_MANIFEST=data/eval/apps_simple_method1_dpo_v2_ld_alpha0_canary_v3_paired30_manifest.json
export DEV_SUMMARY=data/eval/apps_simple_method1_dpo_v2_ld_alpha0_canary_v3_dev_paired30_summary.json
export DEV_REPORT=docs/method1/13-apps-dpo-v2-ld-alpha0-canary-v3-paired30-results.md
export GPU=2
export MAX_PAIRS=400
export MAX_LENGTH=3072
export EPOCHS=1
export LEARNING_RATE=5e-7
export BETA=0.2
export LD_ALPHA=0.0
export SAVE_STEPS=10
export VERIFY_TIMEOUT=30
export VERIFY_WORKERS=4

exec bash scripts/method1/run_method1_apps_dpo_v2_canary.sh
