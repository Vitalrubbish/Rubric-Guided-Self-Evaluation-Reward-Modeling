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
export OUTPUT_DIR=outputs/apps_simple_method1_dpo_v2_semantic_fenced_canary_lora_v2
export DEV_RESPONSES=data/responses/apps_simple_method1_dpo_v2_semantic_fenced_canary_v2_dev.jsonl
export DEV_LABELED=data/responses/apps_simple_method1_dpo_v2_semantic_fenced_canary_v2_dev_labeled.jsonl
export DEV_SUMMARY=data/eval/apps_simple_method1_dpo_v2_semantic_fenced_canary_v2_dev_summary.json
export DEV_REPORT=docs/method1/11-apps-dpo-v2-semantic-fenced-canary-v2-results.md
export PAIRED_BASE_LABELED=data/responses/apps_simple_method1_dpo_dev_v2_base_greedy_semantic_fenced_canary_v2_paired_labeled.jsonl
export PAIRED_VERIFY_MANIFEST=data/eval/apps_simple_method1_dpo_v2_semantic_fenced_canary_v2_paired_verification_manifest.json
export GPU=2
export MAX_PAIRS=400
export MAX_LENGTH=3072
export EPOCHS=1
export LEARNING_RATE=5e-7
export BETA=0.2
export SAVE_STEPS=10

exec bash scripts/method1/run_method1_apps_dpo_v2_canary.sh
