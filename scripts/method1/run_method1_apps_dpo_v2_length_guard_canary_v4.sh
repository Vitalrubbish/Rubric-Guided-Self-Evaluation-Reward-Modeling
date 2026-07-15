#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT_DIR"

PAIRS=data/preferences/apps_simple_method1_self_repair_length_matched_termination_guard_canary_dpo_v2.jsonl
EXPECTED_SHA256=8cb036a362a6f84eaa9e98703e820ef94790cb0c928ca1993c50a71447370fb9
ACTUAL_SHA256=$(sha256sum "$PAIRS" | awk '{print $1}')
if [[ "$ACTUAL_SHA256" != "$EXPECTED_SHA256" ]]; then
  echo "v4 canary dataset hash mismatch: $ACTUAL_SHA256" >&2
  exit 1
fi

export PAIRS
export OUTPUT_DIR=outputs/apps_simple_method1_dpo_v2_length_guard_canary_lora_v4
export BASE_RESPONSES=data/responses/apps_simple_method1_dpo_dev_v2_base_greedy_rep105.jsonl
export DEV_RESPONSES=data/responses/apps_simple_method1_dpo_v2_length_guard_canary_v4_rep105_dev.jsonl
export PAIRED_BASE_LABELED=data/responses/apps_simple_method1_dpo_dev_v2_base_greedy_rep105_length_guard_v4_paired30_labeled.jsonl
export DEV_LABELED=data/responses/apps_simple_method1_dpo_v2_length_guard_canary_v4_rep105_dev_paired30_labeled.jsonl
export PAIRED_VERIFY_MANIFEST=data/eval/apps_simple_method1_dpo_v2_length_guard_canary_v4_rep105_paired30_manifest.json
export DEV_SUMMARY=data/eval/apps_simple_method1_dpo_v2_length_guard_canary_v4_rep105_dev_summary.json
export DEV_REPORT=docs/method1/14-apps-dpo-v2-length-guard-canary-v4-rep105-results.md
export GPU=2
export MAX_PAIRS=400
export MAX_LENGTH=3072
export EPOCHS=2
export LEARNING_RATE=5e-7
export BETA=0.2
export LD_ALPHA=1.0
export REPETITION_PENALTY=1.05
export SAVE_STEPS=6
export VERIFY_TIMEOUT=30
export VERIFY_WORKERS=4

if [[ ! -s "$BASE_RESPONSES" ]] || [[ $(wc -l < "$BASE_RESPONSES") -ne 160 ]]; then
  REPETITION_PENALTY=1.05 \
  RESPONSES="$BASE_RESPONSES" \
  LABELED=data/responses/apps_simple_method1_dpo_dev_v2_base_greedy_rep105_standalone30_labeled.jsonl \
  VERIFY_TIMEOUT=30 \
  VERIFY_WORKERS=4 \
    bash scripts/method1/run_method1_apps_dpo_dev_base_greedy.sh
fi

exec bash scripts/method1/run_method1_apps_dpo_v2_canary.sh
