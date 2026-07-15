#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT_DIR"

PAIRS=data/preferences/apps_simple_method1_two_stage_raw100_syntax_guard11_dpo_v2.jsonl
EXPECTED_SHA256=b3e4792249d53b564419ac6b9e00f2870ad08db64ca23d19f755db1905355fd1
ACTUAL_SHA256=$(sha256sum "$PAIRS" | awk '{print $1}')
if [[ "$ACTUAL_SHA256" != "$EXPECTED_SHA256" ]]; then
  echo "raw syntax-guard canary dataset hash mismatch: $ACTUAL_SHA256" >&2
  exit 1
fi

export PAIRS
export OUTPUT_DIR=outputs/apps_simple_method1_dpo_v2_two_stage_raw_syntax_guard_canary_lora_v11
export BASE_RESPONSES=data/responses/apps_simple_method1_dpo_dev_v2_base_greedy_rep105.jsonl
export DEV_RESPONSES=data/responses/apps_simple_method1_dpo_v2_two_stage_raw_syntax_guard_canary_v11_rep105_dev.jsonl
export PAIRED_BASE_LABELED=data/responses/apps_simple_method1_dpo_dev_v2_base_greedy_rep105_two_stage_raw_syntax_guard_v11_paired60_labeled.jsonl
export DEV_LABELED=data/responses/apps_simple_method1_dpo_v2_two_stage_raw_syntax_guard_canary_v11_rep105_dev_paired60_labeled.jsonl
export PAIRED_VERIFY_MANIFEST=data/eval/apps_simple_method1_dpo_v2_two_stage_raw_syntax_guard_canary_v11_rep105_paired60_manifest.json
export DEV_SUMMARY=data/eval/apps_simple_method1_dpo_v2_two_stage_raw_syntax_guard_canary_v11_rep105_dev_summary.json
export DEV_REPORT=docs/method1/29-apps-dpo-v2-two-stage-raw-syntax-guard-canary-v11-results.md
export GPU=2
export MAX_PAIRS=400
export MAX_LENGTH=3072
export EPOCHS=1
export GRADIENT_ACCUMULATION_STEPS=23
export LEARNING_RATE=5e-7
export BETA=0.2
export LD_ALPHA=1.0
export LOSS_TYPES=sigmoid,sft
export LOSS_WEIGHTS=1.0,0.2
export REPETITION_PENALTY=1.05
export SAVE_STEPS=5
export VERIFY_TIMEOUT=60
export VERIFY_WORKERS=2

exec bash scripts/method1/run_method1_apps_dpo_v2_canary.sh
