#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT_DIR"

export PAIRS=${PAIRS:-data/preferences/apps_simple_method1_loop_v0_rubric_dpo_pairs.jsonl}
export OUTPUT_DIR=${OUTPUT_DIR:-outputs/apps_simple_method1_loop_v0_rubric_dpo_lora}
export DEV_RESPONSES=${DEV_RESPONSES:-data/responses/apps_simple_method1_loop_v0_rubric_dpo_dev.jsonl}
export DEV_LABELED=${DEV_LABELED:-data/responses/apps_simple_method1_loop_v0_rubric_dpo_dev_labeled.jsonl}
export PAIRED_BASE_LABELED=${PAIRED_BASE_LABELED:-data/responses/apps_simple_method1_loop_v0_base_greedy_paired_labeled.jsonl}
export DEV_SUMMARY=${DEV_SUMMARY:-data/eval/apps_simple_method1_loop_v0_rubric_dpo_dev_summary.json}
export DEV_REPORT=${DEV_REPORT:-docs/method1/43-apps-loop-v0-rubric-dpo-canary-results.md}
export PAIRED_VERIFY_MANIFEST=${PAIRED_VERIFY_MANIFEST:-data/eval/apps_simple_method1_loop_v0_rubric_dpo_paired_verification_manifest.json}

export MAX_PAIRS=${MAX_PAIRS:-800}
export LEARNING_RATE=${LEARNING_RATE:-8e-7}
export BETA=${BETA:-0.2}
export SAVE_STEPS=${SAVE_STEPS:-50}
export LOSS_TYPES=${LOSS_TYPES:-sigmoid}

scripts/method1/run_method1_apps_dpo_v2_canary.sh
