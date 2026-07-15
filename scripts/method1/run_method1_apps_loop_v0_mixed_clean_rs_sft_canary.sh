#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT_DIR"

export DATA=${DATA:-data/sft/apps_simple_method1_loop_v0_mixed_clean_rs_sft.jsonl}
export OUTPUT_DIR=${OUTPUT_DIR:-outputs/apps_simple_method1_loop_v0_mixed_clean_rs_sft_lora}
export DEV_RESPONSES=${DEV_RESPONSES:-data/responses/apps_simple_method1_loop_v0_mixed_clean_rs_sft_dev.jsonl}
export DEV_LABELED=${DEV_LABELED:-data/responses/apps_simple_method1_loop_v0_mixed_clean_rs_sft_dev_labeled.jsonl}
export PAIRED_BASE_LABELED=${PAIRED_BASE_LABELED:-data/responses/apps_simple_method1_loop_v0_mixed_clean_rs_sft_base_greedy_paired_labeled.jsonl}
export DEV_SUMMARY=${DEV_SUMMARY:-data/eval/apps_simple_method1_loop_v0_mixed_clean_rs_sft_dev_summary.json}
export DEV_REPORT=${DEV_REPORT:-docs/method1/49-apps-loop-v0-mixed-clean-rs-sft-canary-results.md}
export PAIRED_VERIFY_MANIFEST=${PAIRED_VERIFY_MANIFEST:-data/eval/apps_simple_method1_loop_v0_mixed_clean_rs_sft_paired_verification_manifest.json}

export LEARNING_RATE=${LEARNING_RATE:-5e-7}
export SAVE_STEPS=${SAVE_STEPS:-25}
export EVAL_STEPS=${EVAL_STEPS:-25}

scripts/method1/run_method1_apps_loop_v0_same_problem_rs_sft_canary.sh
