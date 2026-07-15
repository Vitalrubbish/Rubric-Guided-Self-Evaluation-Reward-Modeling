#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

export DATA=${DATA:-data/sft/method2_apps_self_play_critic_repair_v0_3_no_end_marker.jsonl}
export OUTPUT_DIR=${OUTPUT_DIR:-outputs/method2_apps_self_play_critic_repair_sft_lora_v0_3_no_end_marker}
export LEARNING_RATE=${LEARNING_RATE:-5e-7}
export EVAL_STEPS=${EVAL_STEPS:-25}
export SAVE_STEPS=${SAVE_STEPS:-25}

scripts/run_method2_self_play_critic_repair_sft_v0.sh
