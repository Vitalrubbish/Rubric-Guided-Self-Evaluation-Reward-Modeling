#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

export ADAPTER=${ADAPTER:-outputs/method2_apps_self_play_critic_repair_sft_lora_v0_2_format}
export INPUT=${INPUT:-data/sft/method2_apps_self_play_critic_repair_v0_2_format.jsonl}
export VALIDATION_INPUT=${VALIDATION_INPUT:-data/self_play/method2_apps_self_play_v0_2_format_validation_input.jsonl}
export VALIDATION_INPUT_SUMMARY=${VALIDATION_INPUT_SUMMARY:-data/self_play/method2_apps_self_play_v0_2_format_validation_input_summary.json}
export GENERATIONS=${GENERATIONS:-data/self_play/method2_apps_self_play_v0_2_format_validation_generations.jsonl}
export EXTRACTED=${EXTRACTED:-data/self_play/method2_apps_self_play_v0_2_format_validation_extracted.jsonl}
export EXTRACT_SUMMARY=${EXTRACT_SUMMARY:-data/self_play/method2_apps_self_play_v0_2_format_validation_extract_summary.json}
export LABELED=${LABELED:-data/self_play/method2_apps_self_play_v0_2_format_validation_labeled.jsonl}
export SUMMARY=${SUMMARY:-data/self_play/method2_apps_self_play_v0_2_format_validation_repair_gate_summary.json}
export MAX_TOKENS=${MAX_TOKENS:-768}
export STOP_SEQUENCE=${STOP_SEQUENCE:-END_REVISED_CODE}
export STOP_SEQUENCE_2=${STOP_SEQUENCE_2:-$'\nPublic task prompt:'}
export STOP_SEQUENCE_3=${STOP_SEQUENCE_3:-$'\nPrevious failed code:'}

scripts/run_method2_apps_self_play_repair_gate_v0_1_clean.sh
