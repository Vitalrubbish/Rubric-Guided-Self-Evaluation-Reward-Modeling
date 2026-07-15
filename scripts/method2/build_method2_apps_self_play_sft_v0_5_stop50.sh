#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT_DIR"

export SFT_OUTPUT=${SFT_OUTPUT:-data/sft/method2_apps_self_play_critic_repair_v0_5_stop50.jsonl}
export ACCEPTED_OUTPUT=${ACCEPTED_OUTPUT:-data/self_play/method2_apps_self_play_v0_5_stop50_accepted_self_generated.jsonl}
export SUMMARY_OUTPUT=${SUMMARY_OUTPUT:-data/self_play/method2_apps_self_play_v0_5_stop50_summary.json}
export MAX_GENERATED_PER_ID=${MAX_GENERATED_PER_ID:-1}
export MAX_GENERATED_TOTAL=${MAX_GENERATED_TOTAL:-50}
export REQUIRE_FINISH_REASON=${REQUIRE_FINISH_REASON:-stop}
export SOURCE_TAG=${SOURCE_TAG:-method2_v0_5_stop50_self_generated_pass}

scripts/method2/build_method2_apps_self_play_sft_v0_4_iterative.sh
