#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT_DIR"

scripts/method2/build_method2_apps_self_play_sft_v0_5_stop50.sh
scripts/method2/run_method2_apps_self_play_critic_repair_sft_v0_5_stop50.sh
scripts/method2/run_method2_apps_self_play_repair_gate_v0_5_stop50.sh
