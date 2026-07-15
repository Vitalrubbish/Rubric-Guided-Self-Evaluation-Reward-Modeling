#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT_DIR"

# Full Method 2 v0.4 loop:
# v0.3 adapter -> generate train repairs -> verifier filter -> train v0.4 -> repair gate.

scripts/method2/run_method2_apps_self_play_generate_train_candidates_v0_4.sh
scripts/method2/build_method2_apps_self_play_sft_v0_4_iterative.sh
scripts/method2/run_method2_apps_self_play_critic_repair_sft_v0_4_iterative.sh
scripts/method2/run_method2_apps_self_play_repair_gate_v0_4_iterative.sh
