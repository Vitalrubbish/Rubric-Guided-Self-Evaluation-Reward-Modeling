#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT_DIR"

export MAX_WEAK_RUBRIC_PAIRS=0
export RUBRIC_OUTPUT=${RUBRIC_OUTPUT:-data/rubrics/apps_simple_method1_loop_v0_strong_only_rubric.json}
export SCORES_OUTPUT=${SCORES_OUTPUT:-data/rubrics/apps_simple_method1_loop_v0_strong_only_rubric_scores.jsonl}
export PAIRS_OUTPUT=${PAIRS_OUTPUT:-data/preferences/apps_simple_method1_loop_v0_strong_only_dpo_pairs.jsonl}
export SUMMARY_OUTPUT=${SUMMARY_OUTPUT:-data/preferences/apps_simple_method1_loop_v0_strong_only_dpo_pairs_summary.json}

scripts/method1/build_method1_apps_loop_v0_rubric_data.sh
