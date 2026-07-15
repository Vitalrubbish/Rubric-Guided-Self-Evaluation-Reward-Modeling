#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT_DIR"

PYTHON=${PYTHON:-/data2/acm-group-3/miniconda3/envs/rubric/bin/python}

export PATH="$(dirname "$PYTHON"):$PATH"

BUILD_ARGS=(
  "$PYTHON" src/self_play/build_method2_iterative_sft.py
  --base-sft "${BASE_SFT:-data/sft/method2_apps_self_play_critic_repair_v0_3_no_end_marker.jsonl}"
  --generated-labeled "${GENERATED_LABELED:-data/self_play/method2_apps_self_play_v0_4_train_candidates_labeled.jsonl}"
  --sft-output "${SFT_OUTPUT:-data/sft/method2_apps_self_play_critic_repair_v0_4_iterative.jsonl}"
  --accepted-output "${ACCEPTED_OUTPUT:-data/self_play/method2_apps_self_play_v0_4_accepted_self_generated.jsonl}"
  --summary-output "${SUMMARY_OUTPUT:-data/self_play/method2_apps_self_play_v0_4_iterative_summary.json}"
  --max-generated-per-id "${MAX_GENERATED_PER_ID:-1}"
  --source-tag "${SOURCE_TAG:-method2_v0_4_self_generated_pass}"
)
if [[ -n "${MAX_GENERATED_TOTAL:-}" ]]; then
  BUILD_ARGS+=(--max-generated-total "$MAX_GENERATED_TOTAL")
fi
if [[ -n "${REQUIRE_FINISH_REASON:-}" ]]; then
  BUILD_ARGS+=(--require-finish-reason "$REQUIRE_FINISH_REASON")
fi
if [[ -n "${MAX_GENERATED_TOKENS:-}" ]]; then
  BUILD_ARGS+=(--max-generated-tokens "$MAX_GENERATED_TOKENS")
fi
if [[ -n "${MAX_EXTRACTION_NOTES:-}" ]]; then
  BUILD_ARGS+=(--max-extraction-notes "$MAX_EXTRACTION_NOTES")
fi

"${BUILD_ARGS[@]}"

echo "Method2 APPS v0.4 iterative SFT data build complete"
