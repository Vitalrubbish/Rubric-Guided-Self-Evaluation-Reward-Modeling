#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

export PROFILE=${PROFILE:-contrastive_fewshot}
export TAG=${TAG:-hitl_v3_test}
export SPLITS=${SPLITS:-test}
export RUBRIC_PATH=${RUBRIC_PATH:-data/rubrics/phase2/mbpp_hidden_llm_rubric_hitl_v3.json}
export GUIDANCE_PATH=${GUIDANCE_PATH:-data/rubrics/phase2/judge_guidance_score_collapse_v3.json}
export EXAMPLES_PATH=${EXAMPLES_PATH:-data/rubrics/phase2/judge_fewshot_examples_validation_v1.json}
export MAX_FEW_SHOT_EXAMPLES=${MAX_FEW_SHOT_EXAMPLES:-7}
export REQUIRE_TEST_PROBES=1
export MAX_MODEL_LEN=${MAX_MODEL_LEN:-12288}
export MAX_TOKENS=${MAX_TOKENS:-1500}

exec scripts/run_phase2_hitl_judge.sh
