#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

export PROFILE=${PROFILE:-strict_fewshot}
export TAG=${TAG:-hitl_v5_lite_failgate_test}
export SPLITS=${SPLITS:-test}
export RUBRIC_PATH=${RUBRIC_PATH:-data/rubrics/phase2/mbpp_hidden_llm_rubric_hitl_v5_lite.json}
export GUIDANCE_PATH=${GUIDANCE_PATH:-data/rubrics/phase2/judge_guidance_v5_lite.json}
export EXAMPLES_PATH=${EXAMPLES_PATH:-data/rubrics/phase2/judge_fewshot_examples_validation_v1.json}
export MAX_FEW_SHOT_EXAMPLES=${MAX_FEW_SHOT_EXAMPLES:-7}
export REQUIRE_TEST_PROBES=0
export EXECUTION_GATE=failures
export REUSE_RAW_OUTPUT=${REUSE_RAW_OUTPUT:-data/rubrics/phase2/mbpp_hidden_llm_judge_raw_hitl_v5_lite_source_test.jsonl}
export MAX_MODEL_LEN=${MAX_MODEL_LEN:-12288}
export MAX_TOKENS=${MAX_TOKENS:-1500}

exec scripts/run_phase2_hitl_judge.sh
