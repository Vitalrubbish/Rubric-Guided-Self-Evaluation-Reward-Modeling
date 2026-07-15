#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT_DIR"

export LIMIT=400
export MAX_CANDIDATES=400
export K=3
export MIN_SUCCESSFUL_TASKS=100
export INPUT=data/repair/apps_simple_method1_two_stage_full400_k3_v2_prompts.jsonl
export CANDIDATES=data/repair/apps_simple_method1_two_stage_full400_k3_v2_candidates.jsonl
export CANDIDATE_SUMMARY=data/repair/apps_simple_method1_two_stage_full400_k3_v2_candidates_summary.json
export RESPONSES=data/repair/apps_simple_method1_two_stage_full400_k3_v2_responses.jsonl
export LABELED=data/repair/apps_simple_method1_two_stage_full400_k3_v2_labeled.jsonl
export AUDIT=data/repair/apps_simple_method1_two_stage_full400_k3_v2_audit.json
export PAIRS=data/preferences/apps_simple_method1_two_stage_full400_k3_combined_dpo_v2.jsonl
export PAIR_SUMMARY=data/preferences/apps_simple_method1_two_stage_full400_k3_combined_dpo_v2_summary.json
export GPU=2
export PROMPT_FORMAT=chat
export PROMPT_BATCH_SIZE=20
export SPEC_MAX_TOKENS=640
export REPAIR_MAX_TOKENS=1024
export REPAIR_TEMPERATURE=0.2
export REPETITION_PENALTY=1.05
export GPU_MEMORY_UTILIZATION=0.35
export VERIFY_TIMEOUT=30
export VERIFY_WORKERS=4
export EXCLUDE_EVAL_IDS=1

exec bash scripts/method1/run_method1_apps_two_stage_repair_v2.sh
