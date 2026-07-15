#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT_DIR"

PYTHON=${PYTHON:-/data2/acm-group-3/miniconda3/envs/rubric/bin/python}
MODEL=${MODEL:-models/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28}
ADAPTER=outputs/apps_simple_method1_dpo_v2_two_stage_raw_canary_lora_v9
PROMPTS=data/processed/apps_simple_method1_dpo_dev_v2_v9_rep110_badcase_prompts.jsonl
OLD_RAW=data/responses/apps_simple_method1_dpo_v2_two_stage_raw_v9_rep105_badcase_raw.jsonl
NEW_RAW=data/responses/apps_simple_method1_dpo_v2_two_stage_raw_v9_rep110_badcase_raw.jsonl
OLD_LABELED=data/responses/apps_simple_method1_dpo_v2_two_stage_raw_v9_rep105_badcase_paired60_labeled.jsonl
NEW_LABELED=data/responses/apps_simple_method1_dpo_v2_two_stage_raw_v9_rep110_badcase_paired60_labeled.jsonl
VERIFY_MANIFEST=data/eval/apps_simple_method1_dpo_v2_two_stage_raw_v9_rep110_badcase_paired60_manifest.json
SUMMARY=data/eval/apps_simple_method1_dpo_v2_two_stage_raw_v9_rep110_badcase_summary.json

EXPECTED_PROMPT_SHA=e50756af9590941ede2cda062008e40947becaebeadd6950e8d30768539e0969
ACTUAL_PROMPT_SHA=$(sha256sum "$PROMPTS" | awk '{print $1}')
if [[ "$ACTUAL_PROMPT_SHA" != "$EXPECTED_PROMPT_SHA" ]]; then
  echo "bad-case prompt hash mismatch: $ACTUAL_PROMPT_SHA" >&2
  exit 1
fi

export PATH="$(dirname "$PYTHON"):$PATH"
export CUDA_VISIBLE_DEVICES=${GPU:-2}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/data2/acm-group-3/cache}
export HF_HOME=${HF_HOME:-/data2/acm-group-3/cache/huggingface}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-/data2/acm-group-3/cache/huggingface}
export TMPDIR=${TMPDIR:-/data2/acm-group-3/cache/tmp}

"$PYTHON" -m src.generation.vllm_lora_generate \
  --model "$MODEL" \
  --adapter "$ADAPTER" \
  --input "$PROMPTS" \
  --output "$NEW_RAW" \
  --temperature 0 \
  --top-p 1 \
  --repetition-penalty 1.10 \
  --max-tokens 2048 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.35 \
  --prompt-batch-size 25 \
  --max-lora-rank 16

"$PYTHON" -m src.verification.verify_paired_apps_dpo_dev \
  --base-input "$OLD_RAW" \
  --candidate-input "$NEW_RAW" \
  --base-output "$OLD_LABELED" \
  --candidate-output "$NEW_LABELED" \
  --manifest "$VERIFY_MANIFEST" \
  --expected-rows 25 \
  --timeout 60 \
  --workers 2 \
  --process-start-method spawn

"$PYTHON" - "$OLD_LABELED" "$NEW_LABELED" "$SUMMARY" <<'PY'
import json
import sys
from pathlib import Path


def load(path):
    return {row["id"]: row for row in map(json.loads, Path(path).open())}


old = load(sys.argv[1])
new = load(sys.argv[2])
summary = {
    "rows": len(old),
    "rep105_passed": sum(row["passed"] for row in old.values()),
    "rep110_passed": sum(row["passed"] for row in new.values()),
    "rep105_length": sum(row.get("finish_reason") == "length" for row in old.values()),
    "rep110_length": sum(row.get("finish_reason") == "length" for row in new.values()),
    "fail_to_pass": sum(not old[key]["passed"] and new[key]["passed"] for key in old),
    "pass_to_fail": sum(old[key]["passed"] and not new[key]["passed"] for key in old),
    "rep105_syntax": sum(row.get("failure_type") == "syntax_error" for row in old.values()),
    "rep110_syntax": sum(row.get("failure_type") == "syntax_error" for row in new.values()),
}
Path(sys.argv[3]).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))
PY
