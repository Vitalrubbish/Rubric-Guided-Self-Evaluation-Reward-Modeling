#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT_DIR"

PYTHON=${PYTHON:-/data2/acm-group-3/miniconda3/envs/rubric/bin/python}
MODEL=${MODEL:-models/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28}
GPU=${GPU:-2}
PROMPTS=data/processed/apps_simple_method1_dpo_dev_v2_prompts.jsonl
PAIRS=data/preferences/apps_simple_method1_two_stage_full400_k3_semantic_raw_frozen100_dpo_v2.jsonl
ADAPTER=outputs/apps_simple_method1_dpo_v2_two_stage_raw_canary_lora_v9
BASE_RAW=data/responses/apps_simple_method1_dpo_dev_v2_base_greedy_rep110.jsonl
CANDIDATE_RAW=data/responses/apps_simple_method1_dpo_v2_two_stage_raw_canary_v9_rep110_dev.jsonl
BASE_LABELED=data/responses/apps_simple_method1_dpo_dev_v2_base_greedy_rep110_two_stage_raw_v9_paired60_labeled.jsonl
CANDIDATE_LABELED=data/responses/apps_simple_method1_dpo_v2_two_stage_raw_canary_v9_rep110_dev_paired60_labeled.jsonl
VERIFY_MANIFEST=data/eval/apps_simple_method1_dpo_v2_two_stage_raw_canary_v9_rep110_paired60_manifest.json
SUMMARY=data/eval/apps_simple_method1_dpo_v2_two_stage_raw_canary_v9_rep110_dev_summary.json
REPORT=docs/method1/27-apps-dpo-v2-two-stage-raw-canary-v9-rep110-results.md

EXPECTED_PAIR_SHA=b70bc0f39286a1756bbb002d8a4cbfaf4cdcc43519e427a9edfa1b1db417321b
ACTUAL_PAIR_SHA=$(sha256sum "$PAIRS" | awk '{print $1}')
if [[ "$ACTUAL_PAIR_SHA" != "$EXPECTED_PAIR_SHA" ]]; then
  echo "raw full two-stage dataset hash mismatch: $ACTUAL_PAIR_SHA" >&2
  exit 1
fi
if [[ ! -f "$ADAPTER/adapter_model.safetensors" ]]; then
  echo "v9 adapter is incomplete" >&2
  exit 1
fi

export PATH="$(dirname "$PYTHON"):$PATH"
export CUDA_VISIBLE_DEVICES="$GPU"
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/data2/acm-group-3/cache}
export HF_HOME=${HF_HOME:-/data2/acm-group-3/cache/huggingface}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-/data2/acm-group-3/cache/huggingface}
export TMPDIR=${TMPDIR:-/data2/acm-group-3/cache/tmp}

"$PYTHON" src/generation/vllm_smoke_generate.py \
  --model "$MODEL" \
  --input "$PROMPTS" \
  --output "$BASE_RAW" \
  --limit 160 \
  --k 1 \
  --temperature 0 \
  --top-p 1 \
  --repetition-penalty 1.10 \
  --max-tokens 2048 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.35 \
  --prompt-batch-size 32 \
  --seed 42

"$PYTHON" -m src.generation.vllm_lora_generate \
  --model "$MODEL" \
  --adapter "$ADAPTER" \
  --input "$PROMPTS" \
  --output "$CANDIDATE_RAW" \
  --temperature 0 \
  --top-p 1 \
  --repetition-penalty 1.10 \
  --max-tokens 2048 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.35 \
  --prompt-batch-size 32 \
  --max-lora-rank 16 \
  --seed 42

"$PYTHON" -m src.verification.verify_paired_apps_dpo_dev \
  --base-input "$BASE_RAW" \
  --candidate-input "$CANDIDATE_RAW" \
  --base-output "$BASE_LABELED" \
  --candidate-output "$CANDIDATE_LABELED" \
  --manifest "$VERIFY_MANIFEST" \
  --expected-rows 160 \
  --timeout 60 \
  --workers 2 \
  --process-start-method spawn

"$PYTHON" src/analysis-reporting/compare_apps_dpo_dev.py \
  --base-labeled "$BASE_LABELED" \
  --candidate-labeled "$CANDIDATE_LABELED" \
  --training-preferences "$PAIRS" \
  --output "$SUMMARY" \
  --report "$REPORT" \
  --require-paired-verification
