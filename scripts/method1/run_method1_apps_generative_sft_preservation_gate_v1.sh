#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT_DIR"

export ADAPTER=${ADAPTER:-outputs/apps_simple_method1_generative_self_evaluator_sft_lora_v1}
export PAIRS=${PAIRS:-data/preferences/apps_simple_method1_all_train_failures_k5_dpo_v2.jsonl}
export PYTHON=${PYTHON:-/data2/acm-group-3/miniconda3/envs/rubric/bin/python}
export MODEL=${MODEL:-models/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28}
export GPU=${GPU:-2}
export PROMPTS=${PROMPTS:-data/processed/apps_simple_method1_dpo_v2_final523_prompts.jsonl}
export BASE_RESPONSES=${BASE_RESPONSES:-data/responses/apps_simple_method1_dpo_v2_final523_base_rep105.jsonl}
export CANDIDATE_RESPONSES=${CANDIDATE_RESPONSES:-data/responses/apps_simple_method1_generative_sft_v1_final523_rep105.jsonl}
export PAIRED_BASE_LABELED=${PAIRED_BASE_LABELED:-data/responses/apps_simple_method1_dpo_v2_final523_base_rep105_generative_sft_v1_paired_labeled.jsonl}
export CANDIDATE_LABELED=${CANDIDATE_LABELED:-data/responses/apps_simple_method1_generative_sft_v1_final523_rep105_paired_labeled.jsonl}
export PAIRED_VERIFY_MANIFEST=${PAIRED_VERIFY_MANIFEST:-data/eval/apps_simple_method1_generative_sft_v1_final523_paired_manifest.json}
export SUMMARY=${SUMMARY:-data/eval/apps_simple_method1_generative_sft_v1_final523_preservation_summary.json}
export REPORT=${REPORT:-docs/method1/36-generative-sft-v1-final523-preservation-results.md}
export SFT_DATA=${SFT_DATA:-data/sft/apps_simple_method1_generative_self_evaluator_v1.jsonl}
export DPO_DEV=${DPO_DEV:-data/processed/apps_simple_method1_dpo_dev_v2_prompts.jsonl}
export REPETITION_PENALTY=${REPETITION_PENALTY:-1.05}
export VERIFY_TIMEOUT=${VERIFY_TIMEOUT:-60}
export VERIFY_WORKERS=${VERIFY_WORKERS:-2}
export EVAL_GPU_MEMORY_UTILIZATION=${EVAL_GPU_MEMORY_UTILIZATION:-0.35}
export EVAL_PROMPT_BATCH_SIZE=${EVAL_PROMPT_BATCH_SIZE:-32}

export PATH="$(dirname "$PYTHON"):$PATH"
export CUDA_VISIBLE_DEVICES="$GPU"
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/tmp/rubric-cache}
export HF_HOME=${HF_HOME:-/tmp/rubric-cache/huggingface}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-/tmp/rubric-cache/huggingface}
export TMPDIR=${TMPDIR:-/tmp/rubric-tmp}

"$PYTHON" -m src.generation.vllm_lora_generate \
  --model "$MODEL" \
  --adapter "$ADAPTER" \
  --input "$PROMPTS" \
  --output "$CANDIDATE_RESPONSES" \
  --temperature 0.0 \
  --top-p 1.0 \
  --repetition-penalty "$REPETITION_PENALTY" \
  --max-tokens 2048 \
  --max-model-len 8192 \
  --gpu-memory-utilization "$EVAL_GPU_MEMORY_UTILIZATION" \
  --prompt-batch-size "$EVAL_PROMPT_BATCH_SIZE" \
  --max-lora-rank 16 \
  --seed 42

"$PYTHON" -m src.verification.verify_paired_apps_dpo_dev \
  --base-input "$BASE_RESPONSES" \
  --candidate-input "$CANDIDATE_RESPONSES" \
  --base-output "$PAIRED_BASE_LABELED" \
  --candidate-output "$CANDIDATE_LABELED" \
  --manifest "$PAIRED_VERIFY_MANIFEST" \
  --expected-rows 523 \
  --timeout "$VERIFY_TIMEOUT" \
  --workers "$VERIFY_WORKERS" \
  --process-start-method spawn

"$PYTHON" -m src.evaluator.compare_apps_final523_adapter \
  --base-labeled "$PAIRED_BASE_LABELED" \
  --candidate-labeled "$CANDIDATE_LABELED" \
  --sft-data "$SFT_DATA" \
  --dpo-dev "$DPO_DEV" \
  --output "$SUMMARY" \
  --report "$REPORT"
