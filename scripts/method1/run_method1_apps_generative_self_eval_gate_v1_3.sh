#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT_DIR"

PYTHON=${PYTHON:-/data2/acm-group-3/miniconda3/envs/rubric/bin/python}
MODEL=${MODEL:-models/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28}
GPU=${GPU:-2}
ADAPTER=${ADAPTER:-outputs/apps_simple_method1_generative_self_evaluator_sft_lora_v1_3_judge_focused}

INPUT=${INPUT:-data/sft/apps_simple_method1_generative_self_evaluator_v1_3_judge_focused.jsonl}
GENERATIONS=${GENERATIONS:-data/evaluator/apps_simple_method1_generative_self_eval_v1_3_judge_focused_test_generations.jsonl}
SUMMARY=${SUMMARY:-data/evaluator/apps_simple_method1_generative_self_eval_v1_3_judge_focused_test_summary.json}
PREDICTIONS=${PREDICTIONS:-data/evaluator/apps_simple_method1_generative_self_eval_v1_3_judge_focused_test_predictions.jsonl}
REPORT=${REPORT:-docs/method1/41-generative-self-evaluator-v1-3-judge-focused-test-results.md}

export PATH="$(dirname "$PYTHON"):$PATH"
export CUDA_VISIBLE_DEVICES="$GPU"
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/tmp/rubric-cache}
export HF_HOME=${HF_HOME:-/tmp/rubric-cache/huggingface}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-/tmp/rubric-cache/huggingface}
export TMPDIR=${TMPDIR:-/tmp/rubric-tmp}

"$PYTHON" -m src.generation.vllm_lora_prompt_generate \
  --model "$MODEL" \
  --adapter "$ADAPTER" \
  --input "$INPUT" \
  --output "$GENERATIONS" \
  --split "${SPLIT:-test}" \
  --task-type "${TASK_TYPE:-judge_single}" \
  --temperature "${TEMPERATURE:-0.0}" \
  --top-p "${TOP_P:-1.0}" \
  --repetition-penalty "${REPETITION_PENALTY:-1.0}" \
  --max-tokens "${MAX_TOKENS:-256}" \
  --max-model-len "${MAX_MODEL_LEN:-8192}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.35}" \
  --prompt-batch-size "${PROMPT_BATCH_SIZE:-32}" \
  --max-lora-rank "${MAX_LORA_RANK:-16}" \
  --prompt-format "${PROMPT_FORMAT:-raw}" \
  --seed "${SEED:-42}"

"$PYTHON" -m src.evaluator.evaluate_generative_self_eval \
  --generations "$GENERATIONS" \
  --summary-output "$SUMMARY" \
  --predictions-output "$PREDICTIONS" \
  --report-output "$REPORT" \
  --min-parse-rate "${MIN_PARSE_RATE:-0.98}" \
  --max-overacceptance "${MAX_OVERACCEPTANCE:-0.25}" \
  --min-balanced-accuracy "${MIN_BALANCED_ACCURACY:-0.70}"

echo "Generative self-eval v1.3 free-generation gate complete: $SUMMARY"
