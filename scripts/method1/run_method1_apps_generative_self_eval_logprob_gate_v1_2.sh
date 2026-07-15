#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT_DIR"

PYTHON=${PYTHON:-/data2/acm-group-3/miniconda3/envs/rubric/bin/python}
MODEL=${MODEL:-models/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28}
GPU=${GPU:-2}
ADAPTER=${ADAPTER:-outputs/apps_simple_method1_generative_self_evaluator_sft_lora_v1_2_answer_first}

INPUT=${INPUT:-data/sft/apps_simple_method1_generative_self_evaluator_v1_2_answer_first.jsonl}
SCORES=${SCORES:-data/evaluator/apps_simple_method1_generative_self_eval_v1_2_answer_first_logprob_scores.jsonl}
SUMMARY=${SUMMARY:-data/evaluator/apps_simple_method1_generative_self_eval_v1_2_answer_first_logprob_summary.json}
REPORT=${REPORT:-docs/method1/39-generative-self-evaluator-v1-2-answer-first-logprob-results.md}

export PATH="$(dirname "$PYTHON"):$PATH"
export CUDA_VISIBLE_DEVICES="$GPU"
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/tmp/rubric-cache}
export HF_HOME=${HF_HOME:-/tmp/rubric-cache/huggingface}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-/tmp/rubric-cache/huggingface}
export TMPDIR=${TMPDIR:-/tmp/rubric-tmp}

"$PYTHON" -m src.evaluator.score_generative_self_eval_logprob \
  --model "$MODEL" \
  --adapter "$ADAPTER" \
  --input "$INPUT" \
  --scores-output "$SCORES" \
  --summary-output "$SUMMARY" \
  --report-output "$REPORT" \
  --prompt-format "${PROMPT_FORMAT:-raw}" \
  --batch-size "${BATCH_SIZE:-2}" \
  --max-model-len "${MAX_MODEL_LEN:-8192}" \
  --max-overacceptance "${MAX_OVERACCEPTANCE:-0.25}" \
  --min-balanced-accuracy "${MIN_BALANCED_ACCURACY:-0.70}" \
  --dtype "${DTYPE:-bfloat16}"

echo "Generative self-eval v1.2 logprob gate complete: $SUMMARY"
