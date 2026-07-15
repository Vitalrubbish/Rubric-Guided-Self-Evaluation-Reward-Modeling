#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT_DIR"

PYTHON=${PYTHON:-/data2/acm-group-3/miniconda3/envs/rubric/bin/python}
MODEL=${MODEL:-models/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28}
ADAPTER=${ADAPTER:-outputs/apps_simple_method1_dpo_v2_semantic_fenced_canary_lora_v2}
INPUT=${INPUT:-data/processed/apps_simple_method1_dpo_dev_v2_long_regression5_prompts.jsonl}
RESPONSES=${RESPONSES:-data/responses/apps_simple_method1_dpo_v2_rep105_long_regression5.jsonl}
LABELED=${LABELED:-data/responses/apps_simple_method1_dpo_v2_rep105_long_regression5_labeled.jsonl}

export PATH="$(dirname "$PYTHON"):$PATH"
export CUDA_VISIBLE_DEVICES=${GPU:-2}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/data2/acm-group-3/cache}
export HF_HOME=${HF_HOME:-/data2/acm-group-3/cache/huggingface}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-/data2/acm-group-3/cache/huggingface}
export TMPDIR=${TMPDIR:-/data2/acm-group-3/cache/tmp}

"$PYTHON" -m src.generation.vllm_lora_generate \
  --model "$MODEL" \
  --adapter "$ADAPTER" \
  --input "$INPUT" \
  --output "$RESPONSES" \
  --temperature 0 \
  --top-p 1 \
  --repetition-penalty 1.05 \
  --max-tokens 2048 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.35 \
  --prompt-batch-size 5 \
  --max-lora-rank 16

"$PYTHON" -m src.verification.verify_mbpp_smoke \
  --input "$RESPONSES" \
  --output "$LABELED" \
  --timeout 30 \
  --workers 4 \
  --process-start-method spawn

"$PYTHON" - "$LABELED" <<'PY'
import json
import sys
from pathlib import Path

rows = [json.loads(line) for line in Path(sys.argv[1]).read_text().splitlines() if line.strip()]
summary = [
    {
        "id": row["id"],
        "passed": row["passed"],
        "failure_type": row.get("failure_type"),
        "finish_reason": row.get("finish_reason"),
        "generated_token_count": row.get("generated_token_count"),
    }
    for row in rows
]
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY
