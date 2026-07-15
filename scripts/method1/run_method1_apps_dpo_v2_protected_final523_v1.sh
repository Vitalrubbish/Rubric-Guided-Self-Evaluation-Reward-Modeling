#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT_DIR"

PYTHON=${PYTHON:-/data2/acm-group-3/miniconda3/envs/rubric/bin/python}
MODEL=${MODEL:-models/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28}
ADAPTER=outputs/apps_simple_method1_dpo_v2_mixed_sft_canary_lora_v5
PROMPTS=data/processed/apps_simple_method1_dpo_v2_final523_prompts.jsonl
EXPECTED_PROMPTS_SHA256=009f60bb0a69515f5a1343f73810cd19681bee6a410a318e05212befa1d70fb4
TRAINING_PREFERENCES=data/preferences/apps_simple_method1_self_repair_semantic_fenced_canary_dpo_v2.jsonl
DPO_DEV=data/processed/apps_simple_method1_dpo_dev_v2_prompts.jsonl
CANARY_SUMMARY=data/eval/apps_simple_method1_dpo_v2_mixed_sft_v5_protected_routed_rep105_dev_summary.json

RUN_DIR=outputs/apps_simple_method1_dpo_v2_protected_final523_v1
RUN_MANIFEST=$RUN_DIR/run_manifest.json
BASE_RESPONSES=data/responses/apps_simple_method1_dpo_v2_final523_base_rep105.jsonl
ADAPTER_RESPONSES=data/responses/apps_simple_method1_dpo_v2_final523_v5_rep105.jsonl
ROUTED_RESPONSES=data/responses/apps_simple_method1_dpo_v2_final523_v5_protected_routed_rep105.jsonl
ROUTE_MANIFEST=data/eval/apps_simple_method1_dpo_v2_final523_v5_protected_route_manifest.json
BASE_LABELED=data/responses/apps_simple_method1_dpo_v2_final523_base_rep105_paired60_labeled.jsonl
ROUTED_LABELED=data/responses/apps_simple_method1_dpo_v2_final523_v5_protected_routed_rep105_paired60_labeled.jsonl
PAIRED_MANIFEST=data/eval/apps_simple_method1_dpo_v2_final523_v5_protected_paired60_manifest.json
FINAL_SUMMARY=data/eval/apps_simple_method1_dpo_v2_final523_v5_protected_summary.json
FINAL_REPORT=docs/method1/32-apps-dpo-v2-protected-routed-v5-final523-results.md

export PATH="$(dirname "$PYTHON"):$PATH"
export CUDA_VISIBLE_DEVICES=${GPU:-2}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/data2/acm-group-3/cache}
export HF_HOME=${HF_HOME:-/data2/acm-group-3/cache/huggingface}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-/data2/acm-group-3/cache/huggingface}
export TMPDIR=${TMPDIR:-/data2/acm-group-3/cache/tmp}

mkdir -p "$RUN_DIR"

jsonl_complete() {
  local path=$1
  [[ -f "$path" ]] && [[ $(wc -l < "$path") -eq 523 ]]
}

ACTUAL_PROMPTS_SHA256=$(sha256sum "$PROMPTS" | awk '{print $1}')
if [[ "$ACTUAL_PROMPTS_SHA256" != "$EXPECTED_PROMPTS_SHA256" ]]; then
  echo "final523 prompt hash mismatch: $ACTUAL_PROMPTS_SHA256" >&2
  exit 1
fi

"$PYTHON" - "$CANARY_SUMMARY" "$ADAPTER/run_manifest.json" <<'PY'
import json
import sys
from pathlib import Path

canary = json.loads(Path(sys.argv[1]).read_text())
training = json.loads(Path(sys.argv[2]).read_text())
if not canary.get("canary_passed"):
    raise SystemExit("protected routed DPO-dev did not pass; final523 is forbidden")
if training.get("status") != "completed" or training.get("final_global_step") != 13:
    raise SystemExit("v5 training manifest is incomplete")
PY

if [[ -f "$RUN_MANIFEST" ]] && "$PYTHON" - "$RUN_MANIFEST" <<'PY'
import json
import sys
from pathlib import Path
raise SystemExit(0 if json.loads(Path(sys.argv[1]).read_text()).get("status") == "completed" else 1)
PY
then
  echo "final523 run already completed; refusing to execute it twice"
  exit 0
fi

"$PYTHON" - "$RUN_MANIFEST" "$PROMPTS" "$ADAPTER" <<'PY'
import datetime
import hashlib
import json
import sys
from pathlib import Path

manifest_path, prompts, adapter = map(Path, sys.argv[1:])
started_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
if manifest_path.exists():
    prior = json.loads(manifest_path.read_text())
    started_at = prior.get("started_at", started_at)
payload = {
    "status": "running",
    "started_at": started_at,
    "prompts": str(prompts),
    "prompts_sha256": hashlib.sha256(prompts.read_bytes()).hexdigest(),
    "adapter": str(adapter),
    "protocol": {
        "rows": 523,
        "temperature": 0.0,
        "top_p": 1.0,
        "repetition_penalty": 1.05,
        "max_tokens": 2048,
        "seed": 42,
        "paired_timeout": 60,
        "paired_workers": 2,
        "router": "apps_protected_rubric_router_v1",
    },
}
manifest_path.write_text(json.dumps(payload, indent=2) + "\n")
PY

if ! jsonl_complete "$BASE_RESPONSES"; then
  rm -f "$BASE_RESPONSES"
  "$PYTHON" -m src.generation.vllm_smoke_generate \
    --model "$MODEL" \
    --input "$PROMPTS" \
    --output "$BASE_RESPONSES" \
    --limit 523 \
    --temperature 0.0 \
    --top-p 1.0 \
    --repetition-penalty 1.05 \
    --max-tokens 2048 \
    --max-model-len 8192 \
    --gpu-memory-utilization "${EVAL_GPU_MEMORY_UTILIZATION:-0.35}" \
    --prompt-batch-size "${EVAL_PROMPT_BATCH_SIZE:-32}" \
    --k 1 \
    --seed 42
fi

if ! jsonl_complete "$ADAPTER_RESPONSES"; then
  rm -f "$ADAPTER_RESPONSES"
  "$PYTHON" -m src.generation.vllm_lora_generate \
    --model "$MODEL" \
    --adapter "$ADAPTER" \
    --input "$PROMPTS" \
    --output "$ADAPTER_RESPONSES" \
    --temperature 0.0 \
    --top-p 1.0 \
    --repetition-penalty 1.05 \
    --max-tokens 2048 \
    --max-model-len 8192 \
    --gpu-memory-utilization "${EVAL_GPU_MEMORY_UTILIZATION:-0.35}" \
    --prompt-batch-size "${EVAL_PROMPT_BATCH_SIZE:-32}" \
    --max-lora-rank 16 \
    --seed 42
fi

"$PYTHON" -m src.evaluator.route_apps_responses_by_protected_rubric \
  --base-input "$BASE_RESPONSES" \
  --candidate-input "$ADAPTER_RESPONSES" \
  --output "$ROUTED_RESPONSES" \
  --manifest "$ROUTE_MANIFEST" \
  --expected-rows 523

if [[ ! -f "$PAIRED_MANIFEST" ]]; then
  "$PYTHON" -m src.verification.verify_paired_apps_dpo_dev \
    --base-input "$BASE_RESPONSES" \
    --candidate-input "$ROUTED_RESPONSES" \
    --base-output "$BASE_LABELED" \
    --candidate-output "$ROUTED_LABELED" \
    --manifest "$PAIRED_MANIFEST" \
    --expected-rows 523 \
    --timeout 60 \
    --workers 2 \
    --process-start-method spawn
fi

"$PYTHON" -m src.evaluator.compare_apps_dpo_v2_final \
  --base-labeled "$BASE_LABELED" \
  --candidate-labeled "$ROUTED_LABELED" \
  --training-preferences "$TRAINING_PREFERENCES" \
  --dpo-dev "$DPO_DEV" \
  --route-manifest "$ROUTE_MANIFEST" \
  --output "$FINAL_SUMMARY" \
  --report "$FINAL_REPORT"

"$PYTHON" - "$RUN_MANIFEST" "$BASE_RESPONSES" "$ADAPTER_RESPONSES" "$ROUTED_RESPONSES" "$ROUTE_MANIFEST" "$PAIRED_MANIFEST" "$FINAL_SUMMARY" <<'PY'
import datetime
import hashlib
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
payload = json.loads(manifest_path.read_text())
artifacts = {}
for value in sys.argv[2:]:
    path = Path(value)
    if not path.is_file():
        raise SystemExit(f"missing final artifact: {path}")
    artifacts[str(path)] = {
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
summary = json.loads(Path(sys.argv[-1]).read_text())
if summary.get("status") != "completed":
    raise SystemExit("final summary is incomplete")
payload.update(
    {
        "status": "completed",
        "completed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "artifacts": artifacts,
        "combined_metrics": summary["combined"],
        "final_checks": summary["final_checks"],
    }
)
manifest_path.write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
PY
