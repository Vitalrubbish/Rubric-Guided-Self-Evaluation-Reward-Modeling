#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT_DIR"

POLL_SECONDS=${GPU_POLL_SECONDS:-60}
MAX_USED_MIB=${GPU_MAX_USED_MIB:-10000}
MAX_UTIL_PERCENT=${GPU_MAX_UTIL_PERCENT:-10}
STABLE_POLLS=${GPU_STABLE_POLLS:-3}
CANDIDATES=${GPU_CANDIDATES:-0,1,2,3,4,5,6,7}
RUN_ID=${RUN_ID:-apps_simple_method1_dpo_lora_v1_$(date +%Y%m%d_%H%M%S)}
LOG_PATH=${LOG_PATH:-logs/${RUN_ID}.log}
RUN_RUNTIME_SMOKE=${RUN_RUNTIME_SMOKE:-1}
RUN_HELDOUT_EVAL=${RUN_HELDOUT_EVAL:-1}
SMOKE_OUTPUT_DIR=${SMOKE_OUTPUT_DIR:-outputs/${RUN_ID}_runtime_smoke2}

mkdir -p logs
IFS=',' read -r -a candidate_list <<<"$CANDIDATES"
declare -A stable_counts

echo "[$(date --iso-8601=seconds)] waiting for a GPU: used<=${MAX_USED_MIB}MiB util<=${MAX_UTIL_PERCENT}% stable_polls=${STABLE_POLLS}"
echo "[$(date --iso-8601=seconds)] candidates=${CANDIDATES} log=${LOG_PATH}"

while true; do
  snapshot=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total \
    --format=csv,noheader,nounits)
  echo "[$(date --iso-8601=seconds)] GPU snapshot"
  echo "$snapshot"

  for idx in "${candidate_list[@]}"; do
    line=$(awk -F',' -v wanted="$idx" '$1 + 0 == wanted {print; exit}' <<<"$snapshot")
    [[ -n "$line" ]] || continue
    util=$(awk -F',' '{gsub(/ /, "", $2); print $2}' <<<"$line")
    used=$(awk -F',' '{gsub(/ /, "", $3); print $3}' <<<"$line")

    if (( used <= MAX_USED_MIB && util <= MAX_UTIL_PERCENT )); then
      stable_counts[$idx]=$(( ${stable_counts[$idx]:-0} + 1 ))
    else
      stable_counts[$idx]=0
    fi

    if (( ${stable_counts[$idx]} < STABLE_POLLS )); then
      continue
    fi

    lock_path="/tmp/acm-group-3-rubric-dpo-gpu${idx}.lock"
    exec {lock_fd}>"$lock_path"
    if ! flock -n "$lock_fd"; then
      echo "[$(date --iso-8601=seconds)] GPU ${idx} passed telemetry gate but group lock is held"
      continue
    fi

    sleep 10
    confirm=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used \
      --format=csv,noheader,nounits | awk -F',' -v wanted="$idx" '$1 + 0 == wanted {print}')
    confirm_util=$(awk -F',' '{gsub(/ /, "", $2); print $2}' <<<"$confirm")
    confirm_used=$(awk -F',' '{gsub(/ /, "", $3); print $3}' <<<"$confirm")
    if (( confirm_used > MAX_USED_MIB || confirm_util > MAX_UTIL_PERCENT )); then
      echo "[$(date --iso-8601=seconds)] GPU ${idx} changed during confirmation; releasing lock"
      flock -u "$lock_fd"
      stable_counts[$idx]=0
      continue
    fi

    echo "[$(date --iso-8601=seconds)] claiming physical GPU ${idx}; starting formal DPO"
    export CUDA_VISIBLE_DEVICES="$idx"
    export SKIP_PREPARE=1
    if [[ "$RUN_RUNTIME_SMOKE" == "1" ]]; then
      echo "[$(date --iso-8601=seconds)] running two-pair DPO runtime smoke"
      set +e
      OUTPUT_DIR="$SMOKE_OUTPUT_DIR" MAX_PAIRS=2 GRAD_ACCUM=1 SAVE_STEPS=1000 \
        scripts/method1/run_method1_apps_dpo.sh 2>&1 | tee -a "$LOG_PATH"
      smoke_status=${PIPESTATUS[0]}
      set -e
      if (( smoke_status != 0 )); then
        echo "[$(date --iso-8601=seconds)] runtime smoke failed with status=${smoke_status}; formal run not started"
        exit "$smoke_status"
      fi
      if [[ ! -s "$SMOKE_OUTPUT_DIR/adapter_model.safetensors" ]]; then
        echo "runtime smoke returned success without an adapter artifact" >&2
        exit 4
      fi
      echo "[$(date --iso-8601=seconds)] runtime smoke passed; starting 1198-pair formal run"
    fi
    set +e
    scripts/method1/run_method1_apps_dpo.sh 2>&1 | tee -a "$LOG_PATH"
    status=${PIPESTATUS[0]}
    set -e
    echo "[$(date --iso-8601=seconds)] training exited with status=${status}"
    if (( status == 0 )) && [[ "$RUN_HELDOUT_EVAL" == "1" ]]; then
      echo "[$(date --iso-8601=seconds)] starting held-out APPS validation/test evaluation"
      set +e
      scripts/method1/run_method1_apps_dpo_eval.sh 2>&1 | tee -a "$LOG_PATH"
      status=${PIPESTATUS[0]}
      set -e
      echo "[$(date --iso-8601=seconds)] held-out evaluation exited with status=${status}"
    fi
    exit "$status"
  done

  sleep "$POLL_SECONDS"
done
