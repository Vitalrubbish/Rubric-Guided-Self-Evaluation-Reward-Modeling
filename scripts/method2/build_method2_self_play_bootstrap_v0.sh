#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT_DIR"

PYTHON=${PYTHON:-/data2/acm-group-3/miniconda3/envs/rubric/bin/python}

export PATH="$(dirname "$PYTHON"):$PATH"

"$PYTHON" src/self_play/build_method2_bootstrap_data.py \
  --llm-critic-pairs "${LLM_CRITIC_PAIRS:-data/self_play/llm_critic_pairs_mbpp_train_logic_n20_k5.jsonl}" \
  --proxy-pairs "${PROXY_PAIRS:-data/self_play/self_play_pairs_from_protected_revision.jsonl}" \
  --sft-output "${SFT_OUTPUT:-data/sft/method2_self_play_critic_repair_v0.jsonl}" \
  --dpo-output "${DPO_OUTPUT:-data/preferences/method2_self_play_critic_repair_pairs_v0.jsonl}" \
  --summary-output "${SUMMARY_OUTPUT:-data/self_play/method2_self_play_bootstrap_v0_summary.json}" \
  --validation-percent "${VALIDATION_PERCENT:-20}"

echo "Method2 self-play bootstrap v0 complete"
