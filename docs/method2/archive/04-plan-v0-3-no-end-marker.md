# Method 2 APPS v0.3 No End Marker

## Reason

v0.2 reduced `finish_reason=length`, but introduced early empty outputs:

- v0.2 repair gate: `13/38 = 34.2%`
- `empty_revised_code`: `9/38`
- those empty outputs had `generated_token_count=4` and stopped on `END_REVISED_CODE`

So v0.3 removes `END_REVISED_CODE` from both training targets and generation stop sequences.

## Changes

- Keep the v0.1-style completion ending: no explicit end marker.
- Add a prompt-final answer delimiter: `Repair response:`.
- Keep prompt-echo stops only:
  - `\nPublic task prompt:`
  - `\nPrevious failed code:`
- Preserve v0.1 training coverage by default: no revised-code length filter unless `MAX_REVISED_CODE_CHARS` is explicitly set.

## Data

Command:

```bash
scripts/method2/build_method2_apps_self_play_bootstrap_v0_3_no_end_marker.sh
```

Result:

- SFT: `data/sft/method2_apps_self_play_critic_repair_v0_3_no_end_marker.jsonl`
- DPO pairs: `data/preferences/method2_apps_self_play_critic_repair_pairs_v0_3_no_end_marker.jsonl`
- summary: `data/self_play/method2_apps_self_play_bootstrap_v0_3_no_end_marker_summary.json`
- rows: `373`
- split: `335 train / 38 validation`
- skipped: `2 duplicate_pair`
- all prompts end with `Repair response:`
- no completion contains `END_REVISED_CODE`

## Token Dry Run

`max_length=4096`, raw prompt format:

- train rows after token gate: `330/335`
- validation rows after token gate: `38/38`
- train total token p95/p99: `2629 / 2837`
- train completion token p95/p99: `308 / 2102`
- validation total token p99: `2971`

This matches the v0.1 training shape closely while avoiding v0.2's early stop target.

## Next Commands

Run when a GPU is available:

```bash
GPU=1 scripts/method2/run_method2_apps_self_play_critic_repair_sft_v0_3_no_end_marker.sh
GPU=1 scripts/method2/run_method2_apps_self_play_repair_gate_v0_3_no_end_marker.sh
```

Expected comparison target:

- must beat v0.2 `13/38`
- should approach or beat v0.1 postextract `23/38`
- should reduce v0.1 `finish_reason=length = 15/38` without producing v0.2-style empty code
