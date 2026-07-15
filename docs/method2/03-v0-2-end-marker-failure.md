# Method 2 APPS v0.2 Format Stability

## Purpose

v0.2 keeps the Method 2 objective unchanged: given a failed APPS solution, output `ERROR_FINDINGS` and one corrected `REVISED_CODE`.

This version only strengthens format stability:

- add explicit `END_REVISED_CODE` to SFT completions;
- stop vLLM generation on `END_REVISED_CODE` and prompt-echo markers;
- reject overly long revised-code targets during bootstrap construction;
- make revised-code extraction handle marker variants, end markers, inline repeated markers, and direct-code fallback.

## Data Build

Command:

```bash
scripts/build_method2_apps_self_play_bootstrap_v0_2_format.sh
```

Result:

- SFT: `data/sft/method2_apps_self_play_critic_repair_v0_2_format.jsonl`
- DPO pairs: `data/preferences/method2_apps_self_play_critic_repair_pairs_v0_2_format.jsonl`
- summary: `data/self_play/method2_apps_self_play_bootstrap_v0_2_format_summary.json`
- rows: `363`
- split: `325 train / 38 validation`
- skipped: `10 revised_code_too_long`, `2 duplicate_pair`
- all `363` completions end with `END_REVISED_CODE`

## Token Dry Run

`max_length=4096`, raw prompt format:

- train rows after token gate: `323/325`
- validation rows after token gate: `38/38`
- train total token p95/p99: `2627 / 2807`
- validation total token p99: `2973`

## Extractor Regression Replay

Using the old v0.1 validation generations with the strengthened extractor:

- pass rate: `24/38 = 63.2%`
- prior post-extract replay was `23/38 = 60.5%`
- syntax failures reduced to `2`
- remaining format misses: `5 missing_revised_code_marker`

The remaining misses are generation-structure failures, so v0.2 should be judged by the new SFT adapter plus repair gate, not by extractor replay alone.

## Next Commands

```bash
GPU=1 scripts/run_method2_apps_self_play_critic_repair_sft_v0_2_format.sh
GPU=1 scripts/run_method2_apps_self_play_repair_gate_v0_2_format.sh
```
