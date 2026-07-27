# Method 2 APPS v0.5b Targeted50

## Purpose

v0.5b keeps the v0.5 canary size but replaces the final selection rule.

v0.5 selected the first 50 clean verifier-passing `finish_reason=stop` repairs after sorting by base SFT id. That kept the data small, but it concentrated the added rows in an early APPS id range and did not improve any v0.3 validation failures.

v0.5b tests whether a cleaner and broader 50-row mix is more useful.

## Selection Policy

v0.5b keeps generated rows only when:

- verifier passed;
- Method 2 extraction status is `ok`;
- `finish_reason == stop`;
- generated code is parseable;
- generated code is not a duplicate of the base SFT target;
- at most one generated repair is selected per original train prompt;
- the raw completion contains at least 2 explicit `ERROR_FINDINGS`;
- extraction notes are at most 1;
- the raw completion contains only one `ERROR_FINDINGS`/`REVISED_CODE` pair.

After filtering, v0.5b selects up to 50 rows with round-robin balancing over:

```text
selection_reason + problem_id decile
```

This avoids the v0.5 behavior where the cap mostly picked the earliest eligible APPS ids.

## Data Build

Command:

```bash
scripts/method2/build_method2_apps_self_play_sft_v0_5b_targeted50.sh
```

Current result:

- base rows: `373`
- generated selected rows: `50`
- combined rows: `423`
- split: `385 train / 38 validation`
- generated finish counts: `50 stop`
- generated io modes: `42 function_call / 8 stdin_stdout`
- quality-filtered candidates before cap: `338`
- overlap with v0.5 selected rows: `17/50`
- v0.5 problem id range: `2409-2726`
- v0.5b problem id range: `2409-4741`

Outputs:

```text
data/sft/method2_apps_self_play_critic_repair_v0_5b_targeted50.jsonl
data/self_play/method2_apps_self_play_v0_5b_targeted50_accepted_self_generated.jsonl
data/self_play/method2_apps_self_play_v0_5b_targeted50_summary.json
```

## Next Commands

Run when a GPU is available:

```bash
GPU=1 scripts/method2/run_method2_apps_self_play_critic_repair_sft_v0_5b_targeted50.sh
GPU=1 scripts/method2/run_method2_apps_self_play_repair_gate_v0_5b_targeted50.sh
```

The v0.5b repair gate defaults to `VERIFY_WORKERS=1` to reduce timeout flakes on the 38-row validation gate.

After the gate finishes, compare transitions against v0.3:

```bash
python src/analysis-reporting/compare_method2_repair_gates.py \
  --baseline data/self_play/method2_apps_self_play_v0_3_no_end_marker_validation_labeled.jsonl \
  --candidate data/self_play/method2_apps_self_play_v0_5b_targeted50_validation_labeled.jsonl \
  --output data/self_play/method2_apps_self_play_v0_5b_targeted50_vs_v0_3_compare.json
```

## Acceptance Criteria

Compare against v0.3 and v0.5:

- v0.3 pass: `24/38 = 63.2%`
- v0.5 raw pass: `22/38 = 57.9%`
- v0.5 timeout-adjusted pass: approximately `23/38`

v0.5b is useful only if it:

- keeps extraction at `38/38 ok`;
- reaches at least `24/38`;
- has no true v0.3 `P->F` semantic regressions;
- creates at least one v0.3 `F->P` improvement;
- avoids verifier timeout regressions after the single-worker gate.
