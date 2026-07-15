# Method 2 APPS v0.5 Selective Stop50 Canary

## Purpose

v0.5 tests the conservative next step after v0.4:

- keep the v0.3 no-end-marker prompt and validation gate;
- reuse existing v0.4 self-generated train candidates;
- add only a small number of high-confidence generated repairs;
- avoid full-weight mixing of all verifier-passing self-generated rows.

This is a data-selection canary, not a format-control change.

## Selection Policy

The v0.5-stop50 SFT build keeps generated rows only when:

- verifier passed;
- Method 2 extraction status is `ok`;
- `finish_reason == stop`;
- generated code is parseable;
- generated code is not a duplicate of the base SFT target;
- at most one generated repair is selected per original train prompt;
- at most 50 generated repairs are added.

## Data Build

Command:

```bash
scripts/method2/build_method2_apps_self_play_sft_v0_5_stop50.sh
```

Result:

- base rows: `373`
- generated selected rows: `50`
- combined rows: `423`
- split: `385 train / 38 validation`
- generated finish counts: `50 stop`
- skipped length-finished candidates: `304`

Outputs:

```text
data/sft/method2_apps_self_play_critic_repair_v0_5_stop50.jsonl
data/self_play/method2_apps_self_play_v0_5_stop50_accepted_self_generated.jsonl
data/self_play/method2_apps_self_play_v0_5_stop50_summary.json
```

## Next Commands

Run when a GPU is available:

```bash
GPU=1 scripts/method2/run_method2_apps_self_play_critic_repair_sft_v0_5_stop50.sh
GPU=1 scripts/method2/run_method2_apps_self_play_repair_gate_v0_5_stop50.sh
```

## Acceptance Criteria

Compare against v0.3:

- v0.3 pass: `24/38 = 63.2%`
- v0.3 extraction: `38/38 ok`
- v0.3 failures: `12 logic_error`, `1 syntax_error`, `1 runtime_error`

v0.5-stop50 is useful only if it:

- keeps extraction at `38/38 ok`;
- matches or beats `24/38`;
- does not add timeout regressions;
- ideally reduces logic failures without increasing syntax/runtime failures.
