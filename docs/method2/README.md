# Method 2 APPS Self-Play Repair

Method 2 is the active APPS route.

## Objective

Train a model to repair failed APPS solutions:

```text
public task + failed code -> ERROR_FINDINGS + REVISED_CODE
```

The verifier is used as a gate and data filter, not as hidden-test text in the prompt.

## Current Best

The best held-out repair gate result is v0.3:

- pass: `24/38 = 63.2%`
- extraction: `38/38 ok`
- failures: `12 logic_error`, `1 syntax_error`, `1 runtime_error`
- format strategy: no generated end marker; prompt ends with `Repair response:`

v0.4 successfully implemented iterative self-play data generation, but did not improve the held-out gate:

- pass: `23/38 = 60.5%`
- extraction: `38/38 ok`
- failures: `10 logic_error`, `3 timeout`, `1 syntax_error`, `1 runtime_error`

So v0.3 remains the best checkpoint for gate pass rate. v0.4 is important as a working self-play iteration implementation.

## Version History

- `01-route-switch.md`: why the project switched from Method 1 to Method 2.
- `02-v0-1-clean-baseline.md`: first APPS clean repair baseline.
- `03-v0-2-end-marker-failure.md`: explicit `END_REVISED_CODE` failed by causing early empty outputs.
- `04-v0-3-no-end-marker-best.md`: current best repair gate result.
- `05-v0-4-iterative-selfplay.md`: iterative self-generated repair loop result.
- `06-v0-5-selective-stop50.md`: small stop-finished self-play data mixing canary.

Planning notes and superseded intermediate notes are under `docs/method2/archive/`.

## Canonical Scripts

Build v0.3 data:

```bash
scripts/method2/build_method2_apps_self_play_bootstrap_v0_3_no_end_marker.sh
```

Train v0.3:

```bash
GPU=1 scripts/method2/run_method2_apps_self_play_critic_repair_sft_v0_3_no_end_marker.sh
```

Evaluate v0.3:

```bash
GPU=1 scripts/method2/run_method2_apps_self_play_repair_gate_v0_3_no_end_marker.sh
```

Run the v0.4 iterative loop:

```bash
GPU=1 scripts/method2/run_method2_apps_self_play_generate_train_candidates_v0_4.sh
scripts/method2/build_method2_apps_self_play_sft_v0_4_iterative.sh
GPU=1 scripts/method2/run_method2_apps_self_play_critic_repair_sft_v0_4_iterative.sh
GPU=1 scripts/method2/run_method2_apps_self_play_repair_gate_v0_4_iterative.sh
```

Or run the wrapper:

```bash
GPU=1 scripts/method2/run_method2_apps_self_play_v0_4_iterative_full.sh
```

## Next Direction

Do not add stricter format controls. Format is already stable in v0.3/v0.4.

The next useful work is selective self-play data mixing:

- avoid full-weight mixing of all self-generated rows;
- prefer shorter `finish=stop` generated repairs;
- avoid generated rows with `finish=length`;
- try small canaries with `MAX_GENERATED_TOTAL=50` or `100`;
- target prompts or failure families that v0.3 still misses.

Current selective canary:

```bash
scripts/method2/build_method2_apps_self_play_sft_v0_5_stop50.sh
GPU=1 scripts/method2/run_method2_apps_self_play_critic_repair_sft_v0_5_stop50.sh
GPU=1 scripts/method2/run_method2_apps_self_play_repair_gate_v0_5_stop50.sh
```
