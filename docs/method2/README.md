# Method 2 APPS Self-Play Repair

Method 2 is the active APPS route.

## Objective

Train a model to repair failed APPS solutions:

```text
public task + failed code -> ERROR_FINDINGS + REVISED_CODE
```

The verifier is used as a gate and data filter, not as hidden-test text in the prompt.

## Current Best

The best held-out repair gate result on the corrected 200-row gate is v0.4:

- pass: `41/200 = 20.5%`, Wilson CI95 [15.5%, 26.6%]
- paired bootstrap vs v0.3: +5 rows, CI95 [0, +10], P(not better)=0.032

The older 38-row gate numbers (v0.3 `24/38 = 63.2%`, v0.4 `23/38 = 60.5%`,
v0.5 `22/38 = 57.9%`) are superseded: the 38-row gate was both
statistically underpowered (+-15pp) and selection-biased toward
repairability-enriched problems. On the natural failure distribution the
true repair rates are v0.3 `18.0%`, v0.4 `20.5%`, v0.5 `19.5%`, and the
self-play iterations are mildly helpful rather than harmful. See
`07-gate200-selection-bias.md` for the full analysis.

Self-repair on unseen problems is concentrated on syntax failures
(~24%); logic_error repair remains at 3-4% for all versions.

Historical 38-row details are kept in the per-version notes below, but all
current comparisons use the 200-row gate.

## Version History

- `01-route-switch.md`: why the project switched from Method 1 to Method 2.
- `02-v0-1-clean-baseline.md`: first APPS clean repair baseline.
- `03-v0-2-end-marker-failure.md`: explicit `END_REVISED_CODE` failed by causing early empty outputs.
- `04-v0-3-no-end-marker-best.md`: v0.3 format strategy (no end marker; prompt ends with `Repair response:`).
- `05-v0-4-iterative-selfplay.md`: iterative self-generated repair loop result.
- `06-v0-5-selective-stop50.md`: small stop-finished self-play data mixing canary.
- `06-v0-5b-targeted50.md`: balanced targeted 50-row variant (data built; training pending).
- `07-gate200-selection-bias.md`: expanded 200-row gate, selection-bias correction, and the corrected version ranking (v0.4 best).
- `08-v0-6-second-iteration-plateau.md`: second self-play iteration from v0.4; trajectory 36→41→36 shows the loop oscillates rather than compounds; gold-100 telemetry shows attribution flat and logic repair stuck at ~5%.

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

All evaluation now uses the 200-row gate (`07-gate200-selection-bias.md`),
with Wilson CI and paired bootstrap as the acceptance criteria.

Status after v0.6 (`08-v0-6-second-iteration-plateau.md`): pure self-play
iteration **plateaus at iteration 2** (36→41→36). The loop has no channel
to inject information the model does not already possess. Gold-100
telemetry shows the binding constraint is logic **repair** capability
(~5%), not error finding (74% hit-or-partial), and that ERROR_FINDINGS
barely drive REVISED_CODE on logic errors.

Priority order:

1. v0.7 external-signal arm: inject repair-side demonstrations that map
   correct diagnosis → correct fix (gold repair data), measured against
   the pure self-play trajectory. This is the controlled answer to
   "which errors need external signals".
2. Optional: v0.5b targeted50 can still be trained/gated for
   completeness, but selective-mixing is no longer the main question.
3. Report writing: the trajectory (bootstrap → v0.4 gain → v0.6 plateau),
   the rubric ablation negative result, and the findings→repair causal
   disconnect are the core results.
