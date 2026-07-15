# Method 2 APPS v0.4 Iterative Results

## Commands Run

```bash
GPU=1 scripts/method2/run_method2_apps_self_play_generate_train_candidates_v0_4.sh
scripts/method2/build_method2_apps_self_play_sft_v0_4_iterative.sh
GPU=1 scripts/method2/run_method2_apps_self_play_critic_repair_sft_v0_4_iterative.sh
GPU=1 scripts/method2/run_method2_apps_self_play_repair_gate_v0_4_iterative.sh
```

## Candidate Generation

- train prompts: `335`
- candidates: `1675` (`K=5`)
- extraction: `1675/1675 ok`
- verifier passed: `964/1675 = 57.6%`
- failures: `475 logic_error`, `151 runtime_error`, `57 syntax_error`, `28 timeout`

## v0.4 SFT Build

- base rows: `373`
- selected generated rows: `271`
- combined rows: `644`
- split: `606 train / 38 validation`
- generated rows are train-only
- accepted generated finish: `230 stop / 41 length`

## Training

- training rows after token gate: `598`
- validation rows: `38`
- global step: `75`
- train loss: `1.2296`
- eval loss: `2.1097`
- eval perplexity: `8.25`

## Repair Gate

- passed: `23/38 = 60.5%`
- extraction: `38/38 ok`
- finish: `23 stop / 15 length`
- failures: `10 logic_error`, `3 timeout`, `1 syntax_error`, `1 runtime_error`
- gate passed: `true`

## Comparison To v0.3

v0.3:

- passed: `24/38 = 63.2%`
- failures: `12 logic_error`, `1 syntax_error`, `1 runtime_error`
- extraction: `38/38 ok`

v0.4:

- passed: `23/38 = 60.5%`
- failures: `10 logic_error`, `3 timeout`, `1 syntax_error`, `1 runtime_error`
- extraction: `38/38 ok`

Per-row transition:

- kept pass: `23`
- kept fail: `14`
- regressed: `1`
- improved: `0`

The single regression is `apps/train/2821__method2_apps_repair_self_play_00143`, changing from pass in v0.3 to timeout in v0.4.

## Interpretation

The v0.4 iteration loop worked technically: it generated many verifier-passing repairs and trained successfully. But the first-pass mixing strategy did not improve the held-out repair gate.

Likely issue: adding one generated repair for many train prompts increases imitation of v0.3's own style, but does not target the specific failure modes in the held-out 38-row gate. The next iteration should be more selective, for example:

- add generated rows only for prompts where v0.3 originally failed or where generated candidates are shorter/stop-finished;
- downsample generated rows instead of adding 271 at full weight;
- build a canary with `MAX_GENERATED_TOTAL=50` or `100`;
- avoid candidates with `finish_reason=length` in the augmented data.
