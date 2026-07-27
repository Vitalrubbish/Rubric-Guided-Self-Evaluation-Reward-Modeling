# Method 2 Gate-200: Selection Bias Correction And Version Reranking

Date: 2026-07-23

## Purpose

The original 38-row repair gate had two problems: statistical power
(+-15pp Wilson half-width) and, more importantly, **selection bias** — its
rows came from problems where a passing K=5 repair already existed, i.e. a
repairability-enriched slice of the failure distribution. This document
records the expanded gate and the corrected Method 2 version ranking.

## Expanded Gate Construction

Built by `src/self_play/build_method2_gate_expansion.py`:

- 38 original validation rows kept verbatim (backward-comparable subset);
- 162 new rows sampled (seed 42) from the 998 non-length failed responses
  whose problem ids never appear in the v0.3 SFT build (280 problems), which
  also covers all v0.4/v0.5 generated rows (derived from the same base train
  prompts) — no leakage for any version;
- new-row failure mix: 99 logic_error, 33 syntax_error, 20 runtime_error,
  10 timeout (matches the natural failure distribution).

Artifacts: `data/sft/method2_apps_self_play_critic_repair_gate200_validation.jsonl`,
`data/self_play/method2_apps_self_play_gate200_build_summary.json`.

## Selection Bias Finding (v0.3)

| subset | pass rate | Wilson CI95 |
| --- | --- | --- |
| original 38 | 22/38 = 57.9% | [42.2%, 72.1%] |
| expansion 162 | 14/162 = 8.6% | [5.2%, 14.0%] |
| full 200 | 36/200 = **18.0%** | [13.3%, 23.9%] |

The subset CIs do not overlap: the old gate's 63.2% headline was mostly a
selection-bias artifact. v0.3's true repair rate on the natural failure
distribution is ~18%.

Reproducibility: 36/38 original rows agree with the historical run; both
flips are timeout verifier jitter.

Repair rate by original failure type (expansion subset): syntax 24.2%,
runtime 10.0%, timeout 10.0%, **logic 3.0%** — self-repair on unseen
problems is concentrated on syntax-level failures.

## Corrected Version Ranking (200-row gate)

| version | pass rate | Wilson CI95 | paired bootstrap vs v0.3 |
| --- | --- | --- | --- |
| v0.3 | 36/200 = 18.0% | [13.3%, 23.9%] | — |
| v0.4 | 41/200 = 20.5% | [15.5%, 26.6%] | +5 rows, CI95 [0, +10], P(not better)=0.032 |
| v0.5 | 39/200 = 19.5% | [14.6%, 25.5%] | +3 rows, CI95 [−2, +8], P(not better)=0.165 |

Transitions vs v0.3: v0.4 fixed 6 and broke 1 (McNemar p=0.125); v0.5
fixed 5 and broke 2 (p=0.453). No timeout flakes in either rerun.

**The 38-row gate told the wrong story.** On the unbiased gate, the
self-play iterations were mildly helpful, not harmful: v0.4 is the best
checkpoint with a borderline-significant +2.5pp, and v0.5 sits between
v0.3 and v0.4. The logic_error wall is untouched by all versions
(3-4/99 repaired).

Analysis artifacts:
`data/self_play/method2_apps_self_play_gate200_v0_3_analysis.json`,
`data/self_play/method2_apps_self_play_gate200_bootstrap_stats.json`.

## Reporting Rule Going Forward

- All Method 2 gate comparisons run on the 200-row gate and report Wilson
  CI plus paired bootstrap differences; single-point pass rates on the
  38-row subset are not acceptable evidence.
- The old `min_pass_rate=0.20` gate threshold was calibrated on the biased
  distribution and is retired; decisions use bootstrap comparisons instead.

## Consequences

- v0.4 iterative self-play mixing is rehabilitated as a mildly positive
  result; the "selective mixing" motivation for v0.5/v0.5b was based on a
  biased measurement.
- v0.5b (targeted50) should still be trained and gated, but judged on the
  200-row gate against v0.4, not v0.3.
- The dominant open problem is logic_error repair (3-4%): binary verifier
  reward gives no gradient here, and prompt-injected rubrics are proven
  useless (see `docs/phase2/03-rubric-ablation-pilot300.md`).
