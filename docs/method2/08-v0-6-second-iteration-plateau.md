# Method 2 v0.6: Second Self-Play Iteration — Plateau Result

Date: 2026-07-23

## Design

v0.6 is the second iteration of the self-evolution loop, replicating the
v0.4 recipe exactly with only the starting checkpoint changed:

- v0.4 adapter generated K=5 repair candidates on the same 335 train
  prompts (1675 candidates, 961 verifier-passing = 57.4%);
- SFT: base 373 + 268 accepted self-generated rows = 641 rows, LoRA
  retrained from base (75 steps, eval loss 2.109);
- acceptance: 200-row gate with paired bootstrap vs v0.4, plus gold-100
  telemetry (frontier-LLM-scored attribution grades).

## Gate-200 Trajectory

| version | pass rate | paired bootstrap vs previous |
| --- | --- | --- |
| v0.3 | 36/200 = 18.0% | — |
| v0.4 | 41/200 = 20.5% | +5 rows, CI95 [0, +10], P(better)=96.8% |
| v0.6 | 36/200 = 18.0% | **−5 rows, CI95 [−10, 0], P(not better)=0.989** |

v0.6 transitions vs v0.4: 6 P→F, 1 F→P (McNemar p=0.125) — an almost exact
mirror of v0.4's 6 F→P / 1 P→F over v0.3. Extraction 200/200 ok in all
versions; no timeout flakes in this comparison.

**Verdict: the loop oscillates, it does not compound.** The trajectory
36 → 41 → 36 shows iteration-to-iteration variance of ±5-6 rows with no
net accumulation. The v0.4 gain was real for that sample but not a stable
capability increment; retraining from base on a slightly different
641-row mix re-rolls the same borderline cases.

## Gold-100 Telemetry (v0.4 → v0.6)

Attribution grades (frontier-LLM scored against gold root causes):

| grade | v0.4 | v0.6 |
| --- | --- | --- |
| hit | 30 | 32 |
| partial | 44 | 42 |
| miss | 26 | 26 |

76/100 grades unchanged; transitions are symmetric churn. Category match
73% in both. Attribution ability is flat across the iteration.

Conditional repair rate by attribution grade:

| grade | v0.4 | v0.6 |
| --- | --- | --- |
| hit | 1/30 = 3.3% | 3/32 = 9.4% |
| partial | 3/44 = 6.8% | 2/42 = 4.8% |
| miss | 1/26 = 3.8% | 0/26 = 0.0% |
| overall | 5/100 | 5/100 |

Overall logic repair stays at 5%. (v0.6 shows repair|hit > repair|miss,
a directional hint that the findings→repair link strengthened, but n is
far too small to conclude.)

## Key Structural Finding (from v0.4 telemetry)

Repair success is nearly independent of stated attribution
(hit 3.3% / partial 6.8% / miss 3.8% in v0.4): **ERROR_FINDINGS barely
drive REVISED_CODE on logic errors**. The critic is far ahead of the
repairer (30% hit vs ~5% repair), and the repairer does not use the
diagnosis even when it is correct. The logic wall is a repair-capability
wall, not an error-finding wall.

## Conclusions

1. Pure self-play iteration (STaR-style, fixed 335 prompts, verifier
   filter) plateaus at iteration 2: the loop has no channel to inject
   information the model does not already possess.
2. The capability ceiling is logic repair (~5%); critic accuracy (74%
   hit-or-partial) is not the binding constraint.
3. Remaining levers, in order of expected value:
   - repair-side demonstrations that map correct diagnosis → correct fix
     (gold repair data, i.e. the v0.7 external-signal arm);
   - larger K / new prompts to widen the candidate distribution;
   - process supervision linking findings to repairs.
4. v0.4 remains the best checkpoint (20.5% gate-200). v0.6 is archived as
   the plateau measurement of the self-evolution loop.

## Artifacts

- candidates: `data/self_play/method2_apps_self_play_v0_6_train_candidates_*.jsonl`
- SFT: `data/sft/method2_apps_self_play_critic_repair_v0_6_iterative.jsonl`
- adapter: `outputs/method2_apps_self_play_critic_repair_sft_lora_v0_6_iterative`
- gate200: `data/self_play/method2_apps_self_play_gate200_v0_6_validation_*.jsonl`,
  `..._bootstrap_v0_6_vs_v0_4.json`
- telemetry: `data/self_play/method2_gold100_telemetry_v0_6_*.jsonl`,
  `data/annotation/apps_simple_gold_scoring_v0_6_merged.jsonl`
