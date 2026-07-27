# Project Status And Next Steps

Date: 2026-07-25
Scope: overall progress against `docs/task.md` (Homework 3 + 4), measured
conclusions, current blockers, and the prioritized path forward.

## 1. Goals

- **Homework 3**: error-pattern discovery -> auto rubric -> self-evaluation
  consistency (coverage / AUC / Cohen's Kappa, auto vs human vs random
  rubric).
- **Homework 4**: self-evolving loop. Method 1 (rubric -> DPO), Method 2
  (self-play error discovery), Method 3 (meta-learning, deferred).
- **Converged research question**: can a model self-evolve from its own
  errors, and which components are scale walls vs structural walls?

## 2. Completed Work (with measured results)

### Phase 1: error discovery (done)

- APPS train introductory, 2613 prompts, Qwen2.5-7B k=1 generation,
  verifier pass 42.44%; 1219 safe non-length failures.
- 9-category audited taxonomy (all audits valid). Handoff:
  `data/analysis/apps_simple_phase1/..._taxonomy_refined_for_rubric.yaml`.

### Phase 2: rubric + self-evaluation (done; rubric route falsified)

- 9-dimension audited rubric:
  `data/rubrics/apps_simple_phase2/apps_train_simple_llm_rubric_from_refined_taxonomy.json`.
- Forced-choice logprob judge ablation (300 paired responses, base 7B),
  `docs/phase2/03-rubric-ablation-pilot300.md`:
  - no_rubric AUC 0.8145 / Kappa 0.507 (strongest self-eval signal);
  - auto_rubric 0.7372 vs random_rubric 0.7440: **rubric content carries
    zero judging information**; any injected rubric hurts at 7B
    (dilution + content cost, monotone no > short > random ≈ auto).
- Same ablation at 32B (Coder-32B-AWQ): no_rubric AUC 0.8700 / Kappa
  0.640; dilution penalty gone; **auto 0.8577 vs random 0.8602 — still no
  separation**. Rubric-content uselessness is scale-invariant so far.

### Measurement infrastructure (done)

- 200-row leakage-free gate (original 38 + 162 expansion);
  `docs/method2/07-gate200-selection-bias.md`. The old 38-row gate was
  selection-biased (repairability-enriched): v0.3 measured 57.9% there
  vs 8.6% on the natural failure distribution.
- Paired bootstrap / McNemar / Wilson CI tooling:
  `src/analysis-reporting/bootstrap_method2_repair_gate_stats.py`.
- 100-row frontier-LLM gold logic-error attribution set (gate-clean):
  `data/annotation/apps_simple_logic_attribution_100.jsonl`
  (97 high / 3 medium confidence; LLM annotation, human spot-check still
  recommended).

### Method 1 (done, archived negative)

Direct generator SFT/DPO repeatedly regressed; archive retained as
negative results (`docs/method1/`).

### Method 2 at 7B (done, plateau measured)

- Gate-200 trajectory: v0.3 18.0% -> v0.4 20.5% (+5, P=96.8%) -> v0.6
  18.0% (−5, P(not better)=98.9%). **The loop oscillates; it does not
  compound.** Iteration variance ±5-6 rows; net accumulation zero.
  `docs/method2/08-v0-6-second-iteration-plateau.md`.
- Gold-100 telemetry (v0.4): attribution hit 30%, hit-or-partial 74%,
  logic repair 5%; repair rate by attribution grade is FLAT
  (hit 3.3% / partial 6.8% / miss 3.8%) — **ERROR_FINDINGS barely drive
  REVISED_CODE at 7B; the wall is repair capability, not error finding.**

### 32B static capability (done, inference-only, AWQ)

| metric | 7B (best, trained) | 32B (bare, no training) |
| --- | --- | --- |
| gate200 repair | 20.5% | **48.5%** |
| gold100 logic repair | 5% | **27%** |
| attribution hit | 30% | 46% |
| **repair given hit** | **3.3%** | **43.5%** |
| judge AUC (no rubric) | 0.8145 | 0.8700 |
| judge Kappa | 0.507 | 0.640 |

The findings->repair causal link, absent at 7B, is established at 32B.

## 3. Conclusions So Far

| wall | verdict |
| --- | --- |
| repair capability | scale wall (5% -> 27%) |
| judge capability | scale wall (AUC 0.81 -> 0.87, Kappa 0.51 -> 0.64) |
| rubric content validity | **structural** (auto ≈ random at 7B and 32B) |
| self-play compounding | falsified at 7B; **undecided at 32B** |

Interim thesis: small-scale self-evolution is information-bounded (the
loop has no new-information channel beyond 1-bit verifier selection);
scale raises the ceiling but does not remove it. The 32B loop is the
decisive test of how high that ceiling sits.

## 4. Current Blockers

- **GPU memory**: cluster 8xA800 all occupied by other tenants; max free
  ~43GB. bf16 32B training/inference needs ~70GB free.
- AWQ quantization caveat: all 32B numbers are AWQ 4-bit; headline values
  need one bf16 confirmation run when a GPU frees.
- Human-rubric upper-bound row of the Homework 3 metrics table still
  empty (report completeness item).

## 5. Next Steps (prioritized)

**Strategy v2 (main line)**: `docs/strategy-v2-executable-rubric.md` —
replace natural-language rubrics with model-written EXECUTABLE tests as
the reward signal, 32B Coder as the new baseline, test library as the
evolving artifact. First step: test-quality probe (Section 6 of that
doc).

1. ~~32B self-play loop r0 -> r1 -> r2~~ **DONE (2026-07-26, QLoRA)**:
   plateau confirmed — 48.5 -> 46.0 -> 48.5 -> 48.5, telemetry flat,
   candidate pass rate saturated (83.7%/83.9%). Full writeup:
   `docs/method2/10-coder32b-self-play-plateau.md`. The verifier-filtered
   self-play loop does not compound at 7B or 32B.
2. ~~Doc-faithful DPO arm~~ **DONE (2026-07-26)**: 46.5% vs bare 48.5%
   (CI [-13,+5], flat-to-slightly-negative). Method 2 falsified at 32B in
   both SFT and preference-pair forms (see doc 10 addendum).
3. ~~Execution-feedback probe + internalization~~ **DONE (2026-07-26)**:
   in-context feedback WORKS (+6.5pp, 55.0% vs 48.5% ceiling, p~1e-4),
   but r1F distillation does NOT internalize it (46.5%, flat). Conclusion:
   the gain is information-theoretic, not capability-theoretic; evolution
   is viable at the SYSTEM level (model+executor), not at the weight
   level. See `docs/method2/11-execution-feedback-probe.md`.
4. **Self-generated curriculum**: model-generated problem variants to
   escape the saturated 335-prompt pool.
5. **v0.7 external-signal arm**: gold diagnosis->repair demonstrations vs
   the pure self-play trajectory — the controlled answer to "which errors
   need external signals".
6. **72B-AWQ judge ablation probe** (~1h, when ~41GB+ contiguous frees):
   last data point for the rubric question (auto vs random separation).
7. **bf16 confirmation** of 32B headline numbers when a GPU frees.
8. **Human-rubric arm** to fill the last Homework 3 table row.
9. **Report writing**: selection-biased gate correction, plateaus at two
   scales, rubric ablation negative result, findings->repair disconnect
   at 7B and its recovery at 32B, saturated self-play at 32B.

## 6. Key Artifact Index

- docs: `docs/phase2/03-rubric-ablation-pilot300.md`,
  `docs/method2/07-gate200-selection-bias.md`,
  `docs/method2/08-v0-6-second-iteration-plateau.md`,
  `docs/method2/09-coder32b-self-play-runbook.md`,
  `docs/method2/10-coder32b-self-play-plateau.md`
- gates: `data/sft/method2_apps_self_play_critic_repair_gate200_validation.jsonl`
- gold set: `data/annotation/apps_simple_logic_attribution_100.jsonl`
- telemetry scorings: `data/annotation/apps_simple_gold_scoring_{v0_4,v0_6,coder32b}_merged.jsonl`
- 32B results: `data/evaluator/..._coder32b_awq.json`,
  `data/self_play/..._coder32b_awq_*.jsonl`
- models: `models/models--Qwen--Qwen2.5-Coder-32B-Instruct` (bf16),
  `...-AWQ`, `models--Qwen--Qwen2.5-72B-Instruct-AWQ`
