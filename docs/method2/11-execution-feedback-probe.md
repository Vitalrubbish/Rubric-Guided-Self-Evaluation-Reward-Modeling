# Execution-Feedback Probe: First Signal Above The Factory Ceiling

Date: 2026-07-26

## Motivation

Every self-play variant (SFT rounds, doc-faithful DPO, at 7B and 32B)
left the model exactly at its bare level: the loop had no information
inflow beyond 1-bit verifier selection. This probe tests the remaining
untried channel: N-bit public-test execution feedback in the repair
prompt.

## Design

- Base: 103 gate200 rows where bare Coder-32B-AWQ failed to repair
  (no training, test-time only).
- Public cases: LLM-extracted from task text only (strict literal rule),
  then **validated by execution against the canonical solution**
  (`src/self_play/build_exec_feedback_round2.py`); hidden `input_output`
  is never shown to the model.
- Round-2 prompt: original critic+repair prompt + previous repair code +
  per-case behavior (`fn(args) -> expected X, got Y` / exception), with
  the same ERROR_FINDINGS + REVISED_CODE output contract.

Row groups (103 bare-failed):

| group | rows | meaning |
| --- | --- | --- |
| public cases available, previous repair fails publicly | 59 | feedback directly usable (probe set) |
| previous repair passes public cases, fails hidden | 13 | coverage limit of public feedback |
| no usable public cases / extraction failed | 31 | feedback blind spot |

## Results

- Recovered: **13/59 = 22.0%** (binomial CI95 [12%, 35%]); all 59 started
  as failures, 13 rescued, 0 regressed -> paired p ~= 1e-4.
- Projected gate200: **110/200 = 55.0%** vs the 48.5% ceiling that no
  training variant ever moved (paired bootstrap confirmed all of
  r0/r1/r2/DPO in the 46-48.5% band).
- Extraction: 58/59 ok.

## Reward-Hacking Audit

- Final scoring is on the hidden suite, so public-case hardcoding cannot
  inflate the metric.
- Literal scan of all 13 recovered repairs: **0 contain public input
  literals** (len >= 6). No hardcoding evidence.
- For the distillation stage: training rows must additionally pass a
  generality filter (no public input literals) on top of the hidden gate.

## Interpretation

- The information-starvation diagnosis is confirmed: given concrete
  counterexamples, the model repairs a fifth of failures it could not
  repair blind. This is the first measured movement above the factory
  ceiling in the project.
- But 78% of publicly-visible failures are true capability ceilings, and
  public feedback covers only 59/103 failures — the channel is real but
  bounded (max realistic rescue ~20-25 rows even at full coverage).

## Next: Internalization Test (Design A)

Does feedback-assisted success internalize into weights? Recipe:

1. Run feedback rounds on the FAILED r1 self-play candidates (335 train
   prompts, leakage-free w.r.t. gate200/gold100), collecting
   hidden-passing "feedback-recovered" repairs.
2. Build r1F SFT: r1 recipe (base 373 + 273 self-generated) **plus**
   feedback-recovered rows (generality-filtered).
3. Train QLoRA from base, evaluate gate200 WITHOUT feedback.
4. Verdict: r1F significantly > r1 (48.5%) -> execution-derived
   information internalizes (a working self-evolution channel);
   flat -> feedback helps only in-context, evolution stays systemic
   (model + executor), not weight-borne.

## Artifacts

- probe: `data/self_play/exec_feedback_probe_round2_{input,generations,extracted,labeled}.jsonl`,
  `..._summary.json`
- public tests: `data/self_play/exec_feedback_tests/part*.jsonl`
- builder: `src/self_play/build_exec_feedback_round2.py`

## Internalization Test Result (r1F, 2026-07-26)

Data production: feedback round on 159 failed r1 self-play candidates
(K=2, r1 adapter): 147/318 = 46.2% hidden-pass in-context. After
generality filter (public-literal scan), candidate dedup, per-problem
cap, and excluding problems already covered by r1's accepted rows:
**+32 SFT rows** (11 on problems the loop never repaired, 21 harder
variants on covered problems). r1F SFT = r1 recipe 646 + 32 = 678 rows,
same QLoRA protocol, prompts contain NO feedback.

Result:

| model | gate200 | gold100 |
| --- | --- | --- |
| r1 | 97/200 = 48.5% | 28 |
| r1F | 93/200 = 46.5% (CI95 [−11, +3], McNemar p=0.39) | 30 |

**No internalization measured.** The feedback-derived rows did not move
bare repair capability; direction is (insignificantly) negative, in the
same 46-48.5% band as every other variant. Caveat: +32 rows is a
canary-sized perturbation (gate resolution ±5 rows); a scaled r1F-full
(100+ feedback rows) was not run after the canary showed no positive
direction.

## Final Synthesis

The execution-feedback gain is **information-theoretic, not
capability-theoretic**: the active ingredient is runtime facts about the
specific instance, which the model can already exploit in-context
(repair|hit 43.5%) — there is no missing skill for SFT to distill.
Facts about instances do not compress into weights at this data scale.

Across the whole project, every weight-update route to self-evolution is
falsified at both scales (self-play SFT loop at 7B and 32B, doc-faithful
DPO, feedback-distilled SFT), while system-level (in-context,
executor-in-the-loop) improvement is real and significant (+6.5pp).
Conclusion: at these scales, evolution is viable at the SYSTEM level
(model + executor + external memory), not at the weight level.
