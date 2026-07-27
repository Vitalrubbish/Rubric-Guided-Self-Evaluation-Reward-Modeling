# Phase 2 / Homework 3 Step 3: Rubric Ablation Self-Evaluation Pilot

Date: 2026-07-23

## Purpose

Measure whether the auto-discovered Phase 2 rubric adds self-evaluation
signal for the base model, using the forced-choice verdict logprob protocol
(the strongest 7B judging protocol in this repo) with the rubric as the
experimental variable.

## Protocol

- Judge: base `Qwen2.5-7B-Instruct` via chat template, no adapter, no training.
- Sample: 300 responses from
  `data/responses/apps_train_simple_executable_qwen25_k1_t2048_full_labeled_nonlength.jsonl`,
  stratified 150 pass / 150 fail, seed 42; 150 validation / 150 test halves
  for threshold selection discipline.
- Arms (same 300 responses in every arm, paired):
  - `no_rubric`: public task + submitted code only;
  - `auto_rubric`: plus the audited 9-dimension Phase 2 rubric;
  - `random_rubric`: plus a structurally identical rubric with dimension
    names deranged across contents (same token budget, zero semantic validity).
- Scoring: logprob margin of ` PASS\n` vs ` FAIL\n` after `Verdict:`.
- Scripts:
  - `src/evaluator/build_rubric_ablation_judge_data.py`
  - `src/evaluator/score_rubric_ablation_logprob.py`
- Artifacts:
  - `data/evaluator/apps_simple_rubric_ablation_pilot300_judge_input.jsonl`
  - `data/evaluator/apps_simple_rubric_ablation_pilot300_scores.jsonl`
  - `data/evaluator/apps_simple_rubric_ablation_pilot300_summary.json`

## Homework 3 Metrics Table

| 方法 | 错误模式覆盖率 | Rubric 区分度（AUC） | 自评与外部一致性 |
| --- | --- | --- | --- |
| 人类编写 rubric（upper bound） | 未评估 | 未评估 | 未评估 |
| 模型自动发现 rubric | 1219/1219 失败全覆盖（9 类，审计 valid=true） | 0.7372；与随机 rubric 差 −0.007，CI95 [−0.018, +0.004] → **增量区分度为零** | Kappa 0.280（零阈值）/ 0.333（验证集选阈值，test） |
| 随机 rubric（ablation） | 不适用 | 0.7440 | Kappa 0.273（零阈值）/ 0.413（test） |
| 无 rubric 裸判（参照行） | 不适用 | **0.8145** | Kappa 0.273（零阈值）/ **0.507**（test） |

Paired bootstrap AUC differences (10000 iterations, resampling response ids
jointly):

| 对比 | AUC 差 CI95 | P(a ≤ b) |
| --- | --- | --- |
| auto − no_rubric | [−0.1107, −0.0459] | 1.0 |
| auto − random | [−0.0182, +0.0041] | 0.8902 |
| no_rubric − random | [+0.0395, +0.1024] | 0.0001 |

All arms show the known 7B overacceptance bias at zero threshold
(overacceptance 0.60–0.63, false rejection 0.09–0.12).

## Conclusions

1. The auto-discovered rubric carries **no usable judging information**: it
   is indistinguishable from a semantically shuffled rubric of identical
   structure.
2. Injecting any long rubric block **hurts** the base model's judgment
   (AUC 0.81 → 0.74). The cost is attention dilution, not rubric content.
3. The base model's intrinsic self-evaluation ranking signal is the
   strongest measured in this project so far (AUC 0.8145 on this sample).
4. Implication for Homework 4: rubric-as-reward via **prompt injection is a
   dead end at 7B**. Remaining routes for the rubric to matter: distill it
   into weights, or decompose it (one dimension per judging pass).

## V2 Follow-Up: Short-Rubric Arm (Mechanism Adjudication)

A fourth arm `short_rubric` (dimension names only, no definitions or
checklists) was added to separate length dilution from content cost, on the
same 300 paired responses. The three original arms reproduced exactly
(AUC 0.8145 / 0.7372 / 0.7440), confirming protocol determinism.

| arm | AUC | test Kappa (selected threshold) |
| --- | --- | --- |
| no_rubric | 0.8145 | 0.507 |
| short_rubric | 0.7917 | 0.467 |
| random_rubric | 0.7440 | 0.413 |
| auto_rubric | 0.7372 | 0.333 |

| 对比 | AUC 差 CI95 | 判决 |
| --- | --- | --- |
| short − no | [−0.0418, −0.0045], P=0.9917 | short still significantly worse than no rubric |
| short − auto | [+0.0301, +0.0814], P≈0 | short significantly better than full rubric |

Verdict: both mechanisms exist and stack monotonically
(no > short > random ≈ auto). Length dilution is the main effect (full →
names-only recovers +0.055); a smaller pure content cost remains (even
dimension names alone hurt by −0.023). Rubric semantic validity contributes
nothing at any granularity.

**Rubric-as-judge route closed**: the best any prompt-injected rubric can
achieve is to approach the no-rubric baseline, never exceed it. The rubric
remains valuable as the documented taxonomy artifact and as potential
training signal, not as prompt context.

Artifacts: `data/evaluator/apps_simple_rubric_ablation_pilot300_v2_*.json*`.

## Caveats

- 300-row pilot; AUC standard errors are roughly ±0.03 per arm, but arm
  deltas are paired and tightly estimated.
- Rubric construction and evaluation share the same response distribution
  (APPS train introductory). This inflates absolute auto_rubric numbers if
  anything; arm deltas are internally valid because all arms see identical
  rows.
- Human-rubric upper bound not yet run; it needs a hand-written checklist
  evaluated under the same protocol.

## Next Options

- Single-dimension decomposition arm: judge one rubric dimension per pass
  and aggregate, testing whether the dilution effect disappears.
- Full-sample rerun (2283 non-length responses) if the pilot conclusions
  need tighter absolute numbers for the report.
- Human-rubric arm to complete the upper-bound row of the metrics table.
