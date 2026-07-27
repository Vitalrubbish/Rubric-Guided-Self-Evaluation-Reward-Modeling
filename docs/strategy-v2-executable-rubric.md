# Strategy v2: Executable Rubrics — Self-Generated Tests As The Reward Signal

Date: 2026-07-26
Status: proposed main line, pending the test-quality probe (Section 5)
Supersedes: natural-language rubric routes (falsified at 7B and 32B)

## 1. Core Goal (unchanged)

Train a model that discovers error patterns from its own failures,
defines its OWN rubric for self-evaluation, and uses it as a learnable
reward signal for self-evolution — without human-provided scoring
standards (`docs/task.md`).

What changes is the **implementation form of "rubric"**, not the goal.

## 2. Evidence Constraints (all measured in this project)

| finding | consequence |
| --- | --- |
| Natural-language rubric injection: auto ≈ random at 7B (AUC 0.737/0.744) and 32B (0.858/0.860) | rubric-as-prose carries zero judging information; rubric must not be prompt text |
| 1-bit verifier reward: self-play SFT oscillates (7B: 36→41→36), pins at ceiling (32B: 48.5%), DPO flat (46.5%) | reward must be denser than pass/fail |
| Execution feedback (expected-vs-actual on public tests): +6.5pp in-context, p≈1e-4 | grounded, per-instance reward works |
| Feedback distillation into weights: flat (r1F 46.5%) | facts do not compress into weights; treat weight updates as conservative, gated, secondary |
| repair|hit at 32B: 43.5% (7B: 3.3%); judge AUC 0.87 (7B: 0.81) | the critic→repair link and judging only exist at 32B; 7B cannot host the loop |
| Writing checks is easier than writing solutions | the model should do what it is relatively best at: find errors and write tests; the executor adjudicates |

## 3. Re-Interpretation Of The Goal's Three Clauses

1. **"自定义 rubric" → executable rubric.** The artifact the model
   induces from each failure pattern is a set of *test cases /
   assertions*, not scoring prose. "Carry-propagation errors" becomes
   `assert add(999, 1) == 1000`, not "check carry handling".
2. **"作为 reward signal" → grounded by the executor.** The reward is
   the model's own tests, executed. This stays inside the project's
   red line: tests are model-written (no human rubric), the executor is
   environment, not a teacher.
3. **"自我进化" → scored at two levels.** System level: the evolving
   test library (an external memory that grows/prunes/refines — exactly
   the "rubric evolution" tracking the assignment asks for). Weight
   level: conservative LoRA updates accepted only by the 200-row gate
   with paired bootstrap.

## 4. Base Model Decision

- **32B Coder is the new baseline** (Qwen2.5-Coder-32B-Instruct; AWQ for
  inference, QLoRA-NF4 for training — both validated on the shared
  cluster). 7B is retired from the main line: it lacks the
  findings→repair link (3.3% conditional repair) that any self-repair
  loop needs.
- 72B deferred: it would raise the plateau, not answer a new question.

## 5. The New Main Loop

```
1. FAILURE COLLECTION   model solves/repairs; verifier labels failures
2. EXECUTABLE RUBRIC    model writes a test suite per failed problem
3. TEST QUALITY GATE    tests are validated against canonical/verifier
                        labels (precision/recall); bad tests are dropped
                        — a rubric that scores wrong is worse than none
4. REWARD               K repair candidates per problem are ranked by
                        the model's own tests (per-dimension, executable,
                        N-bit), not by prose judgment or 1-bit verdict
5. TRAINING             top-ranked repairs join SFT data (QLoRA,
                        unchanged conservative protocol)
6. TELEMETRY (two layers)
   - system: test-library evolution — coverage, precision, per-taxonomy
     growth curve (the demonstrable "rubric is evolving" evidence)
   - weights: bare gate200 + paired bootstrap; gold-100 attribution and
     conditional-repair telemetry (unchanged discipline)
```

Anti-hacking: tests validated against canonical solutions before use;
repairs must pass the hidden suite to enter training data; public-literal
scan on all training rows (both gates already implemented).

## 6. Decisive First Experiment: Test-Quality Probe

Before any training code, measure the loop's premise:

1. 32B writes 3-5 tests per problem for 50-100 problems (gold-100 or
   train prompts);
2. score the self-written tests against verifier labels: model-failed
   code should fail the tests, correct code should pass → precision /
   recall of the executable rubric;
3. if precision ≥ ~80%: use the tests to rank K=5 candidates
   (best-of-K by self-tests) on gate200; success = statistically
   beating the 48.5% bare ceiling.

Outcomes:

- tests precise + best-of-K above ceiling → the loop's premise holds;
  this is the first positive evidence for "model-defined rubric as
  reward" in the assignment's original sense; proceed to the full loop;
- tests imprecise → executable-rubric route closes; project concludes at
  the system-level result (model+executor evolution without weight
  evolution).

## 7. What Is Preserved

- Phase 1 taxonomy (9 categories) — the organizing artifact rubrics
  (now tests) are indexed by;
- measurement stack: gate200 + paired bootstrap, gold-100 telemetry,
  reward-hacking audits;
- the falsified-route registry (NL rubric, self-play SFT, DPO, feedback
  distillation) as the report's negative-results chapter.
