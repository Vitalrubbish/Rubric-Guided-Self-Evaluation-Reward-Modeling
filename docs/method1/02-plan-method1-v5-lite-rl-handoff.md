# Method 1 Handoff: v3 Baseline and v5-Lite Training Signal

## Task Alignment

This document is the Method 1 continuation of Phase 2.

It implements the next stage of **作业 4 方法 1：Error-Pattern -> Rubric -> RL 闭环**:

```text
generated responses
-> verifier failure discovery
-> taxonomy/rubric generation
-> v3 no-gate self-evaluation baseline
-> v5-lite verifier-gated training signal
-> Method 1 RL/DPO/critic training
-> no-gate self-evaluation after training
```

Phase 2 is complete once v3 and v5-lite failures are produced. Method 1 owns all subsequent training.

## Inputs From Phase 2

### v3 Baseline

Role: pre-RL self-evaluation baseline.

Full test metrics:

- AUC: `0.618196080942623`
- Accuracy: `0.5866666666666667`
- Kappa: `0.18481119175344474`
- Overacceptance: `0.7018229166666666`
- False rejection: `0.11065573770491803`

### v5-Lite Failures Signal

Role: verifier-gated teacher/scaffold for reward and preference construction.

Full test metrics:

- AUC: `0.9506062158469946`
- Accuracy: `0.9433333333333334`
- Kappa: `0.8862918847186995`
- Overacceptance: `0.0`
- False rejection: `0.11612021857923498`

This is not a pure self-evaluation score because it uses execution results in post-processing.

## Method 1 Next Steps

1. Generate v5-lite failures signals for the train split only.
2. Construct preference pairs from train responses:
   - verifier-pass responses as preferred over verifier-fail responses;
   - v5-lite failure dimensions as auxiliary reward/error features;
   - no validation/test examples in training.
3. Train either:
   - a generator with DPO/preferences; or
   - a self-evaluator/critic to predict rubric scores without execution gate.
4. Evaluate the trained model with execution gate disabled.
5. Compare no-gate post-RL self-evaluation against the v3 baseline.

## Reporting Rule

- Report v3 as the Method 1 pre-RL baseline.
- Report v5-lite failures as the Method 1 teacher signal.
- Report post-RL no-gate self-evaluation as the actual self-evolving result.
- Do not claim v5-lite failures accuracy as model self-evaluation ability.
