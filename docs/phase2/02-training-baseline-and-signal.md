# Phase 2 Training Baseline and Signal

## Current Decision

- Use `hitl_v3` as the self-evaluation baseline for later training.
- Use `v5-lite failures` as the reliable training signal source.
- Do not use v4 as a baseline because its required self-probe gate causes severe false rejection.

## Baseline: v3 Self-Evaluation

v3 is the baseline for measuring whether later RL improves rubric-based self-evaluation without external execution gates.

Full test metrics:

- AUC: `0.618196080942623`
- Accuracy: `0.5866666666666667`
- Kappa: `0.18481119175344474`
- Overacceptance: `0.7018229166666666`
- False rejection: `0.11065573770491803`
- Confusion: `TN=229, FP=539, FN=81, TP=651`

Run entrypoint:

```bash
scripts/run_phase2_hitl_v3_judge.sh
```

## Training Signal: v5-Lite Failures

v5-lite failures is not a pure LLM self-evaluation metric. It is a verifier-failure-gated rubric judge used to construct lower-noise training rewards and preference data.

Full test metrics:

- AUC: `0.9506062158469946`
- Accuracy: `0.9433333333333334`
- Kappa: `0.8862918847186995`
- Overacceptance: `0.0`
- False rejection: `0.11612021857923498`
- Confusion: `TN=768, FP=0, FN=85, TP=647`

Run entrypoint:

```bash
scripts/run_phase2_hitl_v5_lite_failures_judge.sh
```

## Reporting Rule

Report v3 as the pre-RL self-evaluation baseline. Report v5-lite failures as a teacher/scaffold for reward construction, not as evidence that the model can self-evaluate without external execution evidence.
