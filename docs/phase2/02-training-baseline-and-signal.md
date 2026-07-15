# Phase 2 Training Baseline and Signal

## Alignment With `task.md`

This Phase 2 setup produces the handoff artifacts for **作业 4 方法 1：Error-Pattern -> Rubric -> RL 闭环**. Phase 2 itself ends after the baseline and training-signal artifacts are produced.

Mapping:

1. Generate responses: Phase 1 generated MBPP code responses.
2. Discover error patterns: Phase 1 clustered verifier-failed train responses and refined them into an operational taxonomy.
3. Generate/update rubric: Phase 2 converts that taxonomy into an 8-dimension rubric.
4. Handoff to Method 1: v5-lite failures creates verifier-gated rubric signals for reward/preference construction.
5. Method 1 training and post-RL evaluation happen outside Phase 2.

External verifier labels are allowed as the failure-discovery and evaluation signal in `task.md`, but the final self-evaluation claim must be measured without an external execution gate.

## Current Decision

- Use `hitl_v3` as the **pre-RL self-evaluation baseline**.
- Use `v5-lite failures` as the **reliable training signal source**.
- Do not use v5-lite failures as proof of pure self-evaluation accuracy, because it uses verifier execution results in post-processing.
- Treat Phase 2 as complete once both artifacts are generated and audited.

## Baseline: v3 Self-Evaluation

v3 is the baseline for measuring whether later RL improves rubric-based self-evaluation without external execution gates. This is the number the post-RL evaluator must beat.

Full test metrics:

- AUC: `0.618196080942623`
- Accuracy: `0.5866666666666667`
- Kappa: `0.18481119175344474`
- Overacceptance: `0.7018229166666666`
- False rejection: `0.11065573770491803`
- Confusion: `TN=229, FP=539, FN=81, TP=651`

Run entrypoint:

```bash
scripts/phase2/run_phase2_hitl_v3_judge.sh
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
scripts/phase2/run_phase2_hitl_v5_lite_failures_judge.sh
```

## Reporting Rule

Report v3 as the pre-RL self-evaluation baseline. Report v5-lite failures as a teacher/scaffold for reward construction, not as evidence that the model can self-evaluate without external execution evidence.

## Method 1 Handoff

The next loop belongs to Method 1 and should use only train split data for training:

```text
train responses
-> verifier labels and safe diagnostics
-> v5-lite failures rubric signal
-> preference/reward construction
-> RL/DPO or critic training
-> no-gate self-evaluation on validation/test
-> compare against v3
```

Success criterion:

- The trained model's no-gate self-evaluation improves over v3 on held-out validation/test.
- The improvement is reported separately from the verifier-gated v5-lite teacher score.
