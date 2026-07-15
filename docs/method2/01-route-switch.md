# Method 2 Route Switch

## Decision

Switch the active route from Method 1 direct generator updates to Method 2:
Self-Play Error Discovery.

Method 1 is currently blocked because direct solver SFT/DPO repeatedly perturbs
the 7B code-generation distribution. The latest APPS canaries regressed pass@1,
syntax errors, or length finishes even after stricter cleaning and lower LR.

Method 2 is more feasible now because the repo already has grounded self-play
evidence:

- protected proxy self-play: 178 fail->pass repairs, repair precision given edit
  0.583607, all-pass preservation 1.0, mostly syntax cleanup;
- real LLM critic multi-candidate self-play: 20 MBPP logic failures, 7 repaired
  tasks, 13/100 passing revised candidates, gate passed.

Method 3 is deferred. It needs stable self-evaluation across multiple tasks
before meta-learning is meaningful. Current generative self-evaluation has useful
ranking signal but has not passed the held-out gate.

## New Mainline

Method 2 loop-v0:

1. Start from a failed response A.
2. Ask the same model to produce explicit error findings.
3. Ask it to produce revised response B.
4. Verify B externally.
5. Keep `(A < B)` only when A fails and B passes.
6. Track repair rate, false-positive/harmful edit rate, and success by failure
   type before doing any policy update.

The first training artifact is not plain code SFT. It is critic+repair SFT:

```text
public task + failed code -> ERROR_FINDINGS + REVISED_CODE
```

Preference pairs use the same critic+repair prompt, so DPO cannot silently
degrade into generic solver imitation.

## Bootstrap Artifacts

Build with:

```bash
scripts/build_method2_apps_self_play_bootstrap_v0_1_clean.sh
```

Outputs:

```text
data/sft/method2_apps_self_play_critic_repair_v0_1_clean.jsonl
data/preferences/method2_apps_self_play_critic_repair_pairs_v0_1_clean.jsonl
data/self_play/method2_apps_self_play_bootstrap_v0_1_clean_summary.json
```

The earlier `method2_self_play_critic_repair_v0` artifact is MBPP bootstrap
data and should not be used for the current APPS mainline.

The earlier APPS `method2_apps_self_play_critic_repair_v0` artifact should also
be superseded by v0.1 clean because it preserved some function-call demo
harnesses in `REVISED_CODE` and lacked top-level verifier metadata.

Policy:

- real LLM critic rows and deterministic protected proxy rows are kept as
  separate sources;
- only train-source rows are used by default;
- proxy rows are bootstrap data, not final evidence of self-discovery;
- next gate is repair effectiveness, not APPS greedy pass@1.
