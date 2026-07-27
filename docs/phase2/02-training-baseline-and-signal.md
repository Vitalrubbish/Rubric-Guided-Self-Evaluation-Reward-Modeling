# Phase 2 Training Baseline And Signal Status

Date: 2026-07-12

## Current Status

The active dataset has changed to APPS simple. Therefore the old MBPP `hitl_v3` and `v5-lite failures` metrics are no longer current training-baseline evidence for this project state.

Current completed APPS artifacts:

```text
data/responses/apps_train_simple_executable_qwen25_k1_t2048_full_labeled.jsonl
data/responses/apps_train_simple_executable_qwen25_k1_t2048_full_labeled_nonlength.jsonl
data/analysis/apps_simple_phase1/apps_train_simple_qwen25_k1_t2048_taxonomy_refined_for_rubric.yaml
data/rubrics/apps_simple_phase2/apps_train_simple_llm_rubric_from_refined_taxonomy.json
```

What exists now:

- verifier pass/fail labels for 2613 APPS simple responses;
- safe non-length failure set with 1219 failures;
- 9-category refined taxonomy;
- 9-dimension audited rubric.

What does not yet exist for APPS simple:

- no-gate LLM self-evaluation baseline;
- verifier-gated rubric teacher signal;
- trained evaluator/critic;
- post-training no-gate self-evaluation comparison.

## Available Training Signal

The strongest current signal is the external verifier label:

| Metric | Value |
| --- | ---: |
| Full labeled responses | 2613 |
| Passed | 1109 |
| Failed | 1504 |
| Pass rate | 42.44% |
| Non-length labeled responses | 2283 |
| Non-length failures used for taxonomy | 1219 |

This can train or validate a pass/fail critic, but it is not a human rubric label and not a 1-5 dimension-level target.

The refined taxonomy assignments provide failure-type supervision for the 1219 non-length failures:

| Category | Failure assignments |
| --- | ---: |
| syntax_parseability_truncation | 362 |
| numeric_formula_arithmetic_error | 210 |
| edge_case_handling | 194 |
| output_type_or_container_shape | 130 |
| runtime_api_type_misuse | 103 |
| interface_name_signature_mismatch | 92 |
| sequence_collection_transformation_error | 92 |
| predicate_branch_condition_error | 25 |
| string_regex_pattern_logic | 11 |

These assignments are suitable as critic/evaluator training features or auxiliary labels for failed responses. They should not be represented as human ground-truth rubric scores. The old broad algorithmic category has been split before training so the labels expose distinct numeric, transformation, and predicate failures.

## Baseline Definition For The Next Stage

For Method 1, the APPS simple baseline should be defined before training:

```text
APPS simple responses
-> no-gate LLM rubric judge using the Phase 2 rubric
-> compare predicted_pass/score against verifier labels
```

This APPS no-gate run replaces the old MBPP v3 role.

If a verifier-gated rubric teacher is built, it must be reported separately:

```text
verifier labels + rubric dimensions
-> teacher/scaffold signal for training
```

That teacher signal is useful for reward/preference construction, but it is not evidence that the model can self-evaluate without execution evidence.

## Method 1 Handoff

Recommended next sequence:

1. Run a no-gate APPS simple rubric judge baseline on held-out evaluation data or a held-out split carved from APPS train.
2. Build a verifier-gated teacher signal only for train examples.
3. Train either a critic/evaluator or a generator preference model.
4. Evaluate the trained model with execution gate disabled.
5. Compare against the APPS no-gate baseline, not the archived MBPP v3 number.

## Reporting Rule

Report current APPS artifacts as:

- verifier-labeled APPS simple generation baseline;
- Phase 1 APPS error taxonomy;
- Phase 2 APPS rubric.

Do not report archived MBPP v3/v5-lite metrics as current APPS training evidence.
