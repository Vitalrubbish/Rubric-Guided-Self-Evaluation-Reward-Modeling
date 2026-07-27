# Phase 2 Label Assets And Training Readiness

Date: 2026-07-12

## Conclusion

The active label assets now come from APPS simple verifier execution, not MBPP v3/v5-lite judge runs.

They are suitable for:

- pass/fail critic training;
- preference construction based on verifier outcomes;
- failure-category auxiliary supervision for non-length failed responses;
- evaluator/critic distillation experiments.

They are not suitable for:

- claiming human-ground-truth rubric scores;
- claiming pure self-evaluation ability without a no-gate evaluator run;
- training on APPS official test.

## Active Label Files

Full verifier labels:

```text
data/responses/apps_train_simple_executable_qwen25_k1_t2048_full_labeled.jsonl
```

Non-length labels used for taxonomy:

```text
data/responses/apps_train_simple_executable_qwen25_k1_t2048_full_labeled_nonlength.jsonl
```

Failure-category assignments:

```text
data/analysis/apps_simple_phase1/apps_train_simple_qwen25_k1_t2048_taxonomy_refined_response_assignments.jsonl
```

Phase 2 rubric:

```text
data/rubrics/apps_simple_phase2/apps_train_simple_llm_rubric_from_refined_taxonomy.json
```

## Verifier Label Summary

Full labeled APPS simple responses:

| Metric | Value |
| --- | ---: |
| Rows | 2613 |
| Passed | 1109 |
| Failed | 1504 |
| Pass rate | 42.44% |
| Length-finished | 330 |
| Length rate | 12.63% |

Non-length taxonomy subset:

| Metric | Value |
| --- | ---: |
| Rows | 2283 |
| Passed | 1064 |
| Failed | 1219 |
| Pass rate | 46.61% |

Failure types in the taxonomy source:

| Type | Count |
| --- | ---: |
| logic_error | 676 |
| syntax_error | 311 |
| runtime_error | 183 |
| timeout | 48 |
| generation_failure | 1 |

## Failure Category Labels

The refined category assignments cover 1219 non-length failures:

| Category | Count |
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

These are model-assisted taxonomy labels with deterministic audit, not human annotations. The former broad algorithmic label has been removed from active assignments and split into the three more specific logic-failure labels above.

## Leakage Rules

Training or judge prompts may use:

- task prompt;
- public starter code;
- public interface name/signature;
- generated answer or extracted code;
- rubric text.

Training or judge prompts must not include:

- verifier labels;
- exact expected outputs;
- private diagnostics;
- hidden tests;
- benchmark identity as a model-facing hint;
- APPS official test examples in training.

## Training Readiness

Ready:

- APPS simple verifier labels;
- safe failure artifacts;
- refined taxonomy assignments;
- audited Phase 2 rubric;
- local Qwen2.5-7B-Instruct environment.

Not ready until explicitly built:

- APPS no-gate rubric judge baseline;
- verifier-gated APPS teacher signal;
- train/validation/test split policy for critic training;
- critic/evaluator training script and evaluation protocol.

## Go/No-Go

Current decision:

```text
Go: train or evaluate a verifier-supervised critic/evaluator using APPS simple train-derived data.
Go: use taxonomy assignments as auxiliary failure labels for non-length failures.
No-Go: claim human-GT rubric quality.
No-Go: claim self-evaluation improvement before a no-gate APPS baseline and post-training comparison exist.
```
