# Phase 1: APPS Simple Error Discovery Plan And Result

Date: 2026-07-12

## Objective

Build a non-leaking error-discovery pipeline for the current APPS simple code-generation dataset. The phase takes verifier-failed responses, asks the model to summarize visible root causes, clusters those summaries, consolidates raw clusters into higher-level categories, and refines those categories into rubric-ready dimensions for Phase 2.

## Dataset Decision

The project no longer uses MBPP as the active taxonomy source. The current source is:

```text
APPS official train
-> executable prompt conversion
-> deterministic metadata filter
-> difficulty=introductory
-> 2613 prompts
```

Input and summary:

```text
data/processed/apps_train_simple_executable_prompts_unified.jsonl
data/processed/apps_train_simple_executable_prompts_unified_summary.json
```

Counts:

| Property | Count |
| --- | ---: |
| Selected APPS train introductory rows | 2613 |
| function_call | 2489 |
| stdin_stdout | 124 |

This choice gives a formal 2k-3k scale dataset without arbitrary subsampling. The APPS official test split remains held out.

## Generation Result

The active response files are:

```text
data/responses/apps_train_simple_executable_qwen25_k1_t2048_full.jsonl
data/responses/apps_train_simple_executable_qwen25_k1_t2048_full_labeled.jsonl
data/responses/apps_train_simple_executable_qwen25_k1_t2048_full_labeled_nonlength.jsonl
```

Full verifier metrics:

| Metric | Value |
| --- | ---: |
| Responses | 2613 |
| Passed | 1109 |
| Failed | 1504 |
| Pass rate | 42.44% |
| Length-finished responses | 330 |
| Length-finished rate | 12.63% |

The taxonomy source excludes length-finished generations:

| Metric | Value |
| --- | ---: |
| Non-length responses | 2283 |
| Non-length passed | 1064 |
| Non-length failed | 1219 |
| Non-length pass rate | 46.61% |

Rationale: `finish_reason=length` often means truncation or runaway generation. It is a generation-budget phenomenon, not reliable evidence of ordinary code logic failure.

## Failure Artifacts

Safe failure artifacts:

```text
data/analysis/apps_simple_phase1/apps_train_simple_qwen25_k1_t2048_failures_safe.jsonl
data/analysis/apps_simple_phase1/apps_train_simple_qwen25_k1_t2048_summary.json
data/analysis/apps_simple_phase1/apps_train_simple_qwen25_k1_t2048_taxonomy_initial_safe.yaml
```

Failure types:

| Type | Count |
| --- | ---: |
| logic_error | 676 |
| syntax_error | 311 |
| runtime_error | 183 |
| timeout | 48 |
| generation_failure | 1 |

Top rule-pattern diagnostics:

| Pattern | Count |
| --- | ---: |
| logic_wrong_output | 676 |
| syntax_malformed_code | 229 |
| syntax_duplicate_function_after_return | 72 |
| timeout_nonterminating_or_too_slow | 48 |
| runtime_name_error | 43 |
| runtime_value_error | 39 |
| runtime_type_error | 33 |
| runtime_other_exception | 30 |

These rule patterns are diagnostics only. The formal discovery step uses LLM summaries plus clustering, not a fixed rule taxonomy.

## LLM Discovery

Inputs:

```text
data/analysis/apps_simple_phase1/apps_train_simple_qwen25_k1_t2048_failures_safe.jsonl
```

Outputs:

```text
data/analysis/apps_simple_phase1/apps_train_simple_qwen25_k1_t2048_failures_with_safe_llm_summaries.jsonl
data/analysis/apps_simple_phase1/apps_train_simple_qwen25_k1_t2048_discovered_clusters_safe.jsonl
data/analysis/apps_simple_phase1/apps_train_simple_qwen25_k1_t2048_discovered_taxonomy_safe.yaml
data/analysis/apps_simple_phase1/apps_train_simple_qwen25_k1_t2048_discovered_taxonomy_summary_safe.json
```

Result:

```text
failure samples = 1219
raw clusters = 42
largest raw cluster = 25.3%
recursive split = one oversized cluster split into 15 sub-clusters
```

## Consolidation

Consolidation merges raw clusters into fewer rubric-operational groups while the script enforces raw-cluster coverage.

Outputs:

```text
data/analysis/apps_simple_phase1/apps_train_simple_qwen25_k1_t2048_taxonomy_consolidated.yaml
data/analysis/apps_simple_phase1/apps_train_simple_qwen25_k1_t2048_taxonomy_consolidated_audit.json
data/analysis/apps_simple_phase1/apps_train_simple_qwen25_k1_t2048_taxonomy_consolidated_cluster_mapping.jsonl
data/analysis/apps_simple_phase1/apps_train_simple_qwen25_k1_t2048_taxonomy_consolidated_response_assignments.jsonl
data/analysis/apps_simple_phase1/apps_train_simple_qwen25_k1_t2048_taxonomy_consolidation_raw_response.txt
```

Audit:

```text
raw_cluster_count = 42
category_count = 9
covered_cluster_count = 42
missing_clusters = []
duplicate_clusters = []
unknown_clusters = []
private_leakage_flags = []
broad_category_flags = []
valid = true
```

## Refinement

Refinement keeps assignments fixed and turns each consolidated category into a rubric seed with operational definitions, mechanisms, checklists, boundaries, and 1-5 anchors.

Outputs:

```text
data/analysis/apps_simple_phase1/apps_train_simple_qwen25_k1_t2048_taxonomy_refined_for_rubric.yaml
data/analysis/apps_simple_phase1/apps_train_simple_qwen25_k1_t2048_taxonomy_refined_for_rubric_audit.json
data/analysis/apps_simple_phase1/apps_train_simple_qwen25_k1_t2048_taxonomy_refined_response_assignments.jsonl
data/analysis/apps_simple_phase1/apps_train_simple_qwen25_k1_t2048_taxonomy_refinement_raw_response.txt
```

Current refined taxonomy:

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

The former `algorithmic_wrong_value` category was too broad for training and judging. It has been retired as an active label and deterministically split by raw-cluster evidence into numeric formula/arithmetic, sequence or collection transformation, and predicate/branch condition errors.

Audit:

```text
assignment_count = 1219
source_category_count = 9
refined_category_count = 9
missing_category_ids = []
duplicate_category_ids = []
unknown_category_ids = []
schema_flags = []
generic_text_flags = []
private_leakage_flags = []
valid = true
```

Three categories used deterministic template fallback after LLM candidates were rejected by quality gates:

```text
syntax_parseability_truncation
output_type_or_container_shape
edge_case_handling
```

This is acceptable because the final schema, category coverage, generic-text, and private-leakage audits pass.

## Phase 1 Completion Boundary

Phase 1 is complete for the current APPS simple dataset. Its handoff artifact is:

```text
data/analysis/apps_simple_phase1/apps_train_simple_qwen25_k1_t2048_taxonomy_refined_for_rubric.yaml
```

Phase 2 consumes this file to generate the formal rubric. Training, DPO, RL, and post-training self-evaluation belong to Method 1, not Phase 1.
