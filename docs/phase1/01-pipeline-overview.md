# Phase 1 Pipeline Overview

Date: 2026-07-12

## Current Dataset

Phase 1 now uses the APPS official train split, restricted by the official difficulty label `introductory` and by deterministic verifier-readiness filtering.

This is not an arbitrary sample. The current working set is:

```text
data/processed/apps_train_simple_executable_prompts_unified.jsonl
```

Dataset summary:

| Field | Value |
| --- | ---: |
| Source rows seen | 3771 |
| Selected rows | 2613 |
| Split | train |
| Difficulty | introductory |
| function_call tasks | 2489 |
| stdin_stdout tasks | 124 |

The APPS official test split remains held out and is not used for taxonomy or rubric construction.

## Scope

Phase 1 discovers coding-error patterns from verifier-failed APPS simple responses and turns them into a rubric-operational taxonomy.

Core scripts:

```text
src/data-prep/select_prompts_by_metadata.py
src/generation/vllm_smoke_generate.py
src/verification/verify_mbpp_smoke.py
src/error-analysis/build_failure_artifacts.py
src/error-analysis/discover_error_taxonomy.py
src/error-analysis/consolidate_taxonomy.py
src/error-analysis/refine_taxonomy_for_rubric.py
```

`verify_mbpp_smoke.py` is still the verifier entrypoint name, but it now contains APPS adapters for both `function_call` and `stdin_stdout` modes.

## Current Pipeline

```text
APPS train executable prompts
-> deterministic metadata filter: dataset=apps, split=train, difficulty=introductory
-> unified coding prompts
-> Qwen2.5-7B-Instruct k=1 generation
-> APPS verifier labeling
-> separate finish_reason=length cases
-> build safe non-length failures
-> LLM root-cause summaries
-> TF-IDF/SVD/HDBSCAN raw clustering
-> LLM consolidation with deterministic coverage audit
-> LLM refinement with schema/private-leakage/generic-text audit
-> refined taxonomy for Phase 2
```

The model-facing prompt does not include benchmark name, verifier labels, hidden tests, exact expected values, private diagnostics, or dataset identity. Dataset identity remains only as JSON metadata for routing outputs to the correct verifier.

## Generation And Verification

Generation configuration:

| Setting | Value |
| --- | --- |
| Model | local Qwen2.5-7B-Instruct snapshot |
| k | 1 |
| temperature | 0.7 |
| top_p | 0.9 |
| max_tokens | 2048 |
| max_model_len | 12288 for main run; 16384 for remaining long prompt |
| GPU | CUDA device 2 |

Generation was completed in two parts because one late prompt exceeded the original 12288-token context:

```text
data/responses/apps_train_simple_executable_qwen25_k1_t2048.jsonl
data/responses/apps_train_simple_executable_qwen25_k1_t2048_remaining_2305_2613.jsonl
data/responses/apps_train_simple_executable_qwen25_k1_t2048_full.jsonl
```

Final verifier output:

```text
data/responses/apps_train_simple_executable_qwen25_k1_t2048_full_labeled.jsonl
```

Verification summary:

| Metric | Value |
| --- | ---: |
| Total responses | 2613 |
| Passed | 1109 |
| Failed | 1504 |
| Pass rate | 42.44% |
| `finish_reason=length` | 330 |
| Length rate | 12.63% |

`finish_reason=length` indicates possible truncation or overgeneration. These cases are reported separately and are not used as normal algorithmic failure evidence for taxonomy construction.

## Phase 1 Taxonomy Source

The core taxonomy source excludes length-finished responses:

```text
data/responses/apps_train_simple_executable_qwen25_k1_t2048_full_labeled_nonlength.jsonl
```

Non-length summary:

| Metric | Value |
| --- | ---: |
| Non-length responses | 2283 |
| Passed | 1064 |
| Failed | 1219 |
| Pass rate | 46.61% |

Safe failures:

```text
data/analysis/apps_simple_phase1/apps_train_simple_qwen25_k1_t2048_failures_safe.jsonl
```

Failure distribution:

| Failure type | Count |
| --- | ---: |
| logic_error | 676 |
| syntax_error | 311 |
| runtime_error | 183 |
| timeout | 48 |
| generation_failure | 1 |

## Discovery, Consolidation, Refinement

Raw discovery:

```text
data/analysis/apps_simple_phase1/apps_train_simple_qwen25_k1_t2048_failures_with_safe_llm_summaries.jsonl
data/analysis/apps_simple_phase1/apps_train_simple_qwen25_k1_t2048_discovered_clusters_safe.jsonl
data/analysis/apps_simple_phase1/apps_train_simple_qwen25_k1_t2048_discovered_taxonomy_safe.yaml
data/analysis/apps_simple_phase1/apps_train_simple_qwen25_k1_t2048_discovered_taxonomy_summary_safe.json
```

Discovery result:

| Metric | Value |
| --- | ---: |
| Failure assignments | 1219 |
| Raw clusters | 42 |
| Oversized clusters repaired | 1 cluster split into 15 sub-clusters |

Consolidated taxonomy:

```text
data/analysis/apps_simple_phase1/apps_train_simple_qwen25_k1_t2048_taxonomy_consolidated.yaml
data/analysis/apps_simple_phase1/apps_train_simple_qwen25_k1_t2048_taxonomy_consolidated_audit.json
data/analysis/apps_simple_phase1/apps_train_simple_qwen25_k1_t2048_taxonomy_consolidated_response_assignments.jsonl
```

Consolidation audit:

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

Refined taxonomy, consumed by Phase 2:

```text
data/analysis/apps_simple_phase1/apps_train_simple_qwen25_k1_t2048_taxonomy_refined_for_rubric.yaml
data/analysis/apps_simple_phase1/apps_train_simple_qwen25_k1_t2048_taxonomy_refined_for_rubric_audit.json
data/analysis/apps_simple_phase1/apps_train_simple_qwen25_k1_t2048_taxonomy_refined_response_assignments.jsonl
```

Current refined categories:

| Category | Assigned failures |
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

The previous broad `algorithmic_wrong_value` bucket is no longer an active category; it has been split into numeric formula, sequence/collection transformation, and predicate/branch condition failures.

Refinement audit:

```text
assignment_count = 1219
source_category_count = 9
refined_category_count = 9
schema_flags = []
generic_text_flags = []
private_leakage_flags = []
valid = true
```

## Quality Gates

Phase 1 is valid for Phase 2 only if:

- the current prompt file has 2613 APPS train introductory rows;
- APPS official test is not used in construction;
- safe failure artifacts exclude hidden tests, private diagnostics, and exact expected values;
- `finish_reason=length` cases are not treated as normal logic failures;
- raw cluster assignments cover the safe failure set;
- consolidated taxonomy covers every raw cluster exactly once;
- refined taxonomy preserves all 1219 response assignments;
- taxonomy audits report `valid = true`.

Current status satisfies these gates.
