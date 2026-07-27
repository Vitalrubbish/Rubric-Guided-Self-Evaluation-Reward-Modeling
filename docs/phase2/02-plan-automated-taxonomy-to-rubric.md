# Phase 2: Automated Taxonomy-to-Rubric Pipeline

Date: 2026-07-12

## Objective

Phase 2 converts the Phase 1 APPS simple refined taxonomy into a formal, judge-ready rubric.

It covers the rubric-generation part of `docs/task.md`:

```text
discovered error taxonomy
-> formal rubric dimensions
-> 1-5 score anchors
-> schema/private-leakage audit
```

The current Phase 2 run has generated and audited the rubric. It has not yet rerun the old MBPP v3/v5-lite judge experiments on the new APPS simple dataset. Any old v3/v5-lite numbers are archived MBPP results and are not current APPS evidence.

## Phase 1 Input

Current formal input:

```text
data/analysis/apps_simple_phase1/apps_train_simple_qwen25_k1_t2048_taxonomy_refined_for_rubric.yaml
data/analysis/apps_simple_phase1/apps_train_simple_qwen25_k1_t2048_taxonomy_refined_for_rubric_audit.json
data/analysis/apps_simple_phase1/apps_train_simple_qwen25_k1_t2048_taxonomy_refined_response_assignments.jsonl
```

Source status:

```text
assignment_count = 1219
source_category_count = 9
refined_category_count = 9
schema_flags = []
generic_text_flags = []
private_leakage_flags = []
valid = true
```

Refined categories:

```text
syntax_parseability_truncation
output_type_or_container_shape
edge_case_handling
interface_name_signature_mismatch
runtime_api_type_misuse
string_regex_pattern_logic
numeric_formula_arithmetic_error
sequence_collection_transformation_error
predicate_branch_condition_error
```

## Rubric Generation

Implemented in:

```text
src/rubric/generate_llm_rubric_from_taxonomy.py
```

The script uses the LLM as a controlled rubric writer. Deterministic code preserves:

- exact dimension ids;
- one dimension per refined taxonomy category;
- equal weights;
- all 1-5 anchors;
- critical-gate flags;
- source statistics;
- schema and private-leakage constraints.

The script now accepts `--rubric-name` so non-MBPP runs do not inherit the old `mbpp_hidden` rubric name.

Run command used for the current APPS simple rubric:

```bash
PATH=/data2/acm-group-3/miniconda3/envs/rubric/bin:$PATH \
CUDA_VISIBLE_DEVICES=2 \
XDG_CACHE_HOME=/tmp/rubric-cache \
HF_HOME=/tmp/rubric-cache/huggingface \
TRANSFORMERS_CACHE=/tmp/rubric-cache/huggingface \
TMPDIR=/tmp/rubric-tmp \
/data2/acm-group-3/miniconda3/envs/rubric/bin/python \
  src/rubric/generate_llm_rubric_from_taxonomy.py \
  --taxonomy data/analysis/apps_simple_phase1/apps_train_simple_qwen25_k1_t2048_taxonomy_refined_for_rubric.yaml \
  --source-audit data/analysis/apps_simple_phase1/apps_train_simple_qwen25_k1_t2048_taxonomy_refined_for_rubric_audit.json \
  --output data/rubrics/apps_simple_phase2/apps_train_simple_llm_rubric_from_refined_taxonomy.json \
  --audit-output data/rubrics/apps_simple_phase2/apps_train_simple_llm_rubric_from_refined_taxonomy_audit.json \
  --raw-llm-output data/rubrics/apps_simple_phase2/apps_train_simple_llm_rubric_from_refined_taxonomy_raw_response.txt \
  --rubric-name apps_train_simple_llm_rubric_from_refined_taxonomy_v2_split_algorithmic \
  --model models/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.30 \
  --temperature 0.2 \
  --top-p 0.95 \
  --max-tokens 4096
```

Outputs:

```text
data/rubrics/apps_simple_phase2/apps_train_simple_llm_rubric_from_refined_taxonomy.json
data/rubrics/apps_simple_phase2/apps_train_simple_llm_rubric_from_refined_taxonomy_audit.json
data/rubrics/apps_simple_phase2/apps_train_simple_llm_rubric_from_refined_taxonomy_raw_response.txt
```

Current audit:

```text
used_llm = true
used_existing_llm_output = false
dimension_count = 9
duplicate_dimension_ids = []
missing_anchor_dimensions = []
private_leakage_flags = []
generic_dimension_flags = []
used_deterministic_fallback = false
valid = true
```

Rubric name:

```text
apps_train_simple_llm_rubric_from_refined_taxonomy_v2_split_algorithmic
```

## Completion Boundary

Current Phase 2 is complete for taxonomy-to-rubric generation.

Completed:

- consumed the current APPS simple refined taxonomy;
- generated a 9-dimension rubric;
- passed source-taxonomy validation;
- passed dimension coverage checks;
- passed anchor completeness checks;
- passed private-leakage and generic-dimension checks.

Not yet completed on the APPS simple dataset:

- no-gate LLM rubric judge baseline;
- verifier-gated training-signal run;
- evaluator/critic training;
- post-training self-evaluation comparison.

Those steps belong to the next Method 1 training/evaluator phase.

## Acceptance Criteria

The current rubric artifact is accepted only if:

- `source_audit.valid == true`;
- the generated rubric has exactly 9 dimensions;
- dimension ids exactly match the refined taxonomy ids;
- every dimension has anchors `1` through `5`;
- no private verifier fields appear in the rubric;
- audit reports `valid = true`.

The current APPS simple rubric satisfies these criteria.
