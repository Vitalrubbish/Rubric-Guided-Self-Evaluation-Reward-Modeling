# Phase 2: Automated Taxonomy-to-Rubric Pipeline

Date: 2026-07-10

## Objective And Task Alignment

Phase 2 implements the rubric-generation and rubric-judging part of `task.md`.

It covers **作业 3 Step 2--3**:

- generate a rubric from discovered error patterns;
- use the generated rubric for self-evaluation;
- compare self-evaluation with an external verifier.

It also prepares the handoff artifacts for **作业 4 方法 1：Error-Pattern -> Rubric -> RL 闭环**:

- v3 provides the pre-RL self-evaluation baseline;
- v5-lite failures provides a verifier-gated teacher signal for reward/preference construction;
- the next RL stage belongs to Method 1, not Phase 2.

Phase 2 no longer owns taxonomy consolidation or refined-taxonomy generation. Both steps are now part of Phase 1, which produces the refined rubric-operational taxonomy consumed by Phase 2.

The Phase 2 objective is to automatically generate a formal rubric from the Phase 1 refined taxonomy, then use an LLM rubric judge to evaluate held-out responses against verifier labels. The retained flow is:

```text
Phase 1 refined taxonomy
-> LLM rubric generation
-> deterministic rubric schema/leakage audit + repair/fallback
-> v3 no-gate LLM rubric judge baseline
-> v5-lite verifier-gated rubric signal
-> deterministic judge schema/leakage audit + repair/fallback
-> verifier-alignment metrics
```

Completion boundary:

```text
Phase 2 ends when v3 baseline metrics and v5-lite failures signal are produced.
RL/DPO/critic training starts in Method 1.
```

## Completed Inputs

Phase 1 has produced the consolidated taxonomy as an audited intermediate artifact:

```text
data/analysis/phase1/mbpp_hidden_train_qwen25_k3_taxonomy_consolidated.yaml
data/analysis/phase1/mbpp_hidden_train_qwen25_k3_taxonomy_consolidated_audit.json
data/analysis/phase1/mbpp_hidden_train_qwen25_k3_taxonomy_consolidated_cluster_mapping.jsonl
data/analysis/phase1/mbpp_hidden_train_qwen25_k3_taxonomy_consolidated_response_assignments.jsonl
```

Phase 1 has also produced the refined rubric-operational taxonomy, which is the formal Phase 2 input:

```text
data/analysis/phase1/mbpp_hidden_train_qwen25_k3_taxonomy_refinement_raw_response.txt
data/analysis/phase1/mbpp_hidden_train_qwen25_k3_taxonomy_refined_for_rubric.yaml
data/analysis/phase1/mbpp_hidden_train_qwen25_k3_taxonomy_refined_for_rubric_audit.json
data/analysis/phase1/mbpp_hidden_train_qwen25_k3_taxonomy_refined_response_assignments.jsonl
```

Consolidation audit:

```text
raw_cluster_count = 17
category_count = 8
covered_cluster_count = 17
missing_clusters = []
duplicate_clusters = []
unknown_clusters = []
private_leakage_flags = []
broad_category_flags = []
valid = true
```

Refinement audit:

```text
source_category_count = 8
refined_category_count = 8
assignment_count = 519
schema_flags = []
generic_text_flags = []
private_leakage_flags = []
valid = true
initial_accepted_categories = 2
revised_accepted_categories = 4
targeted_repair_accepted_categories = 2
template_fallback_categories = []
```

Current operational categories:

```text
numeric_formula_correctness
output_type_container_shape
algorithmic_wrong_value
syntax_parseability_or_output_format
runtime_api_type_misuse
string_regex_pattern_logic
edge_case_boundary_handling
interface_name_signature_mismatch
```

## Automation Boundary

Phase 1 has completed:

- merging raw clusters into higher-level categories;
- naming categories;
- generating rubric-operational refinement for each category;
- deterministic schema validation;
- raw-cluster coverage checks;
- duplicate/unknown cluster checks;
- private-leakage checks;
- response-level assignment inheritance;
- automatic splitting and repair when LLM output is too broad;
- multi-candidate refinement quality gates, bad-phrase masks, generic-name rejection, and category-conditioned blocked terms;
- automatic revision or targeted repair when refinement output is too generic, truncated, or has unreliable score anchors.

The Phase 2 LLM is responsible for:

- converting the 8 refined taxonomy categories into formal rubric dimensions;
- scoring held-out responses with the formal rubric;
- emitting structured scores, category-level rationales, and an overall decision.

The Phase 2 program is responsible for:

- schema validation;
- private-leakage checks;
- rubric category coverage checks;
- sanitized judge-input construction;
- LLM output repair/fallback;
- execution-gated post-processing for v5-lite failures;
- AUC, accuracy, Cohen's Kappa, and per-category score distribution computation.

## Implemented Rubric Generation

Rubric generation is implemented in:

```text
src/rubric/generate_llm_rubric_from_taxonomy.py
scripts/phase2/run_phase2_rubric_generation.sh
```

Input:

```text
data/analysis/phase1/mbpp_hidden_train_qwen25_k3_taxonomy_refined_for_rubric.yaml
```

Outputs:

```text
data/rubrics/phase2/mbpp_hidden_llm_rubric_from_refined_taxonomy.json
data/rubrics/phase2/mbpp_hidden_llm_rubric_from_refined_taxonomy_audit.json
data/rubrics/phase2/mbpp_hidden_llm_rubric_from_refined_taxonomy_raw_response.txt
```

Run command:

```bash
GPU=2 scripts/phase2/run_phase2_rubric_generation.sh
```

Use deterministic fallback for CPU-only schema debugging:

```bash
DETERMINISTIC_ONLY=1 scripts/phase2/run_phase2_rubric_generation.sh
```

Current rubric audit status:

```text
valid = true
dimension_count = 8
private_leakage_flags = []
generic_dimension_flags = []
used_existing_llm_output = true
```

## Current Rubric Judge Entrypoints

Rubric judging is implemented in:

```text
src/rubric/evaluate_llm_rubric_judge.py
scripts/phase2/run_phase2_hitl_judge.sh
```

The active versioned entrypoints are:

```text
scripts/phase2/run_phase2_hitl_v3_judge.sh
scripts/phase2/run_phase2_hitl_v5_lite_failures_judge.sh
```

Evaluation input:

```text
data/responses/phase1_mbpp_hidden_qwen25_k3_labeled.jsonl
```

## Active Baseline: v3

v3 is the baseline for later RL because it measures rubric-based self-evaluation without an external execution gate. It corresponds to the `task.md` self-evaluation metric before self-evolving training.

Run command:

```bash
GPU=2 scripts/phase2/run_phase2_hitl_v3_judge.sh
```

Full test metrics:

| Metric | Value |
| --- | ---: |
| AUC | 0.618196 |
| Accuracy | 0.586667 |
| Cohen's Kappa | 0.184811 |
| Overacceptance | 0.701823 |
| False rejection | 0.110656 |

## Active Training Signal: v5-Lite Failures

v5-lite failures is a verifier-failure-gated rubric judge. It is used to produce lower-noise reward and preference construction signals for the Method 1 RL loop, not as proof that the LLM can self-evaluate without external execution evidence.

Run command:

```bash
scripts/phase2/run_phase2_hitl_v5_lite_failures_judge.sh
```

Full test metrics:

| Metric | Value |
| --- | ---: |
| AUC | 0.950606 |
| Accuracy | 0.943333 |
| Cohen's Kappa | 0.886292 |
| Overacceptance | 0.0 |
| False rejection | 0.116120 |

## Reporting Rule

Report v3 as the pre-RL self-evaluation baseline. Report v5-lite failures as a teacher/scaffold for reward construction. Do not report v5-lite failures as pure LLM self-evaluation, because it uses verifier execution results in post-processing.

## Completion Status

Phase 2 is complete for the current Method 1 handoff:

- v3 baseline is available.
- v5-lite failures signal is available.
- obsolete intermediate judge/calibration/HITL implementations were removed.
- Method 1 owns the next RL/DPO/critic training loop.

## Acceptance Criteria

The retained Phase 2 scoring pipeline is accepted only if:

- the score and raw output files contain the expected row count;
- `audit.valid == true`;
- `prompt_leakage_count == 0`;
- JSON parse failures and repaired outputs are explained by the audit;
- judge prompts do not receive verifier labels, hidden tests, exact assertions, or private diagnostics;
- metrics are reported against verifier pass/fail labels;
- v3 and v5-lite failures are reported under separate roles.
