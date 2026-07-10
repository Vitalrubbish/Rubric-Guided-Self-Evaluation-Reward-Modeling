# Phase 2: Automated Taxonomy-to-Rubric Pipeline

Date: 2026-07-10

## Objective

Phase 2 no longer owns taxonomy consolidation or refined-taxonomy generation. Both steps are now part of Phase 1, which produces the refined rubric-operational taxonomy consumed by Phase 2.

The Phase 2 objective is to automatically generate a formal rubric from the Phase 1 refined taxonomy, then use an LLM rubric judge to evaluate held-out validation/test responses against verifier labels. The flow remains fully automated:

```text
Phase 1 refined taxonomy
-> LLM rubric generation
-> deterministic rubric schema/leakage audit + repair/fallback
-> LLM rubric judge
-> deterministic judge schema/leakage audit + repair/fallback
-> verifier-alignment metrics
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
- AUC, accuracy, Cohen's Kappa, and per-category score distribution computation.

## Implemented Rubric Generation

Rubric generation is implemented in:

```text
src/rubric/generate_llm_rubric_from_taxonomy.py
scripts/run_phase2_rubric_generation.sh
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
GPU=2 scripts/run_phase2_rubric_generation.sh
```

Use deterministic fallback for CPU-only schema debugging:

```bash
DETERMINISTIC_ONLY=1 scripts/run_phase2_rubric_generation.sh
```

Current rubric audit status:

```text
valid = true
dimension_count = 8
private_leakage_flags = []
generic_dimension_flags = []
used_existing_llm_output = true
```

## Implemented LLM Rubric Judge

Rubric judging is implemented in:

```text
src/rubric/evaluate_llm_rubric_judge.py
scripts/run_phase2_llm_judge.sh
```

Evaluation input:

```text
data/responses/phase1_mbpp_hidden_qwen25_k3_labeled.jsonl
```

Outputs:

```text
data/rubrics/phase2/mbpp_hidden_llm_judge_scores_validation_test.jsonl
data/rubrics/phase2/mbpp_hidden_llm_judge_metrics_validation_test.json
data/rubrics/phase2/mbpp_hidden_llm_judge_audit_validation_test.json
data/rubrics/phase2/mbpp_hidden_llm_judge_raw_validation_test.jsonl
```

Full held-out command:

```bash
GPU=2 scripts/run_phase2_llm_judge.sh
```

Small LLM smoke command:

```bash
GPU=2 LIMIT=4 TAG=smoke_llm BATCH_SIZE=4 MAX_NUM_SEQS=4 scripts/run_phase2_llm_judge.sh
```

CPU-only schema and metrics debugging command:

```bash
DETERMINISTIC_ONLY=1 LIMIT=4 TAG=smoke_det scripts/run_phase2_llm_judge.sh
```

Evaluation requirements:

- the rubric is generated from the Phase 1 refined taxonomy, which was induced from train failures;
- the judge evaluates validation/test responses only;
- judge prompts do not include verifier labels, assertions, `test_list`, or `private_diagnostics`;
- outputs include AUC, accuracy, Cohen's Kappa, and per-category score distributions.

Current LLM judge smoke status:

```text
num_samples = 4
prompt_leakage_count = 0
json_parse_failed_count = 0
repaired_record_count = 0
used_visible_code_fallback_count = 0
valid = true
```

The smoke run proves prompt sanitization, vLLM execution, JSON parsing, repair/audit, and metric writing. Formal conclusions must use the full validation/test run.

## Full Held-Out Run

The full validation/test LLM judge run completed on 2026-07-10:

```bash
GPU=2 GPU_MEMORY_UTILIZATION=0.25 BATCH_SIZE=16 MAX_NUM_SEQS=16 MAX_TOKENS=768 scripts/run_phase2_llm_judge.sh
```

The raw outputs were then reprocessed without another GPU run after two audit/parser fixes:

- leakage audit now treats private fields as JSON/metadata keys, so public MBPP identifiers such as a function parameter named `test_list` are not false positives;
- JSON parsing now prefers the actual judge object with `dimension_scores` when the model first repeats a public-input JSON object.

Reprocess command:

```bash
REUSE_RAW_OUTPUT=data/rubrics/phase2/mbpp_hidden_llm_judge_raw_validation_test.jsonl scripts/run_phase2_llm_judge.sh
```

Final full-run audit:

```text
num_requested_rows = 1770
score_rows = 1770
raw_rows = 1770
prompt_leakage_count = 0
json_parse_failed_count = 0
repaired_record_count = 205
used_visible_code_fallback_count = 203
valid = true
```

Final full-run metrics:

| Metric | Value |
| --- | ---: |
| LLM judge AUC | 0.603597 |
| Cohen's Kappa | 0.152540 |
| Accuracy | 0.572881 |
| Mean score, passed responses | 4.735021 |
| Mean score, failed responses | 4.452407 |
| Predicted pass rate | 0.696610 |
| True pass rate | 0.489831 |

Interpretation:

- The end-to-end Phase 2 pipeline is operational: full held-out scoring completed, row counts match, prompt leakage is zero, JSON parse failures are zero, and the audit is valid.
- The judge is still too lenient. It assigns high scores to both passed and failed responses, which explains the weak Kappa and high predicted pass rate.
- 205/1770 records required repair, with 203 falling back to visible-code heuristics because the model repeated the public input instead of emitting the required judge JSON. The runner now supports raw-output reprocessing, and the prompt has been tightened for future reruns. A targeted rerun of those 205 repaired rows is the recommended next robustness step when a GPU is actually idle.

## Score Calibration Update

A no-GPU score calibration pass was added after the first full run. It reuses the existing raw judge outputs and improves prediction in two stages:

1. `calibrated_t475`: deterministic code clamps plus programmatic pass/fail calculation with a validation-selected score threshold of `4.75`;
2. `validation_logistic`: a small L2 logistic calibrator trained only on validation rows, then applied to test/all rows.

Calibration artifacts:

```text
src/rubric/calibrate_llm_judge_scores.py
scripts/run_phase2_score_calibration.sh
data/rubrics/phase2/mbpp_hidden_llm_judge_scores_validation_test_calibrated_t475.jsonl
data/rubrics/phase2/mbpp_hidden_llm_judge_metrics_validation_test_calibrated_t475.json
data/rubrics/phase2/mbpp_hidden_llm_judge_scores_validation_logistic.jsonl
data/rubrics/phase2/mbpp_hidden_llm_judge_metrics_validation_logistic.json
```

Current comparison:

| Setting | Split | AUC | Accuracy | Kappa | Predicted pass rate |
| --- | --- | ---: | ---: | ---: | ---: |
| Original LLM boolean | all | 0.603597 | 0.572881 | 0.152540 | 0.696610 |
| Deterministic + threshold | all | 0.655815 | 0.624294 | 0.253477 | 0.661017 |
| Validation logistic calibrator | all | 0.660290 | 0.641808 | 0.290174 | 0.727119 |
| Original LLM boolean | test | 0.615479 | 0.578000 | 0.164052 | 0.700667 |
| Deterministic + threshold | test | 0.666387 | 0.628667 | 0.263111 | 0.663333 |
| Validation logistic calibrator | test | 0.666834 | 0.647333 | 0.302148 | 0.723333 |

The recommended current score artifact is:

```text
data/rubrics/phase2/mbpp_hidden_llm_judge_scores_validation_logistic.jsonl
```

The logistic calibrator improves Kappa and accuracy, but it is still permissive: the predicted pass rate remains higher than the verifier pass rate. It should be reported as an optimized calibration baseline, not as a solved judge.

## Acceptance Criteria

The full Phase 2 run should be accepted only if:

- the score file contains the expected validation+test row count;
- `audit.valid == true`;
- `prompt_leakage_count == 0`;
- JSON parse failures and repaired outputs are either zero or explained by the audit;
- the judge never receives verifier labels, hidden tests, exact assertions, or private diagnostics;
- metrics are reported against verifier pass/fail labels.

## No Manual Intervention

If LLM rubric generation or LLM judge output has formatting errors, the scripts should repair or fall back automatically. Manual editing must not be part of taxonomy or rubric generation; humans should only read final reports and audits.

## Judge Improvement Plan

The current best calibrated judge is useful as a Phase 2 baseline, but it is not reliable enough to replace the verifier or to create full training labels. The main weakness is still over-acceptance: many verifier-failed responses receive high judge scores or high calibrated pass probability.

The next improvements should keep the data boundary fixed:

```text
train responses      -> allowed for taxonomy, rubric induction, preference construction, and training
validation responses -> allowed for threshold selection, calibration, and ablation selection
test responses       -> final evaluation only; no tuning and no training
```

### 1. Targeted rerun for malformed judge outputs

The full run still contains 205 repaired rows, including 203 rows where the model repeated public input instead of producing the required judgment JSON. These rows should be rerun with a stricter prompt when GPU is idle.

Acceptance criteria:

- repaired row count decreases substantially;
- JSON parse failures remain zero;
- prompt leakage remains zero;
- test metrics are reported after applying the same validation-selected calibration procedure.

### 2. Stronger evidence-first judge prompt

The current judge can assign high scores without demonstrating evidence. The next prompt should require each dimension to include:

- visible code evidence;
- an explicit possible failure risk;
- the final 1-5 score.

The prompt should forbid score 5 unless the judge names concrete evidence from the submitted code. This is especially important for `algorithmic_wrong_value`, `numeric_formula_correctness`, `edge_case_boundary_handling`, and `string_regex_pattern_logic`.

### 3. Deterministic checker expansion

Machine-checkable dimensions should not rely only on LLM judgment. The deterministic layer should expand beyond the current syntax/interface/runtime clamps:

- AST parseability and duplicated top-level definitions;
- required function/class name and arity from public signatures;
- missing imports for common namespaces such as `re`, `math`, `heapq`, `itertools`, and `collections`;
- markdown fence or explanation pollution only when extraction cannot recover valid code;
- explicit stubs such as `pass`, `TODO`, or `NotImplementedError`;
- suspicious multiple definitions of the same public function.

These checks should clamp only when the evidence is unambiguous, to avoid reducing agreement with the verifier because of harmless formatting.

### 4. Task-specific public checklist

The taxonomy-level rubric is too generic to fully evaluate task-specific semantics. For each held-out task, generate a checklist from public information only:

```text
task description + public interface -> task-specific checklist
```

The checklist should include:

- expected input and output shape;
- core transformation or predicate;
- boundary cases implied by the public wording;
- common pitfalls inferred from the task wording;
- regex/string behavior when relevant.

The checklist must not use `test_list`, hidden assertions, expected values, private diagnostics, or verifier failure messages.

### 5. Lightweight public sanity simulation

For semantic dimensions, the judge should construct 1-3 simple sanity cases from the public task description and manually simulate the submitted code. This is not a hidden-test verifier; it is a public reasoning aid.

Expected benefit:

- fewer high-score verifier failures;
- better separation for algorithmic, numeric, edge-case, and regex/string dimensions;
- more useful rationales for error analysis.

Risk:

- the LLM may create incorrect sanity cases. The prompt should require simple cases and explain that they are heuristic evidence, not ground-truth tests.

### 6. Validation-only calibration

Any threshold, gate, or calibrator must be selected on validation only. Test is reserved for final evaluation.

Current best calibration:

```text
source score file = mbpp_hidden_llm_judge_scores_validation_test_calibrated_t475.jsonl
calibrator = L2 logistic regression
calibration split = validation
threshold = 0.535
test AUC = 0.666834
test accuracy = 0.647333
test Kappa = 0.302148
```

Future calibrators may use richer features, but they must not use test labels during feature selection, threshold selection, or model selection.

### 7. High-confidence training boundary

Until judge quality improves, judge outputs should not be used as full training labels. Safe uses are:

- high-confidence failure mining;
- auxiliary reward features;
- ranking diagnostics;
- selecting candidates for human or verifier review.

Unsafe uses are:

- judge-only positive labels for DPO;
- full replacement of verifier pass/fail labels;
- training on validation/test judge labels or verifier labels;
- repeated test-driven calibration.

For preference construction, use train split only and prefer verifier-confirmed pairs. Judge-only signals should be filtered by high-confidence thresholds and audited before entering training.
