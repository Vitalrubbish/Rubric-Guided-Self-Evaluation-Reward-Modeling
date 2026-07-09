# Phase 1 Pipeline Overview

Date: 2026-07-09

## Scope

Phase 1 is the MBPP hidden-tests error-discovery and taxonomy-bootstrap pipeline. The current source tree keeps only the scripts needed for this stage.

Core scripts:

```text
src/data-prep/prepare_coding_prompts.py
src/generation/vllm_smoke_generate.py
src/verification/verify_mbpp_smoke.py
src/error-analysis/build_failure_artifacts.py
src/error-analysis/discover_error_taxonomy.py
src/error-analysis/consolidate_taxonomy.py
src/error-analysis/refine_taxonomy_for_rubric.py
```

Optional reporting/baseline scripts:

```text
src/analysis-reporting/compute_coding_task_metrics.py
src/rubric/generate_auto_rubric.py
src/rubric/evaluate_rubric_static.py
```

Training, self-play, math/GSM8K transfer, revision, preference-building, and final-report aggregation scripts are intentionally outside the Phase 1 source surface.

## Run Command

Use the pipeline runner from the repository root:

```bash
SKIP_EXISTING=1 GPU=2 scripts/run_phase1_pipeline.sh
```

Useful environment variables:

| Variable | Default | Meaning |
| --- | --- | --- |
| `PYTHON` | `/data2/acm-group-3/miniconda3/envs/rubric/bin/python` | Python executable |
| `MODEL` | local Qwen2.5-7B-Instruct snapshot | vLLM model path |
| `GPU` | `2` | CUDA device used by generation and LLM taxonomy steps |
| `K` | `3` | responses per MBPP task |
| `LIMIT` | `964` | number of MBPP prompts |
| `SKIP_EXISTING` | `0` | set to `1` to reuse existing artifacts |
| `DRY_RUN` | `0` | set to `1` to print commands without running them |
| `RUN_REPORTS` | `1` | set to `0` to skip response/task metrics |
| `RUN_STATIC_BASELINE` | `1` | set to `0` to stop after refined taxonomy |

For a command preview:

```bash
DRY_RUN=1 scripts/run_phase1_pipeline.sh
```

## Core Pipeline

The conceptual Phase 1 pipeline is:

```text
Prepare hidden-test MBPP prompts
-> Generate k=3 responses
-> Verify responses and build safe train failures
-> Discover raw error taxonomy with LLM attribution + clustering
-> Consolidate raw clusters into rubric categories
-> Refine categories into rubric-operational taxonomy
```

The model never receives MBPP `test_list`, concrete `assert` statements, exact expected values, or private diagnostics during attribution, clustering, consolidation, or refinement.

## Stage Meanings

| Stage | Script(s) | Meaning | Main output |
| --- | --- | --- | --- |
| Prepare hidden-test MBPP prompts | `prepare_coding_prompts.py` | Converts MBPP raw data into model prompts that expose task text and public interface only. This prevents visible-test or assert leakage. | `data/processed/coding_prompts.jsonl` |
| Generate k=3 responses | `vllm_smoke_generate.py` | Samples three model answers per task to create enough behavioral diversity and failure examples for clustering. | `data/responses/mbpp_hidden_qwen25_k3.jsonl` |
| Verify responses and build safe train failures | `verify_mbpp_smoke.py`, `build_failure_artifacts.py` | Uses MBPP tests only inside the verifier, labels pass/fail, then strips private verifier details from failure artifacts. The train split is used for taxonomy discovery. | safe failure JSONL files |
| Discover raw error taxonomy | `discover_error_taxonomy.py` | First asks the LLM for free-form root-cause summaries, then clusters those summaries with TF-IDF/SVD/HDBSCAN. This is the model-driven error discovery step. | discovered clusters and raw taxonomy YAML |
| Consolidate raw clusters | `consolidate_taxonomy.py` | Uses an LLM to merge raw clusters into fewer rubric-usable categories, while deterministic audit enforces one-to-one raw-cluster coverage. | consolidated taxonomy and response assignments |
| Refine rubric-operational taxonomy | `refine_taxonomy_for_rubric.py` | Converts each consolidated category into rubric-ready definitions, mechanisms, checklists, score anchors, and boundaries. Quality gates trigger LLM revision/targeted repair when output is too broad. | refined taxonomy for Phase 2 |

This is the minimal Phase 1 artifact needed by Phase 2:

```text
data/analysis/phase1/mbpp_hidden_train_qwen25_k3_taxonomy_refined_for_rubric.yaml
```

## Optional Reports And Baselines

The runner can also produce reporting artifacts and a static rubric sanity check:

| Optional step | Script | Why it exists | Disable with |
| --- | --- | --- | --- |
| Response/task metrics | `compute_coding_task_metrics.py` | Reports response pass rate, task pass@k, split-level pass rates, and failure distributions. It is useful for experiment reporting but not required to build the taxonomy. | `RUN_REPORTS=0` |
| Initial rule taxonomy side output | `build_failure_artifacts.py` | Provides rule-pattern counts as a diagnostic baseline. These labels are not used as the main clustering feature. | Not separately disabled; it is emitted with safe failures. |
| Static rubric baseline | `generate_auto_rubric.py`, `evaluate_rubric_static.py` | Provides a weak sanity-check baseline for rubric coverage and static discriminability. It is not the final Phase 2 LLM rubric judge. | `RUN_STATIC_BASELINE=0` |

For the simplified core run:

```bash
RUN_REPORTS=0 RUN_STATIC_BASELINE=0 SKIP_EXISTING=1 GPU=2 scripts/run_phase1_pipeline.sh
```

## Core Outputs

Generation and verification:

```text
data/processed/coding_prompts.jsonl
data/responses/mbpp_hidden_qwen25_k3.jsonl
data/responses/phase1_mbpp_hidden_qwen25_k3_labeled.jsonl
```

Safe failures:

```text
data/analysis/phase1/mbpp_hidden_qwen25_k3_failures_safe.jsonl
data/analysis/phase1/mbpp_hidden_train_qwen25_k3_failures_safe.jsonl
```

Raw discovery:

```text
data/analysis/phase1/mbpp_hidden_train_qwen25_k3_failures_with_safe_llm_summaries.jsonl
data/analysis/phase1/mbpp_hidden_train_qwen25_k3_discovered_clusters_safe.jsonl
data/analysis/phase1/mbpp_hidden_train_qwen25_k3_discovered_taxonomy_safe.yaml
data/analysis/phase1/mbpp_hidden_train_qwen25_k3_discovered_taxonomy_summary_safe.json
```

Consolidated and refined taxonomy:

```text
data/analysis/phase1/mbpp_hidden_train_qwen25_k3_taxonomy_consolidated.yaml
data/analysis/phase1/mbpp_hidden_train_qwen25_k3_taxonomy_consolidated_audit.json
data/analysis/phase1/mbpp_hidden_train_qwen25_k3_taxonomy_consolidated_response_assignments.jsonl
data/analysis/phase1/mbpp_hidden_train_qwen25_k3_taxonomy_refined_for_rubric.yaml
data/analysis/phase1/mbpp_hidden_train_qwen25_k3_taxonomy_refined_for_rubric_audit.json
data/analysis/phase1/mbpp_hidden_train_qwen25_k3_taxonomy_refined_response_assignments.jsonl
```

Optional report and static rubric baseline outputs:

```text
data/analysis/phase1/mbpp_hidden_qwen25_k3_task_metrics.json
data/rubrics/phase1/mbpp_hidden_auto_rubric_refined.json
data/rubrics/phase1/mbpp_hidden_generic_rubric.json
data/rubrics/phase1/mbpp_hidden_random_rubric_ablation.json
data/rubrics/phase1/mbpp_hidden_auto_rubric_eval_metrics.json
data/rubrics/phase1/mbpp_hidden_generic_rubric_eval_metrics.json
data/rubrics/phase1/mbpp_hidden_random_rubric_eval_metrics.json
```

## Quality Gates

Phase 1 must satisfy these invariants before its artifacts are used by Phase 2:

- `data/processed/coding_prompts.jsonl` contains 964 MBPP rows.
- Prompt text contains no verifier `assert`.
- Safe failure artifacts do not contain `test_list`, `test_setup_code`, `private_diagnostics`, or exact expected values.
- Cluster assignments and train failures align by `response_id`.
- Consolidated taxonomy covers every raw cluster exactly once.
- Refined taxonomy covers every consolidated category and preserves all response assignments.
- Refinement audit reports no schema flags, generic text flags, or private leakage flags.

Current refined taxonomy status:

```text
valid = true
initial_accepted_categories = 2
revised_accepted_categories = 4
targeted_repair_accepted_categories = 2
template_fallback_categories = []
```

## Notes

`evaluate_rubric_static.py` is a static sanity-check baseline, not the final LLM rubric judge. Phase 2 should consume `mbpp_hidden_train_qwen25_k3_taxonomy_refined_for_rubric.yaml` to generate rubric text and evaluate it with a real LLM judge on held-out responses.
