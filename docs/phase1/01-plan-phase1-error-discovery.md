# Phase 1: MBPP Hidden-Tests k=3 Error Discovery, Attribution Clustering, and Refined Taxonomy Plan

## Objective

The formal Phase 1 objective is to build a non-leaking error-discovery pipeline for MBPP code-generation tasks. The pipeline generates diverse model responses, labels failures with an external verifier, discovers a raw error taxonomy from train failures, performs semantic consolidation, and then refines the taxonomy into rubric-operational categories that Phase 2 can consume directly.

Phase 1 may keep the static rubric baseline as an engineering sanity check. Formal LLM rubric generation and LLM rubric judging belong to Phase 2.

HumanEval+ is no longer part of the formal Phase 1 evaluation set. HumanEval+ results should be treated only as earlier exploration and should not be used as the current Phase 1 conclusion.

## Core Experiment Contract

| Item | Current formal setting |
| --- | --- |
| Dataset | MBPP only |
| Split | train 374 + test 500 + validation 90 = 964 |
| Prompt | hidden-tests mode; prompts contain no `assert` |
| Verifier | MBPP `test_list`, used only inside verification |
| Sampling scale | k=3 |
| Response count | 964 x 3 = 2892 |
| Unique key | `response_id = id + sample_id` |
| Model | Qwen2.5-7B-Instruct |
| Generation parameters | temperature=0.7, top_p=0.9, max_tokens=512 |

## Current Status

Date: 2026-07-09

Phase 1 has completed the formal k=3 run, raw taxonomy discovery, automatic consolidation, and rubric-operational refinement. Phase 1.5 has completed a k=5 stability replication. Current conclusions should use the safe artifacts under the `phase1` and `phase1_5` directories, not older visible-tests or MBPP+HumanEval+ mixed results.

| Stage | Setting | Responses | Pass | Fail | Response pass rate | Task pass@k | Train failures | Taxonomy |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Phase 1 | MBPP hidden-tests, k=3 | 2892 | 1470 | 1422 | 0.508299 | 0.594398 | 519 | 17 raw clusters -> 8 refined categories |
| Phase 1.5 | MBPP hidden-tests, k=5 | 4820 | 2427 | 2393 | 0.503527 | 0.614108 | 867 | 24 clusters |

Completed key artifacts:

- `data/responses/phase1_mbpp_hidden_qwen25_k3_labeled.jsonl`
- `data/analysis/phase1/mbpp_hidden_qwen25_k3_failures_safe.jsonl`
- `data/analysis/phase1/mbpp_hidden_train_qwen25_k3_failures_with_safe_llm_summaries.jsonl`
- `data/analysis/phase1/mbpp_hidden_train_qwen25_k3_discovered_taxonomy_safe.yaml`
- `data/analysis/phase1/mbpp_hidden_train_qwen25_k3_taxonomy_consolidated.yaml`
- `data/analysis/phase1/mbpp_hidden_train_qwen25_k3_taxonomy_consolidated_audit.json`
- `data/analysis/phase1/mbpp_hidden_train_qwen25_k3_taxonomy_consolidated_response_assignments.jsonl`
- `data/analysis/phase1/mbpp_hidden_train_qwen25_k3_taxonomy_refined_for_rubric.yaml`
- `data/analysis/phase1/mbpp_hidden_train_qwen25_k3_taxonomy_refined_for_rubric_audit.json`
- `data/analysis/phase1/mbpp_hidden_train_qwen25_k3_taxonomy_refined_response_assignments.jsonl`
- `data/responses/phase1_5_mbpp_hidden_qwen25_k5_labeled.jsonl`
- `data/analysis/phase1_5/mbpp_hidden_train_qwen25_k5_discovered_taxonomy_safe_v2.yaml`
- `data/rubrics/phase1/mbpp_hidden_auto_rubric_refined.json`

Safe artifacts do not contain `test_list`, `test`, `test_setup_code`, or `private_diagnostics`. Debug copies with exact assertions, actual values, and expected values are kept only under `data/private_diagnostics/`; they must not enter attribution, clustering, training, or report-facing pipelines.

## Key Corrections

### 1. Preventing test-directed generation

The older MBPP prompt exposed assertions from `test_list` to the model. That made the measured rate a visible-test pass rate, allowed special-casing against the visible assertions, polluted the error taxonomy with visible-test behavior, and made rubric evaluation unsuitable for hidden-test self-evaluation.

The current formal prompt contains only:

- natural-language task description;
- public interface signatures extracted from canonical solutions;
- no concrete assertions, input/output examples, or expected values.

`test_list` remains in JSONL records only for the verifier.

### 2. k=3 downstream alignment

With k=3, each problem id has three responses. All downstream files must align by `response_id`, not only by `id`.

Standard fields:

```json
{
  "response_id": "mbpp/train/601__sample0",
  "id": "mbpp/train/601",
  "sample_id": 0,
  "dataset": "mbpp",
  "split": "train",
  "interface_signatures": ["class Pair", "  def __init__(self, a, b)", "def max_chain_length(arr, n)"]
}
```

### 3. Avoiding rule-label leakage in discovery

The rule label `error_pattern` may be used as a baseline and output metadata, but it is not a formal clustering feature and does not directly determine cluster names.

The formal discovery script is `discover_error_taxonomy.py`: it asks the LLM for free-form root-cause summaries, then clusters `llm_summary + failure_type + error` using TF-IDF/SVD/HDBSCAN. `error_pattern` is retained only for audit and interpretation.

## Step 1: Data Preparation

Command:

```bash
/data2/acm-group-3/miniconda3/envs/rubric/bin/python \
  src/data-prep/prepare_coding_prompts.py \
  --raw-dir data/raw \
  --output data/processed/coding_prompts.jsonl
```

Expected result:

- 964 prompts;
- every row has `dataset == "mbpp"`;
- every row has `prompt_mode == "mbpp_hidden_tests"`;
- prompt text contains zero `assert` occurrences;
- prompt text includes public interface signatures;
- each record still retains `test_list` for verifier-only use.

Verification command:

```bash
python - <<'PY'
import json
rows = [json.loads(line) for line in open("data/processed/coding_prompts.jsonl", encoding="utf-8") if line.strip()]
print(len(rows))
print(sorted(set(row["dataset"] for row in rows)))
print(sum("assert " in row["prompt"] for row in rows))
print(sorted(set(row.get("prompt_mode") for row in rows)))
PY
```

Current verified result:

```text
964
['mbpp']
0
['mbpp_hidden_tests']
```

## Step 2: k=3 Response Generation

Command template:

```bash
CUDA_VISIBLE_DEVICES=<GPU_ID> \
PATH=/data2/acm-group-3/miniconda3/envs/rubric/bin:$PATH \
XDG_CACHE_HOME=/tmp/rubric-cache \
HF_HOME=/tmp/rubric-cache/huggingface \
TRANSFORMERS_CACHE=/tmp/rubric-cache/huggingface \
TMPDIR=/tmp/rubric-tmp \
/data2/acm-group-3/miniconda3/envs/rubric/bin/python \
  src/generation/vllm_smoke_generate.py \
  --model models/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28 \
  --input data/processed/coding_prompts.jsonl \
  --output data/responses/mbpp_hidden_qwen25_k3.jsonl \
  --limit 964 \
  --k 3 \
  --temperature 0.7 \
  --top-p 0.9 \
  --max-tokens 512 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.40
```

Expected output:

```text
2892 responses
```

Each response must include:

- `response_id`
- `id`
- `sample_id`
- `prompt_mode`
- `interface_signatures`
- `generated_code`
- `temperature`
- `top_p`
- `seed`

## Step 3: Verifier Labeling

Command:

```bash
/data2/acm-group-3/miniconda3/envs/rubric/bin/python \
  src/verification/verify_mbpp_smoke.py \
  --input data/responses/mbpp_hidden_qwen25_k3.jsonl \
  --output data/responses/phase1_mbpp_hidden_qwen25_k3_labeled.jsonl \
  --timeout 5
```

Output fields:

- `response_id`
- `passed`
- `failure_type`
- `error`
- `extracted_code`
- `safe_diagnostics`
- `private_diagnostics`

The current verifier is a lightweight multiprocessing executor, not a Docker sandbox. `safe_diagnostics` may be used for attribution; `private_diagnostics` is only for local debugging.

## Step 4: Safe Failure Artifacts and Initial Rule Taxonomy

Command:

```bash
/data2/acm-group-3/miniconda3/envs/rubric/bin/python \
  src/error-analysis/build_failure_artifacts.py \
  --input data/responses/phase1_mbpp_hidden_qwen25_k3_labeled.jsonl \
  --failure-output data/analysis/phase1/mbpp_hidden_qwen25_k3_failures_safe.jsonl \
  --summary-output data/analysis/phase1/mbpp_hidden_qwen25_k3_summary.json \
  --taxonomy-output data/analysis/phase1/mbpp_hidden_qwen25_k3_taxonomy_initial_safe.yaml
```

The generated `error_pattern` is a rule baseline, not the sole evidence for the final model-discovered taxonomy. Default outputs exclude hidden tests and private diagnostics.

## Step 5: LLM Attribution and Model-Discovered Raw Taxonomy

Default command:

```bash
/data2/acm-group-3/miniconda3/envs/rubric/bin/python \
  src/error-analysis/discover_error_taxonomy.py \
  --failures data/analysis/phase1/mbpp_hidden_train_qwen25_k3_failures_safe.jsonl \
  --stage1-output data/analysis/phase1/mbpp_hidden_train_qwen25_k3_failures_with_safe_llm_summaries.jsonl \
  --assignments-output data/analysis/phase1/mbpp_hidden_train_qwen25_k3_discovered_clusters_safe.jsonl \
  --taxonomy-output data/analysis/phase1/mbpp_hidden_train_qwen25_k3_discovered_taxonomy_safe.yaml \
  --summary-output data/analysis/phase1/mbpp_hidden_train_qwen25_k3_discovered_taxonomy_summary_safe.json \
  --summarize-model models/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.35 \
  --temperature 0.0 \
  --max-tokens 128 \
  --batch-size 64 \
  --min-cluster-size 8 \
  --min-samples 3 \
  --max-cluster-ratio 0.25
```

Outputs should include cluster frequencies, failure-type distributions, rule-pattern distributions for auxiliary audit, representative examples, top terms, and `response_id`-level assignments.

## Step 6: Taxonomy Consolidation and Rubric-Operational Refinement

This step is part of Phase 1. The final core Phase 1 artifact is not the raw discovered taxonomy; it is the refined rubric-operational taxonomy:

```text
data/analysis/phase1/mbpp_hidden_train_qwen25_k3_taxonomy_refined_for_rubric.yaml
```

Raw clusters must not be used directly as the Phase 2 rubric input. The full Phase 1 taxonomy pipeline is:

```text
safe train failures
-> LLM root-cause summaries
-> TF-IDF/SVD/HDBSCAN clustering
-> raw discovered taxonomy
-> LLM semantic consolidation + deterministic coverage audit
-> consolidated taxonomy
-> per-category LLM refinement + deterministic quality audit + LLM revision/targeted repair
-> rubric-operational taxonomy
```

Consolidation command:

```bash
/data2/acm-group-3/miniconda3/envs/rubric/bin/python \
  src/error-analysis/consolidate_taxonomy.py \
  --taxonomy data/analysis/phase1/mbpp_hidden_train_qwen25_k3_discovered_taxonomy_safe.yaml \
  --raw-assignments data/analysis/phase1/mbpp_hidden_train_qwen25_k3_discovered_clusters_safe.jsonl \
  --output data/analysis/phase1/mbpp_hidden_train_qwen25_k3_taxonomy_consolidated.yaml \
  --audit-output data/analysis/phase1/mbpp_hidden_train_qwen25_k3_taxonomy_consolidated_audit.json \
  --cluster-mapping-output data/analysis/phase1/mbpp_hidden_train_qwen25_k3_taxonomy_consolidated_cluster_mapping.jsonl \
  --response-assignments-output data/analysis/phase1/mbpp_hidden_train_qwen25_k3_taxonomy_consolidated_response_assignments.jsonl \
  --raw-llm-output data/analysis/phase1/mbpp_hidden_train_qwen25_k3_taxonomy_consolidation_raw_response.txt \
  --model models/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.30 \
  --temperature 0.0 \
  --max-tokens 2048 \
  --min-categories 6 \
  --max-categories 8
```

Refinement command:

```bash
/data2/acm-group-3/miniconda3/envs/rubric/bin/python \
  src/error-analysis/refine_taxonomy_for_rubric.py \
  --taxonomy data/analysis/phase1/mbpp_hidden_train_qwen25_k3_taxonomy_consolidated.yaml \
  --assignments data/analysis/phase1/mbpp_hidden_train_qwen25_k3_taxonomy_consolidated_response_assignments.jsonl \
  --failures data/analysis/phase1/mbpp_hidden_train_qwen25_k3_failures_safe.jsonl \
  --output data/analysis/phase1/mbpp_hidden_train_qwen25_k3_taxonomy_refined_for_rubric.yaml \
  --audit-output data/analysis/phase1/mbpp_hidden_train_qwen25_k3_taxonomy_refined_for_rubric_audit.json \
  --response-assignments-output data/analysis/phase1/mbpp_hidden_train_qwen25_k3_taxonomy_refined_response_assignments.jsonl \
  --raw-llm-output data/analysis/phase1/mbpp_hidden_train_qwen25_k3_taxonomy_refinement_raw_response.txt \
  --model models/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.30 \
  --temperature 0.4 \
  --max-tokens 2048 \
  --max-num-seqs 6 \
  --candidates-per-category 3 \
  --revision-candidates 2 \
  --targeted-repair-candidates 2 \
  --max-examples-per-category 5
```

Each refined taxonomy category must contain an operational definition, failure mechanism, common manifestations, judge checklist, 1-5 score anchors, and positive/negative boundaries.

The quality audit must guarantee complete category coverage, unchanged response-assignment count, no private/test leakage, no overly generic refinement text, and complete schema coverage.

Current Phase 1 refined taxonomy audit is `valid = true` and covers 519/519 train failure responses. The latest refinement accepted 2 categories initially, 4 after revision, and 2 after targeted repair, with 0 template fallback categories.

## Step 7: Static Rubric Baseline

`generate_auto_rubric.py` is still a semi-automatic template-based generator with relatively fixed dimensions. Formal reporting should call it:

```text
taxonomy-informed rubric template
```

It should not be described as a fully free-form model-discovered rubric.

Command:

```bash
/data2/acm-group-3/miniconda3/envs/rubric/bin/python \
  src/rubric/generate_auto_rubric.py \
  --taxonomy data/analysis/phase1/mbpp_hidden_train_qwen25_k3_taxonomy_refined_for_rubric.yaml \
  --output data/rubrics/phase1/mbpp_hidden_auto_rubric_refined.json \
  --generic-output data/rubrics/phase1/mbpp_hidden_generic_rubric.json \
  --random-output data/rubrics/phase1/mbpp_hidden_random_rubric_ablation.json
```

## Step 8: Static Self-Evaluation Baseline

`evaluate_rubric_static.py` is a static heuristic scorer, not a true LLM rubric judge. It is useful as an engineering sanity check, but it is not final evidence that a model can read a rubric and self-evaluate.

Command:

```bash
/data2/acm-group-3/miniconda3/envs/rubric/bin/python \
  src/rubric/evaluate_rubric_static.py \
  --labeled data/responses/phase1_mbpp_hidden_qwen25_k3_labeled.jsonl \
  --failures data/analysis/phase1/mbpp_hidden_qwen25_k3_failures_safe.jsonl \
  --rubric data/rubrics/phase1/mbpp_hidden_auto_rubric_refined.json \
  --scores-output data/rubrics/phase1/mbpp_hidden_auto_rubric_scores_static.jsonl \
  --metrics-output data/rubrics/phase1/mbpp_hidden_auto_rubric_eval_metrics.json
```

Current static baseline result:

| Rubric | Coverage | Static AUC | Kappa | Accuracy |
| --- | ---: | ---: | ---: | ---: |
| auto taxonomy-informed | 1.000 | 0.596642 | 0.189652 | 0.600277 |
| generic | 0.000 | 0.508823 | 0.017383 | 0.501383 |
| random ablation | 1.000 | 0.596642 | 0.189652 | 0.600277 |

The static scorer does not actually read rubric text semantics, which is why the auto and random rubric variants score identically. It is only a sanity check, not the main self-evaluation result.

Formal self-evaluation is Phase 2 and requires an LLM to score each response after reading the generated rubric, without exposing verifier labels or assertions, then compute AUC, Cohen's Kappa, and accuracy against verifier pass/fail labels.

## Completed Code-Level Fixes

- `prepare_coding_prompts.py` defaults to MBPP-only hidden-tests mode.
- `vllm_smoke_generate.py` defaults to k=3 and writes `response_id`.
- `verify_mbpp_smoke.py` outputs safe/private diagnostics and aligns by `response_id`.
- `build_failure_artifacts.py` emits safe artifacts by default; private test fields require an explicit flag.
- `discover_error_taxonomy.py` supports safe attribution, private-field rejection, and recursive cluster-size control.
- `consolidate_taxonomy.py` supports raw-cluster consolidation, coverage audit, broad-category repair, and response-level assignment inheritance.
- `refine_taxonomy_for_rubric.py` supports per-category multi-candidate LLM refinement, bad-phrase masking, revision, targeted repair, category-conditioned quality gates, raw-output reuse, and rubric-operational schema audit.
- `generate_auto_rubric.py` is compatible with the newer `error_patterns` field and fine-grained error labels.

## Current Risks

1. Older k=1, HumanEval+, and visible-tests results cannot be used as current Phase 1 conclusions.
2. The verifier is not a full sandbox; executing untrusted code still carries risk.
3. The static rubric scorer is not a true LLM self-evaluation judge.
4. Raw cluster names remain keyword-like machine-discovery labels. The formal taxonomy is the audited LLM-consolidated and rubric-operationally refined version.
5. Direct vLLM runs with the environment Python require the conda environment `bin` directory in `PATH`; otherwise FlashInfer JIT can fail if it cannot find `ninja`.
