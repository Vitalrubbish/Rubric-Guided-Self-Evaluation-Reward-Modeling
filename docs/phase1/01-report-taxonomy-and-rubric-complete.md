# B/C Work Complete: Refined Taxonomy and Auto Rubric

Date: 2026-07-02

## 1. Inputs

The work starts from the corrected full baseline:

```text
data/responses/coding_all_qwen25_vllm_k1_labeled_v2.jsonl
data/analysis/coding_failures_qwen25_k1.jsonl
data/analysis/coding_error_taxonomy_initial.yaml
```

Baseline recap:

| Dataset | Total | Passed | Failed | Pass@1 |
|---|---:|---:|---:|---:|
| MBPP | 964 | 504 | 460 | 52.28% |
| HumanEval+ | 164 | 73 | 91 | 44.51% |
| Total | 1128 | 577 | 551 | 51.15% |

## 2. B Work: Failure Clustering and Refined Taxonomy

### Method

The refined taxonomy is built with a reproducible unsupervised pipeline:

```text
failure text
  -> TF-IDF ngrams
  -> TruncatedSVD
  -> normalized dense features
  -> HDBSCAN, with KMeans fallback
  -> cluster labeling from dominant rule pattern, failure type, and top terms
```

The failure text uses:

- dataset,
- failure type,
- initial rule pattern,
- verifier error,
- prompt,
- extracted/generated code.

### Outputs

```text
data/analysis/failure_clusters_qwen25_k1.jsonl
data/analysis/coding_error_taxonomy_refined.yaml
data/analysis/coding_error_taxonomy_refined_summary.json
```

### Result

The 551 failures were grouped into 18 clusters.

Top clusters:

| Cluster | Name | Count | Ratio |
|---|---|---:|---:|
| cluster_00 | syntax_malformed_code | 92 | 16.70% |
| cluster_01 | logic_wrong_output | 85 | 15.43% |
| cluster_02 | syntax_error_mixed_string_return_split | 43 | 7.80% |
| cluster_03 | logic_wrong_output | 39 | 7.08% |
| cluster_04 | mixed_arr_arr arr_left_right | 32 | 5.81% |
| cluster_05 | syntax_malformed_code | 32 | 5.81% |
| cluster_06 | mixed_result_return result_lst_append | 30 | 5.44% |
| cluster_07 | logic_wrong_output | 30 | 5.44% |
| cluster_08 | mixed_count_return count_count return_code | 26 | 4.72% |
| cluster_09 | syntax_error_mixed_list_int_lst | 25 | 4.54% |

The refined taxonomy is ready for manual inspection or LLM-assisted naming in the next pass.

## 3. C Work: Auto Rubric Generation

### Method

The auto rubric is generated from the refined taxonomy and initial rule patterns.

It contains six task-specific dimensions:

1. Functional Correctness and Edge-Case Coverage
2. Syntax Validity and Parseability
3. Interface and Test Contract Compliance
4. Runtime Dependency and API Safety
5. Termination and Complexity Control
6. Output Cleanliness and Single-Solution Formatting

Each dimension contains:

- dimension name,
- description,
- linked error patterns,
- 1-5 scoring criteria,
- positive example,
- negative examples taken from taxonomy clusters.

### Outputs

```text
data/rubrics/auto_rubric_refined.json
data/rubrics/generic_rubric.json
data/rubrics/random_rubric_ablation.json
```

## 4. Rubric Evaluation

### Scoring Notes

Two evaluation modes are provided:

1. `static_*`: a weak self-evaluation proxy that does not run unit tests. It checks parseability, interface compliance, output cleanliness, obvious missing dependencies, and simple risk patterns.
2. `verifier_informed_*`: an upper-bound reference that maps known verifier failures to rubric dimensions. This is not a true self-evaluation score.

The static score is the useful early signal for rubric discriminability.

### Outputs

```text
data/rubrics/auto_rubric_scores_static.jsonl
data/rubrics/auto_rubric_eval_metrics.json
data/rubrics/generic_rubric_scores_static.jsonl
data/rubrics/generic_rubric_eval_metrics.json
data/rubrics/random_rubric_scores_static.jsonl
data/rubrics/random_rubric_eval_metrics.json
```

### Results

| Rubric | Coverage | Static AUC | Static Kappa | Static Accuracy |
|---|---:|---:|---:|---:|
| Auto rubric | 100.00% | 0.8013 | 0.5249 | 76.51% |
| Generic rubric | 0.00% | 0.6597 | 0.3164 | 65.51% |
| Random ablation | 100.00% | 0.8013 | 0.5249 | 76.51% |

Important caveat:

The current static scorer does not use the shuffled pattern links, so the random ablation is not meaningful yet. A real random-rubric ablation should be evaluated with an LLM judge or a dimension-specific scorer that actually reads the rubric text.

## 5. Scripts Added

```text
scripts/refine_error_taxonomy.py
scripts/generate_auto_rubric.py
scripts/evaluate_rubric_static.py
```

Run commands:

```bash
python scripts/refine_error_taxonomy.py \
  --failures data/analysis/coding_failures_qwen25_k1.jsonl \
  --assignments-output data/analysis/failure_clusters_qwen25_k1.jsonl \
  --taxonomy-output data/analysis/coding_error_taxonomy_refined.yaml \
  --summary-output data/analysis/coding_error_taxonomy_refined_summary.json

python scripts/generate_auto_rubric.py \
  --taxonomy data/analysis/coding_error_taxonomy_refined.yaml \
  --output data/rubrics/auto_rubric_refined.json \
  --generic-output data/rubrics/generic_rubric.json \
  --random-output data/rubrics/random_rubric_ablation.json

python scripts/evaluate_rubric_static.py \
  --labeled data/responses/coding_all_qwen25_vllm_k1_labeled_v2.jsonl \
  --failures data/analysis/coding_failures_qwen25_k1.jsonl \
  --rubric data/rubrics/auto_rubric_refined.json \
  --scores-output data/rubrics/auto_rubric_scores_static.jsonl \
  --metrics-output data/rubrics/auto_rubric_eval_metrics.json
```

## 6. Next Hand-Off

For the next phase:

- B can manually/LLM-refine cluster names and merge clusters with the same semantic cause.
- C can replace static scoring with an LLM rubric judge on a 100-200 sample subset.
- D can use `auto_rubric_scores_static.jsonl` to construct early preference pairs, but should treat it as a proxy signal, not final reward.

The immediate next technical target is:

```text
LLM judge scores 200 responses using auto_rubric_refined.json,
then compare against verifier pass/fail with AUC and Cohen's Kappa.
```
