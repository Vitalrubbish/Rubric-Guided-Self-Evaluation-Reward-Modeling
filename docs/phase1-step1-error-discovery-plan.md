# Phase 1, Step 1: Error Pattern Discovery Pipeline — Action Plan

## Goal

Build a pipeline that enables a model to generate a large number of responses on GSM8K/MT-Bench, identify failures via external verifiers, and then cluster/attribute those failures to produce an error type taxonomy.

## Environment Setup

- [ ] Set up Python 3.10+ virtual environment
- [ ] Install dependencies: `transformers`, `datasets`, `vllm` (or `text-generation-inference`), `scikit-learn`, `openai` (if using API), `wandb` (logging)
- [ ] Configure model access: choose base model (e.g., LLaMA-3-8B, Mistral-7B, or DeepSeek-7B) and set up inference (local GPU or API)
- [ ] Create project structure:
  ```
  src/
    data/           # Dataset loading, preprocessing
    inference/      # Model response generation
    evaluation/     # External verifiers, correctness checking
    analysis/       # Clustering, error attribution, taxonomy
    rubric/         # Rubric generation (Phase 1, Step 2)
  data/
    raw/            # Raw benchmark datasets
    responses/      # Generated model responses
    analysis/       # Error clusters, taxonomies
  configs/          # YAML config files
  scripts/          # Runnable pipeline scripts
  ```

## Step 1.1: Data Preparation

- [ ] Load GSM8K dataset (train + test splits) via HuggingFace `datasets`
- [ ] Load MT-Bench questions (multi-turn prompt format)
- [ ] Preprocess: standardize prompt templates, tokenize, truncate to model context window
- [ ] Sample 500+ questions from GSM8K for response generation
- [ ] Store in `data/raw/` with consistent format

## Step 1.2: Response Generation

- [ ] Implement batch inference script using vLLM for efficient generation
- [ ] Generate responses for all sampled questions
- [ ] Set generation parameters: temperature=0.7, top_p=0.9, max_tokens=512/1024
- [ ] Save each response with metadata (prompt, model, timestamp, generation params) to `data/responses/gsm8k_responses.jsonl`
- [ ] Repeat for MT-Bench with multi-turn conversation handling

## Step 1.3: Failure Labeling

- [ ] Implement external verifier for GSM8K: exact match on final answer (extract numeric answer from model output, compare to ground truth)
- [ ] Implement external verifier for MT-Bench: use GPT-4 as judge (or AlpacaEval 2.0 pipeline) for multi-dimensional scoring
- [ ] Label each response as `correct` or `incorrect`
- [ ] Filter: keep only incorrect responses for error analysis
- [ ] Output: `data/responses/gsm8k_failures.jsonl` with failure annotations
- [ ] Expected yield: ~200-300 failure samples (depending on base model accuracy)

## Step 1.4: Error Clustering and Attribution

### Approach A: LLM-Prompted Clustering

- [ ] Design a clustering prompt:

  > "Here are N incorrect math problem responses. Each response is paired with the correct answer. Group these errors into categories. For each category, provide: (1) category name, (2) description, (3) 2-3 examples."

- [ ] Send failure samples in batches (e.g., 50 per batch due to context limits)
- [ ] Use the LLM itself to perform attribution for each failed response
- [ ] Merge batch results, deduplicate overlapping categories

### Approach B: Embedding-Based Clustering

- [ ] Embed each failure case using sentence-transformers or the base model's hidden states
- [ ] Apply dimensionality reduction (UMAP) for visualization
- [ ] Cluster with HDBSCAN or K-Means (K chosen by silhouette score)
- [ ] For each cluster, prompt LLM to generate a cluster label and description

### Combined Approach (Recommended)

1. Use embedding-based clustering as initial grouping
2. For each cluster, use LLM to generate human-readable error type descriptions
3. Validate: check that cluster members share the same error mechanism

## Step 1.5: Taxonomy Generation

- [ ] Consolidate error categories into a structured taxonomy (JSON/YAML)
- [ ] Expected format:

```yaml
error_taxonomy:
  - name: "Calculation Error"
    description: "Arithmetic mistakes in intermediate steps"
    examples: [...]
    frequency: 0.15
  - name: "Multi-step Reasoning Loss"
    description: "Model loses track of premises across reasoning steps"
    examples: [...]
    frequency: 0.12
  - name: "Irrelevant Answer"
    description: "Response does not address the question"
    examples: [...]
    frequency: 0.08
  ...
```

- [ ] Compute per-category frequency statistics
- [ ] Generate visualizations: bar chart of error type distribution, UMAP of error embeddings colored by category

## Step 1.6: Validation

- [ ] Human review: randomly sample 50 errors and verify category assignments
- [ ] Compute inter-annotator agreement between LLM clustering and human review
- [ ] Refine taxonomy based on review feedback
- [ ] Coverage check: what percentage of failures are covered by the taxonomy? Target: >90%

## Deliverables

| Deliverable | Path |
|-------------|------|
| Generated responses | `data/responses/gsm8k_responses.jsonl` |
| Labeled failures | `data/responses/gsm8k_failures.jsonl` |
| Error taxonomy | `data/analysis/error_taxonomy.yaml` |
| Clustering notebook | `notebooks/01_error_clustering.ipynb` |
| Analysis report | `docs/reports/phase1-step1-report.md` |

## Timeline

| Task | Estimated Effort |
|------|-----------------|
| Environment setup + data prep | 0.5 day |
| Response generation | 0.5 day |
| Failure labeling | 0.5 day |
| Error clustering (embedding + LLM) | 1.5 days |
| Taxonomy generation + visualization | 0.5 day |
| Validation + report | 0.5 day |
| **Total** | **~4 days** |

## Dependencies for Next Steps

- Error taxonomy from this step is the **required input** for Phase 1, Step 2 (Rubric Generation)
- The response generation pipeline will be reused in Phase 2 (Self-Evolving loop)
- External verifier implementations will serve as ground truth for evaluating self-evaluation accuracy
