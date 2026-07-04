# Phase 1, Step 1: Error Pattern Discovery Pipeline — Action Plan

## Goal

Build a pipeline that enables a model to generate responses at scale on coding benchmarks (MBPP + HumanEval+), identify failures via unit test execution, and then cluster/attribute those failures to produce an error type taxonomy.

## Dataset Overview

| Dataset | Domain | Size | Verifier | Path |
|---------|--------|------|----------|------|
| **MBPP** (full) | Python code generation | train 374 + test 500 + validation 90 = **964** usable | 3 assert tests/problem | `data/raw/mbpp_{split}.jsonl` |
| **HumanEval+** | Python code generation | test **164** | ~764 tests/problem (mutation-fuzzed) | `data/raw/humanevalplus_test.jsonl` |

Total: **1,128** problems. All verification via code execution, **zero API cost**. Moderate scale: large enough for statistically meaningful clustering, small enough to keep computation manageable.

## Model

| Model | Size | Path | Rationale |
|-------|------|------|-----------|
| **Qwen2.5-7B-Instruct** | ~15 GB (fp16) | `models/models--Qwen--Qwen2.5-7B-Instruct/` | Non-gated; general-purpose (not code-specific), expected MBPP/HumanEval+ pass rate ~40–55%, sufficient failure samples for clustering |

GPU: 8× NVIDIA A800 80 GB. A single GPU (GPU0: 58.6 GB free / GPU6: 57.8 GB free) is sufficient for loading and inference.

## Environment Setup

- [x] Python 3.10+ conda environment (`rubric`) — `/home/acm-group-3/miniconda3/envs/rubric/`
- [x] Installed: `transformers`, `datasets`, `huggingface_hub`
- [ ] To install: `vllm`, `scikit-learn`, `sentence-transformers`, `umap-learn`, `wandb`
- [x] Model: Qwen2.5-7B-Instruct downloaded to `models/`
- [x] Datasets: MBPP + HumanEval+ downloaded to `data/raw/`
- [ ] Set up sandboxed code execution environment (Docker container: 5 s timeout, 512 MB memory, no network, restricted filesystem)
- [ ] Create project directory structure:
  ```
  src/
    data/           # Dataset loading, preprocessing
    inference/      # Model response generation
    evaluation/     # Test executor, correctness checking
    analysis/       # Clustering, error attribution, taxonomy
    rubric/         # Rubric generation (Phase 1, Step 2)
  data/
    raw/            # Raw benchmark datasets (populated)
    responses/      # Generated model responses
    analysis/       # Error clusters, taxonomies
  models/           # Downloaded model weights
  configs/          # YAML config files
  scripts/          # Runnable pipeline scripts
  sandbox/          # Dockerfile + code executor
  ```

> **HF Mirror**: Direct connection to HuggingFace is blocked. Set `HF_ENDPOINT="https://hf-mirror.com"` for all downloads.

## Step 1.1: Data Preparation

- [x] Datasets downloaded to `data/raw/`:
  - `mbpp_train.jsonl` (374), `mbpp_test.jsonl` (500), `mbpp_validation.jsonl` (90), `mbpp_prompt.jsonl` (10)
  - `humanevalplus_test.jsonl` (164)
- [ ] Merge MBPP splits: train (374) + test (500) + validation (90) = **964 problems**. Discard prompt split (10 problems, few-shot only).
- [ ] Unify prompt format:
  - MBPP: construct prompt from `text` field as `"""def function_name(args):\n    \"\"\"{text}\"\"\"\n"""` — model completes the function body
  - HumanEval+: use `prompt` field directly (function signature + docstring)
  - Qwen2.5-7B-Instruct context window = 32K tokens — all prompts fit without truncation
- [ ] Unified schema:
  ```json
  {"id": "mbpp/0", "dataset": "mbpp", "prompt": "def remove_Occ...", "test_list": ["assert ..."], "test_setup_code": ""}
  {"id": "humaneval+/0", "dataset": "humanevalplus", "prompt": "from typing import...", "entry_point": "has_close_elements", "test": "def check(...)..."}
  ```
- [ ] Output: `data/raw/coding_prompts.jsonl` (1,128 unified prompts)

## Step 1.2: Response Generation

- [ ] Implement batch inference using vLLM with Qwen2.5-7B-Instruct
  - Model path: `models/models--Qwen--Qwen2.5-7B-Instruct/`
  - Target GPU: GPU0 or GPU6 (~58 GB free each)
- [ ] Generate code completions for all 1,128 problems
- [ ] Generation parameters: temperature=0.7, top_p=0.9, max_tokens=512
- [ ] Generate **k=3** samples per problem to capture diverse error patterns → ~3,384 responses total
- [ ] Record per response: `problem_id`, `dataset`, `prompt`, `generated_code`, `model="Qwen2.5-7B-Instruct"`, `timestamp`, `sample_id`, `seed`
- [ ] Fix random seed (`seed=42`) for reproducibility
- [ ] Boundary handling:
  - Empty/whitespace-only output → `generation_failure` (logged, excluded from clustering)
  - Truncated at max_tokens → flag `truncated: true`
  - Comment-only output / refusal → flag `non_code_output`
- [ ] Output: `data/responses/coding_responses.jsonl`

## Step 1.3: Failure Labeling

- [ ] Implement sandboxed test executor (`sandbox/executor.py`):
  - Run inside Docker container: 5 s timeout, 512 MB memory, no network, restricted filesystem
  - Wrap generated code into executable module, import required libraries
  - Execute MBPP `test_list` or HumanEval+ extended test cases
  - Record per test case: pass/fail, exception type + message, traceback

- [ ] Label:
  - `pass`: all test cases pass → `correct`
  - `fail`: any test fails or execution error → `incorrect`
  - `generation_failure`: empty/non-code output → excluded from clustering

- [ ] Auto-classify failure type:
  - `syntax_error` — code cannot be parsed by Python (Python SyntaxError)
  - `runtime_error` — exception at runtime (NameError, TypeError, IndexError, KeyError, ValueError, ZeroDivisionError, AttributeError, ImportError, ...)
  - `logic_error` — no exception but assertion fails (wrong output)
  - `timeout` — exceeds 5 s limit
  - `incomplete` — stub/placeholder (pass, ..., #TODO)

- [ ] Output: `data/responses/coding_failures.jsonl`
- [ ] Expected yield: ~1,500–2,000 failure samples (Qwen2.5-7B-Instruct is not code-specific, estimated pass rate ~40–55%)

## Step 1.4: Error Clustering and Attribution

### Approach A: LLM-Prompted Clustering (auxiliary)

- [ ] Send batches of ~30 failure samples per call (full code + error traces consume many tokens)
- [ ] Prompt:
  > "Here are N incorrect code solutions, each with the problem description, the generated code, and the test failure details. Group these errors into categories. For each category, provide: (1) category name, (2) description, (3) 2–3 representative examples."

- [ ] After all batches: **consolidation pass** — send all batch-level categories to the LLM to merge overlapping ones into a unified taxonomy

### Approach B: Embedding-Based Clustering (primary)

- [ ] Construct embedding text per failure:
  ```
  [Problem]: {first 200 chars of description}
  [Generated Code]: {first 400 chars of code}
  [Failure]: {failure type} | {error message / first failing assert}
  ```
- [ ] Embed using `jina-embeddings-v2-base-code` or `all-MiniLM-L6-v2`
- [ ] UMAP reduction (n_components=5 for clustering, 2 for visualization)
- [ ] **HDBSCAN** (min_cluster_size=10, min_samples=5) — no preset K needed, handles noise automatically. Tune to keep noise ratio < 20%.
- [ ] LLM generates label and description for each cluster

### Recommended: Combined Approach

1. Approach B as primary clustering → produce clusters
2. Approach A for refinement — generate human-readable labels and representative examples per cluster
3. For top-5 clusters: randomly sample 5 members each, manually verify coherence
4. Merge clusters with description cosine similarity < 0.85
5. Noise ratio > 20% → adjust embedding strategy or HDBSCAN parameters

## Step 1.5: Taxonomy Generation

- [ ] Consolidate into structured YAML taxonomy:

```yaml
coding_error_taxonomy:
  - name: "Syntax Error"
    description: "Generated code has invalid Python syntax (missing colons, unmatched brackets, wrong indentation, etc.)"
    failure_type: syntax_error
    frequency: 0.18
    examples:
      - problem_id: "MBPP/123"
        snippet: |
          if x > 0
              return x  # SyntaxError: expected ':'

  - name: "Runtime Exception"
    description: "Code parses but raises a runtime exception during test execution"
    failure_type: runtime_error
    frequency: 0.22
    examples:
      - problem_id: "HumanEval+/42"
        snippet: |
          result.append(item)  # NameError: name 'result' is not defined

  - name: "Logic Error — Wrong Algorithm"
    description: "Code runs without exception but produces incorrect output; core algorithmic approach is flawed"
    failure_type: logic_error
    frequency: 0.25
    examples:
      - problem_id: "MBPP/301"
        snippet: |
          # Hash-map O(n) required; model uses O(n^2) with wrong comparison logic

  - name: "Logic Error — Edge Case Miss"
    description: "Works on typical inputs, fails on boundaries (empty, single element, negatives, None, duplicates, large inputs)"
    failure_type: logic_error
    frequency: 0.12
    examples:
      - problem_id: "HumanEval+/7"
        snippet: |
          for i in range(len(numbers)-1):  # IndexError on empty list

  - name: "Incomplete / Stub Implementation"
    description: "Model outputs a placeholder instead of a full solution"
    failure_type: incomplete
    frequency: 0.08
    examples:
      - problem_id: "MBPP/567"
        snippet: |
          def solve(x):
              pass

  - name: "API / Library Misuse"
    description: "Incorrect use of Python stdlib or builtins (wrong argument order, misunderstanding return semantics)"
    failure_type: logic_error | runtime_error
    frequency: 0.08
    examples:
      - problem_id: "HumanEval+/15"
        snippet: |
          return lst.sort()  # .sort() returns None

  - name: "Time Limit Exceeded"
    description: "Functionally correct but too slow, exceeds 5 s timeout"
    failure_type: timeout
    frequency: 0.04
    examples:
      - problem_id: "MBPP/234"
        snippet: |
          # O(2^n) recursion without memoization on n=30 input

  - name: "Other"
    description: "Errors that do not fit the above categories"
    failure_type: mixed
    frequency: 0.03
```

- [ ] Compute per-category frequency with 95% confidence intervals
- [ ] Visualizations: error type distribution bar chart, UMAP scatter plot colored by category
- [ ] Output: `data/analysis/coding_error_taxonomy.yaml`

## Step 1.6: Validation

- [ ] Human review: randomly sample 50 failures, verify category assignments
- [ ] Compute Cohen's Kappa for categories with ≥5 samples (LLM vs. human)
- [ ] Refine taxonomy based on review:
  - Categories < 3% frequency with no clear pattern → merge into "Other"
  - Categories conflating distinct error mechanisms → split
- [ ] Cluster quality check: top-5 clusters intra-cluster cosine similarity ≥ 0.6
- [ ] Coverage: percentage of failures covered by taxonomy (excluding noise)? Target > 90%

## Deliverables

| Deliverable | Path |
|-------------|------|
| Generated coding responses | `data/responses/coding_responses.jsonl` |
| Labeled failures | `data/responses/coding_failures.jsonl` |
| Coding error taxonomy | `data/analysis/coding_error_taxonomy.yaml` |
| Clustering notebook | `notebooks/01_error_clustering.ipynb` |
| Docker sandbox | `sandbox/Dockerfile`, `sandbox/executor.py` |
| Analysis report | `docs/reports/phase1-step1-report.md` |

## Timeline

| Task | Estimated Effort |
|------|-----------------|
| Environment setup + data prep | 0.5 day |
| Docker sandbox setup | 0.5 day |
| Response generation | 0.5 day |
| Failure labeling (test execution) | 0.5 day |
| Error clustering (embedding + LLM) | 1.5 days |
| Taxonomy generation + visualization | 0.5 day |
| Validation + report | 0.5 day |
| **Total** | **~4.5 days** |

## Dependencies for Next Steps

- Error taxonomy from this step is the **required input** for Phase 1, Step 2 (Rubric Generation)
- Response generation pipeline reused in Phase 2 (Self-Evolving loop)
- Test executor serves as ground truth for Phase 1, Step 3 (self-evaluation accuracy validation)
