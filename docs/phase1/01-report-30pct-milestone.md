# 30% Milestone: Baseline, Verification, and Initial Error Taxonomy

Date: 2026-07-02

## 1. What 30% Means

This milestone completes the project foundation:

1. The dataset and model environment are usable on A800.
2. Full k=1 baseline inference has been run.
3. Unit-test verification works for MBPP and HumanEval+.
4. A corrected labeled result file is available.
5. Failure samples and an initial rule-based error taxonomy are available for error discovery and rubric generation.

The project is now ready for parallel Phase 1 work:

- error clustering and attribution,
- rubric generation,
- rubric scoring,
- preference-pair construction.

## 2. Dataset

The current dataset is code generation:

| Dataset | Split | Count |
|---|---|---:|
| MBPP | train | 374 |
| MBPP | test | 500 |
| MBPP | validation | 90 |
| HumanEval+ | test | 164 |
| Total | - | 1128 |

Unified prompt file:

```text
data/processed/coding_prompts.jsonl
```

## 3. Model and Environment

Model:

```text
models/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28
```

Conda environment:

```text
/data2/acm-group-3/miniconda3/envs/rubric
```

Key packages:

```text
torch 2.11.0+cu130
transformers 5.12.1
vllm 0.24.0
peft 0.19.1
trl 1.7.0
scikit-learn / sentence-transformers / umap-learn / hdbscan
```

Always use `/data2` cache paths:

```bash
export XDG_CACHE_HOME=/data2/acm-group-3/cache
export HF_HOME=/data2/acm-group-3/cache/huggingface
export TRANSFORMERS_CACHE=/data2/acm-group-3/cache/huggingface
export TMPDIR=/data2/acm-group-3/cache/tmp
```

## 4. Baseline Result

Final corrected labeled file:

```text
data/responses/coding_all_qwen25_vllm_k1_labeled_v2.jsonl
```

Overall result:

| Dataset | Total | Passed | Failed | Pass@1 |
|---|---:|---:|---:|---:|
| MBPP | 964 | 504 | 460 | 52.28% |
| HumanEval+ | 164 | 73 | 91 | 44.51% |
| Total | 1128 | 577 | 551 | 51.15% |

By split:

| Split | Total | Passed | Failed | Pass@1 |
|---|---:|---:|---:|---:|
| MBPP train | 374 | 216 | 158 | 57.75% |
| MBPP test | 500 | 239 | 261 | 47.80% |
| MBPP validation | 90 | 49 | 41 | 54.44% |
| HumanEval+ test | 164 | 73 | 91 | 44.51% |

Failure types:

| Dataset | Logic | Syntax | Runtime | Timeout |
|---|---:|---:|---:|---:|
| MBPP | 215 | 220 | 24 | 1 |
| HumanEval+ | 27 | 60 | 3 | 1 |

Important correction:

- The first HumanEval+ verification pass undercounted performance because code extraction removed leading indentation from function-body completions.
- After preserving indentation and testing both `generated_only` and `prompt_plus_completion`, HumanEval+ pass@1 improved from 12.20% to 44.51%.

## 5. Initial Error Taxonomy

Generated files:

```text
data/analysis/coding_failures_qwen25_k1.jsonl
data/analysis/coding_baseline_summary_qwen25_k1.json
data/analysis/coding_error_taxonomy_initial.yaml
```

Initial rule-based pattern distribution:

| Error Pattern | Count | Ratio among failures |
|---|---:|---:|
| logic_wrong_output | 242 | 43.92% |
| syntax_malformed_code | 194 | 35.21% |
| syntax_duplicate_function_after_return | 79 | 14.34% |
| runtime_name_error | 14 | 2.54% |
| syntax_truncated_or_unclosed_block | 7 | 1.27% |
| runtime_type_error | 6 | 1.09% |
| runtime_index_error | 3 | 0.54% |
| runtime_value_error | 2 | 0.36% |
| timeout_nonterminating_or_too_slow | 2 | 0.36% |
| runtime_attribute_error | 1 | 0.18% |
| runtime_other_exception | 1 | 0.18% |

This taxonomy is only the initial rule layer. The next step should refine it with embedding clustering and LLM attribution.

## 6. Scripts Added

```text
scripts/prepare_coding_prompts.py
scripts/filter_coding_prompts.py
scripts/vllm_smoke_generate.py
scripts/verify_mbpp_smoke.py
scripts/sft_lora_smoke_train.py
scripts/build_failure_artifacts.py
```

The most important reusable commands:

```bash
python scripts/prepare_coding_prompts.py \
  --raw-dir data/raw \
  --output data/processed/coding_prompts.jsonl

CUDA_VISIBLE_DEVICES=0 python scripts/vllm_smoke_generate.py \
  --model models/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28 \
  --input data/processed/coding_prompts.jsonl \
  --output data/responses/coding_all_qwen25_vllm_k1.jsonl \
  --limit 1128 \
  --max-tokens 512 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.40

python scripts/verify_mbpp_smoke.py \
  --input data/responses/coding_all_qwen25_vllm_k1.jsonl \
  --output data/responses/coding_all_qwen25_vllm_k1_labeled_v2.jsonl \
  --timeout 5

python scripts/build_failure_artifacts.py \
  --input data/responses/coding_all_qwen25_vllm_k1_labeled_v2.jsonl \
  --failure-output data/analysis/coding_failures_qwen25_k1.jsonl \
  --summary-output data/analysis/coding_baseline_summary_qwen25_k1.json \
  --taxonomy-output data/analysis/coding_error_taxonomy_initial.yaml
```

## 7. Next Parallel Work Packages

### A. Inference Owner

Next target:

- add `k > 1` generation,
- add resume support,
- add stop sequences to reduce duplicated function bodies,
- compare `temperature=0.0` vs `temperature=0.7`.

Suggested immediate experiment:

```text
MBPP validation, k=3, max_tokens=384
```

### B. Error Discovery Owner

Input:

```text
data/analysis/coding_failures_qwen25_k1.jsonl
data/analysis/coding_error_taxonomy_initial.yaml
```

Tasks:

- run embedding clustering over failure samples,
- compare clusters to the initial rule taxonomy,
- produce `data/analysis/coding_error_taxonomy_refined.yaml`.

### C. Rubric Owner

Input:

```text
data/analysis/coding_error_taxonomy_initial.yaml
```

Tasks:

- generate rubric dimensions from each error pattern,
- score a held-out subset,
- compute AUC / agreement against unit-test pass/fail.

### D. Training Owner

Input:

```text
data/responses/coding_all_qwen25_vllm_k1_labeled_v2.jsonl
data/analysis/coding_failures_qwen25_k1.jsonl
```

Tasks:

- construct simple chosen/rejected pairs from pass/fail outputs,
- scale `sft_lora_smoke_train.py` into LoRA-SFT or DPO,
- first train on 100-500 MBPP examples only.

## 8. Current Risk Notes

1. Many syntax failures are duplicated function bodies or malformed continuations. Stop sequences and extraction cleanup can improve baseline without training.
2. HumanEval+ verification is sensitive to indentation and prompt-completion composition. Keep the corrected verifier.
3. GPU utilization was already high on the shared A800. Prefer small batches and avoid long training while all GPUs show 100% utilization.
4. Root filesystem is almost full. Do not write model/cache/temp files outside `/data2`.
