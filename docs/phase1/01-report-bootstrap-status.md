# Bootstrap Status: Coding Error Discovery Pipeline

Date: 2026-07-02

## 1. Current Dataset

The downloaded dataset is for code generation, not GSM8K.

Raw files:

| File | Count | Meaning |
|---|---:|---|
| `data/raw/mbpp_train.jsonl` | 374 | MBPP train |
| `data/raw/mbpp_test.jsonl` | 500 | MBPP test |
| `data/raw/mbpp_validation.jsonl` | 90 | MBPP validation |
| `data/raw/mbpp_prompt.jsonl` | 10 | MBPP few-shot prompt examples, not used in the merged file |
| `data/raw/humanevalplus_test.jsonl` | 164 | HumanEval+ test |

Merged prompt file:

```text
data/processed/coding_prompts.jsonl
```

It contains 1128 prompts:

- MBPP train/test/validation: 964
- HumanEval+: 164

## 2. Environment

Conda environment:

```text
/data2/acm-group-3/miniconda3/envs/rubric
```

Key installed packages:

```text
torch 2.11.0+cu130
transformers 5.12.1
vllm 0.24.0
peft 0.19.1
trl 1.7.0
scikit-learn
sentence-transformers
umap-learn
hdbscan
evaluate
```

Important: the root filesystem is almost full. Always put cache and temp files under `/data2`:

```bash
export XDG_CACHE_HOME=/data2/acm-group-3/cache
export HF_HOME=/data2/acm-group-3/cache/huggingface
export TRANSFORMERS_CACHE=/data2/acm-group-3/cache/huggingface
export TMPDIR=/data2/acm-group-3/cache/tmp
```

## 3. Model

Available local model:

```text
models/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28
```

The Llama cache directory is present but does not contain full weights.

## 4. Smoke Tests Completed

### 4.1 Merge Data

```bash
python scripts/prepare_coding_prompts.py \
  --raw-dir data/raw \
  --output data/processed/coding_prompts.jsonl
```

Result:

```text
wrote 1128 prompts to data/processed/coding_prompts.jsonl
```

### 4.2 vLLM Inference

```bash
CUDA_VISIBLE_DEVICES=0 \
XDG_CACHE_HOME=/data2/acm-group-3/cache \
HF_HOME=/data2/acm-group-3/cache/huggingface \
TRANSFORMERS_CACHE=/data2/acm-group-3/cache/huggingface \
TMPDIR=/data2/acm-group-3/cache/tmp \
python scripts/vllm_smoke_generate.py \
  --model models/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28 \
  --input data/processed/coding_prompts.jsonl \
  --output data/responses/vllm_smoke_responses_v2.jsonl \
  --limit 1 \
  --max-tokens 256 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.35
```

Result:

```text
wrote 1 responses to data/responses/vllm_smoke_responses_v2.jsonl
```

### 4.3 MBPP Unit-Test Verification

```bash
python scripts/verify_mbpp_smoke.py \
  --input data/responses/vllm_smoke_responses_v2.jsonl \
  --output data/responses/vllm_smoke_labeled_v2.jsonl
```

Result:

```text
evaluated 1 responses, passed=1, failed=0
```

### 4.4 LoRA SFT Smoke Training

```bash
CUDA_VISIBLE_DEVICES=0 \
XDG_CACHE_HOME=/data2/acm-group-3/cache \
HF_HOME=/data2/acm-group-3/cache/huggingface \
TRANSFORMERS_CACHE=/data2/acm-group-3/cache/huggingface \
TMPDIR=/data2/acm-group-3/cache/tmp \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python scripts/sft_lora_smoke_train.py \
  --model models/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28 \
  --data data/processed/coding_prompts.jsonl \
  --output-dir outputs/sft_lora_smoke \
  --limit 1 \
  --max-length 768
```

Result:

```text
trainable params: 20,185,088 || all params: 7,635,801,600 || trainable%: 0.2643
step=0 id=mbpp/train/601 loss=0.7098
saved LoRA smoke adapter to outputs/sft_lora_smoke
```

## 5. Next Parallel Tasks

### A: Inference Pipeline

Scale `vllm_smoke_generate.py` into full batch generation:

- support `k` samples per problem
- support resume if output file already contains some ids
- generate all MBPP first, then HumanEval+

Recommended first real run:

```text
MBPP validation, k=1, max_tokens=512
```

### B: Verifier Pipeline

Upgrade `verify_mbpp_smoke.py` into a full verifier:

- MBPP full assert execution
- HumanEval+ `check(candidate)` execution
- timeout
- failure type labels: syntax, runtime, logic, timeout, generation_failure

### C: Error Analysis

Use verified failures to create:

```text
data/analysis/coding_failures.jsonl
data/analysis/coding_error_taxonomy.yaml
```

Start with rule-based failure types, then add embedding clustering.

### D: Training Pipeline

Scale `sft_lora_smoke_train.py` into actual LoRA training:

- use 100-500 MBPP examples first
- save adapter under `outputs/sft_lora_mbpp_*`
- compare base vs adapter with the verifier

Do not start full training while all GPUs show `GPU-Util=100%`.
