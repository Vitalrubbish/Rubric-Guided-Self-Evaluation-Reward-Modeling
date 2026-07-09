# Phase 1：MBPP Hidden-Tests k=3 错误发现、归因聚类与 Rubric 基线方案

## 目标

Phase 1 的正式目标是：在 **MBPP 代码生成任务** 上构建一个不泄漏测试断言的错误发现管线，让模型生成多样化 response，经外部 verifier 标注失败样本，再基于失败样本发现错误模式、生成 rubric，并评估 rubric 自评与外部验证结果的一致性。

本阶段不再把 HumanEval+ 纳入正式测试集。HumanEval+ 相关结果只作为早期探索记录，不用于 Phase 1 当前主结论。

## 核心实验约束

| 项目 | 当前正式设置 |
| --- | --- |
| 数据集 | MBPP only |
| Split | train 374 + test 500 + validation 90 = 964 |
| Prompt | hidden-tests：prompt 中不出现 `assert` |
| Verifier | 使用 MBPP `test_list`，但仅在验证阶段使用 |
| 采样规模 | k=3 |
| Response 总数 | 964 × 3 = 2892 |
| 唯一键 | `response_id = id + sample_id` |
| 模型 | Qwen2.5-7B-Instruct |
| 生成参数 | temperature=0.7, top_p=0.9, max_tokens=512 |

## 当前完成状态

Date: 2026-07-09

Phase 1 已按上述协议完成正式 k=3 运行，Phase 1.5 已完成 k=5 稳定性复验。当前结论应以 `phase1` / `phase1_5` 目录中的 safe artifacts 为准，不再引用旧的 visible-tests 或 MBPP+HumanEval+ 混合结果作为主结论。

| 阶段 | 设置 | Response | Pass | Fail | Response Pass Rate | Task Pass@k | Train Failures | Taxonomy |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Phase 1 | MBPP hidden-tests, k=3 | 2892 | 1470 | 1422 | 0.508299 | 0.594398 | 519 | 17 clusters |
| Phase 1.5 | MBPP hidden-tests, k=5 | 4820 | 2427 | 2393 | 0.503527 | 0.614108 | 867 | 24 clusters |

已完成的关键产物：

- `data/responses/phase1_mbpp_hidden_qwen25_k3_labeled.jsonl`
- `data/analysis/phase1/mbpp_hidden_qwen25_k3_failures_safe.jsonl`
- `data/analysis/phase1/mbpp_hidden_train_qwen25_k3_failures_with_safe_llm_summaries.jsonl`
- `data/analysis/phase1/mbpp_hidden_train_qwen25_k3_discovered_taxonomy_safe.yaml`
- `data/analysis/phase1/mbpp_hidden_train_qwen25_k3_taxonomy_consolidated.yaml`
- `data/analysis/phase1/mbpp_hidden_train_qwen25_k3_taxonomy_refined_for_rubric.yaml`
- `data/responses/phase1_5_mbpp_hidden_qwen25_k5_labeled.jsonl`
- `data/analysis/phase1_5/mbpp_hidden_train_qwen25_k5_discovered_taxonomy_safe_v2.yaml`
- `data/rubrics/phase1/mbpp_hidden_auto_rubric_refined.json`

Safe artifacts 不包含 `test_list`、`test`、`test_setup_code` 或 `private_diagnostics`。包含精确断言、actual/expected 的调试副本只保存在 `data/private_diagnostics/`，不得进入归因、聚类、训练或报告主链路。

## 关键修正

### 1. 防止面向测试编程

旧版 MBPP prompt 把 `test_list` 中的 assert 直接给模型，模型会看到要通过哪些测试。这会导致：

- pass rate 变成 visible-test pass rate；
- 模型可能针对三个断言写特例代码；
- 错误 taxonomy 被“可见测试条件下仍失败”的样本污染；
- rubric 评估不能代表隐藏测试下的自评能力。

当前正式 prompt 只包含：

- 自然语言任务描述；
- public interface signatures：从 canonical solution 中抽取函数/类签名，只暴露接口不暴露实现；
- 不包含具体 assert、输入输出样例或 expected value。

`test_list` 仍保存在 JSONL 记录中，但只供 verifier 使用。

### 2. k=3 下游对齐

k=3 时同一个 problem id 会对应三个 response。所有下游文件必须使用 `response_id` 对齐，不能只用 `id`。

标准字段：

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

### 3. 错误发现避免规则标签泄漏

规则标签 `error_pattern` 可作为 baseline 和输出元数据，但不进入当前正式聚类特征，也不直接决定 cluster name。

当前正式错误发现脚本是 `discover_error_taxonomy.py`：先让 LLM 对失败样本自由归因，再对 `llm_summary + failure_type + error` 做 TF-IDF/SVD/HDBSCAN 聚类。`error_pattern` 只保留在输出中用于审计和解释。

## Step 1：数据准备

命令：

```bash
/data2/acm-group-3/miniconda3/envs/rubric/bin/python \
  scripts/data-prep/prepare_coding_prompts.py \
  --raw-dir data/raw \
  --output data/processed/coding_prompts.jsonl
```

期望结果：

- 输出 964 条 prompt；
- 全部 `dataset == "mbpp"`；
- 全部 `prompt_mode == "mbpp_hidden_tests"`；
- prompt 中 `assert` 出现次数为 0；
- prompt 中包含 public interface signatures；
- 每条记录仍保留 `test_list` 供 verifier 使用。

验证命令：

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

当前已验证结果：

```text
964
['mbpp']
0
['mbpp_hidden_tests']
```

## Step 2：k=3 Response 生成

命令模板：

```bash
CUDA_VISIBLE_DEVICES=<GPU_ID> \
PATH=/data2/acm-group-3/miniconda3/envs/rubric/bin:$PATH \
XDG_CACHE_HOME=/tmp/rubric-cache \
HF_HOME=/tmp/rubric-cache/huggingface \
TRANSFORMERS_CACHE=/tmp/rubric-cache/huggingface \
TMPDIR=/tmp/rubric-tmp \
/data2/acm-group-3/miniconda3/envs/rubric/bin/python \
  scripts/generation/vllm_smoke_generate.py \
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

期望输出：

```text
2892 responses
```

每条 response 必须包含：

- `response_id`
- `id`
- `sample_id`
- `prompt_mode`
- `interface_signatures`
- `generated_code`
- `temperature`
- `top_p`
- `seed`

## Step 3：Verifier 标注

命令：

```bash
/data2/acm-group-3/miniconda3/envs/rubric/bin/python \
  scripts/verification/verify_mbpp_smoke.py \
  --input data/responses/mbpp_hidden_qwen25_k3.jsonl \
  --output data/responses/phase1_mbpp_hidden_qwen25_k3_labeled.jsonl \
  --timeout 5
```

输出字段：

- `response_id`
- `passed`
- `failure_type`
- `error`
- `extracted_code`
- `safe_diagnostics`
- `private_diagnostics`

注意：当前 verifier 是基于 multiprocessing 的轻量执行器，不是 Docker sandbox。`safe_diagnostics` 可用于归因，`private_diagnostics` 只能用于本地调试。

## Step 4：Safe Failure Artifacts 与初始规则 Taxonomy

命令：

```bash
/data2/acm-group-3/miniconda3/envs/rubric/bin/python \
  scripts/error-analysis/build_failure_artifacts.py \
  --input data/responses/phase1_mbpp_hidden_qwen25_k3_labeled.jsonl \
  --failure-output data/analysis/phase1/mbpp_hidden_qwen25_k3_failures_safe.jsonl \
  --summary-output data/analysis/phase1/mbpp_hidden_qwen25_k3_summary.json \
  --taxonomy-output data/analysis/phase1/mbpp_hidden_qwen25_k3_taxonomy_initial_safe.yaml
```

这里生成的 `error_pattern` 是规则 baseline，不能作为最终“模型自发现 taxonomy”的唯一证据。默认输出不包含 hidden tests 或 private diagnostics。

## Step 5：LLM 归因与模型自发现 Taxonomy

默认命令：

```bash
/data2/acm-group-3/miniconda3/envs/rubric/bin/python \
  scripts/error-analysis/discover_error_taxonomy.py \
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

输出应包含：

- cluster 频率；
- failure type 分布；
- rule pattern 分布，作为辅助统计；
- representative examples；
- top terms；
- `response_id` 级 assignments。

## Step 6：Taxonomy Consolidation 与 Rubric-Operational Refinement

Raw clusters 不能直接作为 Phase 2 rubric 输入。Phase 1 的完整 taxonomy pipeline 是：

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

Consolidation 命令：

```bash
/data2/acm-group-3/miniconda3/envs/rubric/bin/python \
  scripts/error-analysis/consolidate_taxonomy.py \
  --taxonomy data/analysis/phase1/mbpp_hidden_train_qwen25_k3_discovered_taxonomy_safe.yaml \
  --raw-assignments data/analysis/phase1/mbpp_hidden_train_qwen25_k3_discovered_clusters_safe.jsonl \
  --output data/analysis/phase1/mbpp_hidden_train_qwen25_k3_taxonomy_consolidated.yaml \
  --audit-output data/analysis/phase1/mbpp_hidden_train_qwen25_k3_taxonomy_consolidated_audit.json \
  --cluster-mapping-output data/analysis/phase1/mbpp_hidden_train_qwen25_k3_taxonomy_consolidated_cluster_mapping.jsonl \
  --response-assignments-output data/analysis/phase1/mbpp_hidden_train_qwen25_k3_taxonomy_consolidated_response_assignments.jsonl \
  --raw-llm-output data/analysis/phase1/mbpp_hidden_train_qwen25_k3_taxonomy_consolidation_raw_response.txt
```

Refinement 命令：

```bash
/data2/acm-group-3/miniconda3/envs/rubric/bin/python \
  scripts/error-analysis/refine_taxonomy_for_rubric.py \
  --taxonomy data/analysis/phase1/mbpp_hidden_train_qwen25_k3_taxonomy_consolidated.yaml \
  --assignments data/analysis/phase1/mbpp_hidden_train_qwen25_k3_taxonomy_consolidated_response_assignments.jsonl \
  --failures data/analysis/phase1/mbpp_hidden_train_qwen25_k3_failures_safe.jsonl \
  --output data/analysis/phase1/mbpp_hidden_train_qwen25_k3_taxonomy_refined_for_rubric.yaml \
  --audit-output data/analysis/phase1/mbpp_hidden_train_qwen25_k3_taxonomy_refined_for_rubric_audit.json \
  --response-assignments-output data/analysis/phase1/mbpp_hidden_train_qwen25_k3_taxonomy_refined_response_assignments.jsonl \
  --raw-llm-output data/analysis/phase1/mbpp_hidden_train_qwen25_k3_taxonomy_refinement_raw_response.txt \
  --candidates-per-category 3 \
  --revision-candidates 2 \
  --targeted-repair-candidates 2 \
  --max-examples-per-category 5
```

Refined taxonomy 每个 category 都必须包含：

- operational definition；
- failure mechanism；
- common manifestations；
- judge checklist；
- 1-5 score anchors；
- positive/negative boundary。

质量审计必须保证：

- category coverage 完整；
- response assignment 数量不变；
- 无 private/test leakage；
- 无过泛 refinement 文本；
- schema 完整。

当前 Phase 1 refined taxonomy audit 为 `valid = true`，并覆盖 519/519 train failure responses。最新 refinement 结果为 2 个 initial accepted、4 个 revision accepted、2 个 targeted repair accepted、0 个 template fallback。

## Step 7：Rubric 生成

当前 `generate_auto_rubric.py` 仍是半自动版本，维度模板较固定。正式结论中应将其称为：

```text
taxonomy-informed rubric template
```

而不是完全自由的模型自动归纳 rubric。

命令：

```bash
/data2/acm-group-3/miniconda3/envs/rubric/bin/python \
  scripts/rubric/generate_auto_rubric.py \
  --taxonomy data/analysis/phase1/mbpp_hidden_train_qwen25_k3_taxonomy_refined_for_rubric.yaml \
  --output data/rubrics/phase1/mbpp_hidden_auto_rubric_refined.json \
  --generic-output data/rubrics/phase1/mbpp_hidden_generic_rubric.json \
  --random-output data/rubrics/phase1/mbpp_hidden_random_rubric_ablation.json
```

## Step 8：Self-Evaluation 基线

当前 `evaluate_rubric_static.py` 是静态启发式 scorer，不是真正 LLM rubric judge。它可以作为早期工程 sanity check，但不能作为“模型读 rubric 后自评”的最终证据。

命令：

```bash
/data2/acm-group-3/miniconda3/envs/rubric/bin/python \
  scripts/rubric/evaluate_rubric_static.py \
  --labeled data/responses/phase1_mbpp_hidden_qwen25_k3_labeled.jsonl \
  --failures data/analysis/phase1/mbpp_hidden_qwen25_k3_failures_safe.jsonl \
  --rubric data/rubrics/phase1/mbpp_hidden_auto_rubric_refined.json \
  --scores-output data/rubrics/phase1/mbpp_hidden_auto_rubric_scores_static.jsonl \
  --metrics-output data/rubrics/phase1/mbpp_hidden_auto_rubric_eval_metrics.json
```

当前 static baseline 结果：

| Rubric | Coverage | Static AUC | Kappa | Accuracy |
| --- | ---: | ---: | ---: | ---: |
| auto taxonomy-informed | 1.000 | 0.596642 | 0.189652 | 0.600277 |
| generic | 0.000 | 0.508823 | 0.017383 | 0.501383 |
| random ablation | 1.000 | 0.596642 | 0.189652 | 0.600277 |

解释：static scorer 没有真正读取 rubric 文本语义，因此 auto 与 random 得分相同。它只能作为工程 sanity check，不能作为 self-evaluation 主证据。

正式自评实验还需要补充：

- LLM 读取 rubric 后逐条打分；
- 不向 LLM 暴露 verifier label 或 assert；
- 与 verifier pass/fail 计算 AUC、Cohen's Kappa、accuracy；
- 对比 human rubric、auto rubric、random rubric。

## 当前已完成的代码级修复

- `prepare_coding_prompts.py` 默认 MBPP-only + hidden-tests。
- `vllm_smoke_generate.py` 默认 k=3，并写出 `response_id`。
- `verify_mbpp_smoke.py` 输出 safe/private 两级诊断，并使用 `response_id` 对齐。
- `build_failure_artifacts.py` 默认生成 safe artifacts，私有测试字段需显式开启。
- `discover_error_taxonomy.py` 已支持 safe attribution、私有字段拒绝、递归子聚类规模控制。
- `consolidate_taxonomy.py` 已支持 raw cluster 自动归并、coverage audit、broad category repair、response-level assignment 继承。
- `refine_taxonomy_for_rubric.py` 已支持逐类别多候选 LLM refinement、坏短语 mask、revision、targeted repair、类别条件化质量门、raw output 复用和 rubric-operational schema audit。
- `generate_auto_rubric.py` 已兼容新的 `error_patterns` 字段和细粒度错误标签。

## 当前风险

1. 旧 k=1 / HumanEval+ / visible-tests 结果不能直接作为当前 Phase 1 主结论。
2. 当前 verifier 不是完整 sandbox，执行不可信代码仍有风险。
3. 当前 static rubric scorer 不是真正 LLM self-evaluation。
4. 自动 cluster 名称仍是关键词式标签，适合作为机器发现证据；正式 taxonomy 使用自动 LLM consolidation、rubric-operational refinement 加程序审计生成。
5. 直接调用 env Python 运行 vLLM 时，需要把 conda env 的 `bin` 加入 `PATH`，否则 FlashInfer JIT 可能因找不到可执行 `ninja` 失败。
