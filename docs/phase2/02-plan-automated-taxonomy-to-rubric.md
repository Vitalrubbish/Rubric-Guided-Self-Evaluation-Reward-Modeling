# Phase 2：自动 Taxonomy 到 Rubric 流程

Date: 2026-07-09

## 目标

Phase 2 不引入人工归并。流程应保持自动化：

```text
raw taxonomy
-> LLM semantic consolidation
-> deterministic audit/repair
-> consolidated taxonomy
-> per-category rubric-operational refinement
-> deterministic quality audit + LLM revision/targeted repair
-> refined taxonomy
-> LLM rubric generation
-> LLM rubric judge
-> verifier alignment metrics
```

## 已完成输入

Phase 1 已产出 consolidated taxonomy：

```text
data/analysis/phase1/mbpp_hidden_train_qwen25_k3_taxonomy_consolidated.yaml
data/analysis/phase1/mbpp_hidden_train_qwen25_k3_taxonomy_consolidated_audit.json
data/analysis/phase1/mbpp_hidden_train_qwen25_k3_taxonomy_consolidated_cluster_mapping.jsonl
data/analysis/phase1/mbpp_hidden_train_qwen25_k3_taxonomy_consolidated_response_assignments.jsonl
```

Phase 1 还产出 refined rubric-operational taxonomy：

```text
data/analysis/phase1/mbpp_hidden_train_qwen25_k3_taxonomy_refinement_raw_response.txt
data/analysis/phase1/mbpp_hidden_train_qwen25_k3_taxonomy_refined_for_rubric.yaml
data/analysis/phase1/mbpp_hidden_train_qwen25_k3_taxonomy_refined_for_rubric_audit.json
data/analysis/phase1/mbpp_hidden_train_qwen25_k3_taxonomy_refined_response_assignments.jsonl
```

Audit：

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

Refinement audit：

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

## 自动化边界

LLM 负责：

- 将 raw clusters 归并成高层类别；
- 为类别命名；
- 尝试逐类别生成 rubric-operational refinement。

程序负责：

- schema 校验；
- raw cluster coverage；
- duplicate/unknown cluster 检查；
- private leakage 检查；
- response-level assignment 继承；
- LLM 输出过粗时的自动拆分和修复。
- 多候选 refinement 的 deterministic quality gate、坏短语 mask、过泛维度名拒绝、类别条件化禁词检查。
- LLM refinement 输出过泛、截断或评分锚点不可靠时触发自动 revision/targeted repair；template fallback 只作为所有 LLM 尝试失败后的异常兜底，当前 Phase 1 refined taxonomy 未触发。

## Phase 2 下一步

下一步应实现：

```text
scripts/rubric/generate_llm_rubric_from_taxonomy.py
scripts/rubric/evaluate_llm_rubric_judge.py
```

Rubric generation 输入：

```text
data/analysis/phase1/mbpp_hidden_train_qwen25_k3_taxonomy_refined_for_rubric.yaml
```

Rubric judge 评估输入：

```text
data/responses/phase1_mbpp_hidden_qwen25_k3_labeled.jsonl
```

评估要求：

- rubric 从 train failure taxonomy 生成；
- judge 评估 validation/test responses；
- judge 输入不包含 verifier label、assert、test_list、private_diagnostics；
- 输出 AUC、accuracy、Cohen's Kappa、per-category score distribution。

## 不做人工介入

如果 LLM rubric generation 或 judge 输出格式错误，应由脚本自动 repair 或 fallback，不进入人工编辑流程。人工只阅读最终报告，不参与 taxonomy/rubric 生成。
