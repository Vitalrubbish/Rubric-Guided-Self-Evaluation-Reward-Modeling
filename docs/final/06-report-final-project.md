# Rubric-Guided Self-Evaluation 项目阶段报告

日期：2026-07-04  
远程目录：`/data2/acm-group-3/Rubric-Guided-Self-Evaluation-Reward-Modeling`

最终方法排行榜：

`docs/final_method_leaderboard.md`

## 1. 当前结论

我们已经完成了从数据准备、基线推理、外部 verifier 标注、错误模式发现、rubric 自动生成、自评区分度评估、偏好数据构建，到 rubric-guided self-improvement / DPO ablation 的闭环。主线 benchmark 是 MBPP + HumanEval+ coding；另外补充了 GSM8K n=100 小规模真实闭环、GSM8K-derived rubric -> MATH safe-subset 的 zero-shot transfer，以及 full MATH verifier pressure test。

最关键的实验结果：

| 阶段 | 通过数 | 总数 | pass@1 |
| --- | ---: | ---: | ---: |
| Qwen2.5-7B-Instruct 原始输出 | 577 | 1128 | 51.15% |
| Unprotected rule revision | 745 | 1128 | 66.05% |
| Protected rule revision | 755 | 1128 | 66.93% |

补充 GSM8K appendix：

| 阶段 | 通过数 | 总数 | accuracy |
| --- | ---: | ---: | ---: |
| Qwen2.5-7B-Instruct on GSM8K | 72 | 100 | 72.00% |

补充 GSM8K -> MATH transfer：

| 阶段 | 通过数 | 总数 | accuracy |
| --- | ---: | ---: | ---: |
| Qwen2.5-7B-Instruct on MATH safe subset | 83 | 100 | 83.00% |
| Qwen2.5-7B-Instruct on MATH full-format pressure subset | 43 | 100 | 43.00% |

当前主 baseline 是 protected rule revision：净增通过样本 178 个，绝对提升 15.78 个百分点，且 pass->fail 为 0。旧版 unprotected rule revision 保留为 ablation。

## 2. 实验设置

### 数据集

本项目主线使用 coding benchmark；为贴近原始选题推荐 benchmark，额外补充 GSM8K appendix：

| 数据 | 样本数 |
| --- | ---: |
| MBPP train | 374 |
| MBPP validation | 90 |
| MBPP test | 500 |
| HumanEval+ test | 164 |
| 合计 | 1128 |

| 补充数据 | 样本数 |
| --- | ---: |
| GSM8K test appendix | 100 |
| MATH safe-subset transfer | 100 |
| MATH full-format pressure test | 100 |

统一 prompt 文件：

`data/processed/coding_prompts.jsonl`

### 模型与推理

基座模型：

`models/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28`

推理产物：

`data/responses/coding_all_qwen25_vllm_k1.jsonl`

验证后的主文件：

`data/responses/coding_all_qwen25_vllm_k1_labeled_v2.jsonl`

## 3. 错误模式发现

基线推理总计 1128 条，失败 551 条。失败类型分布：

| 错误模式 | 数量 |
| --- | ---: |
| logic_wrong_output | 242 |
| syntax_malformed_code | 194 |
| syntax_duplicate_function_after_return | 79 |
| runtime_name_error | 14 |
| syntax_truncated_or_unclosed_block | 7 |

进一步聚类后得到 18 个 refined clusters。最大几个簇：

| cluster | 名称 | 数量 |
| --- | --- | ---: |
| cluster_00 | syntax_malformed_code | 92 |
| cluster_01 | logic_wrong_output | 85 |
| cluster_02 | syntax_error_mixed_string_return_split | 43 |
| cluster_03 | logic_wrong_output | 39 |

相关文件：

- `data/analysis/coding_failures_qwen25_k1.jsonl`
- `data/analysis/failure_clusters_qwen25_k1.jsonl`
- `data/analysis/coding_error_taxonomy_refined.yaml`
- `data/analysis/coding_error_taxonomy_refined_summary.json`

## 4. 自动生成 Rubric

自动 rubric 基于 refined taxonomy 生成，共 6 个维度：

1. Functional Correctness and Edge-Case Coverage
2. Syntax Validity and Parseability
3. Interface and Test Contract Compliance
4. Runtime Dependency and API Safety
5. Termination and Complexity Control
6. Output Cleanliness and Single-Solution Formatting

静态自评指标：

| 指标 | 数值 |
| --- | ---: |
| coverage | 1.000 |
| AUC | 0.801 |
| Cohen's Kappa | 0.525 |
| accuracy | 0.765 |
| passed mean score | 4.558 |
| failed mean score | 3.981 |

这说明自动 rubric 对通过/失败样本有较强区分度，可以作为 reward signal 或 revision signal 的第一版。

相关文件：

- `data/rubrics/auto_rubric_refined.json`
- `data/rubrics/auto_rubric_eval_metrics.json`
- `data/rubrics/generic_rubric_eval_metrics.json`
- `data/rubrics/random_rubric_eval_metrics.json`

注意：当前 random rubric ablation 使用同一个静态 scorer，主要用于保留实验接口；后续如果要写进论文式报告，最好改成让模型实际读取 rubric 文本后评分，随机 rubric 才更有解释力。

## 4.1 GSM8K 推荐 Benchmark 附录

为了回应原始选题中推荐 GSM8K/MATH 的要求，我们补充了 GSM8K test n=100 的真实闭环：

| 指标 | 数值 |
| --- | ---: |
| exact-answer accuracy | 72/100 = 72.00% |
| failures | 28 |
| failure pattern coverage | 1.000 |
| static self-eval AUC | 0.849 |
| static Cohen's Kappa@4 | 0.051 |
| verifier-informed upper-bound AUC | 1.000 |

GSM8K 失败模式：

| error pattern | count |
| --- | ---: |
| final_format_violation | 15 |
| arithmetic_or_algebra_slip | 9 |
| wrong_problem_model | 2 |
| reasoning_truncation | 1 |
| ambiguous_final_answer | 1 |

结论：GSM8K rubric 的 static AUC 较高，说明它能排序地区分好坏答案；但 Kappa 很低，说明只靠输出文本做阈值化 self-evaluation 仍不稳定，尤其会高估“推理格式完整但最终数字错误”的样本。

相关文件：

- `docs/gsm8k_alignment_results.md`
- `data/responses/gsm8k_qwen25_k1_n100_labeled.jsonl`
- `data/analysis/gsm8k_error_taxonomy_qwen25_k1_n100.yaml`
- `data/rubrics/gsm8k_auto_rubric_n100.json`
- `data/rubrics/gsm8k_auto_rubric_metrics_n100.json`

## 4.2 GSM8K -> MATH Safe-Subset Transfer

为了进一步补齐 Method 3 的跨数学 benchmark 迁移，我们使用 GSM8K-derived rubric 直接 zero-shot 评估 MATH safe subset responses。

MATH target subset：

| 项目 | 数值 |
| --- | ---: |
| total | 100 |
| Algebra | 50 |
| Prealgebra | 50 |
| Level 1 | 23 |
| Level 2 | 31 |
| Level 3 | 46 |
| gold verifier self-check | 100/100 |
| Qwen2.5-7B accuracy | 83/100 |

Rubric transfer：

| Rubric | Uses MATH failures? | Static AUC | Accuracy@4 | Kappa@4 |
| --- | --- | ---: | ---: | ---: |
| GSM8K-derived rubric | no | 0.883 | 0.850 | 0.181 |
| Generic math rubric | no | 0.883 | 0.850 | 0.181 |
| MATH-derived rubric | yes | 0.883 | 0.910 | 0.596 |

结论：GSM8K-derived rubric 在 MATH safe subset 上有 zero-shot ranking signal，但阈值化 pass/fail 判断较弱；MATH-derived rubric 的 Kappa 更高，说明目标任务失败模式对 rubric 校准有帮助。

相关文件：

- `docs/gsm8k_to_math_transfer_results.md`
- `data/processed/math_transfer_prompts_n100.jsonl`
- `data/responses/math_transfer_qwen25_n100_labeled.jsonl`
- `data/rubrics/math_transfer_gsm8k_rubric_metrics_n100.json`
- `data/rubrics/math_transfer_generic_rubric_metrics_n100.json`
- `data/rubrics/math_transfer_derived_rubric_metrics_n100.json`

## 4.3 Full MATH Verifier Pressure Test

safe subset 是 Method 3 的最小完成版。之后我们继续扩展 MATH verifier，并在 all subjects / Level 1-5 的 n=100 子集上做压力测试。

Verifier 支持：

- 集合、区间、区间并集、根式、`\pi`、多答案、`\pm` / `±`。
- `\boxed{...}` 和 `#### <answer>` 抽取。
- 千位逗号、LaTeX 表达式和 SymPy 等价。

Verifier gate：

| Check | Passed | Total | Accuracy |
| --- | ---: | ---: | ---: |
| full MATH gold verifier self-check | 100 | 100 | 100.00% |
| safe subset regression with full verifier | 83 | 100 | 83.00% |

Full MATH pressure subset：

| 指标 | 数值 |
| --- | ---: |
| total | 100 |
| subjects | 7 |
| Level 4-5 examples | 37 |
| Qwen2.5-7B accuracy | 43/100 = 43.00% |

By level：

| Level | Passed | Total | Accuracy |
| --- | ---: | ---: | ---: |
| Level 1 | 18 | 21 | 85.71% |
| Level 2 | 11 | 21 | 52.38% |
| Level 3 | 7 | 21 | 33.33% |
| Level 4 | 4 | 21 | 19.05% |
| Level 5 | 3 | 16 | 18.75% |

Full MATH rubric transfer：

| Rubric | Uses full MATH failures? | Static AUC | Accuracy@4 | Kappa@4 |
| --- | --- | ---: | ---: | ---: |
| GSM8K-derived rubric | no | 0.873 | 0.510 | 0.123 |
| Generic math rubric | no | 0.879 | 0.820 | 0.651 |
| MATH-derived rubric | yes | 0.879 | 0.820 | 0.651 |

结论：GSM8K-derived rubric 在 full MATH pressure subset 上仍有排序信号，但阈值化 pass/fail 判断很弱。full MATH 的主要瓶颈是复杂符号推理和 final-answer 格式，`ambiguous_final_answer` 占 38/57 个失败。

相关文件：

- `docs/math_full_verifier_results.md`
- `scripts/prepare_math_full_prompts.py`
- `scripts/verify_math_full.py`
- `data/processed/math_full_prompts_n100.jsonl`
- `data/responses/math_full_qwen25_n100_labeled.jsonl`
- `data/rubrics/math_full_gsm8k_rubric_metrics_n100.json`
- `data/rubrics/math_full_generic_rubric_metrics_n100.json`
- `data/rubrics/math_full_derived_rubric_metrics_n100.json`

## 5. 偏好数据

已将所有失败样本构造成 preference pair：

- chosen：canonical solution
- rejected：Qwen2.5-7B-Instruct 的失败输出
- pair 数量：551

文件：

`data/preferences/preference_pairs_qwen25_k1.jsonl`

这个文件可以直接用于 DPO/ORPO，或用于训练 reward model。

## 6. Self-Improvement Baseline

考虑到当前 A800 上 GPU 显存被其他任务大量占用，我们先完成了一轮轻量但可复现的 rubric-guided deterministic revision；后续已经补跑 DPO，见第 7-10 节：

1. 根据错误 taxonomy 和 auto rubric，定位高频可修复错误。
2. 对输出做规则化修正：
   - 截断重复函数体：`truncate_duplicate_function_body`
   - 删除 trailing prose：`drop_trailing_prose`
   - 删除 print/examples：`remove_print_examples`
3. 重新跑 verifier。
4. 与原始输出逐样本对比。

结果：

| transition | 数量 |
| --- | ---: |
| pass->pass | 567 |
| fail->fail | 373 |
| fail->pass | 178 |
| pass->fail | 10 |

按数据集看：

| 数据集 | fail->pass | pass->fail |
| --- | ---: | ---: |
| MBPP | 147 | 8 |
| HumanEval+ | 31 | 2 |

相关文件：

- `scripts/revise_code_outputs.py`
- `data/responses/coding_all_qwen25_vllm_k1_revised.jsonl`
- `data/responses/coding_all_qwen25_vllm_k1_revised_labeled.jsonl`
- `data/revision/revision_comparison_summary.json`

## 7. 训练入口

已完成 LoRA smoke train：

`outputs/sft_lora_smoke/`

已补充并运行 DPO LoRA 脚本：

`scripts/dpo_lora_train.py`

本次实际训练命令：

```bash
export XDG_CACHE_HOME=/data2/acm-group-3/cache
export HF_HOME=/data2/acm-group-3/cache/huggingface
export TRANSFORMERS_CACHE=/data2/acm-group-3/cache/huggingface
export TMPDIR=/data2/acm-group-3/cache/tmp

cd /data2/acm-group-3/Rubric-Guided-Self-Evaluation-Reward-Modeling
/data2/acm-group-3/miniconda3/envs/rubric/bin/python scripts/dpo_lora_train.py \
  --model models/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28 \
  --data data/preferences/preference_pairs_qwen25_k1.jsonl \
  --output-dir outputs/dpo_lora_coding_e1_551_mlen768 \
  --epochs 1 \
  --grad-accum 8 \
  --max-length 768
```

为了省显存，DPO 脚本只加载一个 PEFT policy model，reference logprob 通过临时 disable LoRA adapter 计算，不额外加载第二个 7B 模型。

DPO 训练结果：

| 指标 | 数值 |
| --- | ---: |
| preference pairs | 551 |
| epochs | 1 |
| effective DPO steps | 549 |
| skipped pairs | 2 |
| mean loss | 0.427 |
| preference accuracy | 0.898 |

训练产物：

- `outputs/dpo_lora_coding_e1_551_mlen768/adapter_model.safetensors`
- `outputs/dpo_lora_coding_e1_551_mlen768/train_metrics.json`
- `logs/dpo_lora_coding_e1_551_mlen768_20260702_224226.log`

注意：这里完成的是 DPO adapter 训练本身；随后已补充 MBPP validation 上的 adapter 推理评测，见下一节。全量 1128 条 adapter 评测尚未运行。

## 8. DPO Adapter Validation 评测

已补充 base model + LoRA adapter 推理脚本：

`scripts/generate_with_lora_adapter.py`

该脚本输出格式兼容现有 verifier，可复用 `scripts/verify_mbpp_smoke.py`。

在 MBPP validation 90 条上完成了 DPO adapter 生成与验证：

| 方法 | 推理后端 | 通过数 | 总数 | pass@1 |
| --- | --- | ---: | ---: | ---: |
| Base Qwen2.5-7B | Transformers | 33 | 90 | 36.67% |
| DPO LoRA adapter | Transformers + PEFT | 55 | 90 | 61.11% |
| 原始基线 | vLLM | 49 | 90 | 54.44% |
| Rubric-guided rule revision | verifier 后处理 | 60 | 90 | 66.67% |

关键比较：

- 同后端比较：DPO-HF 比 Base-HF 净增 22 题，说明 DPO adapter 确实学到了 preference signal。
- 与原始 vLLM baseline 比较：DPO-HF 净增 6 题，但这里混入了推理后端差异。
- 与规则修正 baseline 比较：DPO-HF 少 5 题，说明当前一轮 DPO 还没有超过显式错误模式清洗策略。

相关文件：

- `data/responses/dpo_lora_mbpp_validation.jsonl`
- `data/responses/dpo_lora_mbpp_validation_labeled.jsonl`
- `data/eval/dpo_lora_mbpp_validation_summary.json`
- `data/responses/base_hf_mbpp_validation.jsonl`
- `data/responses/base_hf_mbpp_validation_labeled.jsonl`
- `data/eval/base_hf_mbpp_validation_summary.json`
- `data/eval/dpo_vs_base_hf_mbpp_validation_comparison.json`
- `data/eval/dpo_vs_vllm_baseline_mbpp_validation_comparison.json`
- `data/eval/dpo_vs_rule_revision_mbpp_validation_comparison.json`

注意：当前 DPO preference pairs 来自全量失败样本，其中包含 validation 失败样本。因此这个 validation 评测更适合作为 adapter sanity/effectiveness check，而不是严格 held-out 泛化结论。更严谨的下一轮应重新划分 train-only preference pairs，再在 untouched validation/test 上评估。

## 9. Train-Only DPO 严谨评测

为避免 validation/test 泄漏，已重新过滤 preference pairs，只保留 MBPP train 失败样本：

`data/preferences/preference_pairs_qwen25_k1_mbpp_train_only.jsonl`

样本数：158。

训练产物：

`outputs/dpo_lora_mbpp_train_only_e1_158_mlen768`

训练指标：

| 指标 | 数值 |
| --- | ---: |
| train-only pairs | 158 |
| DPO steps | 158 |
| skipped | 0 |
| mean loss | 0.649 |
| preference accuracy | 0.696 |

在 untouched MBPP validation 90 条上的结果：

| 方法 | 是否有 validation 泄漏 | 通过数 | 总数 | pass@1 |
| --- | --- | ---: | ---: | ---: |
| Base Qwen2.5-7B HF | 否 | 33 | 90 | 36.67% |
| Train-only DPO HF | 否 | 33 | 90 | 36.67% |
| Train-only DPO + rule revision | 否 | 49 | 90 | 54.44% |
| 原始 vLLM baseline | 否 | 49 | 90 | 54.44% |
| 单独 rule revision baseline | 否 | 60 | 90 | 66.67% |

结论：

- Train-only DPO adapter 本身没有提升 validation，通过数与 base-HF 持平。
- DPO + rule revision 能从 33/90 提升到 49/90，说明规则修正能补救 adapter 输出中的格式/重复函数体问题。
- 当前最强仍是单独 rule revision baseline 60/90。
- 因 validation gate 未通过，未继续跑 MBPP test 500 条，避免浪费 GPU；后续应先改训练策略。

相关文件：

- `docs/train_only_dpo_execution_plan.md`
- `docs/train_only_dpo_results.md`
- `data/responses/dpo_lora_train_only_mbpp_validation_labeled.jsonl`
- `data/eval/dpo_lora_train_only_mbpp_validation_summary.json`
- `data/eval/dpo_train_only_vs_base_hf_mbpp_validation_comparison.json`
- `data/eval/dpo_train_only_validation_revision_comparison.json`
- `data/eval/dpo_lora_train_only_mbpp_validation_revised_summary.json`

## 10. Augmented Train-Only DPO

为改善 train-only DPO 不泛化的问题，本轮加入 rule-revised successful outputs 作为额外 chosen：

`data/preferences/preference_pairs_qwen25_k1_mbpp_train_augmented.jsonl`

数据构成：

| chosen source | 数量 |
| --- | ---: |
| canonical_solution | 158 |
| rule_revised_success_output | 54 |
| total | 212 |

训练产物：

`outputs/dpo_lora_mbpp_train_augmented_e1_212_mlen768`

训练指标：

| 指标 | 数值 |
| --- | ---: |
| DPO steps | 212 |
| skipped | 0 |
| mean loss | 0.651 |
| preference accuracy | 0.769 |

MBPP validation 90 条结果：

| 方法 | 通过数 | 总数 | pass@1 |
| --- | ---: | ---: | ---: |
| Base-HF | 33 | 90 | 36.67% |
| Train-only DPO | 33 | 90 | 36.67% |
| Augmented DPO | 37 | 90 | 41.11% |
| Train-only DPO + rule revision | 49 | 90 | 54.44% |
| Augmented DPO + rule revision | 53 | 90 | 58.89% |
| 原始 vLLM baseline | 49 | 90 | 54.44% |
| 单独 rule revision baseline | 60 | 90 | 66.67% |

结论：

- 加入 rule-revised successful outputs 后，DPO 单独从 33/90 提升到 37/90，有小幅帮助。
- Augmented DPO + rule revision 达到 53/90，比原始 vLLM baseline 高 4 题，也比 train-only DPO + rule revision 高 4 题。
- 但它仍低于单独 rule revision baseline 的 60/90，说明当前 DPO 训练还没有学会稳定地产生比规则修正更好的代码。
- 因 augmented DPO 单独没有达到 49/90 gate，本轮未跑 MBPP test 500 条。

相关文件：

- `docs/augmented_train_only_dpo_plan.md`
- `docs/augmented_train_only_dpo_results.md`
- `data/preferences/preference_pairs_qwen25_k1_mbpp_train_augmented.jsonl`
- `data/eval/dpo_lora_train_augmented_mbpp_validation_summary.json`
- `data/eval/dpo_train_augmented_validation_revision_comparison.json`
- `data/eval/dpo_train_augmented_revision_vs_vllm_baseline_mbpp_validation_comparison.json`
- `data/eval/dpo_train_augmented_revision_vs_rule_revision_mbpp_validation_comparison.json`

## 11. Protected Rule Revision

为降低 rule revision 的副作用，本轮实现了保护版规则修正：

`scripts/protected_revise_code_outputs.py`

核心策略：只修改 verifier 已判失败的输出，`passed=true` 的样本完全不动。

全量 1128 条结果：

| 方法 | 通过数 | 总数 | pass@1 | pass->fail |
| --- | ---: | ---: | ---: | ---: |
| 原始 Qwen vLLM | 577 | 1128 | 51.15% | - |
| unprotected rule revision | 745 | 1128 | 66.05% | 10 |
| protected rule revision | 755 | 1128 | 66.93% | 0 |

按 split：

| split | protected passed | total | pass@1 |
| --- | ---: | ---: | ---: |
| MBPP train | 270 | 374 | 72.19% |
| MBPP validation | 61 | 90 | 67.78% |
| MBPP test | 320 | 500 | 64.00% |
| HumanEval+ test | 104 | 164 | 63.41% |

DPO validation 上的保护版级联：

| 输入 | before | protected after | unprotected after |
| --- | ---: | ---: | ---: |
| Train-only DPO validation | 33/90 | 50/90 | 49/90 |
| Augmented DPO validation | 37/90 | 54/90 | 53/90 |

结论：

- protected revision 保留了全部已通过样本，消除了 pass->fail。
- 全量结果从 unprotected 的 745/1128 提升到 755/1128。
- MBPP validation 达到 61/90，超过此前单独 rule revision 的 60/90。
- 后续应把 protected rule revision 作为主 baseline，unprotected 只作为 ablation。

相关文件：

- `docs/protected_rule_revision_plan.md`
- `docs/protected_rule_revision_results.md`
- `data/responses/coding_all_qwen25_vllm_k1_protected_revised_labeled.jsonl`
- `data/eval/vllm_baseline_protected_revision_summary.json`
- `data/eval/vllm_baseline_protected_revision_comparison.json`
- `data/eval/dpo_train_augmented_validation_protected_revision_comparison.json`
- `data/eval/dpo_train_only_validation_protected_revision_comparison.json`

## 12. 为什么测试跑得快

这次快主要有四个原因：

1. 推理用的是 vLLM，批处理和 KV cache 管理效率很高。
2. A800 对 7B 模型的短输出推理非常宽裕。
3. 本轮是 `k=1`，不是多采样，不需要每题生成很多候选。
4. verifier 主要是小 Python 单元测试，CPU 子进程执行很快；并且模型和依赖已经缓存/预热过。

所以“跑得快”既有卡好的原因，也有任务设置轻的原因。后续真正 DPO 或多轮 self-evolving 训练会明显更慢，尤其是在显存被占用时。

## 13. 交付文件清单

核心指标汇总：

`data/final/project_metrics_summary.json`

阶段文档：

- `docs/bootstrap_status_2026-07-02.md`
- `docs/B_C_work_complete.md`
- `docs/30_percent_milestone.md`
- `docs/final_project_report.md`

核心脚本：

- `scripts/prepare_coding_prompts.py`
- `scripts/vllm_smoke_generate.py`
- `scripts/verify_mbpp_smoke.py`
- `scripts/build_failure_artifacts.py`
- `scripts/error-analysis/discover_error_taxonomy.py`
- `scripts/generate_auto_rubric.py`
- `scripts/evaluate_rubric_static.py`
- `scripts/build_preference_pairs.py`
- `scripts/filter_preference_pairs.py`
- `scripts/build_augmented_preference_pairs.py`
- `scripts/revise_code_outputs.py`
- `scripts/protected_revise_code_outputs.py`
- `scripts/compare_revision_results.py`
- `scripts/sft_lora_smoke_train.py`
- `scripts/dpo_lora_train.py`
- `scripts/generate_with_lora_adapter.py`
- `scripts/build_self_play_error_discovery.py`
- `scripts/analyze_rubric_evolution.py`

## 14. 作业要求对齐补充

已补充三份面向老师要求的对齐文档：

- `docs/assignment_requirement_alignment.md`
- `docs/rubric_evolution_analysis.md`
- `docs/self_play_error_discovery.md`

当前最诚实的定位：

- Method 1 已完成第一版闭环：错误发现、rubric 生成、自评指标、DPO 训练、protected revision ablation；但多轮在线 self-updating rubric vs fixed rubric 还未完整跑完。
- Method 2 已有两层产物：先生成 verifier-grounded self-play proxy 178 条；随后补跑真实 LLM critic 小样本闭环，MBPP train 16/16 修复成功，生成 16 条 `A=失败输出, critique=LLM 找错, B=LLM 修复输出, A<B` preference pairs。
- Method 3 已完成 MBPP -> HumanEval+ 的最小跨代码迁移审计；本轮又补充 GSM8K n=100 推荐 benchmark appendix、GSM8K -> MATH safe-subset transfer，以及 full MATH verifier n=100 pressure test；但还不是完整 MATH test split 上的跨领域 meta-learning。

下一阶段执行记录：

- `docs/next_phase_execution_plan.md`
- `docs/llm_self_play_critic_results.md`
- `docs/fixed_vs_updated_rubric_ablation.md`
- `docs/meta_transfer_audit.md`

新增机器可读指标：

- `data/self_play/llm_critic_metrics_mbpp_train_n16_v2.json`
- `data/self_play/llm_critic_pairs_mbpp_train_n16_v2.jsonl`
- `data/analysis/fixed_vs_updated_rubric_ablation.json`
- `data/analysis/meta_transfer_audit.json`

## 15. LLM-Critic 54 + DPO

在上一轮 16 条真实 LLM critic 小样本闭环后，本轮扩大到 MBPP train 中 54 条 proxy-success 样本，并合并进无验证集泄漏的 DPO 训练数据。

新增结果：

| 阶段 | 指标 |
| --- | ---: |
| LLM critic attempted | 54 |
| LLM critic repaired | 54 |
| 新增 LLM critic A<B pairs | 54 |
| 合并后 train preference pairs | 266 |
| DPO steps | 266 |
| DPO preference accuracy | 79.70% |
| MBPP validation pass@1 | 43/90 |
| MBPP validation + protected revision | 54/90 |

解读：

- 相比 augmented train-only DPO 单独的 37/90，加入真实 LLM critic pairs 后，DPO 单独提升到 43/90。
- 但 protected revision 后仍为 54/90，与旧 augmented DPO + protected revision 持平，没有超过 protected rule revision baseline 的 61/90。
- 因此下一步不应只继续堆同类 syntax/format 修复 pairs，应转向逻辑错误 critic 和更强的 fixed-vs-updated 多轮对照。

相关文件：

- `docs/llmcritic54_dpo_execution_plan.md`
- `docs/llmcritic54_dpo_results.md`
- `data/self_play/llm_critic_pairs_mbpp_train_n54_v1.jsonl`
- `data/preferences/preference_pairs_qwen25_k1_mbpp_train_augmented_llmcritic54.jsonl`
- `outputs/dpo_lora_mbpp_train_augmented_llmcritic54_e1_mlen768/train_metrics.json`
- `data/final/llmcritic54_dpo_results_summary.json`

## 16. Logic Error Critic Probe

上一轮结论是：syntax/format 类 LLM critic 已经 54/54 修复成功，但 DPO + protected 后没有继续超过 54/90。因此本轮专门抽 MBPP train 的 `logic_error` 做真实 LLM critic。

结果：

| 阶段 | 指标 |
| --- | ---: |
| MBPP train logic_error | 75 |
| protected 后仍失败的 logic_error | 74 |
| logic critic attempted | 20 |
| logic critic repaired | 2 |
| logic critic A<B pairs | 2 |
| repair rate | 10.00% |
| DPO gate | failed |

解读：

- 逻辑错误和 syntax/format 错误难度明显不同。模型常能写出部分正确的 critique，但 revised code 仍无法通过 verifier。
- 因只得到 2 条成功 pairs，未达到预设 gate，本轮没有把它合并进 DPO，避免低质量 semantic pairs 污染训练。
- 下一步应做多候选 self-play + verifier 筛选，或在 prompt 中加入更强的失败断言解释/canonical hint，而不是继续单候选贪心修复。

相关文件：

- `docs/logic_critic_execution_plan.md`
- `docs/logic_critic_n20_results.md`
- `data/self_play/llm_critic_metrics_mbpp_train_logic_n20_v1.json`
- `data/self_play/llm_critic_pairs_mbpp_train_logic_n20_v1.jsonl`

## 17. 下一步建议

## 17. Logic Multi-Candidate Self-Play

在单候选 logic critic 只有 2/20 成功后，本轮对同一批 20 个 MBPP train logic errors 做了 `k=3` 多候选 self-play，并用 verifier 每题筛选第一个通过候选。

| 阶段 | 指标 |
| --- | ---: |
| Attempted tasks | 20 |
| Total candidates | 60 |
| Passed candidates | 9 |
| Repaired tasks | 6 |
| Preference pairs | 6 |
| Task repair rate | 30.00% |
| Candidate pass rate | 15.00% |

对比单候选，任务级修复从 2/20 提升到 6/20，说明多候选 + verifier 筛选对 semantic self-play 有明显帮助。但它刚好达到合并 gate，未达到完整 DPO gate 8/20，所以本轮只合并 preference data，不直接跑完整 DPO。

合并后 train-only preference data：

| Source | Count |
| --- | ---: |
| canonical_solution | 158 |
| rule_revised_success_output | 54 |
| llm_self_play_revised_passed | 54 |
| llm_self_play_logic_multicandidate_revised_passed | 6 |
| total | 272 |

相关文件：

- `docs/logic_multicandidate_execution_plan.md`
- `docs/logic_multicandidate_n20_k3_results.md`
- `data/self_play/llm_critic_metrics_mbpp_train_logic_n20_k3.json`
- `data/self_play/llm_critic_pairs_mbpp_train_logic_n20_k3.jsonl`
- `data/preferences/preference_pairs_qwen25_k1_mbpp_train_augmented_llmcritic54_logic_k3.jsonl`

## 18. 下一步建议

## 18. Logic k=5 Multi-Candidate

在 k=3 修复 6/20 后，本轮追加 seed 404/505，把同一批 20 个 logic tasks 扩展到 `k=5`。

| 阶段 | 指标 |
| --- | ---: |
| Attempted tasks | 20 |
| Total candidates | 100 |
| Passed candidates | 13 |
| Repaired tasks | 7 |
| Preference pairs | 7 |
| Task repair rate | 35.00% |
| Candidate pass rate | 13.00% |

k=5 只比 k=3 多修复 1 个任务：`6/20 -> 7/20`，没有达到完整 DPO gate 8/20。因此本轮只合并 train-only preference data，不跑 DPO。

合并后 train-only preference data：

| Source | Count |
| --- | ---: |
| canonical_solution | 158 |
| rule_revised_success_output | 54 |
| llm_self_play_revised_passed | 54 |
| llm_self_play_logic_multicandidate_revised_passed | 7 |
| total | 273 |

相关文件：

- `docs/logic_k5_execution_plan.md`
- `docs/logic_multicandidate_n20_k5_results.md`
- `data/self_play/llm_critic_metrics_mbpp_train_logic_n20_k5.json`
- `data/self_play/llm_critic_pairs_mbpp_train_logic_n20_k5.jsonl`
- `data/preferences/preference_pairs_qwen25_k1_mbpp_train_augmented_llmcritic54_logic_k5.jsonl`

## 19. 下一步建议

1. 不建议继续直接堆同样 prompt 的候选，k=5 边际收益已经变小。
2. 优先改 prompt：让模型先逐条解释失败断言的输入输出含义，归纳规格，再写 revised code。
3. 达到至少 8/20 或更大样本稳定修复率后，再合并并跑完整 DPO。
4. 做真正的 fixed-rubric vs updated-rubric 两条 DPO/revision 线，而不只停留在 CPU audit。
5. Method 3 若要继续加强，下一步应在 full MATH 上改 prompt / max tokens，降低 `ambiguous_final_answer`，再扩到更大样本；GSM8K -> MATH safe-subset 和 full verifier pressure test 已完成。

## 20. Logic Spec-First Prompt Probe

根据 k=5 边际收益下降的结论，本轮没有继续堆 default prompt 候选，而是新增 `spec_first` prompt mode：要求模型先逐条解释可见断言的输入含义、期望输出含义和隐含规格，再归纳函数规格并写 revised code。

同一批 20 个 MBPP train `logic_error` 样本结果如下：

| 方法 | 修复任务数 | Repair rate |
| --- | ---: | ---: |
| default prompt, single candidate | 2/20 | 10.00% |
| default prompt, k=3 | 6/20 | 30.00% |
| default prompt, k=5 | 7/20 | 35.00% |
| spec-first prompt v1, single candidate | 1/20 | 5.00% |

本轮 spec-first v1 的 critique 提取率是 100%，但 verifier 只确认 1 个 successful repair，且失败样本里出现 3 个 syntax error。因此它没有通过最低有效 gate，不合并进 DPO，也不扩成 spec-first k=3。

解读：

- 单轮长 JSON prompt 能让模型写出更完整的解释，但解释没有稳定约束最终代码。
- 输出越长，代码字段越容易出现格式退化，例如把循环/条件压成一行或混入转义换行。
- 这说明 Method 2 的瓶颈不是“是否让模型解释”这么简单，而是要把错误发现、规格归纳、代码生成拆开并逐步校验。

相关文件：

- `docs/logic_spec_prompt_execution_plan.md`
- `docs/logic_spec_prompt_n20_results.md`
- `data/self_play/llm_critic_metrics_mbpp_train_logic_n20_specfirst_v1.json`
- `data/self_play/llm_critic_pairs_mbpp_train_logic_n20_specfirst_v1.jsonl`

## 21. Updated Next Steps

1. 改成两阶段 prompt：先只生成 assertion analysis + inferred spec，再让模型基于 spec 和 tests 生成代码。
2. 给第二阶段加硬格式约束和 compile/syntax repair pass，先保证候选能被 verifier 正常执行。
3. 两阶段单候选达到 4/20 后再做 k=3；达到 8/20 后再合并 preference data 并跑 DPO。
4. 继续保留 default k=5 的 7 条 logic pairs 作为目前最可靠的 semantic self-play 增量数据。

## 22. Logic Two-Stage Self-Play Critic

根据 spec-first v1 的失败原因，本轮新增两阶段脚本 `scripts/llm_two_stage_self_play_critic.py`：

1. Stage 1 只生成 assertion analysis、inferred spec 和 suspected errors。
2. Stage 2 只基于 Stage 1 的规格和可见测试生成 revised code。
3. 若 revised code 不能 compile，则做一次 syntax/format repair。
4. 最终仍用 MBPP verifier 判断是否形成真实 `A < B` pair。

同一批 20 个 MBPP train `logic_error` 样本结果如下：

| 方法 | 修复任务数 | Repair rate | Syntax error after revision |
| --- | ---: | ---: | ---: |
| default prompt, single candidate | 2/20 | 10.00% | - |
| default prompt, k=3 | 6/20 | 30.00% | - |
| default prompt, k=5 | 7/20 | 35.00% | - |
| spec-first prompt v1, single candidate | 1/20 | 5.00% | 3 |
| two-stage spec-code v1, single candidate | 3/20 | 15.00% | 0 |

成功修复：

- `mbpp/train/648`
- `mbpp/train/650`
- `mbpp/train/661`

结论：

- 两阶段确实解决了 spec-first v1 的格式退化：最终 20 条代码都能 compile，syntax error 为 0。
- 但 successful repairs 只有 3/20，低于预设最低有效 gate 4/20，因此不继续 two-stage k=3，也不合并进 DPO。
- 目前最可靠的 logic 增量数据仍是 default k=5 verifier 筛出的 7 条 pairs。

相关文件：

- `docs/logic_two_stage_execution_plan.md`
- `docs/logic_two_stage_n20_results.md`
- `scripts/llm_two_stage_self_play_critic.py`
- `data/self_play/llm_critic_metrics_mbpp_train_logic_n20_twostage_v1.json`
- `data/self_play/llm_critic_pairs_mbpp_train_logic_n20_twostage_v1.jsonl`

## 23. Updated Next Steps After Two-Stage Probe

1. 不直接扩大 two-stage k=3，不跑 DPO；3 条 successful pairs 太少。
2. 先做失败诊断：对 17 个 fail 样本分类为 `wrong_spec`、`right_spec_wrong_algorithm`、`signature_or_interface`、`insufficient_tests`。
3. 若主要是 `wrong_spec`，加强 Stage 1 的 counterexample/test implication prompt。
4. 若主要是 `right_spec_wrong_algorithm`，在 Stage 2 前加 algorithm sketch，再生成代码。
5. 若主要是 `insufficient_tests`，考虑 retrieval 相似题或 oracle hint，但报告中要明确这不再是纯 self-discovery。

## 24. Two-Stage Failure Diagnosis

本轮对 two-stage v1 的 17 个失败样本做了结构化诊断。诊断输入包括原题、可见测试、Stage 1 inferred spec、Stage 2 revised code、verifier error，以及逐条 assert 的 actual/expected 输出。

分类结果：

| Diagnosis | Count |
| --- | ---: |
| `right_spec_wrong_algorithm` | 17 |

置信度：

| Confidence | Count |
| --- | ---: |
| high | 4 |
| medium | 13 |

代表性证据：

- `mbpp/train/603`: Stage 1 能描述 ludic number，但 Stage 2 过滤逻辑输出空列表。
- `mbpp/train/610`: 任务方向正确，但代码删除了错误下标，actual `[1, 1, 2, 4, 4, 5, 1]` vs expected `[1, 1, 3, 4, 4, 5, 1]`。
- `mbpp/train/659`: 重复元素识别方向正确，但输出顺序错，actual `[20, 30, 60, -20]` vs expected `[20, 30, -20, 60]`。

结论：

- 失败主因不是 Stage 1 规格归纳，也不是接口/格式问题。
- 主因是 Stage 2 没有把规格转成正确算法。
- 下一轮应做 `algorithm sketch -> visible test simulation -> code`，而不是继续改 Stage 1 或进入 retrieval/oracle。

相关文件：

- `docs/logic_two_stage_failure_diagnosis_plan.md`
- `docs/logic_two_stage_failure_diagnosis.md`
- `scripts/diagnose_two_stage_failures.py`
- `data/analysis/two_stage_failure_diagnosis.jsonl`

## 25. Updated Next Steps After Failure Diagnosis

1. 新增 Stage 2 algorithm-sketch 实验：先写算法步骤，再手工模拟至少 2 条可见测试，最后生成代码。
2. 继续使用同一批 20 个 logic samples 和同一 verifier，保持与 2/20、6/20、7/20、1/20、3/20 可比。
3. 单候选若 `<4/20` 停止；`>=4/20` 扩 k=3；`>=8/20` 才合并 preference data 并跑 DPO。
4. 不优先做 retrieval/oracle hint，因为当前诊断没有显示 `insufficient_tests` 是主因。

## 26. Algorithm-Sketch Self-Play Probe

根据 two-stage 失败诊断，本轮新增三阶段脚本 `scripts/llm_algorithm_sketch_self_play_critic.py`：

1. Stage 1 生成 inferred spec。
2. Stage 2a 生成 algorithm sketch，并手工模拟至少 2 条可见测试。
3. Stage 2b 根据 spec + algorithm sketch 生成 revised code。
4. 最终仍由 MBPP verifier 判断是否形成真实 `A < B` pair。

同一批 20 个 MBPP train `logic_error` 样本结果如下：

| 方法 | 修复任务数 | Repair rate | Syntax error after revision |
| --- | ---: | ---: | ---: |
| default prompt, single candidate | 2/20 | 10.00% | - |
| default prompt, k=3 | 6/20 | 30.00% | - |
| default prompt, k=5 | 7/20 | 35.00% | - |
| spec-first prompt v1, single candidate | 1/20 | 5.00% | 3 |
| two-stage spec-code v1, single candidate | 3/20 | 15.00% | 0 |
| algorithm-sketch v1, single candidate | 2/20 | 10.00% | 0 |

algorithm-sketch v1 成功修复：

- `mbpp/train/648`
- `mbpp/train/650`

与 two-stage v1 对比：

- 共同成功：`mbpp/train/648`, `mbpp/train/650`
- 新增成功：none
- 丢失成功：`mbpp/train/661`

结论：

- algorithm sketch 保持了格式稳定，syntax error 为 0。
- 但修复率从 two-stage 的 3/20 降到 2/20，没有新增成功样本。
- 因未达到 4/20 gate，本轮不继续 k=3，不合并 DPO。
- 对这个 7B 模型来说，继续增加解释/草图步骤并没有转化为更强修复，最可靠的 semantic self-play 增量仍是 default k=5 verifier 筛出的 7 条 pairs。

相关文件：

- `docs/logic_algorithm_sketch_execution_plan.md`
- `docs/logic_algorithm_sketch_n20_results.md`
- `scripts/llm_algorithm_sketch_self_play_critic.py`
- `data/self_play/llm_critic_metrics_mbpp_train_logic_n20_algosketch_v1.json`
- `data/self_play/llm_critic_pairs_mbpp_train_logic_n20_algosketch_v1.jsonl`

## 27. Updated Next Steps After Algorithm-Sketch Probe

1. 不继续 algorithm-sketch k=3，不跑 DPO。
2. 报告中保留 default k=5 的 7 条 logic pairs 作为当前最可靠 semantic self-play 数据。
3. 若继续追求 logic 修复率，下一步应做 verifier-feedback repair：生成候选后运行测试，把 actual/expected mismatch 反馈给模型再修一次。
4. verifier-feedback 属于更强外部执行反馈，写 Method 2 时需要明确标注，不要说成纯 self-discovery。

## 28. Logic k=5 Self-Play DPO

本轮把最可靠的 7 条 logic k=5 self-play pairs 合入已有 266 条 LLMCritic54 augmented train-only preference data，得到 273 条 preference pairs，并重新训练 DPO。

训练指标：

| Metric | Value |
| --- | ---: |
| Preference pairs | 273 |
| DPO steps | 273 |
| Skipped | 0 |
| Mean loss | 0.6468 |
| Preference accuracy | 80.59% |

MBPP validation：

| Method | Raw validation | Protected validation |
| --- | ---: | ---: |
| LLMCritic54 DPO | 43/90 | 54/90 |
| LLMCritic54 + logic k=5 DPO | 42/90 | 56/90 |
| Protected rule revision baseline | - | 61/90 |

结论：

- logic k=5 pairs 没有提升 raw DPO：`43/90 -> 42/90`。
- 但 protected cascade 后从 `54/90` 提升到 `56/90`，成为当前最好的无 validation 泄漏 DPO-related 结果。
- 它仍低于 protected rule revision baseline `61/90`，因此不跑 MBPP test。
- Method 2 的最终结论是：显式 self-play error discovery 能提供少量有用 preference signal，但在当前 7B 模型和小样本设置下，DPO 仍需要 protected revision 才能体现收益。

相关文件：

- `docs/logic_k5_dpo_results.md`
- `outputs/dpo_lora_mbpp_train_augmented_llmcritic54_logic_k5_e1_mlen768/train_metrics.json`
- `data/eval/dpo_lora_train_augmented_llmcritic54_logic_k5_mbpp_validation_summary.json`
- `data/eval/dpo_lora_train_augmented_llmcritic54_logic_k5_mbpp_validation_protected_revised_summary.json`

## 29. Updated Next Steps After Logic k=5 DPO

1. Method 2 已有完整链路：explicit error discovery -> improved B -> A<B pairs -> DPO -> validation/protected validation。
2. 不再继续合并低质 logic pairs；后续只在引入 verifier-feedback repair 时继续扩充 logic pairs。
3. 接下来优先整理 Method 1 fixed-vs-updated rubric proxy A/B，以及 Method 3 meta-transfer caveat，使最终报告满足作业结构。

## 30. Method 1 Fixed vs Updated Rubric Summary

Method 1 的完整在线 fixed-rubric vs updated-rubric 双轨 RL 没有运行；本项目采用 proxy A/B：

- Fixed/generic rubric：3 个通用维度，不链接自动发现错误模式。
- Updated/refined rubric：基于 551 个失败样本和 18 个 refined clusters 生成 6 个维度，覆盖 16 个错误模式。

Rubric quality:

| Rubric | Coverage | AUC | Kappa | Accuracy |
| --- | ---: | ---: | ---: | ---: |
| Fixed/generic | 0.000 | 0.660 | 0.316 | 0.655 |
| Updated/refined | 1.000 | 0.801 | 0.525 | 0.765 |

Training/revision proxy:

| Method | Raw validation | Protected validation |
| --- | ---: | ---: |
| Train-only DPO | 33/90 | 50/90 |
| Augmented DPO | 37/90 | 54/90 |
| LLMCritic54 DPO | 43/90 | 54/90 |
| LLMCritic54 + logic k=5 DPO | 42/90 | 56/90 |

结论：

- Updated rubric 明显提升 reward signal 区分度。
- Updated signal 能提升 DPO-related protected validation：`50/90 -> 56/90`。
- 但 DPO 仍未超过 protected rule revision `61/90`，说明小规模 LoRA DPO 还没有完全吸收规则化修复能力。
- protected revision 是 reward-hacking guard：只修改 verifier 失败样本，pass->fail 为 0。

相关文件：

- `docs/method1_fixed_updated_training_ablation.md`
- `docs/fixed_vs_updated_rubric_ablation.md`

## 31. Method 3 Meta-Transfer Summary

本项目已完成三类 Method 3 审计：一是最小跨代码任务迁移，检查 coding refined rubric 是否能从 MBPP 迁移到 HumanEval+；二是 GSM8K-derived rubric 到 MATH safe subset 的 zero-shot 迁移；三是 full MATH verifier n=100 pressure test，覆盖 all subjects / Level 1-5 和复杂答案格式。仍未完成的是完整 MATH test split 上的跨领域 meta-learning。

| Group | N | Auto rubric AUC | Generic rubric AUC |
| --- | ---: | ---: | ---: |
| MBPP train | 374 | 0.798 | 0.653 |
| MBPP validation | 90 | 0.795 | 0.688 |
| MBPP test | 500 | 0.785 | 0.621 |
| HumanEval+ test | 164 | 0.846 | 0.824 |

结论：

- refined coding rubric 不只在 MBPP train 有效，在 MBPP validation/test 和 HumanEval+ 上仍有区分度。
- 这只能作为 minimal meta-transfer evidence，不能声称完成了跨领域 self-evaluation meta-learning。
- 最终展示时应把 GSM8K n=100 写成推荐 benchmark appendix，把 GSM8K -> MATH safe subset 写成已完成 transfer，把 full MATH verifier pressure test 写成增强实验，把 full-scale MATH test split 写成 future work。

相关文件：

- `docs/method3_meta_transfer_final.md`
- `docs/meta_transfer_audit.md`

## 32. Final Submission Status

最终 leaderboard：

- `docs/final_method_leaderboard.md`

最强整体方法：

- Protected rule revision: `755/1128`, pass@1 `66.93%`

最强无泄漏 DPO-related validation 方法：

- LLMCritic54 + logic k=5 DPO + protected revision: `56/90`, pass@1 `62.22%`

项目当前可提交，但需要在答辩/报告中诚实说明：

1. 主 benchmark 是 coding tasks；GSM8K n=100 和 MATH safe subset 是补充 appendix，不是主训练 benchmark。
2. Method 1 的 fixed-vs-updated 是 proxy A/B，不是完整在线双轨 RL。
3. Method 3 包含跨代码迁移、GSM8K -> MATH safe-subset，以及 full MATH verifier n=100 pressure test；完整 MATH test split 仍未覆盖。
4. DPO 有训练闭环，但最终效果仍弱于 protected deterministic revision。
