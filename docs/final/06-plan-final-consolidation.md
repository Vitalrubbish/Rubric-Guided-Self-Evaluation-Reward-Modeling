# Final Consolidation Plan

日期：2026-07-03  
远程目录：`/data2/acm-group-3/Rubric-Guided-Self-Evaluation-Reward-Modeling`

## 0. 目标

把当前所有实验结果收束成统一 leaderboard 和组员可读交付文档。主结论切换为：

- protected rule revision 是当前最强主 baseline
- unprotected rule revision 只作为 ablation
- DPO 相关方法中，augmented DPO + protected revision 最好，但仍低于 protected rule revision

## 1. Step 1：盘点指标来源

指标来源：

- `data/analysis/coding_baseline_summary_qwen25_k1.json`
- `data/revision/revision_comparison_summary.json`
- `data/eval/vllm_baseline_protected_revision_summary.json`
- `data/final/project_metrics_summary.json`
- DPO validation summaries under `data/eval/`

验收标准：

- 所有 JSON 可读
- protected revision 全量为 755/1128
- augmented DPO + protected revision 为 54/90

执行状态：

- 状态：completed
- 检查结果：指标源均存在且 JSON 可解析；protected revision 755/1128；augmented DPO + protected revision 54/90。
- 是否修改后续方案：否。继续生成 leaderboard。

## 2. Step 2：生成 final leaderboard

计划产物：

- `scripts/build_final_leaderboard.py`
- `data/final/final_method_leaderboard.json`
- `docs/final_method_leaderboard.md`

验收标准：

- 脚本 `py_compile` 通过
- 远程运行成功
- leaderboard JSON 可解析
- Markdown 包含 overall、MBPP validation、protected revision ablation 三张表

执行状态：

- 状态：completed
- 检查结果：`scripts/build_final_leaderboard.py` 本地 `py_compile` 通过；远程运行成功；生成 `data/final/final_method_leaderboard.json` 和 `docs/final_method_leaderboard.md`；leaderboard JSON 解析通过，Markdown 包含 Overall、MBPP Validation、Protected Revision Ablation 三张表。
- 是否修改后续方案：否。继续数字一致性检查。

## 3. Step 3：校验 leaderboard 数字

必须检查：

- Overall protected rule revision = 755/1128
- Overall unprotected rule revision = 745/1128
- Original Qwen baseline = 577/1128
- MBPP validation protected rule revision = 61/90
- MBPP validation augmented DPO + protected revision = 54/90
- Full-failure DPO 标注为 leakage=yes

执行状态：

- 状态：completed
- 检查结果：脚本断言通过：Original=577/1128，Unprotected=745/1128，Protected=755/1128，MBPP validation Protected=61/90，Augmented DPO+Protected=54/90，Full-failure DPO 标为 leakage=yes，protected pass->fail=0。
- 是否修改后续方案：否。继续更新组员可读最终交付文档。

## 4. Step 4：更新组员可读交付文档

需要更新：

- `docs/final_project_report.md`
- 新增/更新 `docs/final_method_leaderboard.md`

验收标准：

- final report 中 protected rule revision 是主 baseline
- unprotected revision 被标为 ablation
- DPO 泄漏与无泄漏实验边界清楚

执行状态：

- 状态：completed
- 检查结果：`docs/final_project_report.md` 顶部已切换到 protected rule revision 主 baseline；`docs/final_method_leaderboard.md` 已生成，明确标注 Full-failure DPO 为 leakage=yes，unprotected revision 为 ablation。
- 是否修改后续方案：否。继续最终复核。

## 5. Step 5：最终复核

验收标准：

- 远程 JSON 均可解析
- 关键文档已同步
- 没有遗留训练/生成进程

执行状态：

- 状态：completed
- 检查结果：远程 `data/final/final_method_leaderboard.json` 和 `data/final/project_metrics_summary.json` 均可解析；关键文档已同步；没有遗留训练/生成/verifier 进程。
- 是否修改后续方案：否。final consolidation 完成。
