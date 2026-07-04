#!/usr/bin/env python3
"""Compute a minimal cross-split/domain rubric transfer audit."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from sklearn.metrics import accuracy_score, cohen_kappa_score, roc_auc_score


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def split_from_id(row: dict) -> str:
    row_id = row.get("id", "")
    parts = row_id.split("/")
    if len(parts) >= 3 and parts[0] == "mbpp":
        return f"mbpp/{parts[1]}"
    if row.get("dataset") == "humanevalplus":
        return "humanevalplus/test"
    return row.get("dataset") or "unknown"


def safe_auc(labels: list[int], scores: list[float]) -> float | None:
    if len(set(labels)) < 2:
        return None
    return float(roc_auc_score(labels, scores))


def metrics_for(rows: list[dict]) -> dict:
    labels = [1 if row.get("passed") else 0 for row in rows]
    scores = [float(row.get("static_total_score", 0)) for row in rows]
    preds = [1 if score >= 4.0 else 0 for score in scores]
    return {
        "num_samples": len(rows),
        "passed": sum(labels),
        "failed": len(rows) - sum(labels),
        "auc": safe_auc(labels, scores),
        "kappa": float(cohen_kappa_score(labels, preds)) if len(set(labels)) > 1 else None,
        "accuracy": float(accuracy_score(labels, preds)) if labels else None,
        "mean_score_passed": sum(s for s, y in zip(scores, labels) if y) / max(1, sum(labels)),
        "mean_score_failed": sum(s for s, y in zip(scores, labels) if not y) / max(1, len(rows) - sum(labels)),
    }


def summarize(path: Path) -> dict:
    by_group = defaultdict(list)
    all_rows = []
    for row in read_jsonl(path):
        all_rows.append(row)
        by_group[row.get("dataset") or "unknown"].append(row)
        by_group[split_from_id(row)].append(row)
    return {"all": metrics_for(all_rows), "groups": {key: metrics_for(rows) for key, rows in sorted(by_group.items())}}


def fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--auto-scores", type=Path, default=Path("data/rubrics/auto_rubric_scores_static.jsonl"))
    parser.add_argument("--generic-scores", type=Path, default=Path("data/rubrics/generic_rubric_scores_static.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/analysis/meta_transfer_audit.json"))
    parser.add_argument("--md-output", type=Path, default=Path("docs/meta_transfer_audit.md"))
    args = parser.parse_args()

    result = {
        "type": "minimal_cross_domain_rubric_transfer_audit",
        "auto_rubric": summarize(args.auto_scores),
        "generic_rubric": summarize(args.generic_scores),
        "status": "completed",
        "caveat": (
            "This is not full Method 3 meta-learning. It audits whether the current coding rubric "
            "keeps discriminative power across MBPP splits and HumanEval+. A true Method 3 run still "
            "needs a new task such as GSM8K/MATH and zero-shot rubric generation."
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    md = [
        "# Minimal Meta-Transfer Audit",
        "",
        "## 目的",
        "",
        "这一步只做 Method 3 的最小可落地产物：检查当前 coding rubric 在 MBPP train/validation/test 与 HumanEval+ 上是否仍有区分度。它不是完整的 GSM8K -> MATH meta-learning。",
        "",
        "## Auto Rubric By Group",
        "",
        "| Group | N | Passed | AUC | Kappa | Accuracy |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for group, item in result["auto_rubric"]["groups"].items():
        md.append(f"| {group} | {item['num_samples']} | {item['passed']} | {fmt(item['auc'])} | {fmt(item['kappa'])} | {fmt(item['accuracy'])} |")
    md.extend(
        [
            "",
            "## Generic Rubric By Group",
            "",
            "| Group | N | Passed | AUC | Kappa | Accuracy |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for group, item in result["generic_rubric"]["groups"].items():
        md.append(f"| {group} | {item['num_samples']} | {item['passed']} | {fmt(item['auc'])} | {fmt(item['kappa'])} | {fmt(item['accuracy'])} |")
    md.extend(
        [
            "",
            "## Check",
            "",
            "如果 auto rubric 在 HumanEval+ 上仍明显高于 generic rubric，说明它至少有一定跨代码数据集迁移能力。完整 Method 3 仍需要新增 GSM8K/MATH 或另一个未参与 rubric 发现的新任务。",
            "",
            "输出 JSON：",
            "",
            f"`{args.output}`",
        ]
    )
    args.md_output.parent.mkdir(parents=True, exist_ok=True)
    args.md_output.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "md": str(args.md_output)}, indent=2))


if __name__ == "__main__":
    main()
