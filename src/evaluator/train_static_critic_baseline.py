#!/usr/bin/env python3
"""Train a lightweight no-gate critic baseline for APPS simple.

This is not the final reward model. It is a sanity-check critic that uses only
public task text, visible generated code, generation metadata, and static code
features to predict verifier pass/fail. Verifier failure types, human labels,
and provisional labels are deliberately excluded from inputs.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score, roc_auc_score


PRIVATE_INPUT_KEYS = {
    "passed",
    "correctness_label",
    "failure_type",
    "verifier_summary",
    "deterministic_error_label",
    "deterministic_label_source",
    "deterministic_label_confidence",
    "human_error_label",
    "human_error_confidence",
    "human_evidence",
    "error_attribution_label",
    "error_attribution_source",
    "error_attribution_confidence",
    "error_attribution_trainable",
    "provisional_label_reference",
    "provisional_confidence_reference",
    "provisional_method_reference",
}

CODE_TERMS = [
    "for ",
    "while ",
    "if ",
    "elif ",
    "else",
    "return",
    "print",
    "input(",
    "import ",
    "from ",
    "class ",
    "def ",
    "try:",
    "except",
    "lambda",
    "sort",
    "sorted",
    "Counter",
    "defaultdict",
    "heapq",
    "bisect",
    "math.",
    "re.",
    "List[",
    "Dict[",
    "set(",
    "dict(",
    "append",
    "pop(",
    "split(",
    "join(",
    "replace(",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def short_text(value: Any, limit: int = 500) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 18].rstrip() + " ... [truncated]"


def safe_log1p(value: float | int | None) -> float:
    if value is None:
        return 0.0
    try:
        return math.log1p(max(0.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def parse_ast(code: str) -> ast.AST | None:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            return ast.parse(code or "")
    except SyntaxError:
        return None


def names_from_public_interface(values: list[Any]) -> set[str]:
    names = set()
    for value in values or []:
        text = str(value)
        match = re.search(r"\b(?:def|class)\s+([A-Za-z_][A-Za-z0-9_]*)\b", text)
        if match:
            names.add(match.group(1))
            continue
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", text.strip()):
            names.add(text.strip())
    return names


def has_required_interface(row: dict[str, Any], code: str) -> bool:
    names = names_from_public_interface(row.get("public_interface") or [])
    if not names:
        return True
    return all(re.search(rf"\b(?:def|class)\s+{re.escape(name)}\b", code) for name in names)


class AstCounter(ast.NodeVisitor):
    def __init__(self) -> None:
        self.counts: Counter[str] = Counter()

    def generic_visit(self, node: ast.AST) -> None:
        self.counts[type(node).__name__] += 1
        super().generic_visit(node)


def static_features(row: dict[str, Any]) -> dict[str, float]:
    code = str(row.get("extracted_code") or row.get("generated_code") or "")
    generated = str(row.get("generated_code") or "")
    task = str(row.get("task") or "")
    public_interface = row.get("public_interface") or []
    tree = parse_ast(code)
    code_lines = [line for line in code.splitlines() if line.strip()]

    features: dict[str, float] = {
        "code_chars_log": safe_log1p(len(code)),
        "code_lines_log": safe_log1p(len(code_lines)),
        "task_chars_log": safe_log1p(len(task)),
        "generated_token_count_log": safe_log1p(row.get("generated_token_count")),
        "public_interface_count_log": safe_log1p(len(public_interface)),
        "code_to_task_char_ratio_log": safe_log1p(len(code) / max(1, len(task))),
        "parse_ok": 1.0 if tree is not None else 0.0,
        "parse_failed": 1.0 if tree is None else 0.0,
        "finish_reason_length": 1.0 if row.get("finish_reason") == "length" else 0.0,
        "finish_reason_stop": 1.0 if row.get("finish_reason") == "stop" else 0.0,
        "io_mode_function_call": 1.0 if row.get("io_mode") == "function_call" else 0.0,
        "io_mode_stdin_stdout": 1.0 if row.get("io_mode") == "stdin_stdout" else 0.0,
        "required_interface_missing": 0.0 if has_required_interface(row, code) else 1.0,
        "required_interface_present": 1.0 if has_required_interface(row, code) else 0.0,
        "generated_has_markdown_fence": 1.0 if "```" in generated else 0.0,
        "code_empty": 1.0 if not code.strip() else 0.0,
        "uses_unresolved_list_annotation": 1.0 if "List[" in code and "from typing import List" not in code and "import typing" not in code else 0.0,
        "uses_unresolved_dict_annotation": 1.0 if "Dict[" in code and "from typing import Dict" not in code and "import typing" not in code else 0.0,
    }

    lowered = code.lower()
    for term in CODE_TERMS:
        key = re.sub(r"[^A-Za-z0-9]+", "_", term).strip("_").lower()
        features[f"code_term_{key}_log"] = safe_log1p(lowered.count(term.lower()))

    if tree is not None:
        counter = AstCounter()
        counter.visit(tree)
        for node_name in [
            "FunctionDef",
            "ClassDef",
            "For",
            "While",
            "If",
            "Return",
            "Call",
            "Subscript",
            "Compare",
            "BinOp",
            "BoolOp",
            "ListComp",
            "DictComp",
            "Try",
            "Import",
            "ImportFrom",
        ]:
            features[f"ast_{node_name}_log"] = safe_log1p(counter.counts.get(node_name, 0))
        features["ast_total_nodes_log"] = safe_log1p(sum(counter.counts.values()))
    return features


def build_text(row: dict[str, Any], mode: str) -> str:
    code = str(row.get("extracted_code") or row.get("generated_code") or "")
    task = str(row.get("task") or "")
    if mode == "none":
        return ""
    if mode == "code":
        return code
    if mode == "task_code":
        return f"Task:\n{task}\n\nCode:\n{code}"
    raise ValueError(f"unknown text mode: {mode}")


def split_rows(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("split") == split]


def labels(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.array([1 if row.get("passed") else 0 for row in rows], dtype=int)


def pct(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    if len(set(y_true.tolist())) < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def metrics_at_threshold(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict[str, Any]:
    y_pred = (y_score >= threshold).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": (pct(tp, tp + fn) + pct(tn, tn + fp)) / 2,
        "kappa": float(cohen_kappa_score(y_true, y_pred)) if len(set(y_pred.tolist())) > 1 or len(set(y_true.tolist())) > 1 else 0.0,
        "auc": safe_auc(y_true, y_score),
        "predicted_pass_rate": pct(tp + fp, len(y_true)),
        "true_pass_rate": pct(int(y_true.sum()), len(y_true)),
        "overacceptance_rate": pct(fp, fp + tn),
        "false_rejection_rate": pct(fn, fn + tp),
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def threshold_sweep(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, Any]:
    thresholds = [round(i / 100, 2) for i in range(0, 101)]
    rows = [metrics_at_threshold(y_true, y_score, threshold) for threshold in thresholds]
    best_balanced = max(rows, key=lambda item: (item["balanced_accuracy"], item["accuracy"]))
    best_accuracy = max(rows, key=lambda item: (item["accuracy"], item["balanced_accuracy"]))
    constrained = [
        item for item in rows
        if item["overacceptance_rate"] <= 0.25 and item["false_rejection_rate"] <= 0.25
    ]
    return {
        "best_balanced_accuracy": best_balanced,
        "best_accuracy": best_accuracy,
        "best_with_overacceptance_le_25_false_rejection_le_25": max(constrained, key=lambda item: item["balanced_accuracy"]) if constrained else None,
        "selected_thresholds": [item for item in rows if item["threshold"] in {0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}],
    }


def make_features(
    train_rows: list[dict[str, Any]],
    other_rows: list[dict[str, Any]],
    text_mode: str,
    max_text_features: int,
) -> tuple[csr_matrix, csr_matrix, dict[str, Any]]:
    dict_vectorizer = DictVectorizer(sparse=True)
    x_train_static = dict_vectorizer.fit_transform([static_features(row) for row in train_rows])
    x_other_static = dict_vectorizer.transform([static_features(row) for row in other_rows])

    metadata: dict[str, Any] = {
        "static_feature_count": len(dict_vectorizer.feature_names_),
        "text_mode": text_mode,
    }

    if text_mode == "none":
        return x_train_static, x_other_static, metadata

    train_texts = [build_text(row, text_mode) for row in train_rows]
    other_texts = [build_text(row, text_mode) for row in other_rows]
    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=max_text_features,
        lowercase=True,
        sublinear_tf=True,
    )
    word_vectorizer = TfidfVectorizer(
        analyzer="word",
        token_pattern=r"(?u)\b[A-Za-z_][A-Za-z_0-9]{1,}\b|\b\d+\b",
        ngram_range=(1, 2),
        min_df=2,
        max_features=max_text_features // 2,
        lowercase=True,
        sublinear_tf=True,
    )
    x_train_char = char_vectorizer.fit_transform(train_texts)
    x_other_char = char_vectorizer.transform(other_texts)
    x_train_word = word_vectorizer.fit_transform(train_texts)
    x_other_word = word_vectorizer.transform(other_texts)
    metadata.update(
        {
            "char_feature_count": len(char_vectorizer.get_feature_names_out()),
            "word_feature_count": len(word_vectorizer.get_feature_names_out()),
        }
    )
    return hstack([x_train_static, x_train_char, x_train_word]).tocsr(), hstack([x_other_static, x_other_char, x_other_word]).tocsr(), metadata


def fit_and_evaluate_config(
    name: str,
    train_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    text_mode: str,
    class_weight: str | None,
    max_text_features: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    combined_eval_rows = validation_rows + test_rows
    y_train = labels(train_rows)
    y_validation = labels(validation_rows)
    y_test = labels(test_rows)
    y_eval = labels(combined_eval_rows)

    x_train, x_eval, feature_metadata = make_features(train_rows, combined_eval_rows, text_mode, max_text_features)
    x_validation = x_eval[: len(validation_rows)]
    x_test = x_eval[len(validation_rows) :]

    model = LogisticRegression(
        max_iter=2000,
        solver="liblinear",
        class_weight=class_weight,
        random_state=20260713,
    )
    model.fit(x_train, y_train)

    p_train = model.predict_proba(x_train)[:, 1]
    p_validation = model.predict_proba(x_validation)[:, 1]
    p_test = model.predict_proba(x_test)[:, 1]
    p_eval = model.predict_proba(x_eval)[:, 1]
    validation_sweep = threshold_sweep(y_validation, p_validation)
    selected_threshold = validation_sweep["best_balanced_accuracy"]["threshold"]

    summary = {
        "name": name,
        "text_mode": text_mode,
        "class_weight": class_weight,
        "feature_metadata": feature_metadata,
        "train": metrics_at_threshold(y_train, p_train, selected_threshold),
        "validation_default_0_5": metrics_at_threshold(y_validation, p_validation, 0.5),
        "validation_selected": metrics_at_threshold(y_validation, p_validation, selected_threshold),
        "validation_threshold_sweep": validation_sweep,
        "test_default_0_5": metrics_at_threshold(y_test, p_test, 0.5),
        "test_selected": metrics_at_threshold(y_test, p_test, selected_threshold),
        "eval_selected": metrics_at_threshold(y_eval, p_eval, selected_threshold),
    }

    def build_predictions(rows: list[dict[str, Any]], probabilities: np.ndarray) -> list[dict[str, Any]]:
        values = []
        for row, probability in zip(rows, probabilities):
            values.append(
                {
                    "response_id": row.get("response_id"),
                    "id": row.get("id"),
                    "split": row.get("split"),
                    "passed": bool(row.get("passed")),
                    "critic_pass_probability": float(probability),
                    "critic_predicted_pass": bool(probability >= selected_threshold),
                    "selected_threshold": selected_threshold,
                    "model_name": name,
                }
            )
        return values

    predictions = []
    for row, probability in zip(combined_eval_rows, p_eval):
        predictions.append(
            {
                "response_id": row.get("response_id"),
                "id": row.get("id"),
                "split": row.get("split"),
                "passed": bool(row.get("passed")),
                "critic_pass_probability": float(probability),
                "critic_predicted_pass": bool(probability >= selected_threshold),
                "selected_threshold": selected_threshold,
                "model_name": name,
            }
        )
    all_predictions = build_predictions(train_rows, p_train) + build_predictions(combined_eval_rows, p_eval)
    return summary, predictions, all_predictions


def no_gate_metrics(scores_path: Path) -> dict[str, Any] | None:
    if not scores_path.exists():
        return None
    rows = read_jsonl(scores_path)
    result = {}
    for split_name, split_rows_ in [
        ("validation", [row for row in rows if row.get("split") == "validation"]),
        ("test", [row for row in rows if row.get("split") == "test"]),
        ("eval", rows),
    ]:
        y_true = np.array([1 if row.get("passed") else 0 for row in split_rows_], dtype=int)
        y_pred = np.array([1 if row.get("predicted_pass") else 0 for row in split_rows_], dtype=int)
        y_score = np.array([float(row.get("overall_score") or 0.0) for row in split_rows_], dtype=float)
        if not len(y_true):
            continue
        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        result[split_name] = {
            "num_samples": len(split_rows_),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": (pct(tp, tp + fn) + pct(tn, tn + fp)) / 2,
            "kappa": float(cohen_kappa_score(y_true, y_pred)),
            "auc_overall_score": safe_auc(y_true, y_score),
            "predicted_pass_rate": pct(tp + fp, len(y_true)),
            "true_pass_rate": pct(int(y_true.sum()), len(y_true)),
            "overacceptance_rate": pct(fp, fp + tn),
            "false_rejection_rate": pct(fn, fn + tp),
            "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        }
    return result


def render_markdown(summary: dict[str, Any]) -> str:
    best = summary["best_model"]
    no_gate = summary.get("no_gate_baseline") or {}
    lines = [
        "# Static Critic Baseline",
        "",
        "## Purpose",
        "",
        "This baseline checks whether a supervised no-gate critic trained on visible task/code evidence can reduce the no-gate rubric judge's false-positive problem.",
        "",
        "Inputs deliberately exclude verifier failure type, human attribution labels, and provisional labels.",
        "",
        "## Best Model",
        "",
        f"- name: `{best['name']}`",
        f"- text mode: `{best['text_mode']}`",
        f"- class weight: `{best['class_weight']}`",
        f"- selected threshold: `{best['validation_selected']['threshold']}`",
        "",
        "## Test Metrics",
        "",
        "| Metric | Static critic | No-gate rubric judge |",
        "| --- | ---: | ---: |",
    ]
    test = best["test_selected"]
    no_gate_test = no_gate.get("test", {})
    for key in ["accuracy", "balanced_accuracy", "kappa", "auc", "predicted_pass_rate", "overacceptance_rate", "false_rejection_rate"]:
        ng_key = "auc_overall_score" if key == "auc" else key
        critic_value = test.get(key)
        judge_value = no_gate_test.get(ng_key)
        lines.append(
            f"| {key} | {critic_value:.4f} | {judge_value:.4f} |"
            if isinstance(judge_value, (int, float)) and isinstance(critic_value, (int, float))
            else f"| {key} | {critic_value} | {judge_value} |"
        )
    lines.extend(
        [
            "",
            "Static critic test confusion:",
            "",
            "```text",
            f"TN={test['confusion']['tn']}  FP={test['confusion']['fp']}",
            f"FN={test['confusion']['fn']}   TP={test['confusion']['tp']}",
            "```",
            "",
            "No-gate judge test confusion:",
            "",
            "```text",
            f"TN={no_gate_test.get('confusion', {}).get('tn')}  FP={no_gate_test.get('confusion', {}).get('fp')}",
            f"FN={no_gate_test.get('confusion', {}).get('fn')}   TP={no_gate_test.get('confusion', {}).get('tp')}",
            "```",
            "",
            "## All Configs",
            "",
            "| Model | Val balanced acc | Test balanced acc | Test overacceptance | Test false rejection |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in summary["all_models"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    item["name"],
                    f"{item['validation_selected']['balanced_accuracy']:.4f}",
                    f"{item['test_selected']['balanced_accuracy']:.4f}",
                    f"{item['test_selected']['overacceptance_rate']:.4f}",
                    f"{item['test_selected']['false_rejection_rate']:.4f}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "A lightweight supervised critic is not the final Method 1 reward model, but it gives a lower-cost check of the central project question: whether failure experience plus visible-code evidence can produce a more useful no-gate signal than raw rubric self-scoring.",
            "",
            "If this baseline reduces false positives versus the no-gate rubric judge, the next step should be critic distillation or preference construction. If it does not, the project should first improve the critic target or repair prompt before RL.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train lightweight static/text no-gate critic baselines.")
    parser.add_argument("--input", type=Path, default=Path("data/evaluator/apps_simple_method1_evaluator_training_rows_v1.jsonl"))
    parser.add_argument("--no-gate-scores", type=Path, default=Path("data/rubrics/apps_simple_method1/apps_simple_no_gate_baseline_eval_v1_scores.jsonl"))
    parser.add_argument("--summary-output", type=Path, default=Path("data/evaluator/apps_simple_static_critic_baseline_v1_summary.json"))
    parser.add_argument("--predictions-output", type=Path, default=Path("data/evaluator/apps_simple_static_critic_baseline_v1_predictions.jsonl"))
    parser.add_argument("--all-predictions-output", type=Path, default=Path("data/evaluator/apps_simple_static_critic_baseline_v1_all_predictions.jsonl"))
    parser.add_argument("--markdown-output", type=Path, default=Path("data/evaluator/apps_simple_static_critic_baseline_v1_report.md"))
    parser.add_argument("--max-text-features", type=int, default=50000)
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    train_rows = split_rows(rows, "train")
    validation_rows = split_rows(rows, "validation")
    test_rows = split_rows(rows, "test")
    if not train_rows or not validation_rows or not test_rows:
        raise SystemExit("input must contain train, validation, and test splits")

    configs = [
        ("static_only_balanced", "none", "balanced"),
        ("code_tfidf_static_balanced", "code", "balanced"),
        ("task_code_tfidf_static_balanced", "task_code", "balanced"),
        ("task_code_tfidf_static_unweighted", "task_code", None),
    ]

    all_summaries = []
    predictions_by_model: dict[str, list[dict[str, Any]]] = {}
    all_predictions_by_model: dict[str, list[dict[str, Any]]] = {}
    for name, text_mode, class_weight in configs:
        model_summary, predictions, all_predictions = fit_and_evaluate_config(
            name=name,
            train_rows=train_rows,
            validation_rows=validation_rows,
            test_rows=test_rows,
            text_mode=text_mode,
            class_weight=class_weight,
            max_text_features=args.max_text_features,
        )
        all_summaries.append(model_summary)
        predictions_by_model[name] = predictions
        all_predictions_by_model[name] = all_predictions

    best_model = max(
        all_summaries,
        key=lambda item: (
            item["validation_selected"]["balanced_accuracy"],
            -item["validation_selected"]["overacceptance_rate"],
            item["test_selected"]["balanced_accuracy"],
        ),
    )
    best_predictions = predictions_by_model[best_model["name"]]
    best_all_predictions = all_predictions_by_model[best_model["name"]]
    summary = {
        "dataset": str(args.input),
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "test_rows": len(test_rows),
        "excluded_input_keys": sorted(PRIVATE_INPUT_KEYS),
        "selection_policy": "model and threshold selected by validation balanced accuracy; test is held out",
        "best_model": best_model,
        "all_models": all_summaries,
        "no_gate_baseline": no_gate_metrics(args.no_gate_scores),
    }

    write_json(args.summary_output, summary)
    write_jsonl(args.predictions_output, best_predictions)
    write_jsonl(args.all_predictions_output, best_all_predictions)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(render_markdown(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
