#!/usr/bin/env python3
"""Evaluate a generated rubric with an LLM judge on held-out responses.

The judge prompt is built from a sanitized view only: task text, public
interface, generated response, extracted code, and rubric text. Verifier labels,
tests, diagnostics, and private fields are never included in judge prompts.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, roc_auc_score


PRIVATE_FIELD_NAMES = {
    "test_list",
    "test_setup_code",
    "private_diagnostics",
    "safe_diagnostics",
    "failure_type",
    "passed",
    "first_expected",
    "first_actual",
}

CONFIDENCE_VALUES = {"low", "medium", "high"}
REQUIRED_ANCHORS = {"1", "2", "3", "4", "5"}
CALIBRATED_CRITICAL_IDS = {
    "interface_name_signature_mismatch",
    "runtime_api_type_misuse",
    "syntax_parseability_or_output_format",
    "algorithmic_wrong_value",
    "numeric_formula_correctness",
    "edge_case_boundary_handling",
    "string_regex_pattern_logic",
}


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def response_id(row: dict[str, Any]) -> str:
    if row.get("response_id"):
        return str(row["response_id"])
    return f"{row.get('id')}__sample{row.get('sample_id', 0)}"


def extract_task(prompt: str | None) -> str:
    if not prompt:
        return ""
    match = re.search(r"Task:\s*(.*?)\n\s*\nDefine code matching this public interface:", prompt, re.DOTALL)
    if match:
        return " ".join(match.group(1).split())
    marker = "Return only valid Python code, with no Markdown fences and no explanation."
    if marker in prompt:
        return " ".join(prompt.split(marker, 1)[-1].split())
    return " ".join(prompt.split())


def short_text(text: Any, limit: int) -> str:
    value = str(text or "").strip()
    if len(value) <= limit:
        return value
    return value[: limit - 20].rstrip() + "\n# ... [truncated]"


def schema_match_score(obj: dict[str, Any]) -> int:
    score = 0
    if isinstance(obj.get("dimension_scores"), dict):
        score += 10
    if "overall_score" in obj:
        score += 2
    if "predicted_pass" in obj:
        score += 2
    if "confidence" in obj:
        score += 1
    return score


def parse_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    candidates: list[dict[str, Any]] = []
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            candidates.append(obj)
    except json.JSONDecodeError:
        pass

    for fence in re.finditer(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL):
        try:
            obj = json.loads(fence.group(1))
            if isinstance(obj, dict):
                candidates.append(obj)
        except json.JSONDecodeError:
            pass

    start = text.find("{")
    while start >= 0:
        depth = 0
        in_string = False
        escape = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start : index + 1])
                        if isinstance(obj, dict):
                            candidates.append(obj)
                    except json.JSONDecodeError:
                        pass
                    break
        start = text.find("{", start + 1)
    if not candidates:
        return None
    return max(candidates, key=schema_match_score)


def rubric_dimension_ids(rubric: dict[str, Any]) -> list[str]:
    return [str(dimension["id"]) for dimension in rubric.get("dimensions", [])]


def validate_rubric(rubric: dict[str, Any]) -> list[str]:
    flags = []
    dimensions = rubric.get("dimensions")
    if not isinstance(dimensions, list) or not dimensions:
        return ["rubric has no dimensions"]
    seen = set()
    for dimension in dimensions:
        dimension_id = str(dimension.get("id") or "")
        if not dimension_id:
            flags.append("dimension missing id")
            continue
        if dimension_id in seen:
            flags.append(f"duplicate dimension id: {dimension_id}")
        seen.add(dimension_id)
        anchors = dimension.get("score_anchors")
        if not isinstance(anchors, dict) or set(map(str, anchors.keys())) != REQUIRED_ANCHORS:
            flags.append(f"{dimension_id}: incomplete score_anchors")
    return flags


def compact_rubric_for_prompt(rubric: dict[str, Any]) -> dict[str, Any]:
    dimensions = []
    for dimension in rubric.get("dimensions", []):
        dimensions.append(
            {
                "id": dimension.get("id"),
                "name": dimension.get("name"),
                "definition": dimension.get("definition"),
                "failure_mode": dimension.get("failure_mode"),
                "what_to_check": dimension.get("what_to_check") or [],
                "score_anchors": dimension.get("score_anchors"),
                "critical_failure": bool(dimension.get("critical_failure")),
            }
        )
    return {
        "name": rubric.get("name"),
        "task_type": rubric.get("task_type"),
        "dimensions": dimensions,
        "aggregation": rubric.get("aggregation") or {},
    }


def sanitized_view(row: dict[str, Any], max_response_chars: int, max_code_chars: int) -> dict[str, Any]:
    return {
        "task": extract_task(row.get("prompt")),
        "public_interface": row.get("interface_signatures") or row.get("interface_names") or [],
        "generated_response": short_text(row.get("generated_code"), max_response_chars),
        "extracted_code": short_text(row.get("extracted_code") or row.get("generated_code"), max_code_chars),
    }


def prompt_private_leakage(prompt: str) -> list[str]:
    flags = []
    for field_name in PRIVATE_FIELD_NAMES:
        pattern = rf"['\"]{re.escape(field_name)}['\"]\s*:"
        if re.search(pattern, prompt, flags=re.IGNORECASE):
            flags.append(f'"{field_name}"' if field_name == "passed" else field_name)
    return sorted(flags)


def build_judge_prompt(
    rubric_prompt: dict[str, Any],
    row: dict[str, Any],
    max_response_chars: int,
    max_code_chars: int,
) -> str:
    public_input = sanitized_view(row, max_response_chars, max_code_chars)
    rubric_json = json.dumps(rubric_prompt, ensure_ascii=False, indent=2)
    public_json = json.dumps(public_input, ensure_ascii=False, indent=2)
    dimension_ids = [dimension["id"] for dimension in rubric_prompt["dimensions"]]
    return f"""You are an LLM code judge. Score the submitted Python answer using only the public task, public interface, submitted response, extracted code, and rubric below.

Do not assume any extra reference tests or labels. Evaluate visible code properties only.
Return ONLY valid JSON. No Markdown fences.

Required JSON schema:
{{
  "dimension_scores": {{
    "<dimension_id>": {{"score": 1, "rationale": "short visible-code rationale"}}
  }},
  "overall_score": 1.0,
  "predicted_pass": false,
  "confidence": "low"
}}

Hard constraints:
- Include every dimension id exactly once: {dimension_ids}
- Scores must be integers from 1 to 5.
- If a critical dimension receives 1 or 2, predicted_pass should normally be false.
- Keep rationales concise and based only on the visible task/interface/code.

Rubric:
{rubric_json}

Public input:
{public_json}

Now produce the judgment JSON object only. Do not repeat the rubric or public input.
"""


def extract_code(text: str) -> str:
    fenced = re.search(r"```(?:python)?\s*(.*?)```", text or "", flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip("\n\r")
    return (text or "").strip("\n\r")


def interface_names_from_signatures(signatures: list[Any]) -> set[str]:
    names = set()
    for signature in signatures or []:
        match = re.search(r"\b(?:def|class)\s+([A-Za-z_][A-Za-z0-9_]*)\b", str(signature))
        if match:
            names.add(match.group(1))
    return names


def can_parse(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def deterministic_code_checks(row: dict[str, Any]) -> dict[str, Any]:
    text = row.get("generated_code") or ""
    code = row.get("extracted_code") or extract_code(text)
    parse_ok = can_parse(code)
    names = set(row.get("interface_names") or []) or interface_names_from_signatures(row.get("interface_signatures") or [])
    interface_ok = True
    for name in names:
        if not re.search(rf"\b(def|class)\s+{re.escape(str(name))}\b", code):
            interface_ok = False
            break
    missing_dep = any(token in code and import_text not in code for token, import_text in [("re.", "import re"), ("math.", "import math"), ("heapq.", "import heapq")])
    stub = bool(re.search(r"\b(pass|TODO|NotImplementedError)\b", code))
    return {
        "parse_ok": parse_ok,
        "interface_ok": interface_ok,
        "missing_dep": missing_dep,
        "stub": stub,
        "interface_names": sorted(names),
    }


def visible_code_fallback_scores(row: dict[str, Any], rubric: dict[str, Any]) -> dict[str, int]:
    checks = deterministic_code_checks(row)
    parse_ok = bool(checks["parse_ok"])
    interface_ok = bool(checks["interface_ok"])
    missing_dep = bool(checks["missing_dep"])
    stub = bool(checks["stub"])

    scores = {}
    for dimension in rubric.get("dimensions", []):
        dimension_id = str(dimension.get("id"))
        if dimension_id == "syntax_parseability_or_output_format":
            scores[dimension_id] = 5 if parse_ok else 1
        elif dimension_id == "interface_name_signature_mismatch":
            scores[dimension_id] = 5 if interface_ok else 1
        elif dimension_id == "runtime_api_type_misuse":
            scores[dimension_id] = 2 if missing_dep else (4 if parse_ok else 2)
        elif not parse_ok or not interface_ok:
            scores[dimension_id] = 2
        elif stub:
            scores[dimension_id] = 1
        else:
            scores[dimension_id] = 4
    return scores


def score_to_int(value: Any) -> int | None:
    try:
        score = int(round(float(value)))
    except (TypeError, ValueError):
        return None
    if score < 1 or score > 5:
        return None
    return score


def score_mean(scores: dict[str, int]) -> float:
    return float(np.mean(list(scores.values()))) if scores else 0.0


def predicted_pass_from_scores(scores: dict[str, int], rubric: dict[str, Any]) -> bool:
    critical_ids = set((rubric.get("aggregation") or {}).get("critical_dimension_ids") or [])
    if any(scores.get(dimension_id, 5) <= 2 for dimension_id in critical_ids):
        return False
    threshold = float((rubric.get("aggregation") or {}).get("pass_threshold") or 4.0)
    return score_mean(scores) >= threshold


def calibrated_predicted_pass_from_scores(
    scores: dict[str, int],
    rubric: dict[str, Any],
    threshold_override: float | None = None,
) -> bool:
    if any(scores.get(dimension_id, 5) <= 2 for dimension_id in CALIBRATED_CRITICAL_IDS):
        return False
    if scores.get("algorithmic_wrong_value", 5) <= 3:
        return False
    threshold = threshold_override
    if threshold is None:
        threshold = float((rubric.get("aggregation") or {}).get("pass_threshold") or 4.0)
    return score_mean(scores) >= threshold


def cap_score(
    dimension_scores: dict[str, dict[str, Any]],
    dimension_id: str,
    cap: int,
    reason: str,
    adjustments: list[dict[str, Any]],
) -> None:
    if dimension_id not in dimension_scores:
        return
    old_score = int(dimension_scores[dimension_id]["score"])
    new_score = min(old_score, cap)
    if new_score == old_score:
        return
    dimension_scores[dimension_id]["score"] = new_score
    dimension_scores[dimension_id]["rationale"] = short_text(
        f"{dimension_scores[dimension_id].get('rationale', '')} Deterministic clamp: {reason}",
        240,
    )
    adjustments.append(
        {
            "dimension_id": dimension_id,
            "old_score": old_score,
            "new_score": new_score,
            "reason": reason,
        }
    )


def apply_deterministic_clamps(
    dimension_scores: dict[str, dict[str, Any]],
    row: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    checks = deterministic_code_checks(row)
    adjustments: list[dict[str, Any]] = []

    if not checks["parse_ok"]:
        cap_score(
            dimension_scores,
            "syntax_parseability_or_output_format",
            1,
            "extracted code does not parse with ast.parse",
            adjustments,
        )
        for dimension_id in [
            "numeric_formula_correctness",
            "output_type_container_shape",
            "algorithmic_wrong_value",
            "edge_case_boundary_handling",
            "runtime_api_type_misuse",
            "string_regex_pattern_logic",
        ]:
            cap_score(
                dimension_scores,
                dimension_id,
                2,
                "semantic/runtime behavior cannot be trusted because extracted code is not parseable",
                adjustments,
            )

    if not checks["interface_ok"]:
        cap_score(
            dimension_scores,
            "interface_name_signature_mismatch",
            1,
            "required public interface name is missing from extracted code",
            adjustments,
        )
        cap_score(
            dimension_scores,
            "algorithmic_wrong_value",
            2,
            "solution cannot be called through the required public interface",
            adjustments,
        )

    if checks["missing_dep"]:
        cap_score(
            dimension_scores,
            "runtime_api_type_misuse",
            2,
            "code references a module namespace without an obvious import",
            adjustments,
        )

    if checks["stub"]:
        for dimension_id in [
            "numeric_formula_correctness",
            "algorithmic_wrong_value",
            "edge_case_boundary_handling",
            "string_regex_pattern_logic",
        ]:
            cap_score(
                dimension_scores,
                dimension_id,
                1,
                "code contains an explicit stub marker",
                adjustments,
            )

    return dimension_scores, adjustments


def repair_judgment(
    parsed: dict[str, Any] | None,
    raw_text: str,
    row: dict[str, Any],
    rubric: dict[str, Any],
    calibrated_prediction: bool = False,
    calibrated_pass_threshold: float | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    dimension_ids = rubric_dimension_ids(rubric)
    repair = {
        "json_parse_failed": parsed is None,
        "missing_dimensions": [],
        "invalid_scores": [],
        "used_visible_code_fallback": False,
        "deterministic_adjustments": [],
    }
    fallback_scores = visible_code_fallback_scores(row, rubric)
    dimension_scores: dict[str, dict[str, Any]] = {}

    source_scores = parsed.get("dimension_scores") if isinstance(parsed, dict) else None
    if not isinstance(source_scores, dict):
        source_scores = {}
        repair["used_visible_code_fallback"] = True

    for dimension_id in dimension_ids:
        item = source_scores.get(dimension_id)
        score = None
        rationale = ""
        if isinstance(item, dict):
            score = score_to_int(item.get("score"))
            rationale = short_text(item.get("rationale"), 240)
        elif item is not None:
            score = score_to_int(item)
        if score is None:
            score = fallback_scores.get(dimension_id, 3)
            repair["invalid_scores"].append(dimension_id)
        if not rationale:
            rationale = "Repaired from visible-code fallback because the judge output omitted a valid rationale."
        if dimension_id not in source_scores:
            repair["missing_dimensions"].append(dimension_id)
        dimension_scores[dimension_id] = {"score": int(score), "rationale": rationale}

    if calibrated_prediction:
        dimension_scores, adjustments = apply_deterministic_clamps(dimension_scores, row)
        repair["deterministic_adjustments"] = adjustments

    scores_flat = {key: value["score"] for key, value in dimension_scores.items()}
    overall = None
    predicted = None
    confidence = None
    if isinstance(parsed, dict):
        try:
            overall = float(parsed.get("overall_score"))
        except (TypeError, ValueError):
            overall = None
        predicted = parsed.get("predicted_pass") if isinstance(parsed.get("predicted_pass"), bool) else None
        confidence_value = str(parsed.get("confidence") or "").lower()
        confidence = confidence_value if confidence_value in CONFIDENCE_VALUES else None

    llm_predicted = predicted
    if calibrated_prediction:
        overall = score_mean(scores_flat)
        predicted = calibrated_predicted_pass_from_scores(scores_flat, rubric, calibrated_pass_threshold)
    elif overall is None or overall < 1 or overall > 5:
        overall = score_mean(scores_flat)
    if predicted is None:
        predicted = predicted_pass_from_scores(scores_flat, rubric)
    if confidence is None:
        confidence = "low" if repair["json_parse_failed"] or repair["invalid_scores"] else "medium"

    repaired = {
        "dimension_scores": dimension_scores,
        "overall_score": round(float(overall), 4),
        "predicted_pass": bool(predicted),
        "llm_predicted_pass": llm_predicted,
        "calibrated_prediction": bool(calibrated_prediction),
        "confidence": confidence,
    }
    return repaired, repair


def safe_auc(labels: list[int], scores: list[float]) -> float | None:
    if len(set(labels)) < 2:
        return None
    return float(roc_auc_score(labels, scores))


def compute_metrics(records: list[dict[str, Any]], rubric: dict[str, Any], audit: dict[str, Any]) -> dict[str, Any]:
    labels = [1 if record["passed"] else 0 for record in records]
    totals = [float(record["overall_score"]) for record in records]
    preds = [1 if record["predicted_pass"] else 0 for record in records]
    dimension_ids = rubric_dimension_ids(rubric)
    per_dimension = {}
    for dimension_id in dimension_ids:
        values = [int(record["dimension_scores"][dimension_id]["score"]) for record in records]
        per_dimension[dimension_id] = {
            "distribution": {str(score): values.count(score) for score in range(1, 6)},
            "mean": float(np.mean(values)) if values else None,
            "mean_passed": float(np.mean([v for v, y in zip(values, labels) if y == 1])) if any(y == 1 for y in labels) else None,
            "mean_failed": float(np.mean([v for v, y in zip(values, labels) if y == 0])) if any(y == 0 for y in labels) else None,
        }
    return {
        "rubric": rubric.get("name"),
        "num_dimensions": len(dimension_ids),
        "num_samples": len(records),
        "splits": sorted(set(record["split"] for record in records)),
        "llm_judge_auc": safe_auc(labels, totals),
        "llm_judge_kappa": float(cohen_kappa_score(labels, preds)) if len(set(preds)) > 1 or len(set(labels)) > 1 else 0.0,
        "llm_judge_accuracy": float(accuracy_score(labels, preds)) if records else 0.0,
        "llm_judge_mean_score_passed": float(np.mean([s for s, y in zip(totals, labels) if y == 1])) if any(y == 1 for y in labels) else None,
        "llm_judge_mean_score_failed": float(np.mean([s for s, y in zip(totals, labels) if y == 0])) if any(y == 0 for y in labels) else None,
        "predicted_pass_rate": float(np.mean(preds)) if preds else 0.0,
        "true_pass_rate": float(np.mean(labels)) if labels else 0.0,
        "per_dimension": per_dimension,
        "audit_summary": {
            "prompt_leakage_count": audit["prompt_leakage_count"],
            "json_parse_failed_count": audit["json_parse_failed_count"],
            "repaired_record_count": audit["repaired_record_count"],
            "used_visible_code_fallback_count": audit["used_visible_code_fallback_count"],
            "deterministic_adjusted_record_count": audit.get("deterministic_adjusted_record_count", 0),
        },
    }


def select_rows(path: Path, splits: set[str], limit: int | None, offset: int) -> list[dict[str, Any]]:
    rows = []
    skipped = 0
    for row in read_jsonl(path):
        if str(row.get("split")) not in splits:
            continue
        if skipped < offset:
            skipped += 1
            continue
        rows.append(row)
        if limit is not None and len(rows) >= limit:
            break
    return rows


def run_llm_judge(args: argparse.Namespace, prompts: list[str]) -> list[str]:
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        tensor_parallel_size=1,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
    )
    sampling = SamplingParams(
        n=1,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=42,
    )
    raw_outputs = []
    for start in range(0, len(prompts), args.batch_size):
        batch = prompts[start : start + args.batch_size]
        outputs = llm.generate(batch, sampling)
        raw_outputs.extend(output.outputs[0].text for output in outputs)
    return raw_outputs


def read_raw_outputs(path: Path, rows: list[dict[str, Any]]) -> list[str]:
    raw_by_id = {}
    for record in read_jsonl(path):
        rid = str(record.get("response_id") or "")
        if rid:
            raw_by_id[rid] = str(record.get("raw_judge_output") or "")
    missing = [response_id(row) for row in rows if response_id(row) not in raw_by_id]
    if missing:
        raise SystemExit(f"--reuse-raw-output is missing {len(missing)} requested rows; first missing: {missing[0]}")
    return [raw_by_id[response_id(row)] for row in rows]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an LLM rubric judge on sanitized held-out responses.")
    parser.add_argument("--labeled", type=Path, required=True)
    parser.add_argument("--rubric", type=Path, required=True)
    parser.add_argument("--scores-output", type=Path, default=Path("data/rubrics/phase2/mbpp_hidden_llm_judge_scores_validation_test.jsonl"))
    parser.add_argument("--metrics-output", type=Path, default=Path("data/rubrics/phase2/mbpp_hidden_llm_judge_metrics_validation_test.json"))
    parser.add_argument("--audit-output", type=Path, default=Path("data/rubrics/phase2/mbpp_hidden_llm_judge_audit_validation_test.json"))
    parser.add_argument("--raw-output", type=Path, default=Path("data/rubrics/phase2/mbpp_hidden_llm_judge_raw_validation_test.jsonl"))
    parser.add_argument("--splits", default="validation,test")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-response-chars", type=int, default=1800)
    parser.add_argument("--max-code-chars", type=int, default=2600)
    parser.add_argument("--deterministic-only", action="store_true", help="Use visible-code fallback instead of LLM calls; for smoke tests only.")
    parser.add_argument("--reuse-raw-output", type=Path, help="Recompute scores/metrics from an existing raw judge JSONL without rerunning the LLM.")
    parser.add_argument("--calibrated-prediction", action="store_true", help="Clamp deterministic dimensions and compute predicted_pass from calibrated rubric gates instead of trusting the LLM boolean.")
    parser.add_argument("--calibrated-pass-threshold", type=float, help="Override the rubric pass threshold when --calibrated-prediction is set.")
    parser.add_argument("--model")
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.25)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-num-seqs", type=int, default=16)
    args = parser.parse_args()

    if not args.model and not args.deterministic_only and not args.reuse_raw_output:
        raise SystemExit("--model is required unless --deterministic-only or --reuse-raw-output is set")

    rubric = json.loads(args.rubric.read_text(encoding="utf-8"))
    rubric_flags = validate_rubric(rubric)
    splits = {split.strip() for split in args.splits.split(",") if split.strip()}
    rows = select_rows(args.labeled, splits, args.limit, args.offset)
    rubric_prompt = compact_rubric_for_prompt(rubric)
    prompts = [build_judge_prompt(rubric_prompt, row, args.max_response_chars, args.max_code_chars) for row in rows]
    leakage_by_index = {index: prompt_private_leakage(prompt) for index, prompt in enumerate(prompts)}
    leakage_by_index = {index: flags for index, flags in leakage_by_index.items() if flags}

    if args.reuse_raw_output:
        raw_outputs = read_raw_outputs(args.reuse_raw_output, rows)
    elif args.deterministic_only:
        raw_outputs = ["{}" for _ in rows]
    else:
        raw_outputs = run_llm_judge(args, prompts)

    records = []
    raw_records = []
    audit = {
        "method": "llm_rubric_judge_with_sanitized_prompts",
        "model": args.model,
        "deterministic_only": args.deterministic_only,
        "rubric_flags": rubric_flags,
        "num_requested_rows": len(rows),
        "prompt_leakage_count": len(leakage_by_index),
        "prompt_leakage_examples": [
            {"response_id": response_id(rows[index]), "flags": flags}
            for index, flags in list(leakage_by_index.items())[:10]
        ],
        "json_parse_failed_count": 0,
        "repaired_record_count": 0,
        "used_visible_code_fallback_count": 0,
        "deterministic_adjusted_record_count": 0,
        "calibrated_prediction": bool(args.calibrated_prediction),
        "calibrated_pass_threshold": args.calibrated_pass_threshold,
        "repair_reasons": Counter(),
    }

    for row, raw_text in zip(rows, raw_outputs):
        parsed = parse_json_object(raw_text)
        judgment, repair = repair_judgment(
            parsed,
            raw_text,
            row,
            rubric,
            calibrated_prediction=args.calibrated_prediction,
            calibrated_pass_threshold=args.calibrated_pass_threshold,
        )
        if repair["json_parse_failed"]:
            audit["json_parse_failed_count"] += 1
        if repair["used_visible_code_fallback"]:
            audit["used_visible_code_fallback_count"] += 1
        if repair["deterministic_adjustments"]:
            audit["deterministic_adjusted_record_count"] += 1
        if repair["json_parse_failed"] or repair["missing_dimensions"] or repair["invalid_scores"]:
            audit["repaired_record_count"] += 1
        for key, value in repair.items():
            if isinstance(value, list) and value:
                audit["repair_reasons"][key] += 1
            elif value is True:
                audit["repair_reasons"][key] += 1

        rid = response_id(row)
        record = {
            "response_id": rid,
            "id": row.get("id"),
            "sample_id": row.get("sample_id", 0),
            "dataset": row.get("dataset"),
            "split": row.get("split"),
            "passed": bool(row.get("passed")),
            "dimension_scores": judgment["dimension_scores"],
            "overall_score": judgment["overall_score"],
            "predicted_pass": judgment["predicted_pass"],
            "llm_predicted_pass": judgment["llm_predicted_pass"],
            "calibrated_prediction": judgment["calibrated_prediction"],
            "confidence": judgment["confidence"],
            "repair": repair,
        }
        records.append(record)
        raw_records.append({"response_id": rid, "raw_judge_output": raw_text})

    audit["repair_reasons"] = dict(audit["repair_reasons"])
    audit["valid"] = not rubric_flags and not leakage_by_index and len(records) == len(rows)
    metrics = compute_metrics(records, rubric, audit)

    write_jsonl(args.scores_output, records)
    write_jsonl(args.raw_output, raw_records)
    write_json(args.audit_output, audit)
    write_json(args.metrics_output, metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
