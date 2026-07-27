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
    "syntax_parseability_truncation",
}
STRICT_SEMANTIC_IDS = {
    "numeric_formula_arithmetic_error",
    "sequence_collection_transformation_error",
    "predicate_branch_condition_error",
    "edge_case_handling",
    "output_type_or_container_shape",
    "string_regex_pattern_logic",
}
MAJOR_ERROR_SEVERITIES = {"major", "fatal"}
NON_ACTIONABLE_AMBIGUITY_PHRASES = {
    "",
    "empty",
    "none",
    "n a",
    "na",
    "not applicable",
    "no material ambiguities",
    "no unresolved ambiguity",
    "no unresolved ambiguities",
    "unresolved public contract convention or empty",
}
AMBIGUITY_UNCERTAINTY_TERMS = {
    "ambiguous",
    "ambiguity",
    "not specified",
    "unspecified",
    "not stated",
    "unclear",
    "unresolved",
    "omitted",
    "undefined",
    "unknown",
    "not defined",
}
AMBIGUITY_FIELD_TERMS = {
    "input",
    "domain",
    "nesting",
    "return",
    "output",
    "type",
    "container",
    "tuple",
    "list",
    "bool",
    "boolean",
    "message",
    "string",
    "index",
    "zero based",
    "one based",
    "sentinel",
    "none",
    "coordinate",
    "unit",
    "angle",
    "degree",
    "radian",
    "precision",
    "tolerance",
    "rounding",
    "float",
}
AMBIGUITY_ALTERNATIVE_MARKERS = {" or ", " versus ", " vs ", " either ", "/"}
PROBE_TRACE_MARKERS = {
    "return",
    "evaluat",
    "branch",
    "condition",
    "loop",
    "if ",
    "for ",
    "while ",
    "raise",
    "exception",
    "type",
    "index",
    "slice",
    "assign",
    "append",
    "call",
    "comput",
    "yield",
    "update",
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


def normalize_evidence_text(text: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()


def is_actionable_material_ambiguity(text: Any) -> bool:
    raw = str(text or "").strip().lower()
    normalized = normalize_evidence_text(raw)
    if normalized in NON_ACTIONABLE_AMBIGUITY_PHRASES:
        return False
    has_field = any(term in normalized for term in AMBIGUITY_FIELD_TERMS)
    if not has_field:
        return False
    has_uncertainty = any(term in normalized for term in AMBIGUITY_UNCERTAINTY_TERMS)
    has_alternative = any(marker in raw for marker in AMBIGUITY_ALTERNATIVE_MARKERS)
    return has_uncertainty or has_alternative


def split_material_ambiguities(values: list[str]) -> tuple[list[str], list[str]]:
    actionable = []
    ignored = []
    for value in values:
        if is_actionable_material_ambiguity(value):
            actionable.append(value)
        else:
            ignored.append(value)
    return actionable, ignored


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
        compact_dimension = {
            "id": dimension.get("id"),
            "name": dimension.get("name"),
            "definition": dimension.get("definition"),
            "failure_mode": dimension.get("failure_mode"),
            "what_to_check": dimension.get("what_to_check") or [],
            "score_anchors": dimension.get("score_anchors"),
            "critical_failure": bool(dimension.get("critical_failure")),
        }
        if dimension.get("applicability_instruction"):
            compact_dimension["applicability_instruction"] = dimension.get("applicability_instruction")
        if dimension.get("high_score_evidence_requirement"):
            compact_dimension["high_score_evidence_requirement"] = dimension.get("high_score_evidence_requirement")
        for key in ["aggregation_role", "always_applicable", "known_error_cap", "score_four_five_requirement"]:
            if key in dimension:
                compact_dimension[key] = dimension.get(key)
        dimensions.append(compact_dimension)
    compact = {
        "name": rubric.get("name"),
        "task_type": rubric.get("task_type"),
        "dimensions": dimensions,
        "aggregation": rubric.get("aggregation") or {},
    }
    if rubric.get("revision_policy"):
        compact["revision_policy"] = rubric.get("revision_policy")
    return compact


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


def compact_guidance_for_prompt(guidance: dict[str, Any] | None) -> dict[str, Any]:
    guidance = guidance or {}
    return {
        "bad_rubric_patterns": guidance.get("bad_rubric_patterns") or [],
        "required_judging_sequence": guidance.get("required_judging_sequence") or [],
        "score_policy": guidance.get("score_policy") or {},
        "critical_policy": guidance.get("critical_policy") or {},
        "probe_policy": guidance.get("probe_policy") or {},
        "aggregation_policy": guidance.get("aggregation_policy") or {},
        "public_contract_ambiguity_policy": guidance.get("public_contract_ambiguity_policy") or {},
        "verifier_disagreement_policy": guidance.get("verifier_disagreement_policy") or {},
        "good_judgment_requirements": guidance.get("good_judgment_requirements") or [],
    }


def compact_examples_for_prompt(examples: dict[str, Any] | None, limit: int) -> list[dict[str, Any]]:
    compact = []
    for example in (examples or {}).get("examples", [])[:limit]:
        compact.append(
            {
                "task": example.get("task"),
                "public_interface": example.get("public_interface") or [],
                "code": short_text(example.get("code"), 1000),
                "expected_verdict": example.get("expected_verdict"),
                "critical_errors": example.get("critical_errors") or [],
                "corrected_scores": example.get("corrected_scores") or {},
                "lesson": example.get("lesson"),
            }
        )
    return compact


def strict_public_view(row: dict[str, Any], max_response_chars: int, max_code_chars: int) -> dict[str, Any]:
    public = sanitized_view(row, max_response_chars, max_code_chars)
    generated = extract_code(str(public.get("generated_response") or ""))
    extracted = str(public.get("extracted_code") or "").strip()
    if generated.strip() == extracted:
        public.pop("generated_response", None)
    return public


def build_judge_prompt(
    rubric_prompt: dict[str, Any],
    row: dict[str, Any],
    max_response_chars: int,
    max_code_chars: int,
    prompt_profile: str = "baseline",
    guidance: dict[str, Any] | None = None,
    examples: dict[str, Any] | None = None,
    max_few_shot_examples: int = 7,
) -> str:
    if prompt_profile == "baseline":
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

    public_input = strict_public_view(row, max_response_chars, max_code_chars)
    rubric_json = json.dumps(rubric_prompt, ensure_ascii=False, separators=(",", ":"))
    guidance_json = json.dumps(compact_guidance_for_prompt(guidance), ensure_ascii=False, separators=(",", ":"))
    example_json = json.dumps(
        compact_examples_for_prompt(examples, max_few_shot_examples),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    public_json = json.dumps(public_input, ensure_ascii=False, indent=2)
    dimension_ids = [dimension["id"] for dimension in rubric_prompt["dimensions"]]
    example_block = ""
    if prompt_profile in {"strict_fewshot", "contrastive_fewshot"}:
        example_block = f"""
Calibration examples from the development split only. Imitate their evidence standard and decision logic, not their surface wording:
{example_json}
"""
    contrastive = prompt_profile == "contrastive_fewshot"
    probe_instruction = ""
    probe_schema = ""
    if contrastive:
        probe_instruction = """
Score-collapse prevention rules:
- Before assigning any score, emit exactly three concrete test_probes: ordinary, boundary, and adversarial.
- Each probe must state a public-spec-derived input, expected behavior, and behavior traced from the submitted code.
- If any probe is inconsistent, predicted_pass must be false and every affected semantic score must be at most 2.
- A real edge-case failure is score 2, never score 4. Score 3 is reserved for unresolved uncertainty without a known counterexample.
- Scores 4 and 5 are forbidden unless all three probes are consistent.
- The primary overall score is the minimum applicable semantic score. Structural scores cannot raise it.
- A consistent probe must show code-specific trace evidence in observed_behavior; do not merely copy expected_behavior.
"""
        probe_schema = """
  "test_probes": [
    {"kind": "ordinary|boundary|adversarial", "input": "concrete input", "expected_behavior": "...", "observed_behavior": "behavior traced from code", "consistent": true, "affected_dimensions": ["dimension_id"]}
  ],
"""
    return f"""You are a skeptical but fair Python code judge. Your main failure to avoid is over-acceptance: never mark an answer correct merely because it parses, has the right function name, or looks plausible. Also do not invent unsupported flaws in concise correct code.

Use only the public task, public interface, and submitted code. Never assume hidden tests, reference answers, verifier labels, or private diagnostics.

Required decision order:
1. Restate the input contract, output contract, core rule, and boundary cases.
2. Mark every rubric dimension applicable or not applicable.
3. Trace concrete expressions, branches, loops, return values, and operand types against the contract.
4. Actively attempt a normal-case or boundary-case counterexample for each applicable semantic dimension.
5. List major errors before scoring. A major applicable error vetoes a pass regardless of the mean.
{probe_instruction}

Evidence rules:
- Score 5 requires task-specific code evidence plus a counterexample attempt that the code survives.
- Score 4 requires all required probes to be consistent and permits no known correctness failure; only a non-correctness concern or unresolved minor risk may remain.
- Score 3 means materially unresolved or partially justified and is not a passing semantic score.
- Score 2 is mandatory for any concrete wrong value, wrong shape/type, exception, nontermination, or violated contract, including edge cases.
- Score 1 is reserved for invalid, unusable, unrelated, or ordinary-case core failure.
- For an irrelevant dimension use applicable=false and score=3; it contributes no acceptance evidence.
- A rationale that only paraphrases a rubric anchor is invalid.
- Use an empty list for material_ambiguities when no specific unresolved public convention changes correctness. Generic placeholders such as "empty" or "unresolved public-contract convention" are invalid.

Return ONLY one valid JSON object with this exact top-level schema and no Markdown:
{{
  "specification": {{"input_contract": "...", "output_contract": "...", "core_rule": "...", "boundary_cases": ["..."], "material_ambiguities": []}},
{probe_schema}  "critical_errors": [{{"dimension_id": "...", "severity": "minor|major|fatal", "evidence": "code-specific evidence", "counterexample": "public-spec-derived example or empty string"}}],
  "dimension_scores": {{
    "<dimension_id>": {{"applicable": true, "score": 1, "rationale": "code-specific rationale", "counterexample": "attempted example or empty string"}}
  }},
  "overall_score": 1.0,
  "predicted_pass": false,
  "confidence": "low|medium|high"
}}

Include every dimension exactly once: {dimension_ids}

Human-guidance failure list and scoring policy:
{guidance_json}

Rubric:
{rubric_json}
{example_block}
Public input to judge:
{public_json}

Produce the judgment JSON now. Do not repeat the input.
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
        if dimension_id == "syntax_parseability_truncation":
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


def applicable_score_mean(dimension_scores: dict[str, dict[str, Any]]) -> float:
    values = [
        int(item["score"])
        for item in dimension_scores.values()
        if bool(item.get("applicable", True))
    ]
    if not values:
        values = [int(item["score"]) for item in dimension_scores.values()]
    return float(np.mean(values)) if values else 0.0


def semantic_dimension_ids(rubric: dict[str, Any]) -> set[str]:
    configured = (rubric.get("aggregation") or {}).get("semantic_dimension_ids") or []
    return set(map(str, configured)) or set(STRICT_SEMANTIC_IDS)


def semantic_bottleneck_score(
    dimension_scores: dict[str, dict[str, Any]],
    rubric: dict[str, Any],
) -> float:
    semantic_ids = semantic_dimension_ids(rubric)
    values = [
        int(item["score"])
        for dimension_id, item in dimension_scores.items()
        if dimension_id in semantic_ids and bool(item.get("applicable", True))
    ]
    return float(min(values)) if values else 3.0


def normalize_critical_errors(parsed: dict[str, Any] | None, dimension_ids: set[str]) -> list[dict[str, str]]:
    errors = []
    source = parsed.get("critical_errors") if isinstance(parsed, dict) else None
    if not isinstance(source, list):
        return errors
    for item in source:
        if not isinstance(item, dict):
            continue
        dimension_id = str(item.get("dimension_id") or "")
        severity = str(item.get("severity") or "minor").lower()
        if dimension_id not in dimension_ids or severity not in {"minor", "major", "fatal"}:
            continue
        errors.append(
            {
                "dimension_id": dimension_id,
                "severity": severity,
                "evidence": short_text(item.get("evidence"), 320),
                "counterexample": short_text(item.get("counterexample"), 240),
            }
        )
    return errors


def normalize_specification(parsed: dict[str, Any] | None) -> dict[str, Any]:
    source = parsed.get("specification") if isinstance(parsed, dict) else None
    if not isinstance(source, dict):
        return {}
    boundaries = source.get("boundary_cases")
    if not isinstance(boundaries, list):
        boundaries = []
    ambiguities = source.get("material_ambiguities")
    if not isinstance(ambiguities, list):
        ambiguities = []
    return {
        "input_contract": short_text(source.get("input_contract"), 320),
        "output_contract": short_text(source.get("output_contract"), 320),
        "core_rule": short_text(source.get("core_rule"), 400),
        "boundary_cases": [short_text(value, 180) for value in boundaries[:6]],
        "material_ambiguities": [short_text(value, 240) for value in ambiguities[:6] if str(value).strip()],
    }


def normalize_test_probes(parsed: dict[str, Any] | None, dimension_ids: set[str]) -> list[dict[str, Any]]:
    source = parsed.get("test_probes") if isinstance(parsed, dict) else None
    if not isinstance(source, list):
        return []
    probes = []
    for item in source[:6]:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("kind") or "").lower()
        if kind not in {"ordinary", "boundary", "adversarial"}:
            continue
        consistent = item.get("consistent")
        if not isinstance(consistent, bool):
            continue
        affected = item.get("affected_dimensions") or []
        if not isinstance(affected, list):
            affected = []
        probes.append(
            {
                "kind": kind,
                "input": short_text(item.get("input"), 240),
                "expected_behavior": short_text(item.get("expected_behavior"), 280),
                "observed_behavior": short_text(item.get("observed_behavior"), 280),
                "consistent": consistent,
                "affected_dimensions": [str(value) for value in affected if str(value) in dimension_ids],
            }
        )
    return probes


def weak_probe_trace_reason(probe: dict[str, Any]) -> str | None:
    expected = str(probe.get("expected_behavior") or "").strip()
    observed = str(probe.get("observed_behavior") or "").strip()
    if not observed:
        return "observed_behavior is empty"
    if normalize_evidence_text(expected) == normalize_evidence_text(observed):
        return "observed_behavior repeats expected_behavior without code trace"
    observed_lower = observed.lower()
    if len(observed.split()) < 4:
        return "observed_behavior is too short to show a code trace"
    if not any(marker in observed_lower for marker in PROBE_TRACE_MARKERS):
        return "observed_behavior does not mention a code expression, branch, return, type, or computed value"
    return None


def validate_required_test_probes(
    probes: list[dict[str, Any]],
    strict_trace_evidence: bool = False,
) -> list[str]:
    reasons = []
    required_kinds = {"ordinary", "boundary", "adversarial"}
    kind_counts = Counter(str(probe.get("kind") or "") for probe in probes)
    if len(probes) != 3:
        reasons.append(f"expected exactly 3 probes, received {len(probes)}")
    for kind in sorted(required_kinds):
        if kind_counts[kind] != 1:
            reasons.append(f"expected exactly one {kind} probe, received {kind_counts[kind]}")
    for index, probe in enumerate(probes):
        for field in ["input", "expected_behavior", "observed_behavior"]:
            if not str(probe.get(field) or "").strip():
                reasons.append(f"probe {index} has empty {field}")
        if strict_trace_evidence:
            reason = weak_probe_trace_reason(probe)
            if reason:
                reasons.append(f"probe {index} has weak trace evidence: {reason}")
    return reasons


def apply_probe_inconsistency_clamps(
    dimension_scores: dict[str, dict[str, Any]],
    rubric: dict[str, Any],
    probes: list[dict[str, Any]],
    critical_errors: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    adjustments: list[dict[str, Any]] = []
    semantic_ids = semantic_dimension_ids(rubric)
    for probe in probes:
        if probe.get("consistent") is not False:
            continue
        affected = [value for value in probe.get("affected_dimensions") or [] if value in semantic_ids]
        if not affected:
            fallback = next(
                (
                    dimension_id
                    for dimension_id in [
                        "numeric_formula_arithmetic_error",
                        "sequence_collection_transformation_error",
                        "predicate_branch_condition_error",
                        "edge_case_handling",
                        "output_type_or_container_shape",
                        "string_regex_pattern_logic",
                    ]
                    if dimension_id in semantic_ids
                ),
                next(iter(semantic_ids), "numeric_formula_arithmetic_error"),
            )
            affected = [fallback]
        for dimension_id in affected:
            cap_score(
                dimension_scores,
                dimension_id,
                2,
                f"inconsistent {probe.get('kind')} probe: {probe.get('observed_behavior')}",
                adjustments,
            )
        evidence = short_text(
            f"{probe.get('kind')} probe {probe.get('input')}: expected {probe.get('expected_behavior')}; "
            f"code implies {probe.get('observed_behavior')}",
            320,
        )
        critical_errors.append(
            {
                "dimension_id": affected[0],
                "severity": "major",
                "evidence": evidence,
                "counterexample": short_text(probe.get("input"), 240),
            }
        )
    return adjustments, critical_errors


def strict_predicted_pass_from_judgment(
    dimension_scores: dict[str, dict[str, Any]],
    rubric: dict[str, Any],
    critical_errors: list[dict[str, str]],
    threshold_override: float | None = None,
) -> bool:
    if any(error.get("severity") in MAJOR_ERROR_SEVERITIES for error in critical_errors):
        return False
    critical_ids = set((rubric.get("aggregation") or {}).get("critical_dimension_ids") or [])
    semantic_ids = semantic_dimension_ids(rubric)
    for dimension_id, item in dimension_scores.items():
        if not bool(item.get("applicable", True)):
            continue
        score = int(item["score"])
        if dimension_id in critical_ids and score <= 2:
            return False
        if dimension_id in semantic_ids and score <= 3:
            return False
    threshold = threshold_override
    if threshold is None:
        threshold = float((rubric.get("aggregation") or {}).get("pass_threshold") or 4.0)
    method = str((rubric.get("aggregation") or {}).get("method") or "")
    primary_score = semantic_bottleneck_score(dimension_scores, rubric) if "semantic_bottleneck" in method else applicable_score_mean(dimension_scores)
    return primary_score >= threshold


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
    if any(scores.get(dimension_id, 5) <= 3 for dimension_id in semantic_dimension_ids(rubric)):
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
            "syntax_parseability_truncation",
            1,
            "extracted code does not parse with ast.parse",
            adjustments,
        )
        for dimension_id in [
            "numeric_formula_arithmetic_error",
            "sequence_collection_transformation_error",
            "predicate_branch_condition_error",
            "edge_case_handling",
            "output_type_or_container_shape",
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
            "output_type_or_container_shape",
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
            "numeric_formula_arithmetic_error",
            "sequence_collection_transformation_error",
            "predicate_branch_condition_error",
            "edge_case_handling",
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


def execution_failure_dimension(row: dict[str, Any]) -> str:
    failure_type = str(row.get("failure_type") or "").lower()
    diagnostics = row.get("safe_diagnostics") if isinstance(row.get("safe_diagnostics"), dict) else {}
    diagnostic_kind = str(diagnostics.get("diagnostic_kind") or "").lower()
    first_kind = str(diagnostics.get("first_failure_kind") or "").lower()
    text = " ".join([failure_type, diagnostic_kind, first_kind, str(row.get("error") or "").lower()])
    if "syntax" in text:
        return "syntax_parseability_truncation"
    if "interface" in text or "nameerror" in text or "not defined" in text:
        return "interface_name_signature_mismatch"
    if "type" in text or "exception" in text or "runtime" in text or "timeout" in text:
        return "runtime_api_type_misuse"
    if "wrong_type" in text or "wrong_length" in text:
        return "output_type_or_container_shape"
    return "numeric_formula_arithmetic_error"


def execution_evidence_text(row: dict[str, Any]) -> str:
    diagnostics = row.get("safe_diagnostics") if isinstance(row.get("safe_diagnostics"), dict) else {}
    parts = []
    if row.get("failure_type"):
        parts.append(f"failure_type={row.get('failure_type')}")
    if row.get("error"):
        parts.append(f"error={short_text(row.get('error'), 120)}")
    for key in [
        "diagnostic_kind",
        "test_count",
        "passed_assertions",
        "failed_assertions",
        "first_failed_index",
        "first_failure_kind",
        "first_exception_type",
    ]:
        if key in diagnostics:
            parts.append(f"{key}={diagnostics.get(key)}")
    return "; ".join(parts) if parts else "verifier reported failure"


def apply_execution_failure_gate(
    dimension_scores: dict[str, dict[str, Any]],
    row: dict[str, Any],
    rubric: dict[str, Any],
    critical_errors: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    if bool(row.get("passed")):
        return [], critical_errors
    adjustments: list[dict[str, Any]] = []
    primary_dimension = execution_failure_dimension(row)
    evidence = execution_evidence_text(row)
    cap_score(
        dimension_scores,
        primary_dimension,
        2,
        f"execution gate observed verifier failure: {evidence}",
        adjustments,
    )
    for dimension_id in semantic_dimension_ids(rubric):
        if dimension_id == primary_dimension:
            continue
        if bool(dimension_scores.get(dimension_id, {}).get("applicable", True)):
            cap_score(
                dimension_scores,
                dimension_id,
                3,
                "execution gate prevents semantic pass without verified behavior",
                adjustments,
            )
    critical_errors.append(
        {
            "dimension_id": primary_dimension,
            "severity": "major",
            "evidence": short_text(evidence, 320),
            "counterexample": "",
        }
    )
    return adjustments, critical_errors


def repair_judgment(
    parsed: dict[str, Any] | None,
    raw_text: str,
    row: dict[str, Any],
    rubric: dict[str, Any],
    calibrated_prediction: bool = False,
    calibrated_pass_threshold: float | None = None,
    strict_prediction: bool = False,
    require_test_probes: bool = False,
    execution_gate: str = "none",
) -> tuple[dict[str, Any], dict[str, Any]]:
    dimension_ids = rubric_dimension_ids(rubric)
    repair = {
        "json_parse_failed": parsed is None,
        "missing_dimensions": [],
        "invalid_scores": [],
        "missing_rationales": [],
        "used_visible_code_fallback": False,
        "deterministic_adjustments": [],
        "probe_adjustments": [],
        "material_ambiguity_adjustments": [],
        "execution_gate_adjustments": [],
        "execution_gate_predicted_override": False,
        "ignored_material_ambiguities": [],
        "forced_applicable_dimensions": [],
        "invalid_test_probe_reasons": [],
        "missing_specification": bool(strict_prediction) and not isinstance(parsed.get("specification") if isinstance(parsed, dict) else None, dict),
        "missing_critical_errors_field": bool(strict_prediction) and not isinstance(parsed.get("critical_errors") if isinstance(parsed, dict) else None, list),
        "missing_required_test_probes": False,
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
        counterexample = ""
        applicable = True
        if isinstance(item, dict):
            score = score_to_int(item.get("score"))
            rationale = short_text(item.get("rationale") or item.get("evidence"), 240)
            counterexample = short_text(item.get("counterexample"), 240)
            if isinstance(item.get("applicable"), bool):
                applicable = bool(item.get("applicable"))
        elif item is not None:
            score = score_to_int(item)
        if score is None:
            score = fallback_scores.get(dimension_id, 3)
            repair["invalid_scores"].append(dimension_id)
        if not rationale:
            repair["missing_rationales"].append(dimension_id)
            rationale = "Repaired from visible-code fallback because the judge output omitted a valid rationale."
        if dimension_id not in source_scores:
            repair["missing_dimensions"].append(dimension_id)
        always_applicable_ids = set((rubric.get("aggregation") or {}).get("always_applicable_ids") or [])
        if strict_prediction and dimension_id in always_applicable_ids and not applicable:
            applicable = True
            repair["forced_applicable_dimensions"].append(dimension_id)
        if strict_prediction and not applicable:
            score = 3
        dimension_scores[dimension_id] = {
            "applicable": applicable,
            "score": int(score),
            "rationale": rationale,
            "counterexample": counterexample,
        }

    if calibrated_prediction or strict_prediction:
        dimension_scores, adjustments = apply_deterministic_clamps(dimension_scores, row)
        repair["deterministic_adjustments"] = adjustments

    critical_errors = normalize_critical_errors(parsed, set(dimension_ids))
    specification = normalize_specification(parsed)
    actionable_ambiguities, ignored_ambiguities = split_material_ambiguities(
        specification.get("material_ambiguities") or []
    )
    specification["material_ambiguities"] = actionable_ambiguities
    repair["ignored_material_ambiguities"] = ignored_ambiguities
    test_probes = normalize_test_probes(parsed, set(dimension_ids))
    strict_probe_evidence = bool((rubric.get("aggregation") or {}).get("strict_probe_evidence_action"))
    if require_test_probes:
        repair["invalid_test_probe_reasons"] = validate_required_test_probes(
            test_probes,
            strict_trace_evidence=strict_probe_evidence,
        )
        repair["missing_required_test_probes"] = bool(repair["invalid_test_probe_reasons"])
    if require_test_probes:
        if repair["missing_required_test_probes"]:
            for dimension_id in semantic_dimension_ids(rubric):
                if bool(dimension_scores.get(dimension_id, {}).get("applicable", True)):
                    cap_score(
                        dimension_scores,
                        dimension_id,
                        3,
                        "required ordinary/boundary/adversarial probe evidence is incomplete or invalid",
                        repair["probe_adjustments"],
                    )
        probe_adjustments, critical_errors = apply_probe_inconsistency_clamps(
            dimension_scores,
            rubric,
            test_probes,
            critical_errors,
        )
        repair["probe_adjustments"].extend(probe_adjustments)

    ambiguity_policy_enabled = bool(
        (rubric.get("aggregation") or {}).get("material_contract_ambiguity_action")
    )
    if strict_prediction and ambiguity_policy_enabled and specification.get("material_ambiguities"):
        for dimension_id in semantic_dimension_ids(rubric):
            if bool(dimension_scores.get(dimension_id, {}).get("applicable", True)):
                cap_score(
                    dimension_scores,
                    dimension_id,
                    3,
                    "material public-contract ambiguity requires abstention",
                    repair["material_ambiguity_adjustments"],
                )

    if execution_gate in {"failures", "oracle"}:
        execution_adjustments, critical_errors = apply_execution_failure_gate(
            dimension_scores,
            row,
            rubric,
            critical_errors,
        )
        repair["execution_gate_adjustments"] = execution_adjustments

    scores_flat = {key: value["score"] for key, value in dimension_scores.items()}
    quality_mean = applicable_score_mean(dimension_scores)
    semantic_bottleneck = semantic_bottleneck_score(dimension_scores, rubric)
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
    if strict_prediction:
        aggregation_method = str((rubric.get("aggregation") or {}).get("method") or "")
        overall = semantic_bottleneck if "semantic_bottleneck" in aggregation_method else quality_mean
        structural_ids = {
            dimension_id
            for dimension_id in dimension_ids
            if dimension_id not in semantic_dimension_ids(rubric)
        }
        if any(
            int(dimension_scores[dimension_id]["score"]) <= 2
            for dimension_id in structural_ids
            if bool(dimension_scores[dimension_id].get("applicable", True))
        ):
            overall = min(float(overall), 2.0)
        if repair["missing_required_test_probes"]:
            overall = min(float(overall), 3.0)
        predicted = strict_predicted_pass_from_judgment(
            dimension_scores,
            rubric,
            critical_errors,
            calibrated_pass_threshold,
        )
        if repair["missing_required_test_probes"]:
            predicted = False
        if repair["material_ambiguity_adjustments"]:
            predicted = False
        if llm_predicted is False:
            predicted = False
    elif calibrated_prediction:
        overall = score_mean(scores_flat)
        predicted = calibrated_predicted_pass_from_scores(scores_flat, rubric, calibrated_pass_threshold)
    elif overall is None or overall < 1 or overall > 5:
        overall = score_mean(scores_flat)
    if predicted is None:
        predicted = predicted_pass_from_scores(scores_flat, rubric)
    if execution_gate == "failures" and not bool(row.get("passed")):
        predicted = False
        repair["execution_gate_predicted_override"] = True
    elif execution_gate == "oracle":
        predicted = bool(row.get("passed"))
        repair["execution_gate_predicted_override"] = True
    if confidence is None:
        confidence = "low" if repair["json_parse_failed"] or repair["invalid_scores"] else "medium"

    repaired = {
        "specification": specification,
        "test_probes": test_probes,
        "critical_errors": critical_errors,
        "dimension_scores": dimension_scores,
        "overall_score": round(float(overall), 4),
        "semantic_bottleneck_score": round(float(semantic_bottleneck), 4),
        "quality_mean_score": round(float(quality_mean), 4),
        "predicted_pass": bool(predicted),
        "llm_predicted_pass": llm_predicted,
        "calibrated_prediction": bool(calibrated_prediction),
        "strict_prediction": bool(strict_prediction),
        "confidence": confidence,
    }
    return repaired, repair


def safe_auc(labels: list[int], scores: list[float]) -> float | None:
    if len(set(labels)) < 2:
        return None
    return float(roc_auc_score(labels, scores))


def summarize_score_collapse(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Report high-score concentration separately for verifier-pass and verifier-fail rows."""

    def summarize_group(group: list[dict[str, Any]]) -> dict[str, Any]:
        totals = [float(record["overall_score"]) for record in group]
        all_five_count = 0
        for record in group:
            applicable_scores = [
                int(value["score"])
                for value in record["dimension_scores"].values()
                if bool(value.get("applicable", True))
            ]
            if applicable_scores and all(score == 5 for score in applicable_scores):
                all_five_count += 1
        count = len(group)
        return {
            "count": count,
            "mean_overall_score": float(np.mean(totals)) if totals else None,
            "median_overall_score": float(np.median(totals)) if totals else None,
            "overall_score_ge_4_rate": float(sum(score >= 4.0 for score in totals) / count) if count else None,
            "overall_score_ge_4_5_rate": float(sum(score >= 4.5 for score in totals) / count) if count else None,
            "overall_score_eq_5_rate": float(sum(score == 5.0 for score in totals) / count) if count else None,
            "all_applicable_dimensions_eq_5_rate": float(all_five_count / count) if count else None,
        }

    passed = [record for record in records if bool(record["passed"])]
    failed = [record for record in records if not bool(record["passed"])]
    passed_summary = summarize_group(passed)
    failed_summary = summarize_group(failed)
    mean_gap = None
    if passed_summary["mean_overall_score"] is not None and failed_summary["mean_overall_score"] is not None:
        mean_gap = float(passed_summary["mean_overall_score"] - failed_summary["mean_overall_score"])
    return {
        "passed": passed_summary,
        "failed": failed_summary,
        "mean_score_gap_passed_minus_failed": mean_gap,
    }


def compute_metrics(records: list[dict[str, Any]], rubric: dict[str, Any], audit: dict[str, Any]) -> dict[str, Any]:
    labels = [1 if record["passed"] else 0 for record in records]
    totals = [float(record["overall_score"]) for record in records]
    preds = [1 if record["predicted_pass"] else 0 for record in records]
    tn = sum(1 for label, pred in zip(labels, preds) if label == 0 and pred == 0)
    fp = sum(1 for label, pred in zip(labels, preds) if label == 0 and pred == 1)
    fn = sum(1 for label, pred in zip(labels, preds) if label == 1 and pred == 0)
    tp = sum(1 for label, pred in zip(labels, preds) if label == 1 and pred == 1)
    dimension_ids = rubric_dimension_ids(rubric)
    per_dimension = {}
    for dimension_id in dimension_ids:
        values = [int(record["dimension_scores"][dimension_id]["score"]) for record in records]
        per_dimension[dimension_id] = {
            "distribution": {str(score): values.count(score) for score in range(1, 6)},
            "mean": float(np.mean(values)) if values else None,
            "mean_passed": float(np.mean([v for v, y in zip(values, labels) if y == 1])) if any(y == 1 for y in labels) else None,
            "mean_failed": float(np.mean([v for v, y in zip(values, labels) if y == 0])) if any(y == 0 for y in labels) else None,
            "applicable_rate": float(
                np.mean(
                    [
                        bool(record["dimension_scores"][dimension_id].get("applicable", True))
                        for record in records
                    ]
                )
            ) if records else None,
        }
    metrics = {
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
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "overacceptance_rate": float(fp / (tn + fp)) if tn + fp else 0.0,
        "false_rejection_rate": float(fn / (fn + tp)) if fn + tp else 0.0,
        "score_collapse": summarize_score_collapse(records),
        "per_dimension": per_dimension,
        "audit_summary": {
            "prompt_leakage_count": audit["prompt_leakage_count"],
            "json_parse_failed_count": audit["json_parse_failed_count"],
            "repaired_record_count": audit["repaired_record_count"],
            "used_visible_code_fallback_count": audit["used_visible_code_fallback_count"],
            "deterministic_adjusted_record_count": audit.get("deterministic_adjusted_record_count", 0),
            "probe_adjusted_record_count": audit.get("probe_adjusted_record_count", 0),
            "missing_required_test_probes_count": audit.get("missing_required_test_probes_count", 0),
            "material_contract_ambiguity_count": audit.get("material_contract_ambiguity_count", 0),
            "execution_gate": audit.get("execution_gate", "none"),
            "execution_gate_adjusted_record_count": audit.get("execution_gate_adjusted_record_count", 0),
            "execution_gate_predicted_override_count": audit.get("execution_gate_predicted_override_count", 0),
        },
    }
    raw_llm_rows = [record for record in records if isinstance(record.get("llm_predicted_pass"), bool)]
    if raw_llm_rows:
        raw_labels = [1 if record["passed"] else 0 for record in raw_llm_rows]
        raw_preds = [1 if record["llm_predicted_pass"] else 0 for record in raw_llm_rows]
        raw_tn = sum(1 for label, pred in zip(raw_labels, raw_preds) if label == 0 and pred == 0)
        raw_fp = sum(1 for label, pred in zip(raw_labels, raw_preds) if label == 0 and pred == 1)
        metrics["raw_llm_boolean"] = {
            "num_samples": len(raw_llm_rows),
            "accuracy": float(accuracy_score(raw_labels, raw_preds)),
            "kappa": float(cohen_kappa_score(raw_labels, raw_preds)),
            "predicted_pass_rate": float(np.mean(raw_preds)),
            "overacceptance_rate": float(raw_fp / (raw_tn + raw_fp)) if raw_tn + raw_fp else 0.0,
        }
    return metrics


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
    generation_prompts = prompts
    if args.use_chat_template:
        tokenizer = llm.get_tokenizer()
        generation_prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in prompts
        ]
    raw_outputs = []
    for start in range(0, len(generation_prompts), args.batch_size):
        batch = generation_prompts[start : start + args.batch_size]
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
    parser.add_argument("--prompt-profile", choices=["baseline", "strict", "strict_fewshot", "contrastive_fewshot"], default="baseline")
    parser.add_argument("--judge-guidance", type=Path, help="Human-guidance JSON used by strict prompt profiles.")
    parser.add_argument("--few-shot-examples", type=Path, help="Development-only calibration examples for few-shot prompt profiles.")
    parser.add_argument("--max-few-shot-examples", type=int, default=7)
    parser.add_argument("--deterministic-only", action="store_true", help="Use visible-code fallback instead of LLM calls; for smoke tests only.")
    parser.add_argument("--reuse-raw-output", type=Path, help="Recompute scores/metrics from an existing raw judge JSONL without rerunning the LLM.")
    parser.add_argument("--calibrated-prediction", action="store_true", help="Clamp deterministic dimensions and compute predicted_pass from calibrated rubric gates instead of trusting the LLM boolean.")
    parser.add_argument("--calibrated-pass-threshold", type=float, help="Override the rubric pass threshold when --calibrated-prediction is set.")
    parser.add_argument("--strict-prediction", action="store_true", help="Use applicable-dimension means and hard semantic/critical gates.")
    parser.add_argument("--require-test-probes", action="store_true", help="Require ordinary, boundary, and adversarial probes; missing probes force rejection.")
    parser.add_argument(
        "--execution-gate",
        choices=["none", "failures", "oracle"],
        default="none",
        help=(
            "Use verifier execution results only in post-processing. "
            "'failures' forces verifier-failed rows to fail; 'oracle' lets verifier pass/fail control final predicted_pass."
        ),
    )
    parser.add_argument("--use-chat-template", action="store_true", help="Apply the model tokenizer's chat template before generation.")
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
    guidance = json.loads(args.judge_guidance.read_text(encoding="utf-8")) if args.judge_guidance else None
    examples = json.loads(args.few_shot_examples.read_text(encoding="utf-8")) if args.few_shot_examples else None
    configuration_flags = []
    if args.prompt_profile != "baseline" and not guidance:
        configuration_flags.append("strict prompt profile requires --judge-guidance")
    if args.prompt_profile in {"strict_fewshot", "contrastive_fewshot"} and not examples:
        configuration_flags.append(f"{args.prompt_profile} requires --few-shot-examples")
    if args.prompt_profile == "contrastive_fewshot" and not args.require_test_probes:
        configuration_flags.append("contrastive_fewshot requires --require-test-probes")
    splits = {split.strip() for split in args.splits.split(",") if split.strip()}
    rows = select_rows(args.labeled, splits, args.limit, args.offset)
    rubric_prompt = compact_rubric_for_prompt(rubric)
    prompts = [
        build_judge_prompt(
            rubric_prompt,
            row,
            args.max_response_chars,
            args.max_code_chars,
            prompt_profile=args.prompt_profile,
            guidance=guidance,
            examples=examples,
            max_few_shot_examples=args.max_few_shot_examples,
        )
        for row in rows
    ]
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
        "configuration_flags": configuration_flags,
        "prompt_profile": args.prompt_profile,
        "judge_guidance": str(args.judge_guidance) if args.judge_guidance else None,
        "few_shot_examples": str(args.few_shot_examples) if args.few_shot_examples else None,
        "max_few_shot_examples": args.max_few_shot_examples,
        "use_chat_template": bool(args.use_chat_template),
        "prompt_char_stats": {
            "min": min((len(prompt) for prompt in prompts), default=0),
            "max": max((len(prompt) for prompt in prompts), default=0),
            "mean": float(np.mean([len(prompt) for prompt in prompts])) if prompts else 0.0,
        },
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
        "probe_adjusted_record_count": 0,
        "missing_required_test_probes_count": 0,
        "material_contract_ambiguity_count": 0,
        "execution_gate": args.execution_gate,
        "execution_gate_adjusted_record_count": 0,
        "execution_gate_predicted_override_count": 0,
        "calibrated_prediction": bool(args.calibrated_prediction),
        "strict_prediction": bool(args.strict_prediction),
        "require_test_probes": bool(args.require_test_probes),
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
            strict_prediction=args.strict_prediction,
            require_test_probes=args.require_test_probes,
            execution_gate=args.execution_gate,
        )
        if repair["json_parse_failed"]:
            audit["json_parse_failed_count"] += 1
        if repair["used_visible_code_fallback"]:
            audit["used_visible_code_fallback_count"] += 1
        if repair["deterministic_adjustments"]:
            audit["deterministic_adjusted_record_count"] += 1
        if repair["probe_adjustments"]:
            audit["probe_adjusted_record_count"] += 1
        if repair["missing_required_test_probes"]:
            audit["missing_required_test_probes_count"] += 1
        if repair["material_ambiguity_adjustments"]:
            audit["material_contract_ambiguity_count"] += 1
        if repair["execution_gate_adjustments"]:
            audit["execution_gate_adjusted_record_count"] += 1
        if repair["execution_gate_predicted_override"]:
            audit["execution_gate_predicted_override_count"] += 1
        if (
            repair["json_parse_failed"]
            or repair["missing_dimensions"]
            or repair["invalid_scores"]
            or repair["missing_rationales"]
            or repair["missing_specification"]
            or repair["missing_critical_errors_field"]
            or repair["missing_required_test_probes"]
            or repair["material_ambiguity_adjustments"]
            or repair["execution_gate_adjustments"]
            or repair["execution_gate_predicted_override"]
            or repair["ignored_material_ambiguities"]
            or repair["forced_applicable_dimensions"]
        ):
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
            "specification": judgment["specification"],
            "test_probes": judgment["test_probes"],
            "critical_errors": judgment["critical_errors"],
            "dimension_scores": judgment["dimension_scores"],
            "overall_score": judgment["overall_score"],
            "semantic_bottleneck_score": judgment["semantic_bottleneck_score"],
            "quality_mean_score": judgment["quality_mean_score"],
            "predicted_pass": judgment["predicted_pass"],
            "llm_predicted_pass": judgment["llm_predicted_pass"],
            "calibrated_prediction": judgment["calibrated_prediction"],
            "strict_prediction": judgment["strict_prediction"],
            "prompt_profile": args.prompt_profile,
            "confidence": judgment["confidence"],
            "repair": repair,
        }
        records.append(record)
        raw_records.append({"response_id": rid, "prompt_profile": args.prompt_profile, "raw_judge_output": raw_text})

    audit["repair_reasons"] = dict(audit["repair_reasons"])
    audit["valid"] = not rubric_flags and not configuration_flags and not leakage_by_index and len(records) == len(rows)
    metrics = compute_metrics(records, rubric, audit)

    write_jsonl(args.scores_output, records)
    write_jsonl(args.raw_output, raw_records)
    write_json(args.audit_output, audit)
    write_json(args.metrics_output, metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
