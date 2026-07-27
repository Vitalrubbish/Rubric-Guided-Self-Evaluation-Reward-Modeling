#!/usr/bin/env python3
"""Generate a judge-ready rubric from the Phase 1 refined taxonomy.

The LLM is used as a controlled rubric writer, not as a free-form taxonomy
inventor. Category ids, score anchors, critical gates, and equal weights are
preserved by deterministic repair after generation.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


PRIVATE_TOKENS = {
    "test_list",
    "test_setup_code",
    "private_diagnostics",
    "hidden test",
    "hidden-tests",
    "assert ",
    "assert(",
}

GENERIC_DIMENSION_NAMES = {
    "correctness",
    "functional correctness",
    "logic correctness",
    "completeness",
    "quality",
    "clarity",
    "readability",
    "style",
    "general correctness",
}

GENERIC_BAD_PHRASES = {
    "check the logic",
    "problem requirements",
    "task requirements",
    "correct output",
    "incorrect output",
    "expected output",
    "test cases",
    "given examples",
    "minor errors",
    "major errors",
}

CRITICAL_DIMENSION_IDS = {
    "syntax_parseability_truncation",
    "runtime_api_type_misuse",
    "interface_name_signature_mismatch",
}

REQUIRED_ANCHORS = {"1", "2", "3", "4", "5"}


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass

    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        try:
            obj = json.loads(fence.group(1))
            return obj if isinstance(obj, dict) else None
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
                        return obj if isinstance(obj, dict) else None
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


def compact_counter(counter: dict[str, Any] | None, limit: int = 8) -> dict[str, int]:
    if not counter:
        return {}
    return {str(key): int(value) for key, value in Counter(counter).most_common(limit)}


def clean_text(value: Any, fallback: str = "") -> str:
    text = " ".join(str(value or fallback).split())
    return text


def clean_list(value: Any, fallback: list[str] | None = None) -> list[str]:
    if isinstance(value, list):
        items = [clean_text(item) for item in value]
    elif isinstance(value, str) and value.strip():
        items = [clean_text(value)]
    else:
        items = fallback or []
    return [item for item in items if item]


def normalize_anchors(value: Any, fallback: dict[str, str]) -> tuple[dict[str, str], list[str]]:
    anchors: dict[str, str] = {}
    repairs = []
    source = value if isinstance(value, dict) else {}
    for key in sorted(REQUIRED_ANCHORS):
        text = clean_text(source.get(key) or source.get(int(key)) if source else "")
        if not text:
            text = fallback.get(key, "")
            repairs.append(key)
        anchors[key] = text
    return anchors, repairs


def validate_source_taxonomy(taxonomy: dict[str, Any], source_audit: dict[str, Any] | None) -> list[str]:
    flags = []
    if source_audit and source_audit.get("valid") is False:
        flags.append("source audit valid=false")
    categories = taxonomy.get("categories")
    if not isinstance(categories, list) or not categories:
        flags.append("source taxonomy has no categories")
        return flags
    seen = set()
    for category in categories:
        category_id = str(category.get("id") or "")
        if not category_id:
            flags.append("source category missing id")
            continue
        if category_id in seen:
            flags.append(f"duplicate source category id: {category_id}")
        seen.add(category_id)
        refined = category.get("refined_rubric")
        if not isinstance(refined, dict):
            flags.append(f"{category_id}: missing refined_rubric")
            continue
        anchors = refined.get("score_anchors")
        if not isinstance(anchors, dict) or set(map(str, anchors.keys())) != REQUIRED_ANCHORS:
            flags.append(f"{category_id}: incomplete score_anchors")
    return flags


def build_seed_categories(taxonomy: dict[str, Any]) -> list[dict[str, Any]]:
    seeds = []
    for category in taxonomy.get("categories") or []:
        refined = category.get("refined_rubric") or {}
        anchors, _ = normalize_anchors(refined.get("score_anchors"), {})
        seeds.append(
            {
                "id": str(category.get("id")),
                "name": clean_text(category.get("name")),
                "description": clean_text(category.get("description")),
                "linked_clusters": category.get("linked_clusters") or [],
                "response_count": int(category.get("response_count") or 0),
                "failure_types": compact_counter(category.get("failure_types")),
                "error_patterns": compact_counter(category.get("error_patterns")),
                "refined_rubric": {
                    "rubric_dimension": clean_text(refined.get("rubric_dimension"), category.get("name")),
                    "operational_definition": clean_text(refined.get("operational_definition"), category.get("description")),
                    "failure_mechanism": clean_text(refined.get("failure_mechanism")),
                    "common_manifestations": clean_list(refined.get("common_manifestations")),
                    "judge_checklist": clean_list(refined.get("judge_checklist")),
                    "score_anchors": anchors,
                    "positive_boundary": clean_text(refined.get("positive_boundary")),
                    "negative_boundary": clean_text(refined.get("negative_boundary")),
                    "rubric_generation_notes": clean_text(refined.get("rubric_generation_notes")),
                },
            }
        )
    return seeds


def deterministic_dimension(seed: dict[str, Any]) -> dict[str, Any]:
    refined = seed["refined_rubric"]
    dimension_id = seed["id"]
    return {
        "id": dimension_id,
        "name": refined["rubric_dimension"],
        "definition": refined["operational_definition"],
        "failure_mode": refined["failure_mechanism"],
        "what_to_check": refined["judge_checklist"],
        "common_manifestations": refined["common_manifestations"],
        "score_anchors": refined["score_anchors"],
        "positive_boundary": refined["positive_boundary"],
        "negative_boundary": refined["negative_boundary"],
        "critical_failure": dimension_id in CRITICAL_DIMENSION_IDS,
        "weight": 1.0,
        "source_statistics": {
            "response_count": seed["response_count"],
            "failure_types": seed["failure_types"],
            "error_patterns": seed["error_patterns"],
            "linked_clusters": seed["linked_clusters"],
        },
    }


def build_deterministic_rubric(
    taxonomy_path: Path,
    taxonomy: dict[str, Any],
    seeds: list[dict[str, Any]],
    rubric_name: str,
) -> dict[str, Any]:
    dimensions = [deterministic_dimension(seed) for seed in seeds]
    return {
        "name": rubric_name,
        "source_taxonomy": str(taxonomy_path),
        "source_taxonomy_name": taxonomy.get("name"),
        "generation_method": "taxonomy_controlled_llm_with_deterministic_repair",
        "task_type": "Python code generation",
        "intended_use": "Score a generated Python answer using task text, public interface, raw model response, and extracted code only.",
        "global_judging_instructions": [
            "Score each dimension independently on the 1-5 scale.",
            "Use the task text and public interface as the public contract.",
            "Do not use verifier labels, private diagnostics, held-out answers, or benchmark metadata when scoring.",
            "A low critical dimension score should normally make the predicted pass decision false.",
            "Keep rationales concise and tied to visible code properties.",
        ],
        "dimensions": dimensions,
        "aggregation": {
            "method": "mean_with_critical_gates",
            "dimension_weighting": "equal",
            "pass_threshold": 4.0,
            "critical_dimension_ids": sorted(CRITICAL_DIMENSION_IDS),
            "critical_gate": "If any critical dimension receives a score of 1 or 2, predicted_pass should normally be false.",
            "overall_score": "Mean of dimension scores after considering critical gate rationale.",
        },
        "judge_output_schema": {
            "dimension_scores": {
                "<dimension_id>": {
                    "score": "integer 1-5",
                    "rationale": "short visible-code rationale",
                }
            },
            "overall_score": "number 1-5",
            "predicted_pass": "boolean",
            "confidence": "low|medium|high",
        },
    }


def compact_prompt_categories(seed_categories: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact = []
    for seed in seed_categories:
        refined = seed["refined_rubric"]
        compact.append(
            {
                "id": seed["id"],
                "source_name": seed["name"],
                "critical_failure": seed["id"] in CRITICAL_DIMENSION_IDS,
                "rubric_dimension": refined["rubric_dimension"],
                "operational_definition": refined["operational_definition"],
                "failure_mechanism": refined["failure_mechanism"],
                "judge_checklist": refined["judge_checklist"],
                "common_manifestations": refined["common_manifestations"],
                "score_anchors": refined["score_anchors"],
                "positive_boundary": refined["positive_boundary"],
                "negative_boundary": refined["negative_boundary"],
            }
        )
    return compact


def build_generation_prompt(seed_categories: list[dict[str, Any]], skeleton: dict[str, Any]) -> str:
    seed_json = json.dumps(compact_prompt_categories(seed_categories), ensure_ascii=False, indent=2)
    expected_ids = [category["id"] for category in seed_categories]
    return f"""You are converting a refined coding-error taxonomy into a machine-readable rubric for an LLM code judge.

The taxonomy has already been discovered, consolidated, refined, and audited. Do not invent new categories.

Hard constraints:
- Preserve exactly these dimension ids, once each: {expected_ids}
- Do not merge, split, drop, rename, or reorder ids.
- Keep one rubric dimension per refined taxonomy category.
- Preserve 1-5 score anchors for every dimension. You may improve wording, but not weaken specificity.
- Use equal weights. Do not weight dimensions by frequency.
- Mark only syntax/output-format, runtime/API/type, and interface/name/signature dimensions as critical failures.
- Keep source_statistics out of the prompt-derived text; deterministic repair will attach them later.
- Do not mention hidden benchmark material, verifier labels, exact reference answers, private diagnostics, response ids, or benchmark internals.
- Return ONLY valid JSON. No Markdown fences.

Required output schema:
{{
  "name": "{skeleton.get('name')}",
  "source_taxonomy": "path string",
  "source_taxonomy_name": "name string",
  "generation_method": "taxonomy_controlled_llm_with_deterministic_repair",
  "task_type": "Python code generation",
  "intended_use": "short string",
  "global_judging_instructions": ["instruction"],
  "dimensions": [
    {{
      "id": "same_category_id",
      "name": "judge-ready dimension name",
      "definition": "what visible code property this dimension scores",
      "failure_mode": "causal failure mechanism",
      "what_to_check": ["concrete judge check"],
      "common_manifestations": ["visible manifestation"],
      "score_anchors": {{"1": "...", "2": "...", "3": "...", "4": "...", "5": "..."}},
      "positive_boundary": "when not to penalize this dimension",
      "negative_boundary": "when to penalize this dimension",
      "critical_failure": false,
      "weight": 1.0,
      "source_statistics": {{"response_count": 0, "failure_types": {{}}, "error_patterns": {{}}, "linked_clusters": []}}
    }}
  ],
  "aggregation": {{
    "method": "mean_with_critical_gates",
    "dimension_weighting": "equal",
    "pass_threshold": 4.0,
    "critical_dimension_ids": ["id"],
    "critical_gate": "short rule",
    "overall_score": "short definition"
  }},
  "judge_output_schema": {{
    "dimension_scores": {{"<dimension_id>": {{"score": "integer 1-5", "rationale": "short visible-code rationale"}}}},
    "overall_score": "number 1-5",
    "predicted_pass": "boolean",
    "confidence": "low|medium|high"
  }}
}}

Refined taxonomy categories:
{seed_json}
"""


def text_has_private_leakage(obj: Any) -> list[str]:
    text = json.dumps(obj, ensure_ascii=False).lower()
    return sorted(token for token in PRIVATE_TOKENS if token.lower() in text)


def generic_dimension_flags(dimensions: list[dict[str, Any]]) -> list[str]:
    flags = []
    for dimension in dimensions:
        dimension_id = str(dimension.get("id"))
        name = clean_text(dimension.get("name")).lower()
        body = json.dumps(dimension, ensure_ascii=False).lower()
        if name in GENERIC_DIMENSION_NAMES:
            flags.append(f"{dimension_id}: generic dimension name '{name}'")
        for phrase in GENERIC_BAD_PHRASES:
            if phrase in body:
                flags.append(f"{dimension_id}: generic phrase '{phrase}'")
                break
    return flags


def normalize_dimension(candidate: dict[str, Any], fallback: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    repairs = []
    normalized = dict(fallback)
    for key in [
        "name",
        "definition",
        "failure_mode",
        "positive_boundary",
        "negative_boundary",
    ]:
        value = clean_text(candidate.get(key))
        if value:
            normalized[key] = value
        else:
            repairs.append(f"{fallback['id']}: missing {key}")

    for key in ["what_to_check", "common_manifestations"]:
        values = clean_list(candidate.get(key), fallback.get(key) or [])
        if values:
            normalized[key] = values
        else:
            repairs.append(f"{fallback['id']}: missing {key}")

    anchors, anchor_repairs = normalize_anchors(candidate.get("score_anchors"), fallback["score_anchors"])
    normalized["score_anchors"] = anchors
    repairs.extend(f"{fallback['id']}: repaired score anchor {key}" for key in anchor_repairs)

    # Keep these deterministic to avoid train-frequency overfitting or gate drift.
    normalized["critical_failure"] = fallback["critical_failure"]
    normalized["weight"] = fallback["weight"]
    normalized["source_statistics"] = fallback["source_statistics"]
    return normalized, repairs


def repair_rubric(
    candidate: dict[str, Any] | None,
    skeleton: dict[str, Any],
    expected_ids: list[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    repair_audit: dict[str, Any] = {
        "used_deterministic_fallback": False,
        "fallback_reason": None,
        "missing_dimensions_repaired": [],
        "unknown_dimensions_removed": [],
        "duplicate_dimensions_removed": [],
        "dimension_repairs": [],
        "generic_dimensions_replaced": [],
    }

    if not candidate:
        repair_audit["used_deterministic_fallback"] = True
        repair_audit["fallback_reason"] = "no_parseable_llm_json"
        return skeleton, repair_audit

    if isinstance(candidate.get("rubric"), dict):
        candidate = candidate["rubric"]

    fallback_by_id = {dimension["id"]: dimension for dimension in skeleton["dimensions"]}
    candidate_dimensions = candidate.get("dimensions")
    if not isinstance(candidate_dimensions, list):
        repair_audit["used_deterministic_fallback"] = True
        repair_audit["fallback_reason"] = "missing_dimensions_list"
        return skeleton, repair_audit

    normalized_by_id: dict[str, dict[str, Any]] = {}
    for dimension in candidate_dimensions:
        if not isinstance(dimension, dict):
            continue
        dimension_id = str(dimension.get("id") or "")
        if dimension_id not in fallback_by_id:
            repair_audit["unknown_dimensions_removed"].append(dimension_id or "<missing id>")
            continue
        if dimension_id in normalized_by_id:
            repair_audit["duplicate_dimensions_removed"].append(dimension_id)
            continue
        normalized, repairs = normalize_dimension(dimension, fallback_by_id[dimension_id])
        if generic_dimension_flags([normalized]):
            normalized = fallback_by_id[dimension_id]
            repair_audit["generic_dimensions_replaced"].append(dimension_id)
        normalized_by_id[dimension_id] = normalized
        repair_audit["dimension_repairs"].extend(repairs)

    for dimension_id in expected_ids:
        if dimension_id not in normalized_by_id:
            normalized_by_id[dimension_id] = fallback_by_id[dimension_id]
            repair_audit["missing_dimensions_repaired"].append(dimension_id)

    repaired = dict(skeleton)
    # Source metadata and generation method are deterministic provenance fields;
    # never let the LLM replace them with placeholders.
    for key in ["name", "task_type", "intended_use"]:
        if clean_text(candidate.get(key)):
            repaired[key] = clean_text(candidate.get(key))
    instructions = clean_list(candidate.get("global_judging_instructions"), skeleton["global_judging_instructions"])
    repaired["global_judging_instructions"] = instructions
    repaired["dimensions"] = [normalized_by_id[dimension_id] for dimension_id in expected_ids]
    # Keep aggregation and output schema deterministic for evaluator compatibility.
    repaired["aggregation"] = skeleton["aggregation"]
    repaired["judge_output_schema"] = skeleton["judge_output_schema"]

    leakage = text_has_private_leakage(repaired)
    if leakage:
        repair_audit["used_deterministic_fallback"] = True
        repair_audit["fallback_reason"] = f"private_leakage_after_repair: {leakage}"
        return skeleton, repair_audit

    return repaired, repair_audit


def final_audit(
    rubric: dict[str, Any],
    source_flags: list[str],
    repair_audit: dict[str, Any],
    model: str | None,
    used_llm: bool,
    used_existing_llm_output: bool,
) -> dict[str, Any]:
    dimensions = rubric.get("dimensions") or []
    ids = [dimension.get("id") for dimension in dimensions]
    id_counts = Counter(ids)
    anchor_flags = []
    for dimension in dimensions:
        anchors = dimension.get("score_anchors")
        if not isinstance(anchors, dict) or set(map(str, anchors.keys())) != REQUIRED_ANCHORS:
            anchor_flags.append(str(dimension.get("id")))
    private_flags = text_has_private_leakage(rubric)
    generic_flags = generic_dimension_flags(dimensions)
    return {
        "method": "taxonomy_controlled_llm_rubric_generation_with_deterministic_repair",
        "model": model,
        "used_llm": used_llm,
        "used_existing_llm_output": used_existing_llm_output,
        "source_flags": source_flags,
        "dimension_count": len(dimensions),
        "dimension_ids": ids,
        "duplicate_dimension_ids": sorted(key for key, count in id_counts.items() if count > 1),
        "missing_anchor_dimensions": anchor_flags,
        "private_leakage_flags": private_flags,
        "generic_dimension_flags": generic_flags,
        "repair_audit": repair_audit,
        "valid": (
            not source_flags
            and len(dimensions) == len(set(ids))
            and not anchor_flags
            and not private_flags
            and not generic_flags
        ),
    }


def run_llm(args: argparse.Namespace, prompt: str) -> tuple[dict[str, Any] | None, str]:
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        tensor_parallel_size=1,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=1,
    )
    sampling = SamplingParams(
        n=1,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=42,
    )
    output = llm.generate([prompt], sampling)[0].outputs[0].text
    return parse_json_object(output), output


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an LLM judge rubric from a Phase 1 refined taxonomy.")
    parser.add_argument("--taxonomy", type=Path, required=True)
    parser.add_argument("--source-audit", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/rubrics/phase2/mbpp_hidden_llm_rubric_from_refined_taxonomy.json"))
    parser.add_argument("--audit-output", type=Path, default=Path("data/rubrics/phase2/mbpp_hidden_llm_rubric_from_refined_taxonomy_audit.json"))
    parser.add_argument("--raw-llm-output", type=Path, default=Path("data/rubrics/phase2/mbpp_hidden_llm_rubric_from_refined_taxonomy_raw_response.txt"))
    parser.add_argument("--existing-llm-output", type=Path, help="Parse an existing raw LLM response instead of calling vLLM.")
    parser.add_argument("--deterministic-only", action="store_true", help="Write the deterministic taxonomy-derived rubric skeleton without calling an LLM.")
    parser.add_argument("--rubric-name", default="mbpp_hidden_llm_rubric_from_refined_taxonomy_v1")
    parser.add_argument("--model")
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.30)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=4096)
    args = parser.parse_args()

    taxonomy = load_yaml(args.taxonomy)
    source_audit = load_json(args.source_audit) if args.source_audit else None
    source_flags = validate_source_taxonomy(taxonomy, source_audit)
    seeds = build_seed_categories(taxonomy)
    skeleton = build_deterministic_rubric(args.taxonomy, taxonomy, seeds, args.rubric_name)
    expected_ids = [seed["id"] for seed in seeds]

    used_llm = False
    used_existing = False
    raw_output = ""
    parsed: dict[str, Any] | None = None

    if args.existing_llm_output:
        raw_output = args.existing_llm_output.read_text(encoding="utf-8")
        parsed = parse_json_object(raw_output)
        used_existing = True
    elif args.deterministic_only or not args.model:
        parsed = skeleton
    else:
        prompt = build_generation_prompt(seeds, skeleton)
        parsed, raw_output = run_llm(args, prompt)
        used_llm = True
        if args.raw_llm_output:
            args.raw_llm_output.parent.mkdir(parents=True, exist_ok=True)
            args.raw_llm_output.write_text(raw_output, encoding="utf-8")

    rubric, repair = repair_rubric(parsed, skeleton, expected_ids)
    if args.deterministic_only:
        repair["used_deterministic_fallback"] = True
        repair["fallback_reason"] = "deterministic_only_requested"
    elif not args.model and not args.existing_llm_output:
        repair["used_deterministic_fallback"] = True
        repair["fallback_reason"] = "no_model_provided"

    audit = final_audit(
        rubric=rubric,
        source_flags=source_flags,
        repair_audit=repair,
        model=args.model,
        used_llm=used_llm,
        used_existing_llm_output=used_existing,
    )

    write_json(args.output, rubric)
    write_json(args.audit_output, audit)
    print(json.dumps({"output": str(args.output), "audit": str(args.audit_output), "valid": audit["valid"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
