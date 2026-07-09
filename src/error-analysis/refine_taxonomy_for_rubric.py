#!/usr/bin/env python3
"""Refine a consolidated error taxonomy into rubric-operational dimensions.

This stage keeps the Phase 1 taxonomy assignments fixed. It does not merge,
split, or relabel responses. Its job is to turn each category into a useful
rubric seed with definitions, scoring anchors, checklist items, and boundary
notes. The LLM supplies semantic detail; deterministic repair enforces schema,
coverage, private-field hygiene, and non-generic category descriptions.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml


PRIVATE_TOKENS = {
    "test_list",
    "test_setup_code",
    "private_diagnostics",
    "assert ",
    "assert(",
    "hidden test",
    "hidden-tests",
}

GENERIC_BAD_PHRASES = {
    "automatically operationalized",
    "automatically created",
    "automatically split",
    "check the logic",
    "check this failure mode",
    "distinguish severe, partial, and minor",
    "ensure functions are correctly defined and used as intended",
    "ensure that the types of variables and function arguments are correct",
    "exceeds the problem requirements",
    "problem requirements by providing additional",
    "problem requirements",
    "task requirements",
    "incorrect output value",
    "incorrect output type",
    "incorrect output length",
    "correct output value",
    "correct output type",
    "correct output values",
    "expected value",
    "expected type",
    "expected output",
    "tested inputs",
    "minor errors",
    "severe errors",
    "minor issues",
    "significant issues",
    "most cases",
    "all cases as required",
    "edge cases as specified",
    "handles them correctly",
    "fully handles all relevant",
    "without missing any",
    "some checks in place",
    "required parameters and return type",
    "required parameter signature",
    "minor corrections",
    "unnecessary whitespace",
    "error-free",
    "most api calls",
    "minor runtime errors",
    "no runtime errors, but",
    "some undefined references",
    "when this category should not reduce the score",
    "when this category should reduce the score",
    "try-except",
    "test cases",
    "tested cases",
    "given examples",
    "as specified by the task",
    "specified by the task",
    "task and public interface",
    "correctly implements",
    "correctly handles",
    "right value",
    "for every input",
    "most inputs",
    "no errors",
    "wrong type",
    "correct type",
    "incorrect behavior",
    "used correctly",
}

REQUIRED_REFINED_FIELDS = {
    "rubric_dimension",
    "operational_definition",
    "failure_mechanism",
    "common_manifestations",
    "judge_checklist",
    "score_anchors",
    "positive_boundary",
    "negative_boundary",
    "rubric_generation_notes",
}

CATEGORY_KEYWORDS = {
    "numeric_formula_correctness": {
        "formula",
        "arithmetic",
        "numeric",
        "count",
        "index",
        "round",
        "sequence",
        "aggregation",
        "off-by-one",
    },
    "output_type_container_shape": {
        "return",
        "type",
        "container",
        "shape",
        "nesting",
        "arity",
        "element",
        "ordering",
    },
    "algorithmic_wrong_value": {
        "algorithm",
        "predicate",
        "traversal",
        "filter",
        "comparison",
        "state",
        "selection",
        "transformation",
        "ordering",
    },
    "syntax_parseability_or_output_format": {
        "parse",
        "syntax",
        "indentation",
        "truncated",
        "unclosed",
        "markdown",
        "duplicate",
        "code block",
    },
    "runtime_api_type_misuse": {
        "runtime",
        "api",
        "type",
        "undefined",
        "import",
        "index",
        "unpack",
        "exception",
        "method",
    },
    "string_regex_pattern_logic": {
        "string",
        "regex",
        "substring",
        "token",
        "pattern",
        "match",
        "boundary",
        "capture",
    },
    "edge_case_boundary_handling": {
        "boundary",
        "empty",
        "singleton",
        "duplicate",
        "zero",
        "guard",
        "minimal",
        "degenerate",
    },
    "interface_name_signature_mismatch": {
        "public",
        "interface",
        "signature",
        "callable",
        "name",
        "module",
        "parameter",
        "scope",
    },
}

CATEGORY_ALLOWED_GENERIC_PHRASES = {
    "output_type_container_shape": {
        "correct type",
        "wrong type",
    },
}


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def parse_json_object(text: str) -> dict | None:
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
                        obj = json.loads(text[start:index + 1])
                        return obj if isinstance(obj, dict) else None
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


def compact_counter(counter: dict | None, limit: int = 8) -> dict:
    if not counter:
        return {}
    return dict(Counter(counter).most_common(limit))


def extract_task(prompt: str | None) -> str:
    if not prompt:
        return ""
    match = re.search(r"Task:\s*(.*?)\n\s*\nDefine code matching this public interface:", prompt, re.DOTALL)
    if match:
        return " ".join(match.group(1).split())
    return ""


def short_text(text: str | None, limit: int = 280) -> str:
    text = " ".join(str(text or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def code_excerpt(code: str | None, max_lines: int = 18, max_chars: int = 900) -> str:
    lines = str(code or "").strip().splitlines()
    if not lines:
        return ""
    excerpt = "\n".join(lines[:max_lines])
    return short_text(excerpt, max_chars)


def compact_safe_diagnostics(diagnostics: dict | None) -> dict:
    if not isinstance(diagnostics, dict):
        return {}
    result = {
        "diagnostic_kind": diagnostics.get("diagnostic_kind"),
        "first_failure_kind": diagnostics.get("first_failure_kind"),
    }
    if diagnostics.get("first_actual") is not None:
        result["observed_value_shape"] = diagnostics.get("first_actual")
    if diagnostics.get("first_expected") is not None:
        result["reference_value_shape"] = diagnostics.get("first_expected")
    if diagnostics.get("exception_type"):
        result["exception_type"] = diagnostics.get("exception_type")
    if diagnostics.get("syntax_error_type"):
        result["syntax_error_type"] = diagnostics.get("syntax_error_type")
    return {key: value for key, value in result.items() if value is not None}


def build_failure_lookup(failures_path: Path | None) -> dict[str, dict]:
    if not failures_path:
        return {}
    return {row["response_id"]: row for row in read_jsonl(failures_path)}


def safe_example(row: dict, failure_lookup: dict[str, dict]) -> dict:
    failure = failure_lookup.get(row.get("response_id"), {})
    return {
        "task": short_text(extract_task(failure.get("prompt"))),
        "public_interface": failure.get("interface_signatures") or failure.get("interface_names") or [],
        "failure_type": row.get("failure_type"),
        "error_pattern": row.get("error_pattern"),
        "safe_diagnostic_shape": compact_safe_diagnostics(row.get("safe_diagnostics") or failure.get("safe_diagnostics")),
        "llm_root_cause_summary": short_text(row.get("llm_summary"), 240),
        "generated_code_excerpt": code_excerpt(failure.get("extracted_code")),
    }


def select_examples(rows: list[dict], failure_lookup: dict[str, dict], max_examples: int) -> list[dict]:
    by_pattern: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_pattern[str(row.get("error_pattern") or "unknown")].append(row)

    selected = []
    seen_ids = set()
    for _, pattern_rows in sorted(by_pattern.items(), key=lambda item: (-len(item[1]), item[0])):
        for row in pattern_rows:
            response_id = row.get("response_id")
            if response_id in seen_ids:
                continue
            selected.append(row)
            seen_ids.add(response_id)
            break
        if len(selected) >= max_examples:
            break

    if len(selected) < max_examples:
        for row in rows:
            response_id = row.get("response_id")
            if response_id in seen_ids:
                continue
            selected.append(row)
            seen_ids.add(response_id)
            if len(selected) >= max_examples:
                break

    return [safe_example(row, failure_lookup) for row in selected]


def build_category_evidence(
    taxonomy: dict,
    assignments: list[dict],
    failure_lookup: dict[str, dict],
    max_examples_per_category: int,
) -> list[dict]:
    rows_by_category: dict[str, list[dict]] = defaultdict(list)
    for row in assignments:
        rows_by_category[str(row.get("taxonomy_category_id"))].append(row)

    evidence = []
    for category in taxonomy.get("categories") or []:
        category_id = str(category.get("id"))
        rows = rows_by_category.get(category_id, [])
        failure_types = Counter(row.get("failure_type") for row in rows if row.get("failure_type"))
        error_patterns = Counter(row.get("error_pattern") for row in rows if row.get("error_pattern"))
        evidence.append(
            {
                "id": category_id,
                "current_name": category.get("name"),
                "current_description": category.get("description"),
                "current_rubric_hint": category.get("rubric_hint"),
                "linked_clusters": category.get("linked_clusters") or [],
                "response_count": category.get("response_count", len(rows)),
                "failure_types": compact_counter(dict(failure_types) or category.get("failure_types")),
                "error_patterns": compact_counter(dict(error_patterns) or category.get("error_patterns")),
                "common_failure_signals": category.get("common_failure_signals") or [],
                "representative_safe_examples": select_examples(rows, failure_lookup, max_examples_per_category),
            }
        )
    contrast_summaries = []
    for item in evidence:
        template = category_template({"id": item["id"], "name": item.get("current_name")})
        contrast_summaries.append(
            {
                "id": item["id"],
                "name": item.get("current_name"),
                "response_count": item.get("response_count"),
                "dominant_error_patterns": compact_counter(item.get("error_patterns"), 3),
                "distinguishing_focus": template["failure_mechanism"],
            }
        )
    for item in evidence:
        item["nearby_category_contrasts"] = [
            contrast for contrast in contrast_summaries if contrast["id"] != item["id"]
        ]
    return evidence


def build_refinement_prompt(evidence: list[dict]) -> str:
    evidence_json = json.dumps(evidence, ensure_ascii=False, indent=2)
    category_ids = [item["id"] for item in evidence]
    return f"""You are refining an automatically discovered coding-error taxonomy into rubric-operational dimensions.

Input: consolidated taxonomy categories plus safe representative evidence. The category assignments are fixed.
Task: for every category id, write a precise rubric seed that can later become a 1-5 judging rubric.

Hard constraints:
- Keep exactly these category ids, once each: {category_ids}
- Do not merge, split, rename ids, or reassign clusters.
- Do not mention hidden tests, assert statements, exact expected values, private verifier details, or response ids.
- Avoid broad labels or generic advice such as "logic error", "function error", "type error", "check the logic", or "severe/partial/minor".
- Make each category operational: a judge should know what concrete code property to inspect.
- Score anchors must be specific to that category and must include string keys "1", "2", "3", "4", "5".
- Boundary notes must define concrete in-scope and out-of-scope code conditions using category-specific mechanism language.
- Do not use broad boundary phrases such as "correct type", "wrong type", "right value", "most inputs", "every input", or "specified by the task"; name the concrete contract property instead.
- Rubric dimensions must be descriptive, not one-word or two-word generic labels.

Return ONLY valid JSON with this schema:
{{
  "taxonomy_name": "mbpp_refined_rubric_operational_taxonomy_v1",
  "categories": [
    {{
      "id": "same_category_id",
      "rubric_dimension": "Short rubric dimension name",
      "operational_definition": "Two or three concrete sentences.",
      "failure_mechanism": "The causal mechanism behind this failure mode.",
      "common_manifestations": ["specific manifestation"],
      "judge_checklist": ["concrete check the judge can apply"],
      "score_anchors": {{
        "1": "category-specific anchor",
        "2": "category-specific anchor",
        "3": "category-specific anchor",
        "4": "category-specific anchor",
        "5": "category-specific anchor"
      }},
      "positive_boundary": "Concrete out-of-scope boundary written in category-specific code-property language.",
      "negative_boundary": "Concrete in-scope penalty boundary written in category-specific code-property language.",
      "rubric_generation_notes": "How to convert this category into a standalone rubric dimension.",
      "confidence": "high"
    }}
  ],
  "notes": "short optional note"
}}

Safe category evidence:
{evidence_json}
"""


def build_single_category_prompt(evidence: dict) -> str:
    evidence_json = json.dumps(evidence, ensure_ascii=False, indent=2)
    category_id = evidence["id"]
    quality_floor = category_template({"id": category_id, "name": evidence.get("current_name")})
    quality_floor_json = json.dumps(
        {
            "rubric_dimension": quality_floor["rubric_dimension"],
            "failure_mechanism": quality_floor["failure_mechanism"],
            "common_manifestations": quality_floor["common_manifestations"],
            "judge_checklist": quality_floor["judge_checklist"],
        },
        ensure_ascii=False,
        indent=2,
    )
    return f"""You are refining one coding-error taxonomy category into a rubric-operational dimension.

The category assignment is fixed. Your job is to describe what a future rubric judge should inspect.

Hard constraints:
- Keep exactly this category id: {category_id}
- Return one JSON object, not a list.
- Do not mention hidden tests, assert statements, exact expected values, private verifier details, or response ids.
- Avoid generic advice such as "check the logic", "problem requirements", "task requirements", "incorrect output", "correct output value", "severe/minor errors", or "expected value".
- Do not copy schema placeholder text into any field; boundary fields must contain actual category-specific conditions.
- Avoid starting definitions with generic "Ensures ..." phrasing; define the inspected code property directly.
- Do not use broad boundary phrases such as "correct type", "wrong type", "right value", "most inputs", "every input", or "specified by the task"; name the concrete contract property instead.
- Rubric dimension must be descriptive, not a one-word or two-word generic label.
- Prefer mechanism terms such as "specified relation", "public contract", "semantic rule", "computed numeric content", "state update", or "container contract".
- Make the category operational and code-inspectable: focus on mechanisms, boundaries, and concrete checks.
- Use generated_code_excerpt and safe_diagnostic_shape only to infer mechanisms; do not quote identifiers or examples as if they were tests.
- Use nearby_category_contrasts to explain what belongs in this category instead of adjacent categories.
- Score anchors must be specific to this category and must include string keys "1", "2", "3", "4", "5".
- Score 5 means fully satisfying this category; it must not reward doing extra beyond the task.
- Your output must be at least as concrete as the quality floor below. You may adapt it, but do not make it broader.

Return ONLY valid JSON with this schema:
{{
  "id": "{category_id}",
  "rubric_dimension": "Short rubric dimension name",
  "operational_definition": "Two or three concrete sentences.",
  "failure_mechanism": "The causal mechanism behind this failure mode.",
  "common_manifestations": ["specific manifestation"],
  "judge_checklist": ["concrete check the judge can apply"],
  "score_anchors": {{
    "1": "category-specific anchor",
    "2": "category-specific anchor",
    "3": "category-specific anchor",
    "4": "category-specific anchor",
    "5": "category-specific anchor"
  }},
  "positive_boundary": "Concrete out-of-scope boundary written in category-specific code-property language.",
  "negative_boundary": "Concrete in-scope penalty boundary written in category-specific code-property language.",
  "rubric_generation_notes": "How to convert this category into a standalone rubric dimension.",
  "confidence": "high"
}}

Safe category evidence:
{evidence_json}

Category-specific quality floor:
{quality_floor_json}
"""


def category_template(category: dict) -> dict:
    category_id = str(category.get("id"))
    name = str(category.get("name") or category_id.replace("_", " ").title())

    templates: dict[str, dict] = {
        "numeric_formula_correctness": {
            "rubric_dimension": "Numeric Formula and Calculation Correctness",
            "operational_definition": (
                "Evaluate whether the implementation uses the correct mathematical relation, counting rule, "
                "index arithmetic, rounding behavior, and accumulation logic required by the task. Penalize code "
                "that returns the right general kind of object but computes the wrong numeric content."
            ),
            "failure_mechanism": "The solution encodes an incorrect formula, off-by-one rule, sequence update, or aggregation step.",
            "common_manifestations": [
                "Uses a plausible but incorrect recurrence, divisor rule, index base, or arithmetic expression.",
                "Returns a scalar or collection with numeric values derived from the wrong computation.",
                "Handles simple examples but fails when the task requires exact counting, ordering, or rounding.",
            ],
            "judge_checklist": [
                "Identify the numeric invariant or formula implied by the task statement.",
                "Check loop bounds, index bases, and update order against that invariant.",
                "Check rounding, integer division, and aggregation behavior where applicable.",
                "Confirm numeric values are computed, not merely shaped, according to the requested relation.",
            ],
            "score_anchors": {
                "1": "The numeric computation is unrelated to the required relation or returns arbitrary values.",
                "2": "The code recognizes the numeric task but uses a substantially wrong formula or update rule.",
                "3": "The main numeric idea is present but contains clear off-by-one, rounding, or aggregation errors.",
                "4": "The computation is mostly correct with only narrow boundary or precision weaknesses.",
                "5": "The implementation follows the required numeric relation across normal and boundary inputs.",
            },
            "positive_boundary": "Do not penalize naming or style issues here if the numeric relation and returned values are correct.",
            "negative_boundary": "Penalize when the returned object has acceptable shape but the numbers come from the wrong formula, count, or sequence logic.",
            "rubric_generation_notes": "Use this as a correctness dimension for tasks dominated by arithmetic, counting, sequences, or numeric aggregation.",
        },
        "output_type_container_shape": {
            "rubric_dimension": "Output Type and Container Contract",
            "operational_definition": (
                "Evaluate whether the implementation returns the required type, container nesting, arity, element ordering, "
                "and element kinds described by the public task contract. This category is about the shape of the returned "
                "object, even when some underlying values are plausible."
            ),
            "failure_mechanism": "The solution computes or locates some relevant information but packages it in the wrong return form.",
            "common_manifestations": [
                "Returns a scalar where a list, tuple, string, or structured container is required.",
                "Returns the right container family with the wrong length, nesting, element order, or element type.",
                "Returns None, a generator, or an intermediate helper result instead of the final requested object.",
            ],
            "judge_checklist": [
                "Extract the expected return contract from the task statement and public interface.",
                "Check the top-level return type and whether a value is always returned.",
                "Check container length, nesting, ordering, and element types.",
                "Separate shape errors from pure value errors when the container contract is otherwise satisfied.",
            ],
            "score_anchors": {
                "1": "The solution does not return a usable value for the requested interface.",
                "2": "The top-level return type or container family is incompatible with the task contract.",
                "3": "The container family is plausible but length, nesting, order, or element kinds are wrong.",
                "4": "The return contract is mostly satisfied with only small ambiguity or edge-case shape risk.",
                "5": "The returned object matches the required type, arity, nesting, order, and element kinds.",
            },
            "positive_boundary": "Do not penalize minor implementation style if the returned object contract is exactly satisfied.",
            "negative_boundary": "Penalize when relevant information is present but exposed through the wrong return type, arity, nesting, or ordering.",
            "rubric_generation_notes": "Use this as a separate dimension from value correctness so judges can identify contract-shape failures.",
        },
        "algorithmic_wrong_value": {
            "rubric_dimension": "Algorithmic Semantics and Value Correctness",
            "operational_definition": (
                "Evaluate whether the code implements the task's intended data transformation, selection rule, ordering rule, "
                "or control flow. This category captures executable solutions that return a plausible object but derive it "
                "from the wrong algorithmic semantics."
            ),
            "failure_mechanism": "The solution follows the wrong condition, traversal, filtering, comparison, state update, or result-construction logic.",
            "common_manifestations": [
                "Uses the wrong predicate, sort key, traversal direction, or update condition.",
                "Confuses related concepts in the task and returns values for a neighboring but different problem.",
                "Produces plausible outputs on simple cases while failing the task's core semantic rule.",
            ],
            "judge_checklist": [
                "State the task's required transformation or selection rule in plain terms.",
                "Trace whether the implementation applies that rule to every relevant input element.",
                "Check ordering, filtering, comparison, and state updates for semantic mismatch.",
                "Confirm that correct-looking output shape is not masking the wrong value construction.",
            ],
            "score_anchors": {
                "1": "The implemented algorithm solves a different problem or ignores the requested transformation.",
                "2": "The code contains a recognizable task fragment but applies the wrong core rule.",
                "3": "The main approach is plausible but has substantial semantic errors in conditions, ordering, or updates.",
                "4": "The algorithm is mostly faithful with only narrow semantic or boundary mistakes.",
                "5": "The implementation faithfully applies the required algorithmic rule and returns correct values.",
            },
            "positive_boundary": "Do not penalize under this category for syntax, interface, or return-shape issues when the semantic rule itself is correct.",
            "negative_boundary": "Penalize when the code is executable and well-shaped but computes values using the wrong task logic.",
            "rubric_generation_notes": "Use this as the broad semantic-correctness dimension for nontrivial transformations that are not mainly formula errors.",
        },
        "syntax_parseability_or_output_format": {
            "rubric_dimension": "Python Parseability and Clean Code Output",
            "operational_definition": (
                "Evaluate whether the answer is complete, parseable Python code in a single usable solution body. Penalize "
                "unclosed blocks, duplicated definitions, indentation damage, Markdown artifacts, or prose that prevents direct execution."
            ),
            "failure_mechanism": "The generated answer is structurally malformed before task semantics can be evaluated.",
            "common_manifestations": [
                "Contains Markdown fences, repeated code blocks, or explanatory text that breaks parsing.",
                "Has unclosed parentheses, strings, indentation blocks, or truncated definitions.",
                "Duplicates a function body or appends stray tokens after an otherwise plausible solution.",
            ],
            "judge_checklist": [
                "Check whether the submitted text can be parsed as Python without manual cleanup.",
                "Check for truncation, duplicate definitions, dangling tokens, and indentation damage.",
                "Check that the answer contains code only and exposes one coherent solution.",
                "Distinguish parse failures from runtime failures after parsing succeeds.",
            ],
            "score_anchors": {
                "1": "The answer is not parseable Python or is dominated by non-code content.",
                "2": "Severe formatting, duplication, or truncation prevents direct execution.",
                "3": "The code is close to parseable but requires nontrivial cleanup or reconstruction.",
                "4": "The code is parseable or nearly parseable with only minor harmless formatting artifacts.",
                "5": "The answer is clean, complete, directly parseable Python code.",
            },
            "positive_boundary": "Do not penalize semantic mistakes here if the code is syntactically clean and directly parseable.",
            "negative_boundary": "Penalize when formatting or structural damage prevents evaluating the intended implementation.",
            "rubric_generation_notes": "Use this as an early gating dimension before runtime or semantic scoring.",
        },
        "runtime_api_type_misuse": {
            "rubric_dimension": "Runtime API and Type Safety",
            "operational_definition": (
                "Evaluate whether the implementation can execute on ordinary valid inputs without predictable exceptions from "
                "API misuse, invalid operations, undefined names, or incompatible operand types. This category assumes the public "
                "interface exists but the body is fragile at runtime."
            ),
            "failure_mechanism": "The code calls operations with incompatible values, references unavailable names, or uses APIs with the wrong contract.",
            "common_manifestations": [
                "Applies list, string, dict, regex, math, or iterator APIs to incompatible values.",
                "Indexes, unpacks, casts, or calls values without validating the required shape or type.",
                "References helper names, imports, or variables that are not defined in the submitted code.",
            ],
            "judge_checklist": [
                "Trace whether each API call receives the value type and shape it expects.",
                "Check indexing, unpacking, iteration, casting, and method calls for predictable exceptions.",
                "Check that helper functions, imports, and variables are defined before use.",
                "Separate runtime fragility from wrong-value errors when execution completes cleanly.",
            ],
            "score_anchors": {
                "1": "The function predictably raises before performing the task on ordinary valid inputs.",
                "2": "The code has major undefined-name, import, API, or type-operation failures.",
                "3": "The implementation runs on some cases but has clear runtime fragility on common valid inputs.",
                "4": "The code is mostly runtime-safe with only narrow or unlikely exception risks.",
                "5": "The implementation uses APIs and value types coherently and executes safely on valid inputs.",
            },
            "positive_boundary": "Do not penalize a wrong returned value here if the code executes safely and the issue is semantic.",
            "negative_boundary": "Penalize when normal valid inputs can trigger exceptions due to type, API, name, import, indexing, or unpacking misuse.",
            "rubric_generation_notes": "Use this as the runtime reliability dimension after syntax and interface checks.",
        },
        "string_regex_pattern_logic": {
            "rubric_dimension": "String and Pattern-Matching Logic",
            "operational_definition": (
                "Evaluate whether string processing, substring search, tokenization, regular-expression construction, and match extraction "
                "follow the task's intended pattern semantics. Penalize code that uses the wrong literal, boundary, greediness, case handling, "
                "or extraction target."
            ),
            "failure_mechanism": "The solution misrepresents the pattern to be found or transforms text with incorrect matching semantics.",
            "common_manifestations": [
                "Uses an overly broad, overly narrow, or literal pattern where structured matching is required.",
                "Confuses match existence with match position, span, captured text, or replacement result.",
                "Mishandles case, whitespace, punctuation, token boundaries, or repeated matches.",
            ],
            "judge_checklist": [
                "Identify the exact string relation or pattern semantics required by the task.",
                "Check regex escaping, grouping, boundaries, greediness, and flags where applicable.",
                "Check whether the code returns the requested match artifact rather than an intermediate result.",
                "Check repeated, absent, overlapping, or boundary-position matches when relevant.",
            ],
            "score_anchors": {
                "1": "The string or regex logic is unrelated to the required pattern semantics.",
                "2": "The code searches or transforms text but targets the wrong pattern or artifact.",
                "3": "The pattern logic is plausible but misses important boundaries, repetitions, or extraction details.",
                "4": "The implementation is mostly correct with only narrow text-boundary weaknesses.",
                "5": "The implementation matches and returns exactly the requested string or pattern result.",
            },
            "positive_boundary": "Do not penalize under this category for non-string tasks or for formatting issues unrelated to pattern semantics.",
            "negative_boundary": "Penalize when text-processing code is executable but uses the wrong match, split, replace, or extraction rule.",
            "rubric_generation_notes": "Use this dimension for tasks centered on strings, regexes, substrings, tokenization, or textual extraction.",
        },
        "edge_case_boundary_handling": {
            "rubric_dimension": "Edge Case and Boundary Handling",
            "operational_definition": (
                "Evaluate whether the implementation preserves the intended behavior on small, empty, singleton, duplicate, zero, negative, "
                "limit, or otherwise boundary-shaped inputs. This category captures solutions whose main idea works but whose assumptions "
                "break near the edges of the input domain."
            ),
            "failure_mechanism": "The code hard-codes normal-case assumptions and lacks guards or generalized logic for boundary inputs.",
            "common_manifestations": [
                "Fails on empty or singleton containers, zero-length ranges, duplicates, or repeated values.",
                "Uses strict comparisons, loop bounds, or initialization values that exclude boundary cases.",
                "Special-cases one example while missing nearby variants implied by the task.",
            ],
            "judge_checklist": [
                "List the boundary input shapes implied by the task statement and public interface.",
                "Check initialization, loop bounds, comparisons, and guard clauses against those boundaries.",
                "Check duplicate, absent, minimal, maximal, and degenerate inputs where relevant.",
                "Separate narrow boundary failures from a generally wrong core algorithm.",
            ],
            "score_anchors": {
                "1": "The implementation only works for a narrow hard-coded normal case.",
                "2": "The code fails multiple common boundary shapes or lacks necessary guards.",
                "3": "The main idea works but at least one important boundary family is mishandled.",
                "4": "The implementation handles most boundaries with only rare edge-case risk.",
                "5": "The implementation consistently handles normal and boundary inputs implied by the task.",
            },
            "positive_boundary": "Do not penalize here when a failure reflects the wrong core algorithm rather than a boundary-specific weakness.",
            "negative_boundary": "Penalize when the solution is otherwise plausible but breaks on minimal, empty, duplicate, limit, or absent-case inputs.",
            "rubric_generation_notes": "Use this as an edge-coverage dimension for otherwise plausible implementations.",
        },
        "interface_name_signature_mismatch": {
            "rubric_dimension": "Public Interface and Signature Compliance",
            "operational_definition": (
                "Evaluate whether the submitted code defines the required public function or class with the expected name, callable form, "
                "argument count, and usable return path. This category is about whether the evaluator can invoke the intended entry point."
            ),
            "failure_mechanism": "The solution omits, renames, wraps, or changes the public interface required by the task.",
            "common_manifestations": [
                "Defines a helper or differently named function instead of the required entry point.",
                "Uses the wrong number or order of parameters, or requires extra input not in the public signature.",
                "Places the solution in a class, script body, or nested scope that hides the callable interface.",
            ],
            "judge_checklist": [
                "Check that the required public name is defined at module scope.",
                "Check that the callable accepts the expected argument count and compatible parameter structure.",
                "Check that the implementation returns through the public interface rather than only printing or scripting.",
                "Separate missing-interface failures from runtime errors inside a correctly named callable.",
            ],
            "score_anchors": {
                "1": "The required public callable is missing or impossible to invoke.",
                "2": "A related implementation exists but under the wrong name, scope, or callable form.",
                "3": "The public interface is present but has argument, wrapper, or invocation compatibility issues.",
                "4": "The interface is mostly compliant with only minor ambiguity or extra unused structure.",
                "5": "The code exposes exactly the required public name and compatible signature.",
            },
            "positive_boundary": "Do not penalize semantic mistakes here when the public interface is correctly exposed and callable.",
            "negative_boundary": "Penalize when evaluation cannot reach the intended solution because the name, scope, callable form, or signature is wrong.",
            "rubric_generation_notes": "Use this as an invocation-contract dimension before judging implementation behavior.",
        },
    }

    default = {
        "rubric_dimension": name,
        "operational_definition": (
            f"Evaluate the concrete code property represented by {name}. The judge should inspect the generated code "
            "for the category-specific failure mechanism rather than relying on broad error labels."
        ),
        "failure_mechanism": "The implementation violates a recurring task contract observed in the failure taxonomy.",
        "common_manifestations": [
            "The code is plausible but fails the concrete behavior represented by this category.",
            "The failure appears across multiple tasks rather than as a one-off artifact.",
        ],
        "judge_checklist": [
            "Identify the task contract relevant to this category.",
            "Trace the implementation against that contract.",
            "Check whether any mismatch would affect ordinary valid inputs.",
        ],
        "score_anchors": {
            "1": "The code clearly violates this category's core contract.",
            "2": "The code shows major weaknesses for this category.",
            "3": "The code partially satisfies this category but has important gaps.",
            "4": "The code mostly satisfies this category with narrow risk.",
            "5": "The code fully satisfies this category.",
        },
        "positive_boundary": "Do not penalize unrelated failures under this category.",
        "negative_boundary": "Penalize when the category-specific contract is visibly violated.",
        "rubric_generation_notes": "Convert this into a focused dimension with category-specific evidence and score anchors.",
    }
    return templates.get(category_id, default)


def mask_generic_phrases(value: Any) -> Any:
    """Hide rejected text before feeding a failed candidate back to the LLM."""
    if isinstance(value, dict):
        return {key: mask_generic_phrases(item) for key, item in value.items()}
    if isinstance(value, list):
        return [mask_generic_phrases(item) for item in value]
    if not isinstance(value, str):
        return value
    result = value
    for phrase in sorted(GENERIC_BAD_PHRASES, key=len, reverse=True):
        result = re.sub(re.escape(phrase), "[REJECTED_GENERIC_PHRASE]", result, flags=re.IGNORECASE)
    for token in sorted(PRIVATE_TOKENS, key=len, reverse=True):
        result = re.sub(re.escape(token), "[REJECTED_PRIVATE_TOKEN]", result, flags=re.IGNORECASE)
    return result


def build_revision_prompt(evidence: dict, rejected_candidate: dict, reject_reasons: list[str]) -> str:
    evidence_json = json.dumps(evidence, ensure_ascii=False, indent=2)
    candidate_json = json.dumps(mask_generic_phrases(rejected_candidate), ensure_ascii=False, indent=2)
    reasons_json = json.dumps(mask_generic_phrases(reject_reasons), ensure_ascii=False, indent=2)
    category_id = evidence["id"]
    quality_floor = category_template({"id": category_id, "name": evidence.get("current_name")})
    quality_floor_json = json.dumps(
        {
            "rubric_dimension": quality_floor["rubric_dimension"],
            "failure_mechanism": quality_floor["failure_mechanism"],
            "common_manifestations": quality_floor["common_manifestations"],
            "judge_checklist": quality_floor["judge_checklist"],
            "score_anchors": quality_floor["score_anchors"],
        },
        ensure_ascii=False,
        indent=2,
    )
    return f"""Your previous rubric refinement for category `{category_id}` was rejected by an automatic quality audit.

Rewrite it so it becomes a concrete, code-inspectable rubric dimension.

Reject reasons:
{reasons_json}

Rejected candidate:
{candidate_json}

Hard constraints:
- Keep id exactly `{category_id}`.
- Return ONLY one valid JSON object.
- Do not mention hidden tests, assert statements, exact expected values, private verifier details, or response ids.
- Do not use the rejected generic phrases. Replace them with code-mechanism language.
- Do not use schema placeholder text for positive_boundary or negative_boundary; write actual in-scope and out-of-scope conditions.
- Avoid starting definitions with generic "Ensures ..." phrasing; define the inspected code property directly.
- Do not use broad boundary phrases such as "correct type", "wrong type", "right value", "most inputs", "every input", or "specified by the task"; name the concrete contract property instead.
- Rubric dimension must be descriptive, not a one-word or two-word generic label.
- Use replacements such as "specified relation", "public contract", "semantic rule", "computed numeric content", "state update", "return container contract", or "pattern semantics".
- Score anchors must describe category-specific code properties, not counts of failed examples.
- Score 5 must mean the category is fully satisfied, not that the answer does extra work.
- Use the safe evidence and category-specific quality floor below.

Return JSON with this schema:
{{
  "id": "{category_id}",
  "rubric_dimension": "Short rubric dimension name",
  "operational_definition": "Two or three concrete sentences.",
  "failure_mechanism": "The causal mechanism behind this failure mode.",
  "common_manifestations": ["specific manifestation"],
  "judge_checklist": ["concrete check the judge can apply"],
  "score_anchors": {{"1": "...", "2": "...", "3": "...", "4": "...", "5": "..."}},
  "positive_boundary": "Concrete out-of-scope boundary written in category-specific code-property language.",
  "negative_boundary": "Concrete in-scope penalty boundary written in category-specific code-property language.",
  "rubric_generation_notes": "How to convert this category into a standalone rubric dimension.",
  "confidence": "high"
}}

Safe category evidence:
{evidence_json}

Category-specific quality floor:
{quality_floor_json}
"""


def build_targeted_repair_prompt(evidence: dict, rejected_candidate: dict, reject_reasons: list[str]) -> str:
    evidence_json = json.dumps(evidence, ensure_ascii=False, indent=2)
    candidate_json = json.dumps(mask_generic_phrases(rejected_candidate), ensure_ascii=False, indent=2)
    reasons_json = json.dumps(mask_generic_phrases(reject_reasons), ensure_ascii=False, indent=2)
    category_id = evidence["id"]
    quality_floor = category_template({"id": category_id, "name": evidence.get("current_name")})
    quality_floor_json = json.dumps(quality_floor, ensure_ascii=False, indent=2)
    return f"""Repair one rejected rubric-refinement JSON object for category `{category_id}`.

The automatic audit rejected only the candidate text quality, not the taxonomy assignment. Produce a complete replacement object that passes the audit.

Audit reject reasons with rejected phrases masked:
{reasons_json}

Masked candidate to repair:
{candidate_json}

Repair rules:
- Keep id exactly `{category_id}`.
- Return ONLY one valid JSON object.
- Do not reproduce any `[REJECTED_GENERIC_PHRASE]` or `[REJECTED_PRIVATE_TOKEN]` marker.
- Do not mention hidden tests, assert statements, exact verifier values, private verifier details, or response ids.
- Do not copy schema placeholder text into positive_boundary or negative_boundary.
- Write boundary fields as real category-specific conditions: positive_boundary says what belongs outside this dimension; negative_boundary says what concrete code property belongs inside it.
- Use at least two category-specific mechanism words from the safe evidence or quality floor.
- Score anchors must describe code properties for scores 1 through 5, not counts of examples or verifier outcomes.
- Prefer precise mechanism language over generic "Ensures ..." phrasing.
- Do not use broad boundary phrases such as "correct type", "wrong type", "right value", "most inputs", "every input", or "specified by the task"; name the concrete contract property instead.
- Rubric dimension must be descriptive, not a one-word or two-word generic label.

Required keys: id, rubric_dimension, operational_definition, failure_mechanism, common_manifestations, judge_checklist, score_anchors, positive_boundary, negative_boundary, rubric_generation_notes, confidence.

Safe category evidence:
{evidence_json}

Category-specific quality floor:
{quality_floor_json}
"""


def generic_phrase_hits(obj: Any, category_id: str | None = None) -> list[str]:
    text = json.dumps(obj, ensure_ascii=False).lower()
    allowed = CATEGORY_ALLOWED_GENERIC_PHRASES.get(str(category_id), set()) if category_id else set()
    return sorted(phrase for phrase in GENERIC_BAD_PHRASES if phrase not in allowed and phrase in text)


def has_generic_text(obj: Any, category_id: str | None = None) -> bool:
    return bool(generic_phrase_hits(obj, category_id))


def keyword_hits(category_id: str, obj: Any) -> list[str]:
    keywords = CATEGORY_KEYWORDS.get(category_id, set())
    text = json.dumps(obj, ensure_ascii=False).lower()
    return sorted(keyword for keyword in keywords if keyword in text)


def evaluate_refinement_candidate(candidate: dict | None, category_id: str) -> dict:
    reasons = []
    score = 100
    if not isinstance(candidate, dict):
        return {
            "accepted": False,
            "score": 0,
            "reject_reasons": ["candidate is not a JSON object"],
            "generic_phrase_hits": [],
            "category_keyword_hits": [],
        }

    if str(candidate.get("id")) != category_id:
        reasons.append(f"id mismatch: expected {category_id}, got {candidate.get('id')}")
        score -= 35

    dimension = str(candidate.get("rubric_dimension") or "").strip()
    if len(re.findall(r"[A-Za-z0-9]+", dimension)) < 3:
        reasons.append(f"rubric_dimension is too broad: {dimension!r}")
        score -= 20

    missing_fields = sorted(REQUIRED_REFINED_FIELDS - set(candidate))
    if missing_fields:
        reasons.append(f"missing required fields: {missing_fields}")
        score -= 7 * len(missing_fields)

    anchors = candidate.get("score_anchors")
    if not isinstance(anchors, dict):
        reasons.append("score_anchors is not an object")
        score -= 30
        missing_anchors = [str(score_id) for score_id in range(1, 6)]
    else:
        missing_anchors = [str(score_id) for score_id in range(1, 6) if str(score_id) not in anchors]
        if missing_anchors:
            reasons.append(f"missing score anchors: {missing_anchors}")
            score -= 8 * len(missing_anchors)

    manifestations = candidate.get("common_manifestations")
    if not isinstance(manifestations, list) or len([item for item in manifestations if str(item).strip()]) < 3:
        reasons.append("common_manifestations must contain at least 3 concrete items")
        score -= 15

    checklist = candidate.get("judge_checklist")
    if not isinstance(checklist, list) or len([item for item in checklist if str(item).strip()]) < 4:
        reasons.append("judge_checklist must contain at least 4 concrete checks")
        score -= 15

    phrase_hits = generic_phrase_hits(candidate, category_id)
    if phrase_hits:
        reasons.append(f"generic or test-facing phrases found: {phrase_hits}")
        score -= 10 * len(phrase_hits)

    leak_hits = leakage_flags(candidate)
    if leak_hits:
        reasons.append(f"private/test leakage terms found: {leak_hits}")
        score -= 40

    kw_hits = keyword_hits(category_id, candidate)
    if len(kw_hits) < 2:
        reasons.append(f"category-specific vocabulary too weak: found {kw_hits}")
        score -= 20

    text_lengths = [
        len(str(candidate.get("operational_definition") or "")),
        len(str(candidate.get("failure_mechanism") or "")),
        len(str(candidate.get("positive_boundary") or "")),
        len(str(candidate.get("negative_boundary") or "")),
    ]
    if min(text_lengths) < 40:
        reasons.append("definition/mechanism/boundary text is too short to be operational")
        score -= 10

    accepted = not reasons
    return {
        "accepted": accepted,
        "score": max(score, 0),
        "reject_reasons": reasons,
        "generic_phrase_hits": phrase_hits,
        "private_leakage_flags": leak_hits,
        "category_keyword_hits": kw_hits,
    }


def select_best_candidate(candidates: list[dict], category_id: str) -> tuple[dict | None, dict]:
    evaluated = []
    for index, candidate in enumerate(candidates):
        quality = evaluate_refinement_candidate(candidate, category_id)
        evaluated.append(
            {
                "index": index,
                "candidate": candidate,
                "quality": quality,
            }
        )
    accepted = [item for item in evaluated if item["quality"]["accepted"]]
    pool = accepted or evaluated
    if not pool:
        return None, {
            "accepted": False,
            "best_score": 0,
            "candidate_count": 0,
            "reject_reasons": ["no parseable candidates"],
        }
    best = max(pool, key=lambda item: item["quality"]["score"])
    audit = {
        "accepted": bool(accepted),
        "selected_index": best["index"],
        "best_score": best["quality"]["score"],
        "candidate_count": len(evaluated),
        "reject_reasons": best["quality"]["reject_reasons"],
        "generic_phrase_hits": best["quality"].get("generic_phrase_hits", []),
        "category_keyword_hits": best["quality"].get("category_keyword_hits", []),
    }
    return best["candidate"], audit


def leakage_flags(obj: Any) -> list[str]:
    text = json.dumps(obj, ensure_ascii=False).lower()
    return sorted(token for token in PRIVATE_TOKENS if token in text)


def sanitize_private_text(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: sanitize_private_text(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_private_text(item) for item in value]
    if not isinstance(value, str):
        return value
    replacements = {
        "hidden tests": "private verifier cases",
        "hidden test": "private verifier case",
        "hidden-tests": "private-verifier",
        "assert statements": "verifier checks",
        "assert statement": "verifier check",
        "asserts": "verifier checks",
        "assert ": "verifier check ",
        "assert(": "verifier_check(",
        "test_list": "verifier_cases",
        "test_setup_code": "verifier_setup",
        "private_diagnostics": "verifier_diagnostics",
    }
    result = value
    for old, new in replacements.items():
        result = re.sub(re.escape(old), new, result, flags=re.IGNORECASE)
    return result


def normalize_list(value: Any, fallback: list[str], minimum: int = 2) -> list[str]:
    if isinstance(value, list):
        items = [short_text(str(item), 220) for item in value if str(item).strip()]
    elif value:
        items = [short_text(str(value), 220)]
    else:
        items = []
    for item in fallback:
        if len(items) >= minimum:
            break
        if item not in items:
            items.append(item)
    return items


def normalize_score_anchors(value: Any, fallback: dict[str, str]) -> dict[str, str]:
    anchors = value if isinstance(value, dict) else {}
    return {
        str(score): short_text(str(anchors.get(str(score)) or anchors.get(score) or fallback[str(score)]), 260)
        for score in range(1, 6)
    }


def normalize_refinements(candidate: dict | None, taxonomy: dict) -> tuple[list[dict], dict]:
    by_id = {}
    for item in (candidate or {}).get("categories") or []:
        if isinstance(item, dict) and item.get("id"):
            by_id[str(item["id"])] = item

    categories = []
    repair = {
        "missing_refinements_repaired": [],
        "unknown_refinements_removed": sorted(set(by_id) - {str(cat.get("id")) for cat in taxonomy.get("categories") or []}),
        "generic_refinements_replaced": [],
        "private_text_sanitized": False,
    }
    if isinstance(candidate, dict) and candidate.get("_quality_audit"):
        repair["llm_quality_audit"] = candidate["_quality_audit"]

    for source in taxonomy.get("categories") or []:
        category_id = str(source.get("id"))
        template = category_template(source)
        refinement = by_id.get(category_id)
        if refinement is None:
            repair["missing_refinements_repaired"].append(category_id)
            refinement = template
        elif has_generic_text(refinement, category_id):
            repair["generic_refinements_replaced"].append(category_id)
            refinement = template

        refined = {
            **source,
            "description": short_text(str(refinement.get("operational_definition") or template["operational_definition"]), 700),
            "rubric_hint": short_text(str(refinement.get("rubric_generation_notes") or template["rubric_generation_notes"]), 500),
            "score_focus": "Use the category-specific 1-5 score anchors under refined_rubric.score_anchors.",
            "refined_rubric": {
                "rubric_dimension": short_text(str(refinement.get("rubric_dimension") or template["rubric_dimension"]), 120),
                "operational_definition": short_text(
                    str(refinement.get("operational_definition") or template["operational_definition"]),
                    700,
                ),
                "failure_mechanism": short_text(
                    str(refinement.get("failure_mechanism") or template["failure_mechanism"]),
                    360,
                ),
                "common_manifestations": normalize_list(
                    refinement.get("common_manifestations"),
                    template["common_manifestations"],
                    minimum=3,
                )[:6],
                "judge_checklist": normalize_list(
                    refinement.get("judge_checklist"),
                    template["judge_checklist"],
                    minimum=4,
                )[:6],
                "score_anchors": normalize_score_anchors(refinement.get("score_anchors"), template["score_anchors"]),
                "positive_boundary": short_text(
                    str(refinement.get("positive_boundary") or template["positive_boundary"]),
                    360,
                ),
                "negative_boundary": short_text(
                    str(refinement.get("negative_boundary") or template["negative_boundary"]),
                    360,
                ),
                "rubric_generation_notes": short_text(
                    str(refinement.get("rubric_generation_notes") or template["rubric_generation_notes"]),
                    420,
                ),
                "confidence": str(refinement.get("confidence") or "medium").lower(),
            },
        }
        refined = sanitize_private_text(refined)
        categories.append(refined)

    if leakage_flags(categories):
        repair["private_text_sanitized"] = True
        categories = sanitize_private_text(categories)
    return categories, repair


def object_to_candidate(obj: dict | None) -> dict:
    if not isinstance(obj, dict):
        return {"taxonomy_name": "mbpp_refined_rubric_operational_taxonomy_v1", "categories": []}
    if isinstance(obj.get("categories"), list):
        return obj
    if obj.get("id"):
        return {
            "taxonomy_name": "mbpp_refined_rubric_operational_taxonomy_v1",
            "categories": [obj],
        }
    return {"taxonomy_name": "mbpp_refined_rubric_operational_taxonomy_v1", "categories": []}


def raw_outputs_to_candidate(raw_outputs: list[dict]) -> dict:
    categories = []
    for item in raw_outputs:
        parsed = parse_json_object(str(item.get("text") or ""))
        candidate = object_to_candidate(parsed)
        for category in candidate.get("categories") or []:
            if isinstance(category, dict) and category.get("id"):
                categories.append(category)
    return {
        "taxonomy_name": "mbpp_refined_rubric_operational_taxonomy_v1",
        "categories": categories,
        "notes": "assembled from per-category raw LLM outputs",
    }


def raw_payload_to_candidate(payload: dict) -> dict:
    if payload.get("mode") == "per_category_v2":
        return {
            "taxonomy_name": "mbpp_refined_rubric_operational_taxonomy_v1",
            "categories": payload.get("selected_categories") or [],
            "_quality_audit": payload.get("selection_audit") or {},
            "notes": "assembled from quality-gated per-category LLM outputs",
        }
    if payload.get("mode") == "per_category":
        return raw_outputs_to_candidate(payload.get("raw_outputs") or [])
    return object_to_candidate(payload)


def extract_candidate_from_text(text: str, category_id: str) -> dict | None:
    parsed = parse_json_object(text)
    candidate = object_to_candidate(parsed)
    categories = [item for item in candidate.get("categories") or [] if isinstance(item, dict)]
    exact = next((item for item in categories if str(item.get("id")) == category_id), None)
    if exact is not None:
        return exact
    return categories[0] if categories else None


def validate_refined_categories(categories: list[dict], source_categories: list[dict]) -> dict:
    source_ids = [str(category.get("id")) for category in source_categories]
    refined_ids = [str(category.get("id")) for category in categories]
    duplicate_ids = sorted(category_id for category_id, count in Counter(refined_ids).items() if count > 1)
    missing_ids = sorted(set(source_ids) - set(refined_ids))
    unknown_ids = sorted(set(refined_ids) - set(source_ids))
    schema_flags = []
    generic_flags = []

    for category in categories:
        category_id = str(category.get("id"))
        refined = category.get("refined_rubric") or {}
        missing_fields = sorted(REQUIRED_REFINED_FIELDS - set(refined))
        missing_anchors = sorted(str(score) for score in range(1, 6) if str(score) not in (refined.get("score_anchors") or {}))
        if missing_fields or missing_anchors:
            schema_flags.append(
                {
                    "id": category_id,
                    "missing_fields": missing_fields,
                    "missing_anchors": missing_anchors,
                }
            )
        if has_generic_text(refined, category_id) or has_generic_text(
            {"description": category.get("description"), "rubric_hint": category.get("rubric_hint")},
            category_id,
        ):
            generic_flags.append(category_id)

    leak_flags = leakage_flags(categories)
    return {
        "source_category_count": len(source_categories),
        "refined_category_count": len(categories),
        "missing_category_ids": missing_ids,
        "duplicate_category_ids": duplicate_ids,
        "unknown_category_ids": unknown_ids,
        "schema_flags": schema_flags,
        "generic_text_flags": sorted(set(generic_flags)),
        "private_leakage_flags": leak_flags,
        "valid": not missing_ids and not duplicate_ids and not unknown_ids and not schema_flags and not generic_flags and not leak_flags,
    }


def build_refined_assignments(assignments: list[dict], categories: list[dict]) -> list[dict]:
    dimension_by_id = {
        str(category["id"]): category["refined_rubric"]["rubric_dimension"]
        for category in categories
        if category.get("refined_rubric")
    }
    rows = []
    for row in assignments:
        category_id = str(row.get("taxonomy_category_id"))
        rows.append(
            {
                **row,
                "rubric_dimension": dimension_by_id.get(category_id),
            }
        )
    return rows


def run_llm(args: argparse.Namespace, prompt: str) -> tuple[dict | None, str]:
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
        max_tokens=args.max_tokens,
        seed=42,
    )
    output = llm.generate([prompt], sampling)[0].outputs[0].text
    return parse_json_object(output), output


def run_llm_per_category(args: argparse.Namespace, evidence: list[dict]) -> tuple[dict, str]:
    from vllm import LLM, SamplingParams

    prompts = [build_single_category_prompt(item) for item in evidence]
    llm = LLM(
        model=args.model,
        tensor_parallel_size=1,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=max(
            args.max_num_seqs,
            args.candidates_per_category,
            args.revision_candidates,
            args.targeted_repair_candidates,
        ),
    )
    sampling = SamplingParams(
        n=args.candidates_per_category,
        temperature=max(args.temperature, 0.2) if args.candidates_per_category > 1 else args.temperature,
        max_tokens=args.max_tokens,
        seed=42,
    )
    outputs = llm.generate(prompts, sampling)
    initial_outputs = []
    selected_categories = []
    selection_audit: dict[str, Any] = {
        "initial_accepted_categories": [],
        "revised_accepted_categories": [],
        "targeted_repair_accepted_categories": [],
        "template_fallback_categories": [],
        "per_category": {},
    }
    revision_prompts = []
    revision_sources = []
    targeted_repair_prompts = []
    targeted_repair_sources = []

    for item, output in zip(evidence, outputs):
        category_id = item["id"]
        raw_candidates = []
        parsed_candidates = []
        for candidate_index, candidate_output in enumerate(output.outputs):
            text = candidate_output.text
            candidate = extract_candidate_from_text(text, category_id)
            raw_candidates.append(
                {
                    "candidate_index": candidate_index,
                    "text": text,
                    "parsed": candidate is not None,
                }
            )
            if candidate is not None:
                parsed_candidates.append(candidate)
        best_candidate, audit = select_best_candidate(parsed_candidates, category_id)
        initial_outputs.append(
            {
                "category_id": category_id,
                "candidates": raw_candidates,
                "selection": audit,
            }
        )
        if audit["accepted"] and best_candidate is not None:
            selected_categories.append(best_candidate)
            selection_audit["initial_accepted_categories"].append(category_id)
            selection_audit["per_category"][category_id] = {
                "source": "initial",
                **audit,
            }
            continue

        revision_prompts.append(build_revision_prompt(item, best_candidate or {"id": category_id}, audit["reject_reasons"]))
        revision_sources.append((item, best_candidate, audit))

    revision_outputs = []
    if revision_prompts and args.revision_candidates > 0:
        revision_sampling = SamplingParams(
            n=args.revision_candidates,
            temperature=max(args.temperature, 0.2) if args.revision_candidates > 1 else args.temperature,
            max_tokens=args.max_tokens,
            seed=43,
        )
        outputs = llm.generate(revision_prompts, revision_sampling)
        for (item, _best_initial, initial_audit), output in zip(revision_sources, outputs):
            category_id = item["id"]
            raw_candidates = []
            parsed_candidates = []
            for candidate_index, candidate_output in enumerate(output.outputs):
                text = candidate_output.text
                candidate = extract_candidate_from_text(text, category_id)
                raw_candidates.append(
                    {
                        "candidate_index": candidate_index,
                        "text": text,
                        "parsed": candidate is not None,
                    }
                )
                if candidate is not None:
                    parsed_candidates.append(candidate)
            best_candidate, audit = select_best_candidate(parsed_candidates, category_id)
            revision_outputs.append(
                {
                    "category_id": category_id,
                    "initial_reject_reasons": initial_audit["reject_reasons"],
                    "candidates": raw_candidates,
                    "selection": audit,
                }
            )
            if audit["accepted"] and best_candidate is not None:
                selected_categories.append(best_candidate)
                selection_audit["revised_accepted_categories"].append(category_id)
                selection_audit["per_category"][category_id] = {
                    "source": "revision",
                    "initial_reject_reasons": initial_audit["reject_reasons"],
                    **audit,
                }
            else:
                repair_candidate = best_candidate or _best_initial or {"id": category_id}
                targeted_repair_prompts.append(build_targeted_repair_prompt(item, repair_candidate, audit["reject_reasons"]))
                targeted_repair_sources.append((item, initial_audit, audit))

    targeted_repair_outputs = []
    if targeted_repair_prompts and args.targeted_repair_candidates > 0:
        repair_sampling = SamplingParams(
            n=args.targeted_repair_candidates,
            temperature=max(args.temperature, 0.2) if args.targeted_repair_candidates > 1 else args.temperature,
            max_tokens=args.max_tokens,
            seed=44,
        )
        outputs = llm.generate(targeted_repair_prompts, repair_sampling)
        for (item, initial_audit, revision_audit), output in zip(targeted_repair_sources, outputs):
            category_id = item["id"]
            raw_candidates = []
            parsed_candidates = []
            for candidate_index, candidate_output in enumerate(output.outputs):
                text = candidate_output.text
                candidate = extract_candidate_from_text(text, category_id)
                raw_candidates.append(
                    {
                        "candidate_index": candidate_index,
                        "text": text,
                        "parsed": candidate is not None,
                    }
                )
                if candidate is not None:
                    parsed_candidates.append(candidate)
            best_candidate, audit = select_best_candidate(parsed_candidates, category_id)
            targeted_repair_outputs.append(
                {
                    "category_id": category_id,
                    "initial_reject_reasons": initial_audit["reject_reasons"],
                    "revision_reject_reasons": revision_audit["reject_reasons"],
                    "candidates": raw_candidates,
                    "selection": audit,
                }
            )
            if audit["accepted"] and best_candidate is not None:
                selected_categories.append(best_candidate)
                selection_audit["targeted_repair_accepted_categories"].append(category_id)
                selection_audit["per_category"][category_id] = {
                    "source": "targeted_repair",
                    "initial_reject_reasons": initial_audit["reject_reasons"],
                    "revision_reject_reasons": revision_audit["reject_reasons"],
                    **audit,
                }
            else:
                selection_audit["template_fallback_categories"].append(category_id)
                selection_audit["per_category"][category_id] = {
                    "source": "template_fallback",
                    "initial_reject_reasons": initial_audit["reject_reasons"],
                    "revision_reject_reasons": revision_audit["reject_reasons"],
                    "targeted_repair_reject_reasons": audit["reject_reasons"],
                    "initial_best_score": initial_audit["best_score"],
                    "revision_best_score": revision_audit["best_score"],
                    "targeted_repair_best_score": audit["best_score"],
                }

    for item, _best_initial, initial_audit in revision_sources:
        category_id = item["id"]
        if category_id in selection_audit["per_category"]:
            continue
        selection_audit["template_fallback_categories"].append(category_id)
        selection_audit["per_category"][category_id] = {
            "source": "template_fallback",
            "initial_reject_reasons": initial_audit["reject_reasons"],
            "initial_best_score": initial_audit["best_score"],
            "revision_reject_reasons": ["revision disabled or not attempted"],
        }

    raw_payload = {
        "mode": "per_category_v2",
        "candidates_per_category": args.candidates_per_category,
        "revision_candidates": args.revision_candidates,
        "targeted_repair_candidates": args.targeted_repair_candidates,
        "initial_outputs": initial_outputs,
        "revision_outputs": revision_outputs,
        "targeted_repair_outputs": targeted_repair_outputs,
        "selected_categories": selected_categories,
        "selection_audit": selection_audit,
    }
    return raw_payload_to_candidate(raw_payload), json.dumps(raw_payload, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Refine a consolidated taxonomy into rubric-operational dimensions.")
    parser.add_argument("--taxonomy", type=Path, required=True)
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--failures", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--response-assignments-output", type=Path)
    parser.add_argument("--raw-llm-output", type=Path)
    parser.add_argument("--existing-llm-output", type=Path, help="Parse an existing raw LLM response instead of calling vLLM.")
    parser.add_argument("--model", type=str, default="models/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28")
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.25)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--candidates-per-category", type=int, default=3)
    parser.add_argument("--revision-candidates", type=int, default=2)
    parser.add_argument("--targeted-repair-candidates", type=int, default=2)
    parser.add_argument("--max-examples-per-category", type=int, default=5)
    parser.add_argument("--single-prompt", action="store_true", help="Ask for all category refinements in one prompt.")
    parser.add_argument("--skip-llm", action="store_true", help="Use deterministic rubric-operational templates only.")
    args = parser.parse_args()

    taxonomy = load_yaml(args.taxonomy)
    source_categories = taxonomy.get("categories") or []
    if not source_categories:
        raise ValueError(f"No categories found in {args.taxonomy}")
    assignments = read_jsonl(args.assignments)
    failure_lookup = build_failure_lookup(args.failures)

    parsed: dict | None = None
    raw_llm_text = ""
    used_deterministic_fallback = False
    if args.existing_llm_output:
        raw_llm_text = args.existing_llm_output.read_text(encoding="utf-8")
        parsed = parse_json_object(raw_llm_text)
        if isinstance(parsed, dict) and parsed.get("mode") in {"per_category", "per_category_v2"}:
            parsed = raw_payload_to_candidate(parsed)
        else:
            parsed = object_to_candidate(parsed)
    elif not args.skip_llm:
        evidence = build_category_evidence(taxonomy, assignments, failure_lookup, args.max_examples_per_category)
        if args.single_prompt:
            prompt = build_refinement_prompt(evidence)
            parsed, raw_llm_text = run_llm(args, prompt)
            parsed = object_to_candidate(parsed)
        else:
            parsed, raw_llm_text = run_llm_per_category(args, evidence)
    if parsed is None:
        used_deterministic_fallback = True
        parsed = {"taxonomy_name": "mbpp_refined_rubric_operational_taxonomy_v1", "categories": []}

    refined_categories, repair_audit = normalize_refinements(parsed, taxonomy)
    validation = validate_refined_categories(refined_categories, source_categories)
    rows_by_category = Counter(str(row.get("taxonomy_category_id")) for row in assignments)
    audit = {
        "source_taxonomy": str(args.taxonomy),
        "source_assignments": str(args.assignments),
        "method": "llm_taxonomy_refinement_with_deterministic_schema_and_hygiene_repair",
        "model": None if args.skip_llm else args.model,
        "used_deterministic_fallback": used_deterministic_fallback,
        "assignment_count": len(assignments),
        "assignment_category_counts": dict(rows_by_category.most_common()),
        "repair_audit": repair_audit,
        **validation,
    }

    output_taxonomy = {
        "name": parsed.get("taxonomy_name") or "mbpp_refined_rubric_operational_taxonomy_v1",
        "source_taxonomy": str(args.taxonomy),
        "source_assignments": str(args.assignments),
        "source_method": taxonomy.get("method"),
        "method": "LLM rubric-operational refinement with deterministic audit",
        "total_raw_clusters": taxonomy.get("total_raw_clusters"),
        "total_failures": taxonomy.get("total_failures"),
        "total_tasks": taxonomy.get("total_tasks"),
        "num_categories": len(refined_categories),
        "categories": refined_categories,
        "cluster_mapping": taxonomy.get("cluster_mapping") or [],
        "audit": audit,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(output_taxonomy, allow_unicode=True, sort_keys=False), encoding="utf-8")
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.raw_llm_output:
        args.raw_llm_output.parent.mkdir(parents=True, exist_ok=True)
        args.raw_llm_output.write_text(raw_llm_text, encoding="utf-8")
    if args.response_assignments_output:
        write_jsonl(args.response_assignments_output, build_refined_assignments(assignments, refined_categories))

    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
