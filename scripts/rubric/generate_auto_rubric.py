#!/usr/bin/env python3
"""Generate task-specific rubrics from the refined error taxonomy."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import yaml


DIMENSION_SPECS = [
    {
        "id": "functional_correctness",
        "dimension": "Functional Correctness and Edge-Case Coverage",
        "description": "Evaluate whether the code implements the required behavior and passes representative edge cases.",
        "linked_patterns": ["logic_wrong_output"],
        "score_criteria": {
            "1": "The implementation is mostly unrelated to the task or fails core examples.",
            "2": "The implementation captures a fragment of the task but fails important normal cases.",
            "3": "The main idea is plausible but edge cases or multiple tests fail.",
            "4": "The implementation is mostly correct, with only narrow edge-case weaknesses.",
            "5": "The implementation satisfies the task and tested edge cases.",
        },
    },
    {
        "id": "syntax_parseability",
        "dimension": "Syntax Validity and Parseability",
        "description": "Check whether the generated Python can be parsed and executed as code.",
        "linked_patterns": [
            "syntax_malformed_code",
            "syntax_duplicate_function_after_return",
            "syntax_truncated_or_unclosed_block",
            "syntax_unexpected_indent",
            "syntax_markdown_or_fence_artifact",
        ],
        "score_criteria": {
            "1": "The code cannot be parsed due to severe syntax errors.",
            "2": "The code contains duplicated fragments, unclosed blocks, or indentation errors.",
            "3": "The code is close to parseable but requires nontrivial cleanup.",
            "4": "The code parses after minor formatting cleanup.",
            "5": "The code parses directly as valid Python.",
        },
    },
    {
        "id": "interface_contract_compliance",
        "dimension": "Interface and Test Contract Compliance",
        "description": "Check whether the solution defines the expected function/class names and uses the required signature.",
        "linked_patterns": ["runtime_name_error", "runtime_attribute_error"],
        "score_criteria": {
            "1": "Expected functions/classes are missing or unusable.",
            "2": "The interface is partially present but incompatible with tests.",
            "3": "The interface is mostly present but has naming/signature inconsistencies.",
            "4": "The expected interface is present with minor risk.",
            "5": "The expected interface exactly matches the tests.",
        },
    },
    {
        "id": "runtime_dependency_safety",
        "dimension": "Runtime Dependency and API Safety",
        "description": "Check whether code avoids undefined dependencies, API misuse, and predictable runtime exceptions.",
        "linked_patterns": [
            "runtime_name_error",
            "runtime_type_error",
            "runtime_index_error",
            "runtime_key_error",
            "runtime_value_error",
            "runtime_import_error",
            "runtime_zerodivision_error",
            "runtime_other_exception",
        ],
        "score_criteria": {
            "1": "The code predictably raises exceptions on normal tests.",
            "2": "The code has missing imports, undefined names, or serious API misuse.",
            "3": "The code has possible runtime fragility on some inputs.",
            "4": "The code is mostly runtime-safe with minor concerns.",
            "5": "The code has no obvious runtime dependency or API safety issue.",
        },
    },
    {
        "id": "termination_complexity",
        "dimension": "Termination and Complexity Control",
        "description": "Check whether the implementation terminates and avoids clearly excessive algorithms.",
        "linked_patterns": ["timeout_nonterminating_or_too_slow"],
        "score_criteria": {
            "1": "The code times out or appears non-terminating.",
            "2": "The code likely has severe complexity issues.",
            "3": "The complexity may be acceptable only for small examples.",
            "4": "The algorithm is likely efficient enough for typical tests.",
            "5": "The algorithm clearly terminates and uses appropriate complexity.",
        },
    },
    {
        "id": "output_format_cleanliness",
        "dimension": "Output Cleanliness and Single-Solution Formatting",
        "description": "Check whether the answer contains only one clean Python solution without prose, Markdown, or repeated bodies.",
        "linked_patterns": ["syntax_duplicate_function_after_return", "syntax_markdown_or_fence_artifact"],
        "score_criteria": {
            "1": "The output is not usable code or is dominated by prose/formatting artifacts.",
            "2": "The output contains repeated functions, Markdown, or explanatory text that breaks execution.",
            "3": "The output is usable after cleanup but contains distracting artifacts.",
            "4": "The output is clean except for minor formatting issues.",
            "5": "The output is a single clean Python solution.",
        },
    },
]


GENERIC_RUBRIC = [
    {
        "id": "generic_correctness",
        "dimension": "Correctness",
        "description": "Whether the answer is correct.",
        "linked_patterns": [],
        "score_criteria": {"1": "Incorrect.", "2": "Mostly incorrect.", "3": "Partially correct.", "4": "Mostly correct.", "5": "Correct."},
    },
    {
        "id": "generic_clarity",
        "dimension": "Clarity",
        "description": "Whether the solution is clear and readable.",
        "linked_patterns": [],
        "score_criteria": {"1": "Unclear.", "2": "Hard to follow.", "3": "Somewhat clear.", "4": "Clear.", "5": "Very clear."},
    },
    {
        "id": "generic_completeness",
        "dimension": "Completeness",
        "description": "Whether the solution is complete.",
        "linked_patterns": [],
        "score_criteria": {"1": "Incomplete.", "2": "Major gaps.", "3": "Some gaps.", "4": "Mostly complete.", "5": "Complete."},
    },
]


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def collect_examples(taxonomy: dict) -> dict[str, list[dict]]:
    examples: dict[str, list[dict]] = {}
    for cluster in taxonomy.get("patterns", []):
        for pattern in cluster.get("rule_patterns", {}) or [cluster.get("name")]:
            examples.setdefault(pattern, [])
            examples[pattern].extend(cluster.get("examples", [])[:2])
    return examples


def enrich_dimension(spec: dict, example_map: dict[str, list[dict]]) -> dict:
    positive = "A concise solution that defines the expected interface, parses as Python, runs safely, and passes the tests."
    negative_examples = []
    for pattern in spec["linked_patterns"]:
        negative_examples.extend(example_map.get(pattern, [])[:2])
    return {
        **spec,
        "positive_example": positive,
        "negative_examples": negative_examples[:4],
        "source": "auto_generated_from_refined_error_taxonomy",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--taxonomy", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/rubrics/auto_rubric_refined.json"))
    parser.add_argument("--generic-output", type=Path, default=Path("data/rubrics/generic_rubric.json"))
    parser.add_argument("--random-output", type=Path, default=Path("data/rubrics/random_rubric_ablation.json"))
    args = parser.parse_args()

    taxonomy = load_yaml(args.taxonomy)
    example_map = collect_examples(taxonomy)
    auto_rubric = {
        "name": "auto_rubric_refined_coding_v1",
        "source_taxonomy": str(args.taxonomy),
        "task": "Python code generation on MBPP and HumanEval+",
        "dimensions": [enrich_dimension(spec, example_map) for spec in DIMENSION_SPECS],
    }

    all_patterns = [pattern for spec in DIMENSION_SPECS for pattern in spec["linked_patterns"]]
    shuffled = all_patterns[:]
    random.Random(42).shuffle(shuffled)
    random_dimensions = []
    offset = 0
    for spec in DIMENSION_SPECS:
        count = len(spec["linked_patterns"])
        random_dimensions.append(enrich_dimension({**spec, "linked_patterns": shuffled[offset : offset + count]}, example_map))
        offset += count
    random_rubric = {
        "name": "random_rubric_ablation_coding_v1",
        "source_taxonomy": str(args.taxonomy),
        "task": "Python code generation on MBPP and HumanEval+",
        "dimensions": random_dimensions,
    }
    generic_rubric = {
        "name": "generic_rubric_coding_v1",
        "source_taxonomy": None,
        "task": "Python code generation on MBPP and HumanEval+",
        "dimensions": GENERIC_RUBRIC,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(auto_rubric, ensure_ascii=False, indent=2), encoding="utf-8")
    args.generic_output.write_text(json.dumps(generic_rubric, ensure_ascii=False, indent=2), encoding="utf-8")
    args.random_output.write_text(json.dumps(random_rubric, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(auto_rubric['dimensions'])} dimensions to {args.output}")
    print(f"wrote generic rubric to {args.generic_output}")
    print(f"wrote random ablation rubric to {args.random_output}")


if __name__ == "__main__":
    main()
