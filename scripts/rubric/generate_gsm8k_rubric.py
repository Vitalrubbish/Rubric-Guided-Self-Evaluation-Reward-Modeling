#!/usr/bin/env python3
"""Generate a GSM8K rubric from the discovered failure taxonomy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


DIMENSIONS = {
    "problem_modeling": {
        "name": "Problem modeling",
        "linked_patterns": ["wrong_problem_model"],
        "positive_example": "Correctly identifies all quantities, rates, and relationships before computing.",
        "negative_example": "Combines unrelated quantities or solves for the wrong target value.",
    },
    "calculation_accuracy": {
        "name": "Calculation accuracy",
        "linked_patterns": ["arithmetic_or_algebra_slip"],
        "positive_example": "All arithmetic steps preserve the intended equation and reach the correct number.",
        "negative_example": "The setup is plausible but one addition, multiplication, or simplification is wrong.",
    },
    "stepwise_reasoning_completeness": {
        "name": "Stepwise reasoning completeness",
        "linked_patterns": ["reasoning_truncation", "unclassified_wrong_answer"],
        "positive_example": "The solution states enough intermediate steps to audit the final answer.",
        "negative_example": "The solution jumps to a number or stops before resolving the computation.",
    },
    "final_answer_format": {
        "name": "Final answer format",
        "linked_patterns": ["missing_final_answer", "final_format_violation", "ambiguous_final_answer"],
        "positive_example": "The final line is exactly formatted as #### 42.",
        "negative_example": "The response gives several numbers but no clearly marked final answer.",
    },
}


def score_levels() -> dict[str, str]:
    return {
        "1": "Absent or actively misleading for this dimension.",
        "2": "Major flaw likely to make the final answer wrong or unverifiable.",
        "3": "Partially correct but missing important checks or clarity.",
        "4": "Mostly correct with minor risk or limited explanation.",
        "5": "Correct, explicit, and easy to verify.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--taxonomy", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/rubrics/gsm8k_auto_rubric.json"))
    args = parser.parse_args()

    taxonomy = yaml.safe_load(args.taxonomy.read_text(encoding="utf-8"))
    observed = {pattern["id"] for pattern in taxonomy.get("patterns", [])}
    dimensions = []
    for dim_id, spec in DIMENSIONS.items():
        linked = [pattern for pattern in spec["linked_patterns"] if pattern in observed]
        if linked or dim_id in {
            "problem_modeling",
            "calculation_accuracy",
            "stepwise_reasoning_completeness",
            "final_answer_format",
        }:
            dimensions.append({
                "id": dim_id,
                "name": spec["name"],
                "description": f"Evaluates {spec['name'].lower()} for GSM8K solutions.",
                "score_1_to_5": score_levels(),
                "linked_patterns": spec["linked_patterns"],
                "positive_example": spec["positive_example"],
                "negative_example": spec["negative_example"],
            })

    rubric = {
        "name": "GSM8K failure-derived self-evaluation rubric",
        "dataset": "gsm8k",
        "source_taxonomy": str(args.taxonomy),
        "generation_method": "programmatic rubric generation from discovered failure taxonomy",
        "dimensions": dimensions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rubric, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "num_dimensions": len(dimensions)}, indent=2))


if __name__ == "__main__":
    main()
