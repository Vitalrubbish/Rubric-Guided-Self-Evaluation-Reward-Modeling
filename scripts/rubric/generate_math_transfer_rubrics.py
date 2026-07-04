#!/usr/bin/env python3
"""Create generic and MATH-derived rubrics for transfer comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


LEVELS = {
    "1": "Absent or incorrect.",
    "2": "Major flaw likely to invalidate the answer.",
    "3": "Partially correct but incomplete or weakly justified.",
    "4": "Mostly correct with minor risk.",
    "5": "Correct, explicit, and easy to verify.",
}


GENERIC_DIMS = [
    ("problem_modeling", "Problem modeling", ["wrong_problem_model"]),
    ("calculation_accuracy", "Calculation and symbolic accuracy", ["symbolic_or_arithmetic_error"]),
    ("stepwise_reasoning_completeness", "Reasoning completeness", ["reasoning_truncation"]),
    ("final_answer_format", "Final answer clarity", ["missing_final_answer", "ambiguous_final_answer"]),
]

MATH_DIMS = [
    ("math_problem_modeling", "MATH problem modeling", ["wrong_problem_model"]),
    ("symbolic_transformation_accuracy", "Symbolic transformation accuracy", ["symbolic_or_arithmetic_error"]),
    ("answer_equivalence_and_simplification", "Answer equivalence and simplification", ["symbolic_or_arithmetic_error"]),
    ("reasoning_completeness", "Reasoning completeness", ["reasoning_truncation"]),
    ("final_answer_format", "Final answer format", ["missing_final_answer", "ambiguous_final_answer"]),
]


def make_rubric(name: str, dataset: str, dims: list[tuple[str, str, list[str]]], source_taxonomy: str | None) -> dict:
    return {
        "name": name,
        "dataset": dataset,
        "source_taxonomy": source_taxonomy,
        "dimensions": [
            {
                "id": dim_id,
                "name": dim_name,
                "description": f"Evaluates {dim_name.lower()} for MATH solutions.",
                "score_1_to_5": LEVELS,
                "linked_patterns": linked,
                "positive_example": "The response states the mathematical setup, computes carefully, and gives a clear final answer.",
                "negative_example": "The response contains an unclear or mathematically non-equivalent final answer.",
            }
            for dim_id, dim_name, linked in dims
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--taxonomy", type=Path, required=True)
    parser.add_argument("--generic-output", type=Path, default=Path("data/rubrics/math_transfer_generic_rubric.json"))
    parser.add_argument("--math-output", type=Path, default=Path("data/rubrics/math_transfer_derived_rubric.json"))
    args = parser.parse_args()

    taxonomy = yaml.safe_load(args.taxonomy.read_text(encoding="utf-8"))
    observed = {p["id"] for p in taxonomy.get("patterns", [])}
    math_dims = []
    for dim_id, dim_name, linked in MATH_DIMS:
        if dim_id in {"math_problem_modeling", "symbolic_transformation_accuracy", "answer_equivalence_and_simplification", "final_answer_format"} or observed.intersection(linked):
            math_dims.append((dim_id, dim_name, linked))

    generic = make_rubric("Generic math rubric", "math_transfer", GENERIC_DIMS, None)
    derived = make_rubric("MATH failure-derived rubric", "math_transfer", math_dims, str(args.taxonomy))

    args.generic_output.parent.mkdir(parents=True, exist_ok=True)
    args.generic_output.write_text(json.dumps(generic, indent=2, ensure_ascii=False), encoding="utf-8")
    args.math_output.parent.mkdir(parents=True, exist_ok=True)
    args.math_output.write_text(json.dumps(derived, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "generic_output": str(args.generic_output),
        "math_output": str(args.math_output),
        "generic_dims": len(generic["dimensions"]),
        "math_dims": len(derived["dimensions"]),
    }, indent=2))


if __name__ == "__main__":
    main()
