#!/usr/bin/env python3
"""Create the v3 correctness-first rubric overlay."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any


SEMANTIC_IDS = {
    "numeric_formula_correctness",
    "output_type_container_shape",
    "algorithmic_wrong_value",
    "edge_case_boundary_handling",
    "string_regex_pattern_logic",
}
ALWAYS_APPLICABLE_IDS = {
    "output_type_container_shape",
    "algorithmic_wrong_value",
    "interface_name_signature_mismatch",
    "syntax_parseability_or_output_format",
    "runtime_api_type_misuse",
}
SEMANTIC_ANCHORS = {
    "1": "The program is invalid, unusable, unrelated, or contradicts the core task on ordinary inputs.",
    "2": "A concrete visible input produces a wrong value, shape/type, exception, nontermination, or violated public contract; real edge-case failures also score 2.",
    "3": "No concrete failure is established, but correctness remains materially unresolved or ambiguous; this score is not a pass.",
    "4": "All required probes are consistent and the code trace supports the core rule; only a non-correctness concern or minor residual uncertainty remains.",
    "5": "Code-specific evidence justifies the core rule and relevant boundaries, all required probes are consistent, and no known correctness issue remains.",
}


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a semantic-bottleneck rubric that prevents score-4/5 inflation.")
    parser.add_argument("--base-rubric", type=Path, required=True)
    parser.add_argument("--guidance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    args = parser.parse_args()

    base = json.loads(args.base_rubric.read_text(encoding="utf-8"))
    guidance = json.loads(args.guidance.read_text(encoding="utf-8"))
    rubric = copy.deepcopy(base)
    rubric["name"] = f"{base.get('name', 'coding_rubric')}_score_collapse_fix_v3"
    rubric["base_rubric"] = str(args.base_rubric)
    rubric["human_guidance_source"] = str(args.guidance)
    rubric["human_guidance_name"] = guidance.get("name")
    rubric["revision_policy"] = {
        "type": "correctness_first_score_collapse_fix",
        "required_judging_sequence": guidance.get("required_judging_sequence") or [],
        "probe_policy": guidance.get("probe_policy") or {},
        "score_policy": guidance.get("score_policy") or {},
        "bad_rubric_patterns": guidance.get("bad_rubric_patterns") or [],
    }

    rewritten_anchors = []
    for dimension in rubric.get("dimensions", []):
        dimension_id = str(dimension.get("id") or "")
        dimension["aggregation_role"] = "semantic_bottleneck" if dimension_id in SEMANTIC_IDS else "structural_gate"
        dimension["always_applicable"] = dimension_id in ALWAYS_APPLICABLE_IDS
        dimension["critical_failure"] = True
        if dimension_id in SEMANTIC_IDS:
            dimension["score_anchors"] = dict(SEMANTIC_ANCHORS)
            rewritten_anchors.append(dimension_id)
        dimension["known_error_cap"] = 2
        dimension["score_four_five_requirement"] = (
            "Scores 4 and 5 require all ordinary, boundary, and adversarial probes to be consistent. "
            "Any known wrong output, exception, or contract violation caps this dimension at 2."
        )

    rubric["aggregation"] = {
        "method": "semantic_bottleneck_with_structural_gates",
        "primary_reward": "minimum applicable semantic score",
        "quality_diagnostic": "mean applicable dimension score",
        "semantic_dimension_ids": sorted(SEMANTIC_IDS),
        "always_applicable_ids": sorted(ALWAYS_APPLICABLE_IDS),
        "critical_dimension_ids": [str(dimension["id"]) for dimension in rubric.get("dimensions", [])],
        "not_applicable_score": 3,
        "pass_threshold": 4.0,
        "known_error_score_cap": 2,
        "semantic_score_three_is_not_a_pass": True,
        "decision_rule": guidance.get("critical_policy", {}).get("decision_rule"),
    }

    audit = {
        "valid": bool(rubric.get("dimensions")) and len(rewritten_anchors) == len(SEMANTIC_IDS),
        "method": "correctness_first_human_guidance_overlay",
        "base_rubric": str(args.base_rubric),
        "guidance": str(args.guidance),
        "output_rubric": str(args.output),
        "dimension_count": len(rubric.get("dimensions", [])),
        "semantic_dimension_ids": sorted(SEMANTIC_IDS),
        "always_applicable_ids": sorted(ALWAYS_APPLICABLE_IDS),
        "rewritten_anchor_dimensions": sorted(rewritten_anchors),
        "aggregation_method": rubric["aggregation"]["method"],
        "known_error_score_cap": 2,
        "pass_threshold": 4.0,
    }
    write_json(args.output, rubric)
    write_json(args.audit_output, audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
