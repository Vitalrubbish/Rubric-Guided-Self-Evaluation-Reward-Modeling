#!/usr/bin/env python3
"""Regression tests for v3 score-collapse prevention."""

from __future__ import annotations

import json
import copy
import unittest
from pathlib import Path

from src.rubric.evaluate_llm_rubric_judge import compute_metrics, repair_judgment


ROOT = Path(__file__).resolve().parents[1]
RUBRIC = json.loads(
    (ROOT / "data/rubrics/phase2/mbpp_hidden_llm_rubric_hitl_v3.json").read_text(encoding="utf-8")
)
DIMENSION_IDS = [str(item["id"]) for item in RUBRIC["dimensions"]]
ROW = {
    "id": "unit/task",
    "response_id": "unit/task__sample0",
    "dataset": "mbpp",
    "split": "validation",
    "prompt": (
        "Task: Return the input integer unchanged.\n\n"
        "Define code matching this public interface:\n- def identity(x)\n"
    ),
    "interface_names": ["identity"],
    "interface_signatures": ["def identity(x)"],
    "generated_code": "def identity(x):\n    return x",
    "extracted_code": "def identity(x):\n    return x",
    "passed": True,
}


def scores(value: int = 5, applicable: bool = True) -> dict:
    return {
        dimension_id: {
            "applicable": applicable,
            "score": value,
            "rationale": f"code-specific rationale for {dimension_id}",
            "counterexample": "probe considered",
        }
        for dimension_id in DIMENSION_IDS
    }


def probes(inconsistent: bool = False) -> list[dict]:
    result = []
    for kind in ["ordinary", "boundary", "adversarial"]:
        is_bad = inconsistent and kind == "adversarial"
        result.append(
            {
                "kind": kind,
                "input": "identity(2)",
                "expected_behavior": "returns 2",
                "observed_behavior": "returns 3" if is_bad else "returns 2",
                "consistent": not is_bad,
                "affected_dimensions": ["algorithmic_wrong_value"],
            }
        )
    return result


def traced_probes() -> list[dict]:
    result = []
    for kind, value in [("ordinary", 2), ("boundary", 0), ("adversarial", -3)]:
        result.append(
            {
                "kind": kind,
                "input": f"identity({value})",
                "expected_behavior": f"returns {value}",
                "observed_behavior": f"the return x statement evaluates to {value}",
                "consistent": True,
                "affected_dimensions": ["algorithmic_wrong_value"],
            }
        )
    return result


def parsed_judgment(*, score_value: int = 5, test_probes: list[dict] | None = None) -> dict:
    value = {
        "specification": {
            "input_contract": "one integer",
            "output_contract": "the same integer",
            "core_rule": "return x",
            "boundary_cases": ["zero"],
        },
        "critical_errors": [],
        "dimension_scores": scores(score_value),
        "overall_score": float(score_value),
        "predicted_pass": True,
        "confidence": "high",
    }
    if test_probes is not None:
        value["test_probes"] = test_probes
    return value


class ScoreCollapseFixTests(unittest.TestCase):
    def repair(self, parsed: dict) -> tuple[dict, dict]:
        return repair_judgment(
            parsed,
            json.dumps(parsed),
            ROW,
            RUBRIC,
            strict_prediction=True,
            require_test_probes=True,
        )

    def test_missing_probes_cannot_pass_even_with_all_fives(self) -> None:
        judgment, repair = self.repair(parsed_judgment())
        self.assertTrue(repair["missing_required_test_probes"])
        self.assertFalse(judgment["predicted_pass"])
        self.assertLessEqual(judgment["overall_score"], 3.0)
        self.assertTrue(
            all(
                judgment["dimension_scores"][dimension_id]["score"] <= 3
                for dimension_id in RUBRIC["aggregation"]["semantic_dimension_ids"]
            )
        )

    def test_consistent_probes_allow_supported_high_score(self) -> None:
        judgment, repair = self.repair(parsed_judgment(test_probes=probes()))
        self.assertFalse(repair["missing_required_test_probes"])
        self.assertTrue(judgment["predicted_pass"])
        self.assertEqual(judgment["semantic_bottleneck_score"], 5.0)
        self.assertEqual(judgment["overall_score"], 5.0)

    def test_inconsistent_probe_caps_semantic_score_at_two(self) -> None:
        judgment, repair = self.repair(parsed_judgment(test_probes=probes(inconsistent=True)))
        self.assertTrue(repair["probe_adjustments"])
        self.assertEqual(judgment["dimension_scores"]["algorithmic_wrong_value"]["score"], 2)
        self.assertEqual(judgment["semantic_bottleneck_score"], 2.0)
        self.assertEqual(judgment["overall_score"], 2.0)
        self.assertFalse(judgment["predicted_pass"])

    def test_empty_probe_evidence_is_rejected(self) -> None:
        invalid = probes()
        invalid[1]["expected_behavior"] = ""
        judgment, repair = self.repair(parsed_judgment(test_probes=invalid))
        self.assertTrue(repair["missing_required_test_probes"])
        self.assertIn("probe 1 has empty expected_behavior", repair["invalid_test_probe_reasons"])
        self.assertEqual(judgment["semantic_bottleneck_score"], 3.0)
        self.assertFalse(judgment["predicted_pass"])

    def test_duplicate_probe_kind_is_rejected(self) -> None:
        invalid = probes()
        invalid[2]["kind"] = "ordinary"
        judgment, repair = self.repair(parsed_judgment(test_probes=invalid))
        self.assertTrue(repair["missing_required_test_probes"])
        self.assertTrue(any("adversarial" in reason for reason in repair["invalid_test_probe_reasons"]))
        self.assertFalse(judgment["predicted_pass"])

    def test_structural_fives_cannot_lift_semantic_two(self) -> None:
        parsed = parsed_judgment(test_probes=probes())
        parsed["dimension_scores"]["algorithmic_wrong_value"]["score"] = 2
        judgment, _ = self.repair(parsed)
        self.assertEqual(judgment["overall_score"], 2.0)
        self.assertGreater(judgment["quality_mean_score"], judgment["overall_score"])
        self.assertFalse(judgment["predicted_pass"])

    def test_always_applicable_dimensions_cannot_be_hidden(self) -> None:
        parsed = parsed_judgment(test_probes=probes())
        parsed["dimension_scores"]["algorithmic_wrong_value"]["applicable"] = False
        judgment, repair = self.repair(parsed)
        self.assertIn("algorithmic_wrong_value", repair["forced_applicable_dimensions"])
        self.assertTrue(judgment["dimension_scores"]["algorithmic_wrong_value"]["applicable"])

    def test_metrics_make_failed_high_score_collapse_visible(self) -> None:
        records = []
        for passed, overall, predicted in [(False, 5.0, True), (False, 4.0, True), (True, 5.0, True)]:
            dimension_scores = scores(int(overall))
            records.append(
                {
                    "passed": passed,
                    "overall_score": overall,
                    "predicted_pass": predicted,
                    "split": "validation",
                    "dimension_scores": dimension_scores,
                }
            )
        audit = {
            "prompt_leakage_count": 0,
            "json_parse_failed_count": 0,
            "repaired_record_count": 0,
            "used_visible_code_fallback_count": 0,
        }
        metrics = compute_metrics(records, RUBRIC, audit)
        failed = metrics["score_collapse"]["failed"]
        self.assertEqual(failed["overall_score_ge_4_rate"], 1.0)
        self.assertEqual(failed["overall_score_eq_5_rate"], 0.5)
        self.assertEqual(failed["all_applicable_dimensions_eq_5_rate"], 0.5)

    def test_material_contract_ambiguity_forces_abstention(self) -> None:
        rubric = copy.deepcopy(RUBRIC)
        rubric["aggregation"]["material_contract_ambiguity_action"] = (
            "overall_score<=3; predicted_pass=false; route_to_human"
        )
        parsed = parsed_judgment(test_probes=traced_probes())
        parsed["specification"]["material_ambiguities"] = ["return sentinel is unspecified"]
        judgment, repair = repair_judgment(
            parsed,
            json.dumps(parsed),
            ROW,
            rubric,
            strict_prediction=True,
            require_test_probes=True,
        )
        self.assertTrue(repair["material_ambiguity_adjustments"])
        self.assertEqual(judgment["semantic_bottleneck_score"], 3.0)
        self.assertFalse(judgment["predicted_pass"])

    def test_placeholder_material_ambiguity_is_ignored(self) -> None:
        rubric = copy.deepcopy(RUBRIC)
        rubric["aggregation"]["material_contract_ambiguity_action"] = (
            "overall_score<=3; predicted_pass=false; route_to_human"
        )
        parsed = parsed_judgment(test_probes=traced_probes())
        parsed["specification"]["material_ambiguities"] = ["unresolved public-contract convention or empty"]
        judgment, repair = repair_judgment(
            parsed,
            json.dumps(parsed),
            ROW,
            rubric,
            strict_prediction=True,
            require_test_probes=True,
        )
        self.assertFalse(repair["material_ambiguity_adjustments"])
        self.assertEqual(repair["ignored_material_ambiguities"], ["unresolved public-contract convention or empty"])
        self.assertTrue(judgment["predicted_pass"])

    def test_weak_self_certifying_probes_force_rejection(self) -> None:
        rubric = copy.deepcopy(RUBRIC)
        rubric["aggregation"]["strict_probe_evidence_action"] = (
            "weak_probe_trace_evidence=>overall_score<=3; predicted_pass=false"
        )
        judgment, repair = repair_judgment(
            parsed_judgment(test_probes=probes()),
            json.dumps(parsed_judgment(test_probes=probes())),
            ROW,
            rubric,
            strict_prediction=True,
            require_test_probes=True,
        )
        self.assertTrue(repair["missing_required_test_probes"])
        self.assertTrue(any("weak trace evidence" in reason for reason in repair["invalid_test_probe_reasons"]))
        self.assertLessEqual(judgment["semantic_bottleneck_score"], 3.0)
        self.assertFalse(judgment["predicted_pass"])

    def test_code_traced_probes_allow_supported_high_score(self) -> None:
        rubric = copy.deepcopy(RUBRIC)
        rubric["aggregation"]["strict_probe_evidence_action"] = (
            "weak_probe_trace_evidence=>overall_score<=3; predicted_pass=false"
        )
        judgment, repair = repair_judgment(
            parsed_judgment(test_probes=traced_probes()),
            json.dumps(parsed_judgment(test_probes=traced_probes())),
            ROW,
            rubric,
            strict_prediction=True,
            require_test_probes=True,
        )
        self.assertFalse(repair["missing_required_test_probes"])
        self.assertEqual(judgment["overall_score"], 5.0)
        self.assertTrue(judgment["predicted_pass"])

    def test_execution_failure_gate_forces_failed_rows_to_fail(self) -> None:
        failed_row = copy.deepcopy(ROW)
        failed_row.update(
            {
                "passed": False,
                "failure_type": "logic_error",
                "error": "assertion failed",
                "safe_diagnostics": {
                    "diagnostic_kind": "wrong_output",
                    "test_count": 3,
                    "passed_assertions": 2,
                    "failed_assertions": 1,
                    "first_failure_kind": "wrong_value",
                },
            }
        )
        judgment, repair = repair_judgment(
            parsed_judgment(),
            json.dumps(parsed_judgment()),
            failed_row,
            RUBRIC,
            strict_prediction=True,
            require_test_probes=False,
            execution_gate="failures",
        )
        self.assertTrue(repair["execution_gate_adjustments"])
        self.assertTrue(repair["execution_gate_predicted_override"])
        self.assertFalse(judgment["predicted_pass"])
        self.assertTrue(judgment["critical_errors"])

    def test_execution_oracle_gate_can_override_llm_false_on_passed_rows(self) -> None:
        parsed = parsed_judgment()
        parsed["predicted_pass"] = False
        judgment, repair = repair_judgment(
            parsed,
            json.dumps(parsed),
            ROW,
            RUBRIC,
            strict_prediction=True,
            require_test_probes=False,
            execution_gate="oracle",
        )
        self.assertFalse(repair["execution_gate_adjustments"])
        self.assertTrue(repair["execution_gate_predicted_override"])
        self.assertTrue(judgment["predicted_pass"])


if __name__ == "__main__":
    unittest.main()
