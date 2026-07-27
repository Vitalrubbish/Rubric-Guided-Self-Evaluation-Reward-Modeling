from __future__ import annotations

import unittest

from src.self_play.build_executable_rubric_probe_input import extract_public_prompt
from src.self_play.executable_rubric_utils import all_passed, execute_function_tests
from src.self_play.extract_executable_rubric_tests import evaluate_generation, parse_tests
from src.self_play.score_executable_rubric_tests import confusion, score_candidate, select_suites


class ExecutableRubricInputTests(unittest.TestCase):
    def test_extract_public_prompt_stops_before_visible_suspect_code(self) -> None:
        prompt = (
            "Header\n"
            "Public task prompt:\n"
            "Solve the task.\n\n"
            "Visible suspect code:\n"
            "def solve():\n"
            "    return None\n"
        )

        self.assertEqual(extract_public_prompt(prompt), "Solve the task.")


class ExecutableRubricUtilsTests(unittest.TestCase):
    def test_execute_function_tests_supports_solution_class(self) -> None:
        result = execute_function_tests(
            "class Solution:\n    def add(self, x, y):\n        return x + y\n",
            "add",
            [
                {"args": [2, 3], "expected": 5},
                {"args": [-1, 1], "expected": 0},
            ],
            timeout=5.0,
        )

        self.assertTrue(all_passed(result, 2))


class ExecutableRubricExtractionTests(unittest.TestCase):
    def test_parse_tests_accepts_fenced_json_and_deduplicates(self) -> None:
        raw = """```json
{
  "fn_name": "solve",
  "tests": [
    {"args": [1], "expected": 2},
    {"args": [1], "expected": 2},
    {"args": [5], "expected": 6}
  ]
}
```"""

        cases, status, notes = parse_tests(raw, "solve", min_tests=2, max_tests=5)

        self.assertEqual(status, "ok")
        self.assertEqual(cases, [{"args": [1], "expected": 2}, {"args": [5], "expected": 6}])
        self.assertIn("dropped_duplicate_case", notes)

    def test_parse_tests_recovers_last_suite_after_leading_case_objects(self) -> None:
        raw = """{"args": [1], "expected": 2}
JSON: {"args": [5], "expected": 6}
{
  "fn_name": "solve",
  "tests": [
    {"args": [1], "expected": 2},
    {"args": [5], "expected": 6}
  ]
}
"""

        cases, status, notes = parse_tests(raw, "solve", min_tests=2, max_tests=5)

        self.assertEqual(status, "ok")
        self.assertEqual(cases, [{"args": [1], "expected": 2}, {"args": [5], "expected": 6}])
        self.assertIn("parse:json:balanced", notes)

    def test_evaluate_generation_requires_canonical_pass_and_failed_code_fail(self) -> None:
        source = {
            "id": "apps/train/1__exec_rubric_testgen",
            "source_row_id": "apps/train/1",
            "problem_id": "apps/train/1",
            "metadata": {"fn_name": "solve"},
            "canonical_solution": "def solve(x):\n    return x + 1\n",
            "failed_code": "def solve(x):\n    return x - 1\n",
        }
        generation = {
            "id": "apps/train/1__exec_rubric_testgen",
            "response_id": "suite-1",
            "generated_code": '{"fn_name": "solve", "tests": [{"args": [1], "expected": 2}, {"args": [5], "expected": 6}]}',
        }

        evaluated = evaluate_generation(generation, source, min_tests=2, max_tests=5, timeout=5.0)

        self.assertTrue(evaluated["quality_gate_passed"])
        self.assertTrue(evaluated["canonical_passed_all_tests"])
        self.assertTrue(evaluated["source_failure_caught"])
        self.assertEqual(evaluated["source_failed_pass_count"], 0)

    def test_evaluate_generation_filters_invalid_individual_tests(self) -> None:
        source = {
            "id": "apps/train/1__exec_rubric_testgen",
            "source_row_id": "apps/train/1",
            "problem_id": "apps/train/1",
            "metadata": {"fn_name": "solve"},
            "canonical_solution": "def solve(x):\n    return x + 1\n",
            "failed_code": "def solve(x):\n    return x - 1\n",
        }
        generation = {
            "id": "apps/train/1__exec_rubric_testgen",
            "response_id": "suite-1",
            "generated_code": (
                '{"fn_name": "solve", "tests": ['
                '{"args": [1], "expected": 2}, '
                '{"args": [5], "expected": 123}, '
                '{"args": [10], "expected": 11}'
                "]}"
            ),
        }

        evaluated = evaluate_generation(generation, source, min_tests=2, max_tests=5, timeout=5.0)

        self.assertTrue(evaluated["quality_gate_passed"])
        self.assertFalse(evaluated["canonical_passed_all_tests"])
        self.assertEqual(evaluated["raw_test_count"], 3)
        self.assertEqual(evaluated["test_count"], 2)
        self.assertEqual(evaluated["dropped_test_indices"], [1])
        self.assertEqual(evaluated["tests"], [{"args": [1], "expected": 2}, {"args": [10], "expected": 11}])


class ExecutableRubricScoringTests(unittest.TestCase):
    def test_select_suites_supports_best_and_union_aggregation(self) -> None:
        rows = [
            {
                "problem_id": "apps/train/1",
                "response_id": "suite-a",
                "quality_gate_passed": True,
                "fn_name": "solve",
                "test_count": 1,
                "tests": [{"args": [1], "expected": 2}],
            },
            {
                "problem_id": "apps/train/1",
                "response_id": "suite-b",
                "quality_gate_passed": True,
                "fn_name": "solve",
                "test_count": 2,
                "tests": [{"args": [1], "expected": 2}, {"args": [5], "expected": 6}],
            },
            {
                "problem_id": "apps/train/1",
                "response_id": "suite-c",
                "quality_gate_passed": False,
                "fn_name": "solve",
                "test_count": 1,
                "tests": [{"args": [10], "expected": 11}],
            },
        ]

        best = select_suites(rows)
        union = select_suites(rows, aggregation="union")

        self.assertEqual(best["apps/train/1"]["response_id"], "suite-b")
        self.assertEqual(
            union["apps/train/1"]["tests"],
            [{"args": [1], "expected": 2}, {"args": [5], "expected": 6}],
        )
        self.assertEqual(union["apps/train/1"]["source_suite_count"], 2)

    def test_score_candidate_and_confusion(self) -> None:
        suite = {
            "problem_id": "apps/train/1",
            "response_id": "suite-1",
            "fn_name": "solve",
            "tests": [{"args": [1], "expected": 2}, {"args": [5], "expected": 6}],
        }
        passing = {
            "id": "apps/train/1",
            "response_id": "candidate-pass",
            "problem_id": "apps/train/1",
            "passed": True,
            "generated_code": "def solve(x):\n    return x + 1\n",
        }
        failing = {
            "id": "apps/train/1",
            "response_id": "candidate-fail",
            "problem_id": "apps/train/1",
            "passed": False,
            "failure_type": "logic_error",
            "generated_code": "def solve(x):\n    return x - 1\n",
        }

        scored = [score_candidate(passing, suite, 5.0), score_candidate(failing, suite, 5.0)]
        report = confusion(scored)

        self.assertTrue(scored[0]["predicted_pass_by_tests"])
        self.assertFalse(scored[1]["predicted_pass_by_tests"])
        self.assertEqual(report["tp_pass"], 1)
        self.assertEqual(report["tn_fail"], 1)
        self.assertEqual(report["accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
