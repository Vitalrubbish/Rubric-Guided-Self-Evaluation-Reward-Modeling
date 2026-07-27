from __future__ import annotations

import unittest

from src.self_play.executable_rubric_utils import all_passed, execute_function_tests
from src.self_play.extract_executable_rubric_tests import evaluate_generation, parse_tests
from src.self_play.score_executable_rubric_tests import confusion, score_candidate


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


class ExecutableRubricScoringTests(unittest.TestCase):
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
