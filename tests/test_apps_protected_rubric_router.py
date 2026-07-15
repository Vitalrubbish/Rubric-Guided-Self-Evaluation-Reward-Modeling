import unittest

from src.evaluator.route_apps_responses_by_protected_rubric import (
    assert_unlabeled,
    protected_score,
    route_rows,
)


def row(problem_id: str, code: str, *, interface_names: list[str] | None = None) -> dict:
    return {
        "id": problem_id,
        "prompt": "Write solve.",
        "generated_code": code,
        "interface_names": interface_names or ["solve"],
    }


class ProtectedRubricRouterTests(unittest.TestCase):
    def test_candidate_repairs_syntax_failure(self) -> None:
        base = row("p1", "def solve(:\n    pass")
        candidate = row("p1", "def solve():\n    return 1")

        routed, audit = route_rows([base], [candidate])

        self.assertEqual(routed[0]["generated_code"], candidate["generated_code"])
        self.assertEqual(audit["selection_counts"], {"candidate": 1})

    def test_tie_preserves_base(self) -> None:
        base = row("p1", "def solve():\n    return 1")
        candidate = row("p1", "def solve():\n    return 2")

        routed, audit = route_rows([base], [candidate])

        self.assertEqual(routed[0]["generated_code"], base["generated_code"])
        self.assertEqual(audit["reason_counts"], {"tie_preserves_base": 1})

    def test_candidate_hard_regression_is_blocked(self) -> None:
        base = row("p1", "def solve():\n    return 1")
        candidate = row("p1", "def solve(:\n    pass")

        routed, audit = route_rows([base], [candidate])

        self.assertEqual(routed[0]["generated_code"], base["generated_code"])
        self.assertEqual(audit["reason_counts"], {"candidate_hard_regression_blocked": 1})

    def test_missing_required_interface_scores_two(self) -> None:
        scored = protected_score(row("p1", "def other():\n    return 1"))

        self.assertEqual(scored["score"], 2)
        self.assertEqual(scored["fatal_reason"], "required_interface_missing")

    def test_labeled_input_is_rejected(self) -> None:
        labeled = {**row("p1", "def solve():\n    return 1"), "passed": True}

        with self.assertRaisesRegex(AssertionError, "verifier labels"):
            assert_unlabeled([labeled], "base")


if __name__ == "__main__":
    unittest.main()
