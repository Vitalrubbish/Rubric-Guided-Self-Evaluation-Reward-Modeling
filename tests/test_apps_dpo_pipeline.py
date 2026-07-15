from __future__ import annotations

import unittest

from src.training.build_apps_dpo_preferences import PRIVATE_KEYS, build_pairs


class AppsDpoPreferenceTests(unittest.TestCase):
    def test_only_failed_train_rows_become_pairs(self) -> None:
        source = {
            "p1": {"id": "p1", "split": "train", "prompt": "task one", "canonical_solution": "print(1)"},
            "p2": {"id": "p2", "split": "train", "prompt": "task two", "canonical_solution": "print(2)"},
            "p3": {"id": "p3", "split": "train", "prompt": "task three", "canonical_solution": "print(3)"},
        }
        evaluator_rows = [
            {"id": "p1", "split": "train", "passed": False, "prompt": "task one", "generated_code": "print(0)"},
            {"id": "p2", "split": "train", "passed": True, "prompt": "task two", "generated_code": "print(2)"},
            {"id": "p3", "split": "validation", "passed": False, "prompt": "task three", "generated_code": "print(0)"},
        ]

        pairs, skipped = build_pairs(evaluator_rows, source)

        self.assertEqual([row["id"] for row in pairs], ["p1"])
        self.assertEqual(pairs[0]["chosen"], "print(1)")
        self.assertEqual(pairs[0]["rejected"], "print(0)")
        self.assertEqual(skipped["already_passing"], 1)
        self.assertEqual(skipped["non_train_split"], 1)
        self.assertFalse(PRIVATE_KEYS.intersection(pairs[0]))

    def test_identical_completions_are_skipped(self) -> None:
        source = {"p1": {"id": "p1", "prompt": "task", "canonical_solution": "x = 1\n"}}
        rows = [{"id": "p1", "split": "train", "passed": False, "prompt": "task", "generated_code": "x = 1"}]

        pairs, skipped = build_pairs(rows, source)

        self.assertEqual(pairs, [])
        self.assertEqual(skipped["identical_completions"], 1)


if __name__ == "__main__":
    unittest.main()
