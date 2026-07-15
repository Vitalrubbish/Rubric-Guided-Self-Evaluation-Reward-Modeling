import unittest

from src.evaluator.freeze_apps_dpo_v2_final_prompts import PRIVATE_FIELDS, freeze_rows


class Final523ProtocolTests(unittest.TestCase):
    def source_rows(self) -> list[dict]:
        rows = []
        for index in range(523):
            split = "validation" if index < 261 else "test"
            rows.append(
                {
                    "id": f"apps/train/{index}",
                    "prompt": f"problem {index}",
                    "source_split": "train",
                    "split": split,
                    "eval_split": split,
                    "canonical_solution": "secret",
                    "canonical_solutions": ["secret"],
                    "canonical_verifier": {"private": True},
                }
            )
        return rows

    def test_private_fields_are_removed_and_boundaries_hold(self) -> None:
        frozen, audit = freeze_rows(
            self.source_rows(),
            [{"id": "apps/train/900"}],
            [{"id": "apps/train/901"}],
        )

        self.assertEqual(len(frozen), 523)
        self.assertFalse(any(PRIVATE_FIELDS.intersection(row) for row in frozen))
        self.assertEqual(audit["split_counts"], {"validation": 261, "test": 262})
        self.assertEqual(audit["training_overlap_count"], 0)

    def test_training_overlap_is_rejected(self) -> None:
        with self.assertRaisesRegex(AssertionError, "training IDs"):
            freeze_rows(
                self.source_rows(),
                [{"id": "apps/train/5"}],
                [{"id": "apps/train/901"}],
            )


if __name__ == "__main__":
    unittest.main()
