from __future__ import annotations

import unittest

from src.training.build_apps_dpo_dev import freeze_dev_rows
from src.training.build_apps_dpo_v2_preferences import build_pairs, code_audit, normalize_code
from src.training.freeze_apps_dpo_v2_canary import freeze_canary_rows
from src.training.scale_lora_adapter import scaled_lora_config
from src.training.merge_preference_sets import merge_rows
from src.training.build_apps_dpo_v2_semantic_canary import build_semantic_pairs
from src.training.build_apps_dpo_v2_syntax_guard_canary import build_syntax_guarded_pairs
from src.training.build_apps_dpo_v2_termination_guard_canary import build_guarded_pairs
from src.verification.verify_paired_apps_dpo_dev import build_unique_rows, code_key, materialize_rows


class AppsDpoV2PreferenceTests(unittest.TestCase):
    def original(self, code: str = "def solve(x):\n    return x - 1") -> dict:
        return {
            "response_id": "original-1",
            "id": "apps/train/1",
            "dataset": "apps",
            "split": "train",
            "source_split": "train",
            "passed": False,
            "prompt": "Repair solve(x). Return only code.",
            "generated_code": code,
            "extracted_code": code,
            "model": "qwen",
            "difficulty": "introductory",
            "io_mode": "function_call",
            "failure_type": "logic_error",
        }

    def repair(self, code: str = "def solve(x):\n    return x + 1") -> dict:
        return {
            "response_id": "repair-1",
            "original_response_id": "original-1",
            "repair_candidate_id": "candidate-1",
            "id": "apps/train/1",
            "dataset": "apps",
            "split": "train",
            "passed": True,
            "generated_code": f"```python\n{code}\n```",
            "extracted_code": code,
            "model": "qwen",
            "finish_reason": "stop",
            "prompt_mode": "two_stage",
            "difficulty": "introductory",
            "io_mode": "function_call",
            "interface_names": ["solve"],
        }

    def test_normalizes_both_sides_and_keeps_verified_pair(self) -> None:
        original = self.original("```python\ndef solve(x):\n    return x - 1\n```")
        pairs, skipped, audit = build_pairs(
            [self.repair()],
            {"original-1": original},
        )

        self.assertEqual(len(pairs), 1)
        self.assertNotIn("```", pairs[0]["chosen"])
        self.assertNotIn("```", pairs[0]["rejected"])
        self.assertTrue(pairs[0]["chosen_parseable"])
        self.assertEqual(skipped, {})
        self.assertEqual(audit["normalized_format_counts"]["chosen_fenced"], 0)
        self.assertEqual(audit["normalized_format_counts"]["rejected_fenced"], 0)

    def test_rejects_top_level_demo_in_chosen(self) -> None:
        repair = self.repair("def solve(x):\n    return x + 1\nprint(solve(1))")
        pairs, skipped, audit = build_pairs([repair], {"original-1": self.original()})

        self.assertEqual(pairs, [])
        self.assertEqual(skipped["chosen_top_level_demo"], 1)
        self.assertEqual(audit["candidate_raw_format_counts"]["chosen_fenced"], 1)
        self.assertEqual(audit["raw_format_counts"]["chosen_fenced"], 0)

    def test_rejects_forbidden_dev_id(self) -> None:
        pairs, skipped, _ = build_pairs(
            [self.repair()],
            {"original-1": self.original()},
            forbidden_ids={"apps/train/1"},
        )

        self.assertEqual(pairs, [])
        self.assertEqual(skipped["forbidden_problem_id"], 1)

    def test_code_audit_accepts_solution_method(self) -> None:
        audit = code_audit("class Solution:\n    def solve(self, x):\n        return x", ["solve"])
        self.assertTrue(audit["parseable"])
        self.assertTrue(audit["required_interface_present"])

    def test_normalize_code_extracts_first_fence(self) -> None:
        self.assertEqual(normalize_code("before\n```python\nx = 1  \n```\nafter"), "x = 1")

    def test_plain_lora_scaling_changes_only_global_alpha(self) -> None:
        source = {"r": 16, "lora_alpha": 32, "use_rslora": False, "use_dora": False, "alpha_pattern": {}}

        output, original_alpha, scaled_alpha = scaled_lora_config(source, 0.5)

        self.assertEqual(original_alpha, 32.0)
        self.assertEqual(scaled_alpha, 16.0)
        self.assertEqual(output["lora_alpha"], 16)
        self.assertEqual(source["lora_alpha"], 32)

    def test_preference_merge_retains_primary_and_is_deterministic(self) -> None:
        primary = [{"id": "apps/train/1", "pair_id": "primary-1"}]
        supplements = [
            {"id": "apps/train/2", "pair_id": "supplement-2"},
            {"id": "apps/train/3", "pair_id": "supplement-3"},
        ]

        first, first_added = merge_rows(primary, supplements, target_size=2, seed=7)
        second, second_added = merge_rows(primary, supplements, target_size=2, seed=7)

        self.assertEqual(first, second)
        self.assertEqual(first_added, second_added)
        self.assertIn("apps/train/1", {row["id"] for row in first})
        self.assertEqual(len(first_added), 1)

    def test_normalize_code_removes_unmatched_boundary_fence(self) -> None:
        self.assertEqual(normalize_code("def solve():\n    return 1\n```"), "def solve():\n    return 1")


class AppsDpoDevTests(unittest.TestCase):
    def test_freeze_dev_is_deterministic_and_excludes_training_candidates(self) -> None:
        source_rows = [
            {
                "id": f"apps/train/{index}",
                "split": "train",
                "dataset": "apps",
                "prompt": f"task {index}",
            }
            for index in range(8)
        ]
        source_rows.append({"id": "apps/train/not-train", "split": "validation", "prompt": "held out"})
        base_by_id = {
            row["id"]: {**row, "passed": index % 2 == 0}
            for index, row in enumerate(source_rows)
        }
        excluded = {"apps/train/0", "apps/train/1"}

        first_prompts, first_base = freeze_dev_rows(
            source_rows,
            base_by_id,
            excluded,
            size=3,
            seed=7,
        )
        second_prompts, second_base = freeze_dev_rows(
            source_rows,
            base_by_id,
            excluded,
            size=3,
            seed=7,
        )

        self.assertEqual(first_prompts, second_prompts)
        self.assertEqual(first_base, second_base)
        self.assertFalse({row["id"] for row in first_prompts} & excluded)
        self.assertTrue(all(row["split"] == "dpo_dev" for row in first_prompts))
        self.assertTrue(all(row["source_split"] == "train" for row in first_prompts))


class AppsDpoCanaryFreezeTests(unittest.TestCase):
    def test_two_stage_rows_are_retained_before_supplements(self) -> None:
        rows = [
            {
                "pair_id": f"pair-{index}",
                "id": f"apps/train/{index}",
                "repair_method": "method1_two_stage_public_spec_repair_v2" if index == 4 else "method1_repair_v1",
                "original_failure_type": "logic_error" if index % 2 else "syntax_error",
            }
            for index in range(5)
        ]

        selected = freeze_canary_rows(rows, size=3, seed=7)

        self.assertEqual(len(selected), 3)
        self.assertIn("apps/train/4", {row["id"] for row in selected})
        self.assertEqual(len({row["id"] for row in selected}), 3)

    def test_semantic_filter_matches_format_and_removes_shortcuts(self) -> None:
        base = {
            "pair_id": "pair-1",
            "id": "apps/train/1",
            "chosen": "def solve(x):\n    return x + 1",
            "rejected": "def solve(x):\n    return x - 1",
            "original_failure_type": "logic_error",
            "rejected_parseable": True,
            "rejected_required_interface_present": True,
            "completion_char_ratio": 1.0,
        }
        syntax = {**base, "pair_id": "pair-2", "id": "apps/train/2", "original_failure_type": "syntax_error"}
        demo = {
            **base,
            "pair_id": "pair-3",
            "id": "apps/train/3",
            "rejected": "def solve(x):\n    return x - 1\nprint(solve(1))",
        }

        selected, skipped = build_semantic_pairs([base, syntax, demo], max_length_ratio=3.0)

        self.assertEqual(len(selected), 1)
        self.assertTrue(selected[0]["chosen"].startswith("```python\n"))
        self.assertTrue(selected[0]["rejected"].startswith("```python\n"))
        self.assertEqual(skipped["syntax_error_negative"], 1)
        self.assertEqual(skipped["rejected_top_level_demo"], 1)

    def test_semantic_filter_can_preserve_matched_raw_python(self) -> None:
        row = {
            "pair_id": "pair-raw",
            "id": "apps/train/raw",
            "chosen": "def solve(x):\n    return x + 1",
            "rejected": "def solve(x):\n    return x - 1",
            "original_failure_type": "logic_error",
            "rejected_parseable": True,
            "rejected_required_interface_present": True,
            "completion_char_ratio": 1.0,
        }

        selected, skipped = build_semantic_pairs([row], max_length_ratio=3.0, completion_format="raw")

        self.assertEqual(skipped, {})
        self.assertEqual(selected[0]["completion_format"], "matched_raw_python")
        self.assertNotIn("```", selected[0]["chosen"])
        self.assertNotIn("```", selected[0]["rejected"])

    def test_syntax_guards_require_real_parse_failure_and_close_lengths(self) -> None:
        semantic = {
            "pair_id": "semantic-1",
            "id": "apps/train/1",
            "chosen": "```python\ndef solve(x):\n    return x + 1\n```",
            "rejected": "```python\ndef solve(x):\n    return x - 1\n```",
            "original_failure_type": "logic_error",
        }
        syntax = {
            "pair_id": "syntax-1",
            "id": "apps/train/2",
            "chosen": "def solve(x):\n    return x + 1",
            "rejected": "def solve(x):\n    return x +",
            "original_failure_type": "syntax_error",
        }
        mislabeled = {
            **syntax,
            "pair_id": "syntax-2",
            "id": "apps/train/3",
            "rejected": "def solve(x):\n    return x - 1",
        }

        output, guards, skipped = build_syntax_guarded_pairs(
            [semantic],
            [syntax, mislabeled],
            max_length_ratio=1.5,
        )

        self.assertEqual(len(output), 2)
        self.assertEqual(len(guards), 1)
        self.assertFalse(guards[0]["rejected_parseable"])
        self.assertEqual(guards[0]["chosen"].count("```"), 2)
        self.assertEqual(guards[0]["rejected"].count("```"), 2)
        self.assertEqual(skipped["rejected_actually_parseable"], 1)

    def test_syntax_guards_can_preserve_matched_raw_python(self) -> None:
        semantic = {
            "pair_id": "semantic-raw",
            "id": "apps/train/raw-1",
            "chosen": "def solve(x):\n    return x + 1",
            "rejected": "def solve(x):\n    return x - 1",
            "original_failure_type": "logic_error",
        }
        syntax = {
            "pair_id": "syntax-raw",
            "id": "apps/train/raw-2",
            "chosen": "def solve(x):\n    return x + 1",
            "rejected": "def solve(x):\n    return x +",
            "original_failure_type": "syntax_error",
        }

        output, guards, skipped = build_syntax_guarded_pairs(
            [semantic],
            [syntax],
            max_length_ratio=1.5,
            completion_format="raw",
        )

        self.assertEqual(len(output), 2)
        self.assertEqual(len(guards), 1)
        self.assertEqual(skipped, {})
        self.assertEqual(guards[0]["completion_format"], "matched_raw_python")
        self.assertNotIn("```", guards[0]["chosen"])
        self.assertNotIn("```", guards[0]["rejected"])


class AppsDpoPairedVerifierTests(unittest.TestCase):
    def test_identical_program_is_evaluated_once_and_reused(self) -> None:
        base = [{"id": "apps/train/1", "generated_code": "```python\ndef solve():\n    return 1\n```"}]
        candidate = [{"id": "apps/train/1", "generated_code": "def solve():\n    return 1"}]

        unique, identical = build_unique_rows(base, candidate)
        self.assertEqual(len(unique), 1)
        self.assertEqual(identical, 1)

        key = code_key(base[0])
        result = {
            key: {
                "extracted_code": "def solve():\n    return 1",
                "passed": True,
                "failure_type": None,
                "error": None,
            }
        }
        base_rows = materialize_rows(
            base, result, variant="base", run_id="run", timeout=30.0, workers=2
        )
        candidate_rows = materialize_rows(
            candidate, result, variant="candidate", run_id="run", timeout=30.0, workers=2
        )

        self.assertTrue(base_rows[0]["passed"])
        self.assertTrue(candidate_rows[0]["passed"])
        self.assertEqual(base_rows[0]["paired_code_sha256"], candidate_rows[0]["paired_code_sha256"])
        self.assertEqual(base_rows[0]["paired_verification_run_id"], candidate_rows[0]["paired_verification_run_id"])


class AppsDpoTerminationGuardTests(unittest.TestCase):
    def test_filters_real_pairs_and_adds_matched_repetition_guard(self) -> None:
        rows = [
            {
                "pair_id": f"pair-{index}",
                "id": f"apps/train/{index}",
                "chosen": f"```python\ndef solve_{index}(x):\n    return x + 1\n```",
                "rejected": f"```python\ndef solve_{index}(x):\n    return x - 1\n```",
                "repair_method": "method1_repair_v1",
                "original_failure_type": "logic_error",
            }
            for index in range(3)
        ]

        output, guards, skipped = build_guarded_pairs(
            rows,
            max_length_ratio=1.1,
            termination_guards=1,
            guard_max_code_chars=240,
            seed=42,
        )

        self.assertEqual(len(output), 4)
        self.assertEqual(len(guards), 1)
        self.assertEqual(skipped, {})
        self.assertEqual(guards[0]["repair_method"], "protected_termination_guard_v1")
        self.assertEqual(guards[0]["chosen"].count("```"), 2)
        self.assertEqual(guards[0]["rejected"].count("```"), 2)
        self.assertGreater(len(guards[0]["rejected"]), len(guards[0]["chosen"]))


if __name__ == "__main__":
    unittest.main()
