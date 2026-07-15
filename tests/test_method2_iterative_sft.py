from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class Method2IterativeSftTests(unittest.TestCase):
    def base_row(self, row_id: str, split: str = "train") -> dict:
        return {
            "id": row_id,
            "split": split,
            "source": "apps_repair_self_play",
            "prompt": "Public task prompt:\nWrite solve.",
            "completion": "ERROR_FINDINGS:\n- Wrong formula.\nREVISED_CODE:\ndef solve(x):\n    return x - 1",
            "metadata": {"original_response_id": f"{row_id}__original"},
        }

    def generated_row(self, row_id: str, response_id: str, finish_reason: str) -> dict:
        code = f"def solve(x):\n    return x + {len(response_id)}"
        return {
            "id": row_id,
            "response_id": response_id,
            "sample_id": 0,
            "passed": True,
            "finish_reason": finish_reason,
            "method2_extraction_status": "ok",
            "method2_generated_token_count": 12 if finish_reason == "stop" else 200,
            "method2_extraction_notes": [],
            "method2_raw_completion": f"ERROR_FINDINGS:\n- Fixed formula.\nREVISED_CODE:\n{code}",
            "generated_code": code,
            "io_mode": "function_call",
            "source_mode": "apps_function_call",
        }

    def test_requires_stop_finish_and_total_cap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            base = tmp_path / "base.jsonl"
            generated = tmp_path / "generated.jsonl"
            sft_output = tmp_path / "sft.jsonl"
            accepted = tmp_path / "accepted.jsonl"
            summary = tmp_path / "summary.json"

            write_jsonl(
                base,
                [
                    self.base_row("apps/train/1"),
                    self.base_row("apps/train/2"),
                    self.base_row("apps/train/3", split="validation"),
                ],
            )
            write_jsonl(
                generated,
                [
                    self.generated_row("apps/train/1", "length-candidate", "length"),
                    self.generated_row("apps/train/1", "stop-candidate-1", "stop"),
                    self.generated_row("apps/train/2", "stop-candidate-2", "stop"),
                ],
            )

            subprocess.run(
                [
                    sys.executable,
                    "src/self_play/build_method2_iterative_sft.py",
                    "--base-sft",
                    str(base),
                    "--generated-labeled",
                    str(generated),
                    "--sft-output",
                    str(sft_output),
                    "--accepted-output",
                    str(accepted),
                    "--summary-output",
                    str(summary),
                    "--require-finish-reason",
                    "stop",
                    "--max-generated-total",
                    "1",
                    "--source-tag",
                    "test_stop50",
                ],
                check=True,
                cwd=Path(__file__).resolve().parents[1],
            )

            summary_data = json.loads(summary.read_text(encoding="utf-8"))
            accepted_rows = read_jsonl(accepted)
            sft_rows = read_jsonl(sft_output)

        self.assertEqual(summary_data["generated_selected_rows"], 1)
        self.assertEqual(summary_data["generated_finish_counts"], {"stop": 1})
        self.assertEqual(summary_data["required_finish_reasons"], ["stop"])
        self.assertEqual(summary_data["max_generated_total"], 1)
        self.assertEqual(summary_data["counts"]["skipped:finish_reason:length"], 1)
        self.assertEqual(len(accepted_rows), 1)
        self.assertEqual(accepted_rows[0]["response_id"], "stop-candidate-1")
        self.assertEqual(summary_data["split_counts"], {"train": 3, "validation": 1})
        self.assertEqual(len(sft_rows), 4)


if __name__ == "__main__":
    unittest.main()
