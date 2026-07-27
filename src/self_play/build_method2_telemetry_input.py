#!/usr/bin/env python3
"""Build the Method 2 gold-telemetry generation input.

The gold-attribution sample (data/annotation/) is the fixed per-iteration
health-check set for the self-evolution loop: every checkpoint generates
ERROR_FINDINGS + REVISED_CODE on these logic_error problems, and the
findings are scored against the frontier-LLM gold attributions. The sample
is drawn from a pool disjoint from both the SFT train prompts and the
200-row gate, so it is leakage-free for every Method 2 version.

This script joins the annotation sample back to the full labeled responses
(to recover verifier fields) and renders method2 critic+repair prompts in
the exact v0.3 no-end-marker format.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from src.self_play.build_method2_bootstrap_data import (
    method2_prompt,
    normalize_code,
    sha256_file,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--response-prefix", default="Repair response:")
    args = parser.parse_args()

    responses_by_id = {str(r.get("response_id")): r for r in read_jsonl(args.responses)}
    sample = read_jsonl(args.sample)

    out_rows: list[dict[str, Any]] = []
    missing = 0
    for entry in sample:
        source = responses_by_id.get(str(entry.get("response_id")))
        if source is None:
            missing += 1
            continue
        problem_id = str(source.get("id"))
        failed_code = normalize_code(entry.get("failed_code") or source.get("generated_code"))
        out_rows.append(
            {
                "id": f"{problem_id}__method2_gold_telemetry",
                "problem_id": problem_id,
                "annotation_id": entry.get("annotation_id"),
                "dataset": source.get("dataset"),
                "split": "validation",
                "task_type": "method2_self_play_critic_repair",
                "prompt": method2_prompt(
                    str(source.get("prompt") or ""),
                    failed_code,
                    response_prefix=args.response_prefix,
                ),
                "completion": "",
                "source": "gold_telemetry_logic100",
                "interface_names": source.get("interface_names") or [],
                "interface_signatures": source.get("interface_signatures") or [],
                "starter_code": source.get("starter_code"),
                "input_output": source.get("input_output"),
                "difficulty": source.get("difficulty"),
                "io_mode": source.get("io_mode"),
                "rejected_response_id": source.get("response_id"),
                "metadata": {
                    "problem_id": problem_id,
                    "failure_type": source.get("failure_type"),
                    "gold_annotation_file": str(args.sample),
                    "end_marker": None,
                },
            }
        )

    write_jsonl(args.output, out_rows)
    summary = {
        "responses": str(args.responses),
        "sample": str(args.sample),
        "output": str(args.output),
        "output_sha256": sha256_file(args.output),
        "sample_rows": len(sample),
        "written_rows": len(out_rows),
        "missing_response_ids": missing,
        "io_mode_counts": dict(Counter(str(r.get("io_mode") or "unknown") for r in out_rows)),
        "policy": {
            "role": "fixed per-iteration telemetry set for the self-evolution loop; scored against frontier-LLM gold attributions, never used in training",
            "leakage": "problems are disjoint from the SFT train prompts and the 200-row gate",
        },
    }
    write_json(args.summary_output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
