#!/usr/bin/env python3
"""Create response-style JSONL rows from embedded canonical solutions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def canonical_solutions(row: dict) -> list[str]:
    values = row.get("canonical_solutions")
    if isinstance(values, list):
        return [item for item in values if isinstance(item, str) and item.strip()]
    value = row.get("canonical_solution")
    if isinstance(value, str) and value.strip():
        return [value]
    return []


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-solutions", type=int, default=1)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    prompts = 0
    responses = 0
    with args.output.open("w", encoding="utf-8") as out:
        for row in read_jsonl(args.input):
            prompts += 1
            for sample_id, solution in enumerate(canonical_solutions(row)[: args.max_solutions]):
                record = {
                    "response_id": f"{row['id']}__canonical{sample_id}",
                    "id": row["id"],
                    "dataset": row.get("dataset"),
                    "split": row.get("split"),
                    "prompt_mode": row.get("prompt_mode"),
                    "interface_names": row.get("interface_names"),
                    "interface_signatures": row.get("interface_signatures"),
                    "prompt": row.get("prompt"),
                    "generated_code": solution,
                    "model": "canonical_solution",
                    "sample_id": sample_id,
                    "temperature": 0.0,
                    "top_p": 1.0,
                }
                for key in ("test_list", "test_setup_code", "test", "entry_point", "starter_code", "code_prompt", "libs", "input_output", "difficulty", "io_mode"):
                    if key in row:
                        record[key] = row.get(key)
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                responses += 1
    print(f"wrote {responses} canonical responses from {prompts} prompts to {args.output}")


if __name__ == "__main__":
    main()
