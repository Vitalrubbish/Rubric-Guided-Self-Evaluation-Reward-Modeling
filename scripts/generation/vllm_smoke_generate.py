#!/usr/bin/env python3
"""Small vLLM generation smoke test for local coding prompts."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from vllm import LLM, SamplingParams


def read_jsonl(path: Path, limit: int):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
            if len(rows) >= limit:
                break
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--input", type=Path, default=Path("data/processed/coding_prompts.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/responses/vllm_smoke_responses.jsonl"))
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.35)
    args = parser.parse_args()

    rows = read_jsonl(args.input, args.limit)
    prompts = [row["prompt"] for row in rows]
    llm = LLM(
        model=args.model,
        tensor_parallel_size=1,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=42,
    )
    outputs = llm.generate(prompts, sampling)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    with args.output.open("w", encoding="utf-8") as f:
        for row, out in zip(rows, outputs):
            text = out.outputs[0].text
            record = {
                "id": row["id"],
                "dataset": row["dataset"],
                "prompt": row["prompt"],
                "generated_code": text,
                "model": args.model,
                "timestamp": now,
                "sample_id": 0,
                "seed": 42,
            }
            if "test_list" in row:
                record["test_list"] = row["test_list"]
                record["test_setup_code"] = row.get("test_setup_code", "")
            if "test" in row:
                record["test"] = row["test"]
                record["entry_point"] = row.get("entry_point")
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"wrote {len(outputs)} responses to {args.output}")
    if outputs:
        print(outputs[0].outputs[0].text[:800])


if __name__ == "__main__":
    main()
