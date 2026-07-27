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


def response_id(row, sample_id: int) -> str:
    if row.get("response_id_prefix"):
        return f"{row['response_id_prefix']}__sample{sample_id}"
    problem_id = str(row["id"])
    return f"{problem_id}__sample{sample_id}"


def write_outputs(rows, outputs, output_file, args, timestamp: str) -> int:
    total = 0
    for row, out in zip(rows, outputs):
        for sample_id, completion in enumerate(out.outputs):
            record = {
                "response_id": response_id(row, sample_id),
                "id": row["id"],
                "dataset": row["dataset"],
                "split": row.get("split"),
                "prompt_mode": row.get("prompt_mode"),
                "interface_names": row.get("interface_names"),
                "interface_signatures": row.get("interface_signatures"),
                "prompt": row["prompt"],
                "generated_code": completion.text,
                "model": args.model,
                "timestamp": timestamp,
                "sample_id": sample_id,
                "seed": args.seed,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "repetition_penalty": args.repetition_penalty,
                "max_tokens": args.max_tokens,
                "finish_reason": getattr(completion, "finish_reason", None),
                "stop_reason": getattr(completion, "stop_reason", None),
                "generated_token_count": len(getattr(completion, "token_ids", []) or []),
            }
            if "test_list" in row:
                record["test_list"] = row["test_list"]
                record["test_setup_code"] = row.get("test_setup_code", "")
            if "test" in row:
                record["test"] = row["test"]
                record["entry_point"] = row.get("entry_point")
            for key in ("starter_code", "code_prompt", "libs", "input_output", "difficulty", "io_mode"):
                if key in row:
                    record[key] = row.get(key)
            for key in ("source_split", "eval_split"):
                if key in row:
                    record[key] = row.get(key)
            for key in (
                "repair_candidate_id",
                "original_response_id",
                "original_generated_code",
                "critic_pass_probability",
                "critic_selected_threshold",
                "critic_predicted_pass",
                "selection_reason",
            ):
                if key in row:
                    record[key] = row.get(key)
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            total += 1
    output_file.flush()
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--input", type=Path, default=Path("data/processed/coding_prompts.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/responses/vllm_smoke_responses.jsonl"))
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.35)
    parser.add_argument("--k", type=int, default=3, help="number of completions per prompt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--prompt-batch-size",
        type=int,
        default=128,
        help="number of prompts to send to vLLM per generate() call; outputs are flushed after each batch",
    )
    args = parser.parse_args()

    rows = read_jsonl(args.input, args.limit)
    llm = LLM(
        model=args.model,
        tensor_parallel_size=1,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    sampling = SamplingParams(
        n=args.k,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    total = 0
    with args.output.open("w", encoding="utf-8") as f:
        first_outputs = None
        for start in range(0, len(rows), args.prompt_batch_size):
            batch_rows = rows[start : start + args.prompt_batch_size]
            prompts = [row["prompt"] for row in batch_rows]
            outputs = llm.generate(prompts, sampling)
            if first_outputs is None:
                first_outputs = outputs
            total += write_outputs(batch_rows, outputs, f, args, now)
            print(f"wrote batch {start // args.prompt_batch_size + 1}: {total} responses so far")

    print(f"wrote {total} responses (k={args.k}, {len(rows)} prompts) to {args.output}")
    if first_outputs and first_outputs[0].outputs:
        print(first_outputs[0].outputs[0].text[:800])


if __name__ == "__main__":
    main()
