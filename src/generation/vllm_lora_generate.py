#!/usr/bin/env python3
"""Generate coding responses with a vLLM-served PEFT LoRA adapter."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


COPY_FIELDS = (
    "split",
    "eval_split",
    "source_split",
    "prompt_mode",
    "interface_names",
    "interface_signatures",
    "starter_code",
    "code_prompt",
    "libs",
    "input_output",
    "difficulty",
    "io_mode",
)


def read_jsonl(path: Path, limit: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def batches(rows: list[dict[str, Any]], batch_size: int) -> Iterable[list[dict[str, Any]]]:
    for start in range(0, len(rows), batch_size):
        yield rows[start : start + batch_size]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.70)
    parser.add_argument("--prompt-batch-size", type=int, default=64)
    parser.add_argument("--max-lora-rank", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--stop", action="append", default=[])
    args = parser.parse_args()

    if not (args.adapter / "adapter_model.safetensors").is_file():
        raise FileNotFoundError(f"completed adapter not found: {args.adapter}")
    if args.n < 1:
        raise ValueError("--n must be at least 1")
    rows = read_jsonl(args.input, args.limit)
    if not rows:
        raise RuntimeError("no rows selected for adapter generation")

    llm = LLM(
        model=args.model,
        tensor_parallel_size=1,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_lora=True,
        max_lora_rank=args.max_lora_rank,
    )
    sampling = SamplingParams(
        n=args.n,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        max_tokens=args.max_tokens,
        stop=args.stop or None,
        seed=args.seed,
    )
    lora_request = LoRARequest("apps-simple-method1-dpo-v1", 1, str(args.adapter.resolve()))
    timestamp = datetime.now(timezone.utc).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with args.output.open("w", encoding="utf-8") as handle:
        for batch_index, batch in enumerate(batches(rows, args.prompt_batch_size), start=1):
            outputs = llm.generate(
                [str(row["prompt"]) for row in batch],
                sampling,
                lora_request=lora_request,
            )
            for row, output in zip(batch, outputs):
                for sample_index, completion in enumerate(output.outputs):
                    record = {
                        "response_id": f"{row['id']}__dpo_lora_v1_sample{sample_index}",
                        "id": row["id"],
                        "dataset": row.get("dataset"),
                        "prompt": row["prompt"],
                        "generated_code": completion.text,
                        "model": args.model,
                        "adapter": str(args.adapter),
                        "timestamp": timestamp,
                        "sample_id": sample_index,
                        "seed": args.seed,
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                        "repetition_penalty": args.repetition_penalty,
                        "max_tokens": args.max_tokens,
                        "finish_reason": getattr(completion, "finish_reason", None),
                        "stop_reason": getattr(completion, "stop_reason", None),
                        "stop_sequences": args.stop,
                        "generated_token_count": len(getattr(completion, "token_ids", []) or []),
                        "generation_backend": "vllm_lora",
                    }
                    for key in COPY_FIELDS:
                        if key in row:
                            record[key] = row.get(key)
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    total += 1
            handle.flush()
            print(f"batch={batch_index} generated={total}/{len(rows) * args.n}", flush=True)

    print(json.dumps({"output": str(args.output), "rows": total, "adapter": str(args.adapter), "n": args.n}, indent=2))


if __name__ == "__main__":
    main()
