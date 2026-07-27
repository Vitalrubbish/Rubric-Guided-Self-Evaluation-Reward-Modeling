#!/usr/bin/env python3
"""Generate coding responses with a base (no-adapter) vLLM model.

Same output schema as vllm_lora_generate.py so the downstream
extract/verify pipeline works unchanged; used for bare-capability
baselines of larger models (e.g. Qwen2.5-Coder-32B AWQ) on the Method 2
repair gates.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from vllm import LLM, SamplingParams


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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--stop", action="append", default=[])
    parser.add_argument("--quantization", default=None)
    parser.add_argument("--enforce-eager", action="store_true")
    args = parser.parse_args()

    if args.n < 1:
        raise ValueError("--n must be at least 1")
    rows = read_jsonl(args.input, args.limit)
    if not rows:
        raise RuntimeError("no rows selected for base generation")

    llm_kwargs: dict[str, Any] = {
        "model": args.model,
        "tensor_parallel_size": 1,
        "trust_remote_code": True,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enforce_eager": args.enforce_eager,
    }
    if args.quantization:
        llm_kwargs["quantization"] = args.quantization
    llm = LLM(**llm_kwargs)
    sampling = SamplingParams(
        n=args.n,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        max_tokens=args.max_tokens,
        stop=args.stop or None,
        seed=args.seed,
    )
    timestamp = datetime.now(timezone.utc).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with args.output.open("w", encoding="utf-8") as handle:
        for batch_index, batch in enumerate(batches(rows, args.prompt_batch_size), start=1):
            outputs = llm.generate(
                [str(row["prompt"]) for row in batch],
                sampling,
            )
            for row, output in zip(batch, outputs):
                for sample_index, completion in enumerate(output.outputs):
                    record = {
                        "response_id": f"{row['id']}__base_sample{sample_index}",
                        "id": row["id"],
                        "dataset": row.get("dataset"),
                        "prompt": row["prompt"],
                        "generated_code": completion.text,
                        "model": args.model,
                        "adapter": None,
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
                        "generation_backend": "vllm_base",
                    }
                    for key in COPY_FIELDS:
                        if key in row:
                            record[key] = row.get(key)
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    total += 1
            handle.flush()
            print(f"batch={batch_index} generated={total}/{len(rows) * args.n}", flush=True)

    print(json.dumps({"output": str(args.output), "rows": total, "adapter": None, "n": args.n}, indent=2))


if __name__ == "__main__":
    main()
