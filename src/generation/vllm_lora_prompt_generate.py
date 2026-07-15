#!/usr/bin/env python3
"""Generate generic prompt completions with a vLLM-served PEFT LoRA adapter."""

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
    "task_type",
    "source",
    "completion",
    "metadata",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "prompt" not in row:
                raise ValueError(f"row {line_number} in {path} has no prompt")
            rows.append(row)
    return rows


def filter_rows(
    rows: list[dict[str, Any]],
    *,
    split: str | None,
    task_type: str | None,
    limit: int | None,
) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for row in rows:
        if split is not None and str(row.get("split")) != split:
            continue
        if task_type is not None and str(row.get("task_type")) != task_type:
            continue
        filtered.append(row)
        if limit is not None and len(filtered) >= limit:
            break
    return filtered


def batches(rows: list[dict[str, Any]], batch_size: int) -> Iterable[list[dict[str, Any]]]:
    for start in range(0, len(rows), batch_size):
        yield rows[start : start + batch_size]


def format_prompt(prompt: Any, prompt_format: str) -> str:
    text = str(prompt)
    if prompt_format == "as-is":
        return text
    if prompt_format == "raw":
        return text.strip() + "\n\n"
    raise ValueError(f"unsupported prompt format: {prompt_format}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate prompt completions with a LoRA adapter.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default=None)
    parser.add_argument("--task-type", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.35)
    parser.add_argument("--prompt-batch-size", type=int, default=32)
    parser.add_argument("--max-lora-rank", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora-name", default="method1-generative-self-evaluator")
    parser.add_argument(
        "--prompt-format",
        choices=("as-is", "raw"),
        default="as-is",
        help="Prompt formatting applied before generation. raw matches train_causallm_sft_lora.py.",
    )
    args = parser.parse_args()

    if not (args.adapter / "adapter_model.safetensors").is_file():
        raise FileNotFoundError(f"completed adapter not found: {args.adapter}")
    rows = filter_rows(
        read_jsonl(args.input),
        split=args.split,
        task_type=args.task_type,
        limit=args.limit,
    )
    if not rows:
        raise RuntimeError("no rows selected for generation")

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
        n=1,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )
    lora_request = LoRARequest(args.lora_name, 1, str(args.adapter.resolve()))
    timestamp = datetime.now(timezone.utc).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with args.output.open("w", encoding="utf-8") as handle:
        for batch_index, batch in enumerate(batches(rows, args.prompt_batch_size), start=1):
            outputs = llm.generate(
                [format_prompt(row["prompt"], args.prompt_format) for row in batch],
                sampling,
                lora_request=lora_request,
            )
            for row, output in zip(batch, outputs):
                completion = output.outputs[0]
                record = {
                    "response_id": f"{row.get('id')}__lora_sample0",
                    "id": row.get("id"),
                    "prompt": row.get("prompt"),
                    "generated_text": completion.text,
                    "model": args.model,
                    "adapter": str(args.adapter),
                    "timestamp": timestamp,
                    "sample_id": 0,
                    "seed": args.seed,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "repetition_penalty": args.repetition_penalty,
                    "max_tokens": args.max_tokens,
                    "finish_reason": getattr(completion, "finish_reason", None),
                    "stop_reason": getattr(completion, "stop_reason", None),
                    "generated_token_count": len(getattr(completion, "token_ids", []) or []),
                    "generation_backend": "vllm_lora_prompt",
                    "prompt_format": args.prompt_format,
                }
                for key in COPY_FIELDS:
                    if key in row:
                        record[key] = row.get(key)
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                total += 1
            handle.flush()
            print(f"batch={batch_index} generated={total}/{len(rows)}", flush=True)

    print(
        json.dumps(
            {
                "output": str(args.output),
                "rows": total,
                "adapter": str(args.adapter),
                "split": args.split,
                "task_type": args.task_type,
                "prompt_format": args.prompt_format,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
