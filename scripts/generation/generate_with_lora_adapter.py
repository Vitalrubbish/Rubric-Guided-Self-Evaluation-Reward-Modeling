#!/usr/bin/env python3
"""Generate coding benchmark responses with a base model and optional LoRA adapter."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def read_rows(path: Path, dataset: str | None, split: str | None, limit: int | None) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if dataset and row.get("dataset") != dataset:
                continue
            if split and row.get("split") != split:
                continue
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def copy_test_fields(source: dict, target: dict) -> None:
    for key in ("test_list", "test_setup_code", "test", "entry_point", "split"):
        if key in source:
            target[key] = source[key]


def batched(rows: list[dict], batch_size: int):
    for start in range(0, len(rows), batch_size):
        yield rows[start : start + batch_size]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--adapter", type=str, default=None)
    parser.add_argument("--input", type=Path, default=Path("data/processed/coding_prompts.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-input-length", type=int, default=1536)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for adapter generation")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")

    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map={"": 0},
    )
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id

    rows = read_rows(args.input, args.dataset, args.split, args.limit)
    if not rows:
        raise RuntimeError("No rows selected for generation")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    do_sample = args.temperature > 0
    generated = 0

    with args.output.open("w", encoding="utf-8") as f, torch.inference_mode():
        for batch in batched(rows, args.batch_size):
            prompts = [row["prompt"] for row in batch]
            encoded = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_input_length,
            )
            encoded = {key: value.cuda() for key, value in encoded.items()}
            output_ids = model.generate(
                **encoded,
                max_new_tokens=args.max_new_tokens,
                do_sample=do_sample,
                temperature=args.temperature if do_sample else None,
                top_p=args.top_p if do_sample else None,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            new_tokens = output_ids[:, encoded["input_ids"].shape[1] :]
            texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

            for row, text in zip(batch, texts):
                record = {
                    "id": row["id"],
                    "dataset": row["dataset"],
                    "prompt": row["prompt"],
                    "generated_code": text,
                    "model": args.model,
                    "adapter": args.adapter,
                    "timestamp": now,
                    "sample_id": 0,
                    "seed": args.seed,
                    "generation_backend": "transformers_peft",
                }
                copy_test_fields(row, record)
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                generated += 1
                if generated % 10 == 0:
                    print(f"generated {generated}/{len(rows)}")

    print(f"wrote {generated} responses to {args.output}")
    print(f"adapter={args.adapter or 'none'}")


if __name__ == "__main__":
    main()
