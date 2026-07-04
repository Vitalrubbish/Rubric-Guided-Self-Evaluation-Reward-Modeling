#!/usr/bin/env python3
"""One-step LoRA SFT smoke test on MBPP canonical solutions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


def read_mbpp(path: Path, limit: int):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("dataset") == "mbpp":
                rows.append(row)
            if len(rows) >= limit:
                break
    return rows


def build_example(row: dict) -> tuple[str, str]:
    prompt = row["prompt"]
    target = row["canonical_solution"].strip() + "\n"
    return prompt, target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/processed/coding_prompts.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/sft_lora_smoke"))
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this smoke test")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map={"": 0},
    )
    model.config.use_cache = False

    lora = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    model.train()
    model.print_trainable_parameters()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    rows = read_mbpp(args.data, args.limit)
    if not rows:
        raise RuntimeError("No MBPP rows found")

    for step, row in enumerate(rows):
        prompt, target = build_example(row)
        prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        encoded = tokenizer(
            prompt + target,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_length,
        )
        input_ids = encoded.input_ids.cuda()
        attention_mask = encoded.attention_mask.cuda()
        labels = input_ids.clone()
        mask_len = min(len(prompt_ids), labels.shape[1])
        labels[:, :mask_len] = -100

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        print(f"step={step} id={row['id']} loss={loss.item():.4f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"saved LoRA smoke adapter to {args.output_dir}")


if __name__ == "__main__":
    main()
