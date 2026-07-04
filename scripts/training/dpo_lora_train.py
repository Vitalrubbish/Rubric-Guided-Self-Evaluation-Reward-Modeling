#!/usr/bin/env python3
"""Memory-conscious LoRA DPO training on coding preference pairs.

This uses one PEFT model only. Reference log-probabilities are computed by
temporarily disabling the LoRA adapter, which avoids loading a second 7B model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


def read_jsonl(path: Path, limit: int | None = None) -> Iterable[dict]:
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)
            count += 1
            if limit is not None and count >= limit:
                break


def build_inputs(tokenizer, prompt: str, completion: str, max_length: int, device: torch.device):
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    encoded = tokenizer(
        prompt + completion,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
    )
    input_ids = encoded.input_ids.to(device)
    attention_mask = encoded.attention_mask.to(device)
    labels = input_ids.clone()
    prompt_len = min(len(prompt_ids), labels.shape[1])
    labels[:, :prompt_len] = -100
    return input_ids, attention_mask, labels


def sequence_logprob(model, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor):
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]
    shifted_labels = labels[:, 1:]
    mask = shifted_labels.ne(-100)
    token_count = mask.sum()
    if token_count.item() == 0:
        return None, 0
    safe_labels = shifted_labels.masked_fill(~mask, 0)
    log_probs = torch.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    return (token_log_probs * mask).sum(), int(token_count.item())


def dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta: float):
    policy_margin = policy_chosen - policy_rejected
    ref_margin = ref_chosen - ref_rejected
    logits = beta * (policy_margin - ref_margin)
    return -F.logsigmoid(logits).mean(), logits.detach()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/preferences/preference_pairs_qwen25_k1.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/dpo_lora_coding"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for DPO LoRA training")

    device = torch.device("cuda:0")
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
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    model.train()
    model.print_trainable_parameters()

    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    rows = list(read_jsonl(args.data, args.limit))
    if not rows:
        raise RuntimeError(f"No preference pairs found in {args.data}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stats = {
        "data": str(args.data),
        "num_pairs": len(rows),
        "epochs": args.epochs,
        "lr": args.lr,
        "beta": args.beta,
        "max_length": args.max_length,
        "steps": 0,
        "skipped": 0,
        "mean_loss": None,
        "preference_accuracy": None,
    }

    losses: list[float] = []
    correct = 0
    seen = 0
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(args.epochs):
        for idx, row in enumerate(rows):
            prompt = row["prompt"]
            chosen = row["chosen"].strip() + "\n"
            rejected = row["rejected"].strip() + "\n"

            chosen_inputs = build_inputs(tokenizer, prompt, chosen, args.max_length, device)
            rejected_inputs = build_inputs(tokenizer, prompt, rejected, args.max_length, device)

            with torch.no_grad():
                with model.disable_adapter():
                    ref_chosen, chosen_tokens = sequence_logprob(model, *chosen_inputs)
                    ref_rejected, rejected_tokens = sequence_logprob(model, *rejected_inputs)

            if ref_chosen is None or ref_rejected is None or chosen_tokens == 0 or rejected_tokens == 0:
                stats["skipped"] += 1
                continue

            policy_chosen, _ = sequence_logprob(model, *chosen_inputs)
            policy_rejected, _ = sequence_logprob(model, *rejected_inputs)
            loss, margin = dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, args.beta)
            scaled_loss = loss / args.grad_accum
            scaled_loss.backward()

            losses.append(float(loss.detach().cpu()))
            correct += int(margin.item() > 0)
            seen += 1
            stats["steps"] += 1

            if stats["steps"] % args.grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if stats["steps"] % 10 == 0:
                mean_loss = sum(losses[-10:]) / min(10, len(losses))
                acc = correct / max(1, seen)
                print(
                    f"epoch={epoch} pair={idx + 1}/{len(rows)} "
                    f"step={stats['steps']} loss10={mean_loss:.4f} pref_acc={acc:.3f}"
                )

    if stats["steps"] % args.grad_accum:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    stats["mean_loss"] = sum(losses) / max(1, len(losses))
    stats["preference_accuracy"] = correct / max(1, seen)

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    with (args.output_dir / "train_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"saved DPO LoRA adapter to {args.output_dir}")


if __name__ == "__main__":
    main()
