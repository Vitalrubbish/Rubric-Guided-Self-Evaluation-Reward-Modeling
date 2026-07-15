#!/usr/bin/env python3
"""Train a CausalLM LoRA adapter on prompt/completion SFT rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments


IGNORE_INDEX = -100


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            if "prompt" not in row or "completion" not in row:
                raise ValueError(f"row {line_number} must contain prompt and completion")
            rows.append(row)
    return rows


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def split_rows(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("split") == split]


def percentile(values: list[int], fraction: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * fraction))]


def stats(values: list[int]) -> dict[str, float | int]:
    return {
        "min": min(values, default=0),
        "mean": mean(values) if values else 0.0,
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values, default=0),
    }


def task_type_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row.get("task_type") or "unknown")
        counts[key] = counts.get(key, 0) + 1
    return counts


def format_prompt(tokenizer: Any, prompt: str, prompt_format: str) -> str:
    prompt = prompt.strip()
    if prompt_format == "raw":
        return prompt + "\n\n"
    if prompt_format == "chat":
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    raise ValueError(f"unsupported prompt format: {prompt_format}")


def build_features(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    max_length: int,
    prompt_format: str,
    seed: int,
    max_rows: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected = rows[:]
    random.Random(seed).shuffle(selected)
    if max_rows is not None:
        selected = selected[:max_rows]

    features: list[dict[str, Any]] = []
    prompt_lengths: list[int] = []
    completion_lengths: list[int] = []
    total_lengths: list[int] = []
    skipped_too_long = 0
    skipped_empty_completion = 0
    skipped_prompt_too_long = 0

    eos = tokenizer.eos_token or ""
    for row in selected:
        prompt_text = format_prompt(tokenizer, str(row["prompt"]), prompt_format)
        completion_text = str(row["completion"]).strip()
        if not completion_text:
            skipped_empty_completion += 1
            continue
        completion_text = completion_text + eos
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
        completion_ids = tokenizer(completion_text, add_special_tokens=False).input_ids
        if len(prompt_ids) >= max_length:
            skipped_prompt_too_long += 1
            continue
        if len(prompt_ids) + len(completion_ids) > max_length:
            skipped_too_long += 1
            continue
        input_ids = prompt_ids + completion_ids
        labels = [IGNORE_INDEX] * len(prompt_ids) + completion_ids
        features.append(
            {
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "labels": labels,
                "task_type": row.get("task_type"),
                "id": row.get("id"),
            }
        )
        prompt_lengths.append(len(prompt_ids))
        completion_lengths.append(len(completion_ids))
        total_lengths.append(len(input_ids))

    if not features:
        raise RuntimeError("all SFT rows were removed by token gates")

    audit = {
        "input_rows": len(rows),
        "selected_before_token_gate": len(selected),
        "training_rows": len(features),
        "skipped_too_long": skipped_too_long,
        "skipped_empty_completion": skipped_empty_completion,
        "skipped_prompt_too_long": skipped_prompt_too_long,
        "max_length": max_length,
        "prompt_format": prompt_format,
        "prompt_tokens": stats(prompt_lengths),
        "completion_tokens": stats(completion_lengths),
        "total_tokens": stats(total_lengths),
        "task_type_counts_after_gate": task_type_counts(features),
    }
    return features, audit


@dataclass
class CompletionOnlyCollator:
    tokenizer: Any

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_len = max(len(feature["input_ids"]) for feature in features)
        pad_id = self.tokenizer.pad_token_id
        input_ids = []
        attention_mask = []
        labels = []
        for feature in features:
            pad_len = max_len - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [pad_id] * pad_len)
            attention_mask.append(feature["attention_mask"] + [0] * pad_len)
            labels.append(feature["labels"] + [IGNORE_INDEX] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def strip_metadata(features: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "input_ids": feature["input_ids"],
            "attention_mask": feature["attention_mask"],
            "labels": feature["labels"],
        }
        for feature in features
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CausalLM LoRA SFT on prompt/completion rows.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--prompt-format", choices=("raw", "chat"), default="raw")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--max-validation-rows", type=int, default=None)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.max_length < 512:
        raise ValueError("--max-length must be at least 512")

    rows = read_jsonl(args.data)
    train_rows = split_rows(rows, "train")
    validation_rows = split_rows(rows, "validation")
    if not train_rows or not validation_rows:
        raise SystemExit("data must contain train and validation splits")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_features, train_audit = build_features(
        train_rows,
        tokenizer,
        max_length=args.max_length,
        prompt_format=args.prompt_format,
        seed=args.seed,
        max_rows=args.max_train_rows,
    )
    validation_features, validation_audit = build_features(
        validation_rows,
        tokenizer,
        max_length=args.max_length,
        prompt_format=args.prompt_format,
        seed=args.seed,
        max_rows=args.max_validation_rows,
    )

    manifest = {
        "status": "dry_run" if args.dry_run else "initializing",
        "started_at": utc_now(),
        "git_commit": git_commit(),
        "hostname": platform.node(),
        "model": args.model,
        "data": str(args.data),
        "data_sha256": sha256_file(args.data),
        "trainer_source": str(Path(__file__).resolve()),
        "trainer_source_sha256": sha256_file(Path(__file__).resolve()),
        "output_dir": str(args.output_dir),
        "row_counts": {"train": len(train_rows), "validation": len(validation_rows)},
        "raw_task_type_counts": {
            "train": task_type_counts(train_rows),
            "validation": task_type_counts(validation_rows),
        },
        "token_audit": {"train": train_audit, "validation": validation_audit},
        "hyperparameters": {
            "max_length": args.max_length,
            "prompt_format": args.prompt_format,
            "epochs": args.epochs,
            "max_train_rows": args.max_train_rows,
            "max_validation_rows": args.max_validation_rows,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "per_device_eval_batch_size": args.per_device_eval_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "warmup_ratio": args.warmup_ratio,
            "weight_decay": args.weight_decay,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "seed": args.seed,
        },
        "policy": {
            "model_form": "AutoModelForCausalLM with LoRA",
            "loss": "completion-only causal language modeling",
            "purpose": "joint solve/critique/judge SFT before generator DPO or self-generated rewards",
        },
        "software": {"python": platform.python_version(), "torch": torch.__version__},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "run_manifest.json"
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        return

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for CausalLM SFT training")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        logging_first_step=True,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        remove_unused_columns=False,
        seed=args.seed,
        data_seed=args.seed,
        optim="adamw_torch_fused",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=Dataset.from_list(strip_metadata(train_features)),
        eval_dataset=Dataset.from_list(strip_metadata(validation_features)),
        processing_class=tokenizer,
        data_collator=CompletionOnlyCollator(tokenizer),
    )

    manifest["status"] = "training"
    manifest["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    manifest["gpu_name"] = torch.cuda.get_device_name(0)
    manifest["gpu_total_memory_bytes"] = torch.cuda.get_device_properties(0).total_memory
    write_json(manifest_path, manifest)

    result = trainer.train()
    eval_metrics = trainer.evaluate()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(args.output_dir)
    trainer.save_state()

    train_metrics = dict(result.metrics)
    train_metrics["training_rows"] = train_audit["training_rows"]
    train_metrics["validation_rows"] = validation_audit["training_rows"]
    train_metrics["global_step"] = trainer.state.global_step
    if "eval_loss" in eval_metrics:
        eval_metrics["eval_perplexity"] = float(math.exp(eval_metrics["eval_loss"])) if eval_metrics["eval_loss"] < 20 else None
    write_json(args.output_dir / "train_metrics.json", train_metrics)
    write_json(args.output_dir / "eval_metrics.json", dict(eval_metrics))

    adapter_path = args.output_dir / "adapter_model.safetensors"
    manifest.update(
        {
            "status": "completed",
            "completed_at": utc_now(),
            "train_metrics": train_metrics,
            "eval_metrics": dict(eval_metrics),
            "adapter_model_bytes": adapter_path.stat().st_size if adapter_path.exists() else None,
            "final_global_step": trainer.state.global_step,
        }
    )
    if any(isinstance(value, float) and not math.isfinite(value) for value in train_metrics.values()):
        raise RuntimeError(f"non-finite training metric found: {train_metrics}")
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
