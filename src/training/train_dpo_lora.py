#!/usr/bin/env python3
"""Train a Qwen LoRA adapter with formal DPO on coding preferences."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer


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
            missing = {"prompt", "chosen", "rejected"}.difference(row)
            if missing:
                raise ValueError(f"row {line_number} is missing fields: {sorted(missing)}")
            rows.append(row)
    return rows


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
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def percentile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    if not ordered:
        return 0
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * fraction))]


def token_stats(values: list[int]) -> dict[str, float | int]:
    return {
        "min": min(values, default=0),
        "mean": mean(values) if values else 0.0,
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values, default=0),
    }


def format_prompt(tokenizer: Any, prompt: str, prompt_format: str) -> str:
    if prompt_format == "raw":
        return prompt.strip() + "\n"
    if prompt_format == "chat":
        messages = [{"role": "user", "content": prompt.strip()}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    raise ValueError(f"unsupported prompt format: {prompt_format}")


def capped_token_length(tokenizer: Any, value: str, cap: int) -> int:
    return len(
        tokenizer(
            value,
            add_special_tokens=False,
            truncation=True,
            max_length=cap,
        ).input_ids
    )


def prepare_rows(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    max_length: int,
    min_completion_tokens: int,
    prompt_format: str,
    seed: int,
    max_pairs: int | None,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    rng = random.Random(seed)
    shuffled = rows[:]
    rng.shuffle(shuffled)
    if max_pairs is not None:
        shuffled = shuffled[:max_pairs]

    prepared: list[dict[str, str]] = []
    prompt_lengths: list[int] = []
    chosen_lengths: list[int] = []
    rejected_lengths: list[int] = []
    chosen_total_lengths: list[int] = []
    rejected_total_lengths: list[int] = []
    skipped_prompt_too_long = 0
    skipped_chosen_too_long = 0
    skipped_rejected_too_long = 0

    for row in shuffled:
        prompt = format_prompt(tokenizer, str(row["prompt"]), prompt_format)
        chosen = str(row["chosen"]).strip() + "\n"
        rejected = str(row["rejected"]).strip() + "\n"
        cap = max_length + 1
        prompt_length = capped_token_length(tokenizer, prompt, cap)
        chosen_length = capped_token_length(tokenizer, chosen, cap)
        rejected_length = capped_token_length(tokenizer, rejected, cap)
        if prompt_length > max_length - min_completion_tokens:
            skipped_prompt_too_long += 1
            continue
        if prompt_length + chosen_length > max_length:
            skipped_chosen_too_long += 1
            continue
        if prompt_length + rejected_length > max_length:
            skipped_rejected_too_long += 1
            continue

        prepared.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
        prompt_lengths.append(prompt_length)
        chosen_lengths.append(chosen_length)
        rejected_lengths.append(rejected_length)
        chosen_total_lengths.append(prompt_length + chosen_length)
        rejected_total_lengths.append(prompt_length + rejected_length)

    if not prepared:
        raise RuntimeError("all preference rows were removed by the token-length gate")

    audit = {
        "input_pair_count": len(rows),
        "selected_before_token_gate": len(shuffled),
        "training_pair_count": len(prepared),
        "skipped_prompt_too_long": skipped_prompt_too_long,
        "skipped_chosen_too_long": skipped_chosen_too_long,
        "skipped_rejected_too_long": skipped_rejected_too_long,
        "max_length": max_length,
        "min_completion_tokens": min_completion_tokens,
        "prompt_format": prompt_format,
        "prompt_tokens": token_stats(prompt_lengths),
        "chosen_completion_tokens": token_stats(chosen_lengths),
        "rejected_completion_tokens": token_stats(rejected_lengths),
        "prompt_plus_chosen_tokens": token_stats(chosen_total_lengths),
        "prompt_plus_rejected_tokens": token_stats(rejected_total_lengths),
        "chosen_truncation_count": 0,
        "rejected_truncation_count": 0,
    }
    return prepared, audit


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--min-completion-tokens", type=int, default=128)
    parser.add_argument("--prompt-format", choices=("raw", "chat"), default="raw")
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument(
        "--ld-alpha",
        type=float,
        default=None,
        help="LD-DPO tail-token weight in [0, 1]; 1 is standard DPO and 0 masks unequal-length tails.",
    )
    parser.add_argument(
        "--loss-type",
        action="append",
        choices=(
            "sigmoid",
            "hinge",
            "ipo",
            "exo_pair",
            "nca_pair",
            "robust",
            "bco_pair",
            "sppo_hard",
            "aot",
            "aot_unpaired",
            "apo_zero",
            "apo_down",
            "discopop",
            "sft",
            "sigmoid_norm",
        ),
        help="TRL DPO loss; repeat to combine losses. Defaults to sigmoid.",
    )
    parser.add_argument(
        "--loss-weight",
        action="append",
        type=float,
        help="Weight for each repeated --loss-type, in the same order.",
    )
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.max_length < 256:
        raise ValueError("--max-length must be at least 256")
    if args.min_completion_tokens >= args.max_length:
        raise ValueError("--min-completion-tokens must be smaller than --max-length")
    if args.ld_alpha is not None and not 0.0 <= args.ld_alpha <= 1.0:
        raise ValueError("--ld-alpha must be in [0, 1]")
    loss_types = args.loss_type or ["sigmoid"]
    loss_weights = args.loss_weight
    if loss_weights is not None and len(loss_weights) != len(loss_types):
        raise ValueError("--loss-weight count must match --loss-type count")
    if loss_weights is not None and any(weight < 0 for weight in loss_weights):
        raise ValueError("--loss-weight values must be non-negative")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    source_rows = read_jsonl(args.data)
    prepared_rows, token_audit = prepare_rows(
        source_rows,
        tokenizer,
        max_length=args.max_length,
        min_completion_tokens=args.min_completion_tokens,
        prompt_format=args.prompt_format,
        seed=args.seed,
        max_pairs=args.max_pairs,
    )
    manifest: dict[str, Any] = {
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
        "token_audit": token_audit,
        "hyperparameters": {
            "epochs": args.epochs,
            "max_length": args.max_length,
            "prompt_format": args.prompt_format,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "beta": args.beta,
            "ld_alpha": args.ld_alpha,
            "loss_types": loss_types,
            "loss_weights": loss_weights,
            "warmup_ratio": args.warmup_ratio,
            "weight_decay": args.weight_decay,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "seed": args.seed,
        },
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "run_manifest.json"
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        return

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for DPO training")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    training_args = DPOConfig(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        beta=args.beta,
        ld_alpha=args.ld_alpha,
        loss_type=loss_types,
        loss_weights=loss_weights,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_length=args.max_length,
        truncation_mode="keep_start",
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        use_cache=False,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        logging_first_step=True,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        report_to="none",
        remove_unused_columns=False,
        seed=args.seed,
        data_seed=args.seed,
        optim="adamw_torch_fused",
    )
    dataset = Dataset.from_list(prepared_rows)
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    manifest["status"] = "training"
    manifest["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    manifest["gpu_name"] = torch.cuda.get_device_name(0)
    manifest["gpu_total_memory_bytes"] = torch.cuda.get_device_properties(0).total_memory
    write_json(manifest_path, manifest)

    result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(args.output_dir)
    trainer.save_state()
    metrics = dict(result.metrics)
    metrics["training_pair_count"] = len(prepared_rows)
    metrics["effective_batch_size"] = (
        args.per_device_train_batch_size * args.gradient_accumulation_steps
    )
    metrics["global_step"] = trainer.state.global_step
    if any(isinstance(value, float) and not math.isfinite(value) for value in metrics.values()):
        raise RuntimeError(f"non-finite training metric found: {metrics}")
    write_json(args.output_dir / "train_metrics.json", metrics)

    adapter_path = args.output_dir / "adapter_model.safetensors"
    if not adapter_path.exists() or adapter_path.stat().st_size == 0:
        raise RuntimeError("training ended without a non-empty adapter_model.safetensors")

    manifest.update(
        {
            "status": "completed",
            "completed_at": utc_now(),
            "metrics": metrics,
            "adapter_model_bytes": adapter_path.stat().st_size,
            "final_global_step": trainer.state.global_step,
        }
    )
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
