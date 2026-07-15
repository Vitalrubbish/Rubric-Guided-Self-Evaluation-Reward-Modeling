#!/usr/bin/env python3
"""Create a reproducible inference-only LoRA adapter with scaled effective strength."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scaled_lora_config(config: dict[str, Any], scale: float) -> tuple[dict[str, Any], float, float]:
    if not 0.0 < scale <= 1.0:
        raise ValueError("scale must be in (0, 1]")
    if config.get("use_rslora") or config.get("use_dora") or config.get("alpha_pattern"):
        raise ValueError("only plain LoRA with a single global alpha can be scaled this way")
    rank = float(config.get("r") or 0)
    original_alpha = float(config.get("lora_alpha") or 0)
    if rank <= 0 or original_alpha <= 0:
        raise ValueError("adapter config must contain positive r and lora_alpha")
    scaled_alpha = original_alpha * scale
    if scaled_alpha.is_integer():
        scaled_alpha = int(scaled_alpha)
    output = {**config, "lora_alpha": scaled_alpha}
    return output, original_alpha, float(scaled_alpha)


def main() -> None:
    parser = argparse.ArgumentParser(description="Scale a plain LoRA adapter through its alpha/r multiplier.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scale", type=float, required=True)
    args = parser.parse_args()

    source_config_path = args.input_dir / "adapter_config.json"
    source_model_path = args.input_dir / "adapter_model.safetensors"
    if not source_config_path.is_file() or not source_model_path.is_file():
        raise FileNotFoundError("input adapter must contain adapter_config.json and adapter_model.safetensors")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")

    source_config = json.loads(source_config_path.read_text(encoding="utf-8"))
    output_config, original_alpha, scaled_alpha = scaled_lora_config(source_config, args.scale)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    copy_names = (
        "adapter_model.safetensors",
        "chat_template.jinja",
        "README.md",
        "tokenizer_config.json",
        "tokenizer.json",
    )
    for name in copy_names:
        source = args.input_dir / name
        if source.is_file():
            shutil.copy2(source, args.output_dir / name)
    output_config_path = args.output_dir / "adapter_config.json"
    output_config_path.write_text(json.dumps(output_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    source_run_manifest = args.input_dir / "run_manifest.json"
    if source_run_manifest.is_file():
        shutil.copy2(source_run_manifest, args.output_dir / "source_training_run_manifest.json")

    output_model_path = args.output_dir / "adapter_model.safetensors"
    manifest = {
        "status": "completed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_adapter": str(args.input_dir),
        "output_adapter": str(args.output_dir),
        "scale": args.scale,
        "rank": source_config["r"],
        "original_lora_alpha": original_alpha,
        "scaled_lora_alpha": scaled_alpha,
        "original_effective_alpha_over_r": original_alpha / float(source_config["r"]),
        "scaled_effective_alpha_over_r": scaled_alpha / float(source_config["r"]),
        "source_adapter_model_sha256": sha256_file(source_model_path),
        "output_adapter_model_sha256": sha256_file(output_model_path),
        "source_adapter_config_sha256": sha256_file(source_config_path),
        "output_adapter_config_sha256": sha256_file(output_config_path),
        "policy": "weights are byte-identical; only the plain-LoRA alpha/r inference multiplier is scaled",
    }
    manifest_path = args.output_dir / "scale_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
