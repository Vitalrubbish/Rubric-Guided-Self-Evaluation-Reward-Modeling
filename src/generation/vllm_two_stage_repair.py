#!/usr/bin/env python3
"""Generate APPS self-repairs with an explicit public-spec analysis stage."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vllm import LLM, SamplingParams


def read_jsonl(path: Path, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
            if len(rows) >= limit:
                break
    return rows


def interface_text(row: dict[str, Any]) -> str:
    signatures = row.get("interface_signatures") or []
    names = row.get("interface_names") or []
    values = signatures or names
    return "\n".join(str(value) for value in values) if values else "stdin/stdout program"


def public_task(row: dict[str, Any]) -> str:
    task = str(row.get("task") or "").strip()
    if task:
        return task
    prompt = str(row.get("prompt") or "").strip()
    marker = "Task:\n"
    end_marker = "\n\nPrevious failed code:"
    if marker in prompt and end_marker in prompt:
        return prompt.split(marker, 1)[1].split(end_marker, 1)[0].strip()
    return prompt


def previous_code(row: dict[str, Any]) -> str:
    return str(row.get("previous_code") or row.get("original_generated_code") or "").strip()


def build_spec_prompt(row: dict[str, Any]) -> str:
    return (
        "You are the critic stage of a Python self-repair system. The previous code failed an external verifier.\n"
        "Use only the public task, public interface, and visible failed code. Never invent or request hidden tests.\n"
        "For every example or assertion explicitly present in the public task, explain the input, expected output, "
        "and what behavior it specifies. If none are public, return an empty list.\n"
        "Then infer the complete specification, identify likely root causes in the failed code, and list edge cases.\n"
        "Return one compact JSON object with keys public_case_explanations, inferred_specification, likely_root_causes, "
        "and edge_cases. Do not return revised code in this stage.\n\n"
        f"Public interface:\n{interface_text(row)}\n\n"
        f"Public task:\n{public_task(row)}\n\n"
        f"Previous failed code:\n{previous_code(row)}\n"
    )


def build_repair_prompt(row: dict[str, Any], spec_text: str) -> str:
    return (
        "You are the repair stage of a Python self-repair system.\n"
        "Write a complete corrected solution using the public task and the critic's inferred specification.\n"
        "The evaluator parses your entire completion as one Python file, so the whole completion must pass ast.parse.\n"
        "Preserve the required interface and emit exactly one implementation. Never repeat a function/class definition.\n"
        "Do not include tests, example calls, doctests, Markdown fences, prose, arrows, or critic text.\n"
        "Start with Python code and stop immediately after the final implementation statement.\n\n"
        f"Public interface:\n{interface_text(row)}\n\n"
        f"Public task:\n{public_task(row)}\n\n"
        f"Previous failed code:\n{previous_code(row)}\n\n"
        f"Critic analysis and inferred specification:\n{spec_text.strip()}\n\n"
        "Revised Python code:\n"
    )


def format_model_prompt(tokenizer: Any, prompt: str, prompt_format: str) -> str:
    if prompt_format == "raw":
        return prompt
    if prompt_format == "chat":
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    raise ValueError(f"unsupported prompt format: {prompt_format}")


def completion_record(
    row: dict[str, Any],
    spec_text: str,
    completion: Any,
    args: argparse.Namespace,
    timestamp: str,
    sample_id: int,
) -> dict[str, Any]:
    candidate_id = str(row.get("repair_candidate_id") or row.get("response_id_prefix") or row.get("id"))
    record = {
        "response_id": f"{candidate_id}__two_stage_sample{sample_id}",
        "id": row["id"],
        "dataset": row.get("dataset", "apps"),
        "split": row.get("split"),
        "prompt_mode": f"method1_two_stage_public_spec_repair_v2_{args.prompt_format}",
        "generation_prompt_format": args.prompt_format,
        "interface_names": row.get("interface_names") or [],
        "interface_signatures": row.get("interface_signatures") or [],
        "task": public_task(row),
        "previous_code": previous_code(row),
        "spec_text": spec_text,
        "generated_code": completion.text,
        "model": args.model,
        "timestamp": timestamp,
        "sample_id": sample_id,
        "seed": args.seed,
        "temperature": args.repair_temperature,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "max_tokens": args.repair_max_tokens,
        "finish_reason": getattr(completion, "finish_reason", None),
        "stop_reason": getattr(completion, "stop_reason", None),
        "generated_token_count": len(getattr(completion, "token_ids", []) or []),
    }
    for key in (
        "starter_code",
        "input_output",
        "difficulty",
        "io_mode",
        "repair_candidate_id",
        "original_response_id",
        "original_generated_code",
        "critic_pass_probability",
        "critic_selected_threshold",
        "critic_predicted_pass",
        "selection_reason",
        "original_failure_type",
    ):
        if key in row:
            record[key] = row.get(key)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Two-stage public-spec APPS repair generation with vLLM.")
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/repair/apps_simple_method1_repair_prompts_v2.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/repair/apps_simple_method1_two_stage_repair_v2_responses.jsonl"),
    )
    parser.add_argument("--limit", type=int, default=400)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--spec-max-tokens", type=int, default=640)
    parser.add_argument("--repair-max-tokens", type=int, default=2048)
    parser.add_argument("--repair-temperature", type=float, default=0.2)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--prompt-format", choices=("raw", "chat"), default="chat")
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--max-model-len", type=int, default=12288)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.35)
    parser.add_argument("--prompt-batch-size", type=int, default=20)
    args = parser.parse_args()

    rows = read_jsonl(args.input, args.limit)
    if not rows:
        raise RuntimeError("two-stage repair input is empty")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=1,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    tokenizer = llm.get_tokenizer()
    spec_sampling = SamplingParams(n=1, temperature=0.0, top_p=1.0, max_tokens=args.spec_max_tokens, seed=args.seed)
    repair_sampling = SamplingParams(
        n=args.k,
        temperature=args.repair_temperature,
        top_p=args.top_p,
        max_tokens=args.repair_max_tokens,
        repetition_penalty=args.repetition_penalty,
        seed=args.seed,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    written = 0
    with args.output.open("w", encoding="utf-8") as handle:
        for start in range(0, len(rows), args.prompt_batch_size):
            batch = rows[start : start + args.prompt_batch_size]
            spec_outputs = llm.generate(
                [format_model_prompt(tokenizer, build_spec_prompt(row), args.prompt_format) for row in batch],
                spec_sampling,
            )
            spec_texts = [output.outputs[0].text for output in spec_outputs]
            repair_outputs = llm.generate(
                [
                    format_model_prompt(tokenizer, build_repair_prompt(row, spec), args.prompt_format)
                    for row, spec in zip(batch, spec_texts)
                ],
                repair_sampling,
            )
            for row, spec_text, output in zip(batch, spec_texts, repair_outputs):
                for sample_id, completion in enumerate(output.outputs):
                    record = completion_record(row, spec_text, completion, args, timestamp, sample_id)
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    written += 1
            handle.flush()
            print(f"two-stage batch={start // args.prompt_batch_size + 1} generated={written}/{len(rows) * args.k}")
    print(f"wrote {written} two-stage repair responses to {args.output}")


if __name__ == "__main__":
    main()
