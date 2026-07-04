#!/usr/bin/env python3
"""Run a small LLM critic -> revision loop for failed coding outputs."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


SYSTEM_PROMPT = (
    "You are a careful Python code reviewer for programming benchmark tasks. "
    "Find concrete errors in a failed answer, then write a corrected solution. "
    "Return JSON only."
)


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def clean_code(text: str) -> str:
    text = text or ""
    fenced = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1)
    text = text.strip("\n\r")
    if "\\n" in text and "\n" not in text:
        text = text.replace("\\n", "\n").replace("\\t", "\t")
    return text.strip("\n\r")


def split_from_id(row: dict) -> str | None:
    if row.get("split"):
        return row["split"]
    row_id = row.get("id", "")
    parts = row_id.split("/")
    if len(parts) >= 3 and parts[0] == "mbpp":
        return parts[1]
    if row.get("dataset") == "humanevalplus":
        return "test"
    return None


def copy_test_fields(source: dict, target: dict) -> None:
    for key in ("test_list", "test_setup_code", "test", "entry_point", "split"):
        if key in source:
            target[key] = source[key]


def load_preferred_ids(path: Path | None) -> set[str]:
    if not path:
        return set()
    preferred = set()
    for row in read_jsonl(path):
        if row.get("passed") and row.get("revision_edits"):
            preferred.add(row["id"])
    return preferred


def select_rows(
    rows: list[dict],
    dataset: str | None,
    split: str | None,
    failure_type: str | None,
    limit: int,
    preferred_ids: set[str],
) -> list[dict]:
    candidates = []
    for row in rows:
        if row.get("passed"):
            continue
        if dataset and row.get("dataset") != dataset:
            continue
        if split and split_from_id(row) != split:
            continue
        if failure_type and row.get("failure_type") != failure_type:
            continue
        candidates.append(row)

    if preferred_ids:
        preferred = [row for row in candidates if row.get("id") in preferred_ids]
        others = [row for row in candidates if row.get("id") not in preferred_ids]
        candidates = preferred + others
    return candidates[:limit]


def short_tests(row: dict, max_chars: int = 900) -> str:
    if row.get("dataset") == "mbpp":
        tests = "\n".join(row.get("test_list") or [])
    else:
        tests = row.get("test", "")
    return tests[:max_chars]


def build_user_prompt(row: dict, rubric: dict | None, prompt_mode: str = "default") -> str:
    dims = []
    for dim in (rubric or {}).get("dimensions", [])[:6]:
        dims.append(f"- {dim.get('dimension')}: {dim.get('description')}")
    rubric_text = "\n".join(dims) if dims else "- Correctness\n- Syntax\n- Interface compliance"
    failed_code = clean_code(row.get("generated_code") or row.get("extracted_code") or "")
    logic_hint = ""
    if row.get("failure_type") == "logic_error":
        logic_hint = (
            "\nThis is a semantic/logic failure. Do not only clean formatting. "
            "Infer the intended behavior from the task and tests, identify the wrong algorithm or edge-case handling, "
            "then produce a complete corrected implementation.\n"
        )
    if prompt_mode == "spec_first":
        return f"""Task prompt:
{row.get("prompt", "")[:1800]}

Failed answer A:
```python
{failed_code[:2200]}
```

Verifier failure type: {row.get("failure_type")}
Verifier error: {row.get("error")}
{logic_hint}

Relevant tests:
```python
{short_tests(row)}
```

Rubric dimensions:
{rubric_text}

Before writing code, use the visible tests as executable specifications.
For each assert/test line you can see, explain:
1. what the input means,
2. what the expected output means,
3. what general behavior or edge case the assertion implies.
Then infer the function specification, compare failed answer A against that specification, and only then write revised code.

Hard constraints:
- Preserve the required function name and signature from the task/tests.
- The revised code must be a complete Python implementation, not a patch.
- The revised code should satisfy every visible assert/test, not merely change formatting.
- Return JSON only.

Return JSON with exactly these keys:
{{
  "assertion_analysis": [
    {{
      "assertion": "visible assert or test",
      "input_meaning": "what this input represents",
      "expected_output_meaning": "what the output requires",
      "implied_spec": "behavior or edge case implied by this test"
    }}
  ],
  "inferred_spec": "concise task specification inferred from prompt and tests",
  "error_findings": ["specific issue in answer A relative to the inferred spec"],
  "revised_code": "valid Python code only"
}}
"""
    return f"""Task prompt:
{row.get("prompt", "")[:1800]}

Failed answer A:
```python
{failed_code[:2200]}
```

Verifier failure type: {row.get("failure_type")}
Verifier error: {row.get("error")}
{logic_hint}

Relevant tests:
```python
{short_tests(row)}
```

Rubric dimensions:
{rubric_text}

Return JSON with exactly these keys:
{{
  "error_findings": ["specific issue 1", "specific issue 2"],
  "revised_code": "valid Python code only"
}}
"""


def to_chat_prompt(tokenizer, user_prompt: str) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"{SYSTEM_PROMPT}\n\n{user_prompt}\n\nJSON:"


def parse_output(text: str) -> tuple[list[str], str]:
    text = text.strip()
    candidates = []
    if "{" in text and "}" in text:
        candidates.append(text[text.find("{") : text.rfind("}") + 1])
    candidates.append(text)
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except Exception:  # noqa: BLE001
            continue
        findings = obj.get("error_findings") or obj.get("critique") or []
        if isinstance(findings, str):
            findings = [findings]
        code = obj.get("revised_code") or obj.get("code") or obj.get("improved_code") or ""
        if code:
            return [str(item) for item in findings], clean_code(str(code))

    fenced = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return ["Model did not return strict JSON; code block was extracted as revised_code."], fenced.group(1).strip()
    return ["Model did not return parseable JSON; raw output was used as revised_code."], clean_code(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--labeled", type=Path, default=Path("data/responses/coding_all_qwen25_vllm_k1_labeled_v2.jsonl"))
    parser.add_argument("--rubric", type=Path, default=Path("data/rubrics/auto_rubric_refined.json"))
    parser.add_argument("--prefer-protected-success", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=str, default="mbpp")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--failure-type", type=str, default=None)
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--max-input-length", type=int, default=3072)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt-mode", choices=["default", "spec_first"], default="default")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for LLM critic generation")

    torch.manual_seed(args.seed)
    rows = list(read_jsonl(args.labeled))
    preferred_ids = load_preferred_ids(args.prefer_protected_success)
    selected = select_rows(rows, args.dataset, args.split, args.failure_type, args.limit, preferred_ids)
    if not selected:
        raise RuntimeError("No failed rows selected")

    rubric = json.loads(args.rubric.read_text(encoding="utf-8")) if args.rubric.exists() else None
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map={"": 0},
    )
    model.eval()
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id

    args.output.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    do_sample = args.temperature > 0

    with args.output.open("w", encoding="utf-8") as f, torch.inference_mode():
        for index, row in enumerate(selected, start=1):
            prompt = to_chat_prompt(tokenizer, build_user_prompt(row, rubric, args.prompt_mode))
            encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_input_length)
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
            critic_text = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]
            findings, revised_code = parse_output(critic_text)
            record = {
                "id": row["id"],
                "dataset": row["dataset"],
                "prompt": row.get("prompt"),
                "generated_code": revised_code,
                "original_generated_code": row.get("generated_code"),
                "response_a": clean_code(row.get("generated_code") or row.get("extracted_code") or ""),
                "critique": findings,
                "critic_text": critic_text,
                "model": args.model,
                "timestamp": now,
                "sample_id": 0,
                "seed": args.seed,
                "prompt_mode": args.prompt_mode,
                "generation_backend": "transformers_self_play_critic",
                "source_failure_type": row.get("failure_type"),
                "source_error": row.get("error"),
                "rubric_version": (rubric or {}).get("name"),
            }
            copy_test_fields(row, record)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"generated self-play revision {index}/{len(selected)}: {row['id']}")

    print(f"wrote {len(selected)} self-play critic revisions to {args.output}")


if __name__ == "__main__":
    main()
