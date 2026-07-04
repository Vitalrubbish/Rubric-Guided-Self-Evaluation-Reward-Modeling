#!/usr/bin/env python3
"""Run a spec -> algorithm sketch -> code self-play repair loop."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


SPEC_SYSTEM_PROMPT = (
    "You are a careful Python benchmark specification analyst. "
    "Given a failed solution and visible tests, infer the intended task behavior. "
    "Do not write revised code in this stage. Return JSON only."
)

ALGORITHM_SYSTEM_PROMPT = (
    "You are a Python algorithm designer for programming benchmark tasks. "
    "Given a task, visible tests, and an inferred specification, write a concrete algorithm sketch "
    "and manually simulate visible tests before any code is written. Return JSON only."
)

CODE_SYSTEM_PROMPT = (
    "You are a careful Python implementer for programming benchmark tasks. "
    "Use the inferred specification and algorithm sketch to write a complete corrected solution. "
    "Return JSON only."
)

REPAIR_SYSTEM_PROMPT = (
    "You are a Python syntax and formatting repair assistant. "
    "Fix only syntax, indentation, and escaped-newline formatting issues. "
    "Do not change the algorithmic intent unless needed to make the same code syntactically valid. "
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
    if "\\n" in text and (
        "\n" not in text or text.count("\\n") >= text.count("\n") or ":\\n" in text or "\\n    " in text
    ):
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


def select_rows(
    rows: list[dict],
    dataset: str | None,
    split: str | None,
    failure_type: str | None,
    limit: int,
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
    return candidates[:limit]


def short_tests(row: dict, max_chars: int = 1400) -> str:
    if row.get("dataset") == "mbpp":
        tests = "\n".join(row.get("test_list") or [])
    else:
        tests = row.get("test", "")
    return tests[:max_chars]


def rubric_text(rubric: dict | None) -> str:
    dims = []
    for dim in (rubric or {}).get("dimensions", [])[:6]:
        dims.append(f"- {dim.get('dimension')}: {dim.get('description')}")
    return "\n".join(dims) if dims else "- Correctness\n- Syntax\n- Interface compliance"


def build_spec_prompt(row: dict, rubric: dict | None) -> str:
    failed_code = clean_code(row.get("generated_code") or row.get("extracted_code") or "")
    return f"""Task prompt:
{row.get("prompt", "")[:1800]}

Failed answer A:
```python
{failed_code[:2200]}
```

Verifier failure type: {row.get("failure_type")}
Verifier error: {row.get("error")}

Relevant tests:
```python
{short_tests(row)}
```

Rubric dimensions:
{rubric_text(rubric)}

Analyze the visible tests as executable specifications. For each visible assert/test, explain:
- what the input means,
- what the expected output means,
- what general behavior or edge case it implies.

Then infer the intended function behavior and identify why answer A is likely wrong.

Do not write revised code in this stage.

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
  "inferred_spec": "concise function specification inferred from prompt and tests",
  "suspected_errors": ["specific issue in answer A relative to the inferred spec"],
  "implementation_constraints": ["required function name/signature, imports, output type, edge cases"]
}}
"""


def build_algorithm_prompt(row: dict, spec_text: str) -> str:
    failed_code = clean_code(row.get("generated_code") or row.get("extracted_code") or "")
    return f"""Task prompt:
{row.get("prompt", "")[:1800]}

Failed answer A:
```python
{failed_code[:1600]}
```

Visible tests:
```python
{short_tests(row)}
```

Inferred specification from Stage 1:
```json
{spec_text[:3000]}
```

Design the algorithm before writing code.

Hard constraints:
- Do not write Python code in this stage.
- State the required function name and arguments from the tests.
- Give a concrete algorithm, including indexing conventions, edge cases, output type, and ordering requirements.
- Manually simulate at least two visible tests. For each simulation, show the input, the key intermediate reasoning, and the expected output.
- If the failed answer A uses the wrong algorithm, explicitly say what must change.

Return JSON with exactly these keys:
{{
  "required_interface": "function name and signature implied by tests",
  "algorithm_steps": ["step 1", "step 2", "step 3"],
  "edge_cases": ["edge case handled by the algorithm"],
  "test_simulations": [
    {{
      "assertion": "visible assert",
      "manual_trace": "short trace from input to expected output",
      "expected_output": "expected output"
    }}
  ],
  "changes_from_failed_answer": ["specific algorithmic change required before coding"]
}}
"""


def build_code_prompt(row: dict, spec_text: str, algorithm_text: str) -> str:
    failed_code = clean_code(row.get("generated_code") or row.get("extracted_code") or "")
    return f"""Task prompt:
{row.get("prompt", "")[:1800]}

Failed answer A:
```python
{failed_code[:1600]}
```

Visible tests:
```python
{short_tests(row)}
```

Inferred specification from Stage 1:
```json
{spec_text[:2400]}
```

Algorithm sketch and manual test simulation from Stage 2:
```json
{algorithm_text[:3000]}
```

Now write a complete corrected Python implementation.

Before finalizing the code, silently check that it follows the algorithm sketch and the manual simulations.

Hard constraints:
- Preserve the required function name and call signature implied by the tests.
- Return complete Python code, not a patch.
- Use normal multi-line Python with indentation.
- Do not compress loops, conditionals, or multiple statements into one line with semicolons.
- Do not include Markdown fences in the revised_code string.
- The code should satisfy every visible assert/test and generalize beyond them.

Return JSON with exactly these keys:
{{
  "error_findings": ["specific issue in answer A fixed by the revised code"],
  "algorithm_used": "brief summary of the implemented algorithm",
  "revised_code": "complete valid Python code"
}}
"""


def build_syntax_repair_prompt(row: dict, code: str, compile_error: str) -> str:
    return f"""The following revised code failed Python compilation before benchmark tests.

Task prompt:
{row.get("prompt", "")[:1200]}

Compile error:
{compile_error[:600]}

Code to repair:
```python
{code[:2400]}
```

Fix only syntax, indentation, escaped newline, and formatting problems. Preserve the same function name and intended algorithm.

Return JSON with exactly these keys:
{{
  "repair_notes": ["syntax or formatting issue fixed"],
  "revised_code": "complete valid Python code"
}}
"""


def to_chat_prompt(tokenizer, system_prompt: str, user_prompt: str) -> str:
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"{system_prompt}\n\n{user_prompt}\n\nJSON:"


def generate_text(
    model,
    tokenizer,
    system_prompt: str,
    user_prompt: str,
    max_input_length: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    prompt = to_chat_prompt(tokenizer, system_prompt, user_prompt)
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_length)
    encoded = {key: value.cuda() for key, value in encoded.items()}
    kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if temperature > 0:
        kwargs["temperature"] = temperature
        kwargs["top_p"] = top_p
    output_ids = model.generate(**encoded, **kwargs)
    new_tokens = output_ids[:, encoded["input_ids"].shape[1] :]
    return tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]


def parse_json_object(text: str) -> dict | None:
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
        if isinstance(obj, dict):
            return obj
    return None


def normalize_json_text(text: str) -> tuple[dict | None, str]:
    obj = parse_json_object(text)
    if obj is None:
        return None, text.strip()
    return obj, json.dumps(obj, ensure_ascii=False, indent=2)


def parse_code_output(text: str) -> tuple[list[str], str, str]:
    obj = parse_json_object(text)
    if obj is not None:
        findings = obj.get("error_findings") or obj.get("repair_notes") or obj.get("critique") or []
        if isinstance(findings, str):
            findings = [findings]
        algorithm_used = str(obj.get("algorithm_used") or "")
        code = obj.get("revised_code") or obj.get("code") or obj.get("improved_code") or ""
        if code:
            return [str(item) for item in findings], clean_code(str(code)), algorithm_used

    fenced = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return ["Model did not return strict JSON; code block was extracted as revised_code."], clean_code(fenced.group(1)), ""
    return ["Model did not return parseable JSON; raw output was used as revised_code."], clean_code(text), ""


def compile_error(code: str) -> str | None:
    if not code.strip():
        return "empty output"
    try:
        compile(code, "<candidate>", "exec")
    except SyntaxError as exc:
        return str(exc)
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--labeled", type=Path, default=Path("data/responses/coding_all_qwen25_vllm_k1_labeled_v2.jsonl"))
    parser.add_argument("--rubric", type=Path, default=Path("data/rubrics/auto_rubric_refined.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=str, default="mbpp")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--failure-type", type=str, default=None)
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--max-input-length", type=int, default=6144)
    parser.add_argument("--spec-max-new-tokens", type=int, default=640)
    parser.add_argument("--algorithm-max-new-tokens", type=int, default=768)
    parser.add_argument("--code-max-new-tokens", type=int, default=640)
    parser.add_argument("--repair-max-new-tokens", type=int, default=384)
    parser.add_argument("--syntax-repair-attempts", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for LLM critic generation")

    torch.manual_seed(args.seed)
    rows = list(read_jsonl(args.labeled))
    selected = select_rows(rows, args.dataset, args.split, args.failure_type, args.limit)
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

    with args.output.open("w", encoding="utf-8") as f, torch.inference_mode():
        for index, row in enumerate(selected, start=1):
            spec_text = generate_text(
                model,
                tokenizer,
                SPEC_SYSTEM_PROMPT,
                build_spec_prompt(row, rubric),
                args.max_input_length,
                args.spec_max_new_tokens,
                args.temperature,
                args.top_p,
            )
            spec_obj, normalized_spec_text = normalize_json_text(spec_text)

            algorithm_text = generate_text(
                model,
                tokenizer,
                ALGORITHM_SYSTEM_PROMPT,
                build_algorithm_prompt(row, normalized_spec_text),
                args.max_input_length,
                args.algorithm_max_new_tokens,
                args.temperature,
                args.top_p,
            )
            algorithm_obj, normalized_algorithm_text = normalize_json_text(algorithm_text)

            code_text = generate_text(
                model,
                tokenizer,
                CODE_SYSTEM_PROMPT,
                build_code_prompt(row, normalized_spec_text, normalized_algorithm_text),
                args.max_input_length,
                args.code_max_new_tokens,
                args.temperature,
                args.top_p,
            )
            findings, revised_code, algorithm_used = parse_code_output(code_text)

            syntax_repair_texts = []
            compile_error_before_repair = compile_error(revised_code)
            compile_error_after_repair = compile_error_before_repair
            repair_attempts_used = 0
            while compile_error_after_repair and repair_attempts_used < args.syntax_repair_attempts:
                repair_attempts_used += 1
                repair_text = generate_text(
                    model,
                    tokenizer,
                    REPAIR_SYSTEM_PROMPT,
                    build_syntax_repair_prompt(row, revised_code, compile_error_after_repair),
                    args.max_input_length,
                    args.repair_max_new_tokens,
                    args.temperature,
                    args.top_p,
                )
                repair_findings, repaired_code, _ = parse_code_output(repair_text)
                syntax_repair_texts.append(repair_text)
                if repaired_code:
                    findings.extend(repair_findings)
                    revised_code = repaired_code
                compile_error_after_repair = compile_error(revised_code)

            record = {
                "id": row["id"],
                "dataset": row["dataset"],
                "prompt": row.get("prompt"),
                "generated_code": revised_code,
                "original_generated_code": row.get("generated_code"),
                "response_a": clean_code(row.get("generated_code") or row.get("extracted_code") or ""),
                "critique": findings,
                "spec_text": spec_text,
                "spec_json": spec_obj,
                "normalized_spec_text": normalized_spec_text,
                "algorithm_text": algorithm_text,
                "algorithm_json": algorithm_obj,
                "normalized_algorithm_text": normalized_algorithm_text,
                "code_text": code_text,
                "algorithm_used": algorithm_used,
                "syntax_repair_texts": syntax_repair_texts,
                "syntax_repair_attempts_used": repair_attempts_used,
                "compile_error_before_repair": compile_error_before_repair,
                "compile_error_after_repair": compile_error_after_repair,
                "model": args.model,
                "timestamp": now,
                "sample_id": 0,
                "seed": args.seed,
                "prompt_mode": "spec_algorithm_code",
                "generation_backend": "transformers_algorithm_sketch_self_play_critic",
                "source_failure_type": row.get("failure_type"),
                "source_error": row.get("error"),
                "rubric_version": (rubric or {}).get("name"),
            }
            copy_test_fields(row, record)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(
                f"generated algorithm-sketch revision {index}/{len(selected)}: {row['id']} "
                f"compile_error_after_repair={bool(compile_error_after_repair)}"
            )

    print(f"wrote {len(selected)} algorithm-sketch critic revisions to {args.output}")


if __name__ == "__main__":
    main()
