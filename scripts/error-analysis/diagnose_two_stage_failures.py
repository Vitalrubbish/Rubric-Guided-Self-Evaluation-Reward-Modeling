#!/usr/bin/env python3
"""Diagnose failed two-stage self-play coding revisions."""

from __future__ import annotations

import argparse
import ast
import json
import multiprocessing as mp
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


CATEGORIES = {
    "wrong_spec",
    "right_spec_wrong_algorithm",
    "signature_or_interface",
    "insufficient_tests",
    "unknown",
}


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def truncate(text: object, limit: int = 700) -> str:
    value = "" if text is None else str(text)
    value = value.replace("\r", "")
    return value if len(value) <= limit else value[: limit - 3] + "..."


def extract_test_calls(test_list: list[str]) -> set[str]:
    calls: set[str] = set()
    for test in test_list:
        try:
            tree = ast.parse(test)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    calls.add(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    calls.add(node.func.attr)
    return calls


def extract_defined_functions(code: str) -> set[str]:
    try:
        tree = ast.parse(code or "")
    except SyntaxError:
        return set()
    return {node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}


def safe_repr(value: object, limit: int = 240) -> str:
    try:
        text = repr(value)
    except Exception as exc:  # noqa: BLE001
        text = f"<repr failed: {type(exc).__name__}: {exc}>"
    return truncate(text, limit)


def assertion_expression(test: str) -> ast.Assert | None:
    try:
        tree = ast.parse(test)
    except SyntaxError:
        return None
    if len(tree.body) == 1 and isinstance(tree.body[0], ast.Assert):
        return tree.body[0]
    return None


def evaluate_tests_worker(row: dict, queue: mp.Queue) -> None:
    namespace: dict[str, object] = {}
    code = row.get("generated_code") or row.get("extracted_code") or ""
    results = []
    try:
        setup = row.get("test_setup_code") or ""
        if setup.strip():
            exec(compile(setup, "<test_setup>", "exec"), namespace, namespace)
        exec(compile(code, "<generated_code>", "exec"), namespace, namespace)
    except Exception as exc:  # noqa: BLE001
        queue.put(
            {
                "setup_or_compile_error": f"{type(exc).__name__}: {exc}",
                "assertion_results": [],
            }
        )
        return

    for test in row.get("test_list") or []:
        item = {"assertion": test}
        node = assertion_expression(test)
        try:
            if node and isinstance(node.test, ast.Compare) and len(node.test.ops) == 1 and len(node.test.comparators) == 1:
                left_expr = ast.Expression(node.test.left)
                right_expr = ast.Expression(node.test.comparators[0])
                ast.fix_missing_locations(left_expr)
                ast.fix_missing_locations(right_expr)
                actual = eval(compile(left_expr, "<assert_left>", "eval"), namespace, namespace)
                expected = eval(compile(right_expr, "<assert_right>", "eval"), namespace, namespace)
                item.update(
                    {
                        "actual": safe_repr(actual),
                        "expected": safe_repr(expected),
                        "passed": actual == expected,
                    }
                )
            else:
                exec(compile(test, "<assertion>", "exec"), namespace, namespace)
                item.update({"actual": "<assertion passed>", "expected": "<implicit True>", "passed": True})
        except AssertionError:
            item.update({"actual": "<assertion failed>", "expected": "<implicit True>", "passed": False})
        except Exception as exc:  # noqa: BLE001
            item.update(
                {
                    "actual": f"<{type(exc).__name__}: {exc}>",
                    "expected": "<no value>",
                    "passed": False,
                    "exception": f"{type(exc).__name__}: {exc}",
                }
            )
        results.append(item)
    queue.put({"assertion_results": results})


def evaluate_tests(row: dict, timeout: float) -> dict:
    queue: mp.Queue = mp.Queue()
    process = mp.Process(target=evaluate_tests_worker, args=(row, queue))
    process.start()
    process.join(timeout)
    if process.is_alive():
        process.terminate()
        process.join(1)
        return {"setup_or_compile_error": f"timeout >{timeout}s", "assertion_results": []}
    if queue.empty():
        return {"setup_or_compile_error": "no result from worker", "assertion_results": []}
    return queue.get()


def heuristic_diagnosis(row: dict, observed: dict) -> dict:
    tests = row.get("test_list") or []
    expected_calls = extract_test_calls(tests)
    defined_functions = extract_defined_functions(row.get("generated_code") or "")
    missing_calls = sorted(name for name in expected_calls if name not in defined_functions and name not in dir(__builtins__))
    setup_error = observed.get("setup_or_compile_error")
    exceptions = [item.get("exception", "") for item in observed.get("assertion_results", []) if item.get("exception")]

    if setup_error or missing_calls or any(("NameError" in exc or "TypeError" in exc) for exc in exceptions):
        return {
            "diagnosis": "signature_or_interface",
            "confidence": "high" if missing_calls else "medium",
            "evidence": [
                f"expected calls from tests: {sorted(expected_calls)}",
                f"defined functions in revised code: {sorted(defined_functions)}",
                f"missing calls: {missing_calls}",
                f"setup/compile error: {setup_error}",
            ],
            "recommended_fix": "Add explicit interface extraction before code generation and verify required function names/signatures before running semantic tests.",
        }

    spec_text = json.dumps(row.get("spec_json") or {}, ensure_ascii=False)
    inferred_spec = ""
    if isinstance(row.get("spec_json"), dict):
        inferred_spec = str(row["spec_json"].get("inferred_spec") or "")
    failed_assertions = [item for item in observed.get("assertion_results", []) if not item.get("passed")]

    if not inferred_spec or len(inferred_spec) < 40:
        return {
            "diagnosis": "wrong_spec",
            "confidence": "medium",
            "evidence": ["Stage 1 inferred_spec is missing or too short.", f"inferred_spec: {truncate(inferred_spec, 300)}"],
            "recommended_fix": "Strengthen Stage 1 to produce an explicit natural-language specification before code generation.",
        }

    if len(failed_assertions) == len(tests) and len(tests) >= 3:
        confidence = "medium"
    else:
        confidence = "low"

    task_line = ""
    match = re.search(r"Task:\s*(.*)", row.get("prompt") or "")
    if match:
        task_line = match.group(1).strip()
    evidence = [
        f"task: {truncate(task_line, 240)}",
        f"inferred_spec: {truncate(inferred_spec, 420)}",
        f"failed assertions: {len(failed_assertions)}/{len(tests)}",
    ]
    if failed_assertions:
        first = failed_assertions[0]
        evidence.append(
            "first mismatch: "
            + truncate(
                f"{first.get('assertion')} actual={first.get('actual')} expected={first.get('expected')}",
                420,
            )
        )

    return {
        "diagnosis": "right_spec_wrong_algorithm",
        "confidence": confidence,
        "evidence": evidence,
        "recommended_fix": "Keep the two-stage split, but add an algorithm-sketch stage before writing code and ask the model to manually simulate the visible tests.",
        "spec_text_for_llm": truncate(spec_text, 1200),
    }


def load_model(model_path: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map={"": 0},
    )
    model.eval()
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    return model, tokenizer


def chat_prompt(tokenizer, system_prompt: str, user_prompt: str) -> str:
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"{system_prompt}\n\n{user_prompt}\n\nJSON:"


def model_generate(model, tokenizer, prompt: str, max_input_length: int, max_new_tokens: int) -> str:
    import torch

    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_length)
    encoded = {key: value.cuda() for key, value in encoded.items()}
    with torch.inference_mode():
        output_ids = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
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


def build_llm_prompt(row: dict, observed: dict, heuristic: dict) -> str:
    failed_assertions = [item for item in observed.get("assertion_results", []) if not item.get("passed")]
    compact_failures = []
    for item in failed_assertions[:4]:
        compact_failures.append(
            {
                "assertion": item.get("assertion"),
                "actual": item.get("actual"),
                "expected": item.get("expected"),
                "exception": item.get("exception"),
            }
        )
    return f"""Diagnose why a two-stage self-play repair still failed.

Choose exactly one diagnosis category:
- wrong_spec: Stage 1 inferred the task specification incorrectly or missed a crucial edge case.
- right_spec_wrong_algorithm: Stage 1 specification is mostly correct, but Stage 2 code/algorithm does not satisfy it.
- signature_or_interface: function name, parameters, return type, imports, or interface do not match tests.
- insufficient_tests: visible tests/task text are too weak or ambiguous for self-discovery; external retrieval/oracle hint is likely needed.
- unknown: evidence is insufficient or mixed.

Task prompt:
{truncate(row.get("prompt"), 1400)}

Visible tests:
```python
{truncate(chr(10).join(row.get("test_list") or []), 1000)}
```

Stage 1 inferred specification:
```json
{truncate(row.get("normalized_spec_text") or row.get("spec_text"), 1800)}
```

Stage 2 revised code:
```python
{truncate(row.get("generated_code"), 1600)}
```

Verifier error: {row.get("failure_type")} / {row.get("error")}

Observed visible-test failures:
```json
{json.dumps(compact_failures, ensure_ascii=False, indent=2)}
```

Deterministic precheck:
```json
{json.dumps(heuristic, ensure_ascii=False, indent=2)}
```

Return JSON with exactly these keys:
{{
  "diagnosis": "one category",
  "confidence": "low|medium|high",
  "stage1_spec_quality": "brief assessment",
  "stage2_code_quality": "brief assessment",
  "evidence": ["specific evidence 1", "specific evidence 2"],
  "recommended_fix": "specific next prompt or pipeline change"
}}
"""


def llm_diagnosis(model, tokenizer, row: dict, observed: dict, heuristic: dict, max_input_length: int, max_new_tokens: int) -> tuple[dict, str]:
    system_prompt = (
        "You are a strict coding benchmark failure analyst. "
        "Classify the failure cause using only the provided task, tests, inferred specification, revised code, and observed outputs. "
        "Return JSON only."
    )
    prompt = chat_prompt(tokenizer, system_prompt, build_llm_prompt(row, observed, heuristic))
    text = model_generate(model, tokenizer, prompt, max_input_length, max_new_tokens)
    obj = parse_json_object(text) or {}
    diagnosis = str(obj.get("diagnosis") or "").strip()
    if diagnosis not in CATEGORIES:
        diagnosis = heuristic["diagnosis"]
    evidence = obj.get("evidence") or heuristic.get("evidence") or []
    if isinstance(evidence, str):
        evidence = [evidence]
    result = {
        "diagnosis": diagnosis,
        "confidence": obj.get("confidence") if obj.get("confidence") in {"low", "medium", "high"} else heuristic.get("confidence", "low"),
        "stage1_spec_quality": truncate(obj.get("stage1_spec_quality") or ""),
        "stage2_code_quality": truncate(obj.get("stage2_code_quality") or ""),
        "evidence": [truncate(item, 500) for item in evidence],
        "recommended_fix": truncate(obj.get("recommended_fix") or heuristic.get("recommended_fix") or ""),
    }
    return result, text


def build_markdown(records: list[dict], source: Path) -> str:
    counts = Counter(row["diagnosis"] for row in records)
    confidence_counts = Counter(row["confidence"] for row in records)
    by_category: dict[str, list[dict]] = defaultdict(list)
    for row in records:
        by_category[row["diagnosis"]].append(row)

    lines = [
        "# Logic Two-Stage Failure Diagnosis",
        "",
        "## 输入",
        "",
        f"- Source labeled file: `{source}`",
        f"- Diagnosed failures: {len(records)}",
        "",
        "## 分类分布",
        "",
        "| Diagnosis | Count |",
        "| --- | ---: |",
    ]
    for key, count in counts.most_common():
        lines.append(f"| `{key}` | {count} |")
    lines.extend(["", "## 置信度", "", "| Confidence | Count |", "| --- | ---: |"])
    for key, count in confidence_counts.most_common():
        lines.append(f"| `{key}` | {count} |")

    lines.extend(
        [
            "",
            "## 样本明细",
            "",
            "| ID | Diagnosis | Confidence | First failing assertion | Recommended fix |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for row in records:
        first_fail = ""
        for item in row.get("observed", {}).get("assertion_results", []):
            if not item.get("passed"):
                first_fail = f"{item.get('assertion')} actual={item.get('actual')} expected={item.get('expected')}"
                break
        lines.append(
            "| `{}` | `{}` | {} | {} | {} |".format(
                row["id"],
                row["diagnosis"],
                row["confidence"],
                truncate(first_fail, 160).replace("|", "\\|"),
                truncate(row.get("recommended_fix"), 180).replace("|", "\\|"),
            )
        )

    lines.extend(["", "## 代表样本", ""])
    for category, rows in sorted(by_category.items()):
        lines.append(f"### `{category}`")
        for row in rows[:3]:
            evidence = "; ".join(row.get("evidence") or [])
            lines.extend(
                [
                    "",
                    f"- `{row['id']}` ({row['confidence']}): {truncate(evidence, 500)}",
                    f"  - Next: {truncate(row.get('recommended_fix'), 400)}",
                ]
            )
        lines.append("")

    dominant, dominant_count = counts.most_common(1)[0] if counts else ("none", 0)
    lines.extend(
        [
            "## Gate 决策",
            "",
            f"- Dominant diagnosis: `{dominant}` ({dominant_count}/{len(records)})",
        ]
    )
    if records and dominant == "right_spec_wrong_algorithm" and dominant_count / len(records) >= 0.5:
        lines.append("- 下一步：优先做 Stage 2 algorithm-sketch prompt，再写代码。")
    elif records and dominant == "wrong_spec" and dominant_count / len(records) >= 0.5:
        lines.append("- 下一步：优先加强 Stage 1 counterexample/spec prompt。")
    elif records and counts.get("signature_or_interface", 0) / len(records) >= 0.2:
        lines.append("- 下一步：先加接口提取和签名检查。")
    elif records and counts.get("insufficient_tests", 0) / len(records) >= 0.3:
        lines.append("- 下一步：进入 retrieval/oracle-hint 路线，并在报告中标明不再是纯 self-discovery。")
    else:
        lines.append("- 下一步：没有单一主因，先做混合 prompt 小实验，并继续保持 n=20 gate。")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labeled", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--max-input-length", type=int, default=6144)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    rows = [row for row in read_jsonl(args.labeled) if not row.get("passed")]
    if args.limit is not None:
        rows = rows[: args.limit]

    model = tokenizer = None
    if args.model:
        model, tokenizer = load_model(args.model)

    now = datetime.now(timezone.utc).isoformat()
    records = []
    for index, row in enumerate(rows, start=1):
        observed = evaluate_tests(row, args.timeout)
        heuristic = heuristic_diagnosis(row, observed)
        llm_text = None
        if model is not None and tokenizer is not None:
            diagnosis, llm_text = llm_diagnosis(model, tokenizer, row, observed, heuristic, args.max_input_length, args.max_new_tokens)
        else:
            diagnosis = {
                key: heuristic[key]
                for key in ("diagnosis", "confidence", "evidence", "recommended_fix")
            }
            diagnosis["stage1_spec_quality"] = ""
            diagnosis["stage2_code_quality"] = ""

        record = {
            "id": row["id"],
            "diagnosis": diagnosis["diagnosis"],
            "confidence": diagnosis["confidence"],
            "stage1_spec_quality": diagnosis.get("stage1_spec_quality"),
            "stage2_code_quality": diagnosis.get("stage2_code_quality"),
            "evidence": diagnosis.get("evidence") or [],
            "recommended_fix": diagnosis.get("recommended_fix"),
            "heuristic_diagnosis": heuristic,
            "llm_diagnosis_text": llm_text,
            "observed": observed,
            "prompt": row.get("prompt"),
            "test_list": row.get("test_list"),
            "spec_json": row.get("spec_json"),
            "generated_code": row.get("generated_code"),
            "failure_type": row.get("failure_type"),
            "error": row.get("error"),
            "timestamp": now,
        }
        records.append(record)
        print(f"diagnosed {index}/{len(rows)}: {row['id']} -> {record['diagnosis']} ({record['confidence']})")

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(build_markdown(records, args.labeled), encoding="utf-8")

    print(json.dumps({"diagnosed": len(records), "counts": Counter(row["diagnosis"] for row in records)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
