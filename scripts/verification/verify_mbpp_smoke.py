#!/usr/bin/env python3
"""Run coding benchmark tests for a generated-response JSONL."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import re
import traceback
from pathlib import Path


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def extract_code(text: str) -> str:
    fenced = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip("\n\r")
    return text.strip("\n\r")


def raw_humaneval_prompt(prompt: str) -> str:
    marker = "Return only valid Python code, with no Markdown fences and no explanation.\n\n"
    if marker in prompt:
        return prompt.split(marker, 1)[1]
    for prefix in ("\nfrom ", "\nimport ", "\ndef "):
        index = prompt.find(prefix)
        if index >= 0:
            return prompt[index + 1 :]
    return prompt


def humaneval_sources(row: dict, code: str) -> list[tuple[str, str]]:
    prompt_prefix = raw_humaneval_prompt(row.get("prompt", ""))
    sources = [("generated_only", code)]
    if prompt_prefix.strip():
        sources.append(("prompt_plus_completion", prompt_prefix.rstrip() + "\n" + code))
    return sources


def execute_source(row: dict, source: str, namespace: dict[str, object]) -> None:
    if row.get("dataset") == "mbpp":
        exec(compile(source, "<generated_mbpp>", "exec"), namespace, namespace)
        return

    exec(compile(source, "<generated_humanevalplus>", "exec"), namespace, namespace)
    entry_point = row.get("entry_point")
    if not entry_point or entry_point not in namespace:
        raise NameError(f"entry point not defined: {entry_point}")
    if "check" not in namespace:
        raise NameError("HumanEval+ test does not define check(candidate)")
    namespace["check"](namespace[entry_point])


def run_code(row: dict, code: str, queue: mp.Queue) -> None:
    namespace: dict[str, object] = {}
    try:
        if row.get("dataset") == "mbpp":
            source = "\n".join(
                part
                for part in [row.get("test_setup_code", ""), code, "\n".join(row.get("test_list", []))]
                if part
            )
            execute_source(row, source, namespace)
        elif row.get("dataset") == "humanevalplus":
            errors = []
            for source_name, candidate_source in humaneval_sources(row, code):
                namespace = {}
                source = "\n".join(part for part in [candidate_source, row.get("test", "")] if part)
                try:
                    execute_source(row, source, namespace)
                    queue.put({"passed": True, "failure_type": None, "error": None, "source_mode": source_name})
                    return
                except Exception as exc:  # noqa: BLE001
                    errors.append((source_name, exc, traceback.format_exc(limit=2)))
            if any(isinstance(exc, AssertionError) for _, exc, _ in errors):
                queue.put({"passed": False, "failure_type": "logic_error", "error": "assertion failed"})
            elif all(isinstance(exc, SyntaxError) for _, exc, _ in errors):
                source_name, exc, _ = errors[-1]
                queue.put({"passed": False, "failure_type": "syntax_error", "error": f"{source_name}: {exc}"})
            else:
                source_name, exc, tb = errors[-1]
                queue.put(
                    {
                        "passed": False,
                        "failure_type": "runtime_error",
                        "error": f"{source_name}: {type(exc).__name__}: {exc}",
                        "traceback": tb,
                    }
                )
            return
        else:
            raise ValueError(f"unsupported dataset: {row.get('dataset')}")
        queue.put({"passed": True, "failure_type": None, "error": None})
    except SyntaxError as exc:
        queue.put({"passed": False, "failure_type": "syntax_error", "error": str(exc)})
    except AssertionError:
        queue.put({"passed": False, "failure_type": "logic_error", "error": "assertion failed"})
    except Exception as exc:  # noqa: BLE001
        queue.put(
            {
                "passed": False,
                "failure_type": "runtime_error",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=2),
            }
        )


def evaluate_one(row: dict, timeout: float) -> dict:
    code = extract_code(row.get("generated_code", ""))
    if not code:
        return {**row, "passed": False, "failure_type": "generation_failure", "error": "empty output"}

    queue: mp.Queue = mp.Queue()
    process = mp.Process(target=run_code, args=(row, code, queue))
    process.start()
    process.join(timeout)
    if process.is_alive():
        process.terminate()
        process.join(1)
        result = {"passed": False, "failure_type": "timeout", "error": f">{timeout}s"}
    else:
        result = queue.get() if not queue.empty() else {"passed": False, "failure_type": "runtime_error", "error": "no result"}

    return {
        **row,
        "extracted_code": code,
        "passed": result["passed"],
        "failure_type": result.get("failure_type"),
        "error": result.get("error"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/responses/vllm_smoke_responses.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/responses/vllm_smoke_labeled.jsonl"))
    parser.add_argument("--timeout", type=float, default=5.0)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    passed = 0
    with args.output.open("w", encoding="utf-8") as f:
        for row in read_jsonl(args.input):
            result = evaluate_one(row, args.timeout)
            total += 1
            passed += int(bool(result.get("passed")))
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"evaluated {total} responses, passed={passed}, failed={total - passed}")
    print(f"wrote labels to {args.output}")


if __name__ == "__main__":
    main()
