#!/usr/bin/env python3
"""Run coding benchmark tests for a generated-response JSONL."""

from __future__ import annotations

import argparse
import ast
import builtins
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


def response_id(row: dict) -> str:
    if row.get("response_id"):
        return str(row["response_id"])
    return f"{row.get('id')}__sample{row.get('sample_id', 0)}"


def safe_repr(value: object, limit: int = 240) -> str:
    try:
        text = repr(value)
    except Exception as exc:  # noqa: BLE001
        text = f"<repr failed: {type(exc).__name__}: {exc}>"
    return text if len(text) <= limit else text[: limit - 3] + "..."


def value_summary(value: object) -> dict:
    summary = {"type": type(value).__name__}
    try:
        summary["len"] = len(value)  # type: ignore[arg-type]
    except Exception:  # noqa: BLE001
        pass
    if isinstance(value, (list, tuple, set)):
        summary["container"] = type(value).__name__
        summary["element_types"] = sorted({type(item).__name__ for item in list(value)[:8]})
    elif isinstance(value, dict):
        summary["container"] = "dict"
        summary["key_types"] = sorted({type(item).__name__ for item in list(value.keys())[:8]})
        summary["value_types"] = sorted({type(item).__name__ for item in list(value.values())[:8]})
    return summary


def compare_values(actual: object, expected: object) -> tuple[bool, str, dict]:
    passed = actual == expected
    actual_summary = value_summary(actual)
    expected_summary = value_summary(expected)
    detail = {
        "actual": actual_summary,
        "expected": expected_summary,
    }
    if type(actual) is not type(expected):
        return passed, "wrong_type", detail
    if actual_summary.get("len") != expected_summary.get("len"):
        return passed, "wrong_length", detail
    return passed, "wrong_value", detail


def parse_single_assert(test: str) -> ast.Assert | None:
    try:
        tree = ast.parse(test)
    except SyntaxError:
        return None
    if len(tree.body) == 1 and isinstance(tree.body[0], ast.Assert):
        return tree.body[0]
    return None


def eval_compare_assert(node: ast.Assert, namespace: dict[str, object]) -> tuple[bool, dict, dict]:
    if not isinstance(node.test, ast.Compare) or len(node.test.ops) != 1 or len(node.test.comparators) != 1:
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), "<assertion>", "exec"), namespace, namespace)
        return True, {"kind": "passed"}, {}

    left_expr = ast.Expression(node.test.left)
    right_expr = ast.Expression(node.test.comparators[0])
    ast.fix_missing_locations(left_expr)
    ast.fix_missing_locations(right_expr)
    actual = eval(compile(left_expr, "<assert_left>", "eval"), namespace, namespace)
    expected = eval(compile(right_expr, "<assert_right>", "eval"), namespace, namespace)
    passed, kind, safe_detail = compare_values(actual, expected)
    private_detail = {
        "actual": safe_repr(actual),
        "expected": safe_repr(expected),
    }
    return passed, {"kind": kind, **safe_detail}, private_detail


def extract_defined_functions(code: str) -> set[str]:
    try:
        tree = ast.parse(code or "")
    except SyntaxError:
        return set()
    return {node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}


def missing_required_interfaces(row: dict, namespace: dict[str, object], code: str) -> list[str]:
    required = set(row.get("interface_names") or [])
    if not required:
        return []
    defined = extract_defined_functions(code)
    available = set(namespace)
    builtin_names = set(dir(builtins))
    return sorted(name for name in required if name not in defined and name not in available and name not in builtin_names)


def summarize_assertions(assertions: list[dict]) -> dict:
    failed = [item for item in assertions if not item.get("passed")]
    summary = {
        "test_count": len(assertions),
        "passed_assertions": len(assertions) - len(failed),
        "failed_assertions": len(failed),
    }
    if failed:
        first = failed[0]
        summary["first_failed_index"] = first.get("test_index")
        summary["first_failure_kind"] = first.get("kind")
        for key in ("actual", "expected", "exception_type"):
            if key in first:
                summary[f"first_{key}"] = first[key]
    return summary


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
            setup = row.get("test_setup_code", "")
            if setup:
                exec(compile(setup, "<test_setup>", "exec"), namespace, namespace)
            exec(compile(code, "<generated_mbpp>", "exec"), namespace, namespace)

            missing_interfaces = missing_required_interfaces(row, namespace, code)
            if missing_interfaces:
                queue.put(
                    {
                        "passed": False,
                        "failure_type": "runtime_error",
                        "error": f"required interface not defined: {', '.join(missing_interfaces)}",
                        "safe_diagnostics": {
                            "diagnostic_kind": "missing_required_interface",
                            "missing_interfaces": missing_interfaces,
                            "defined_functions": sorted(extract_defined_functions(code)),
                        },
                        "private_diagnostics": {
                            "missing_interfaces": missing_interfaces,
                            "defined_functions": sorted(extract_defined_functions(code)),
                        },
                    }
                )
                return

            assertion_results = []
            private_assertions = []
            for index, test in enumerate(row.get("test_list") or []):
                item = {"test_index": index}
                private_item = {"test_index": index, "assertion": test}
                node = parse_single_assert(test)
                try:
                    if node is None:
                        exec(compile(test, "<assertion>", "exec"), namespace, namespace)
                        item.update({"passed": True, "kind": "passed"})
                        private_item.update({"passed": True})
                    else:
                        passed, safe_detail, private_detail = eval_compare_assert(node, namespace)
                        item.update({"passed": passed, **safe_detail})
                        private_item.update({"passed": passed, **private_detail})
                    assertion_results.append(item)
                    private_assertions.append(private_item)
                except AssertionError:
                    item.update({"passed": False, "kind": "assertion_failed"})
                    private_item.update({"passed": False, "actual": "<assertion failed>", "expected": "<implicit True>"})
                    assertion_results.append(item)
                    private_assertions.append(private_item)
                except Exception as exc:  # noqa: BLE001
                    item.update(
                        {
                            "passed": False,
                            "kind": "exception_on_case",
                            "exception_type": type(exc).__name__,
                        }
                    )
                    private_item.update(
                        {
                            "passed": False,
                            "exception": f"{type(exc).__name__}: {exc}",
                            "traceback": traceback.format_exc(limit=2),
                        }
                    )
                    assertion_results.append(item)
                    private_assertions.append(private_item)
                    queue.put(
                        {
                            "passed": False,
                            "failure_type": "runtime_error",
                            "error": f"{type(exc).__name__}: {exc}",
                            "safe_diagnostics": {
                                "diagnostic_kind": "exception_on_case",
                                **summarize_assertions(assertion_results),
                            },
                            "private_diagnostics": {
                                "assertions": private_assertions,
                            },
                        }
                    )
                    return

            if any(not item.get("passed") for item in assertion_results):
                queue.put(
                    {
                        "passed": False,
                        "failure_type": "logic_error",
                        "error": "assertion failed",
                        "safe_diagnostics": {
                            "diagnostic_kind": "wrong_output",
                            **summarize_assertions(assertion_results),
                        },
                        "private_diagnostics": {
                            "assertions": private_assertions,
                        },
                    }
                )
                return
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
        queue.put(
            {
                "passed": False,
                "failure_type": "syntax_error",
                "error": str(exc),
                "safe_diagnostics": {"diagnostic_kind": "syntax_error"},
            }
        )
    except AssertionError:
        queue.put({"passed": False, "failure_type": "logic_error", "error": "assertion failed"})
    except Exception as exc:  # noqa: BLE001
        queue.put(
            {
                "passed": False,
                "failure_type": "runtime_error",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=2),
                "safe_diagnostics": {
                    "diagnostic_kind": "runtime_error",
                    "exception_type": type(exc).__name__,
                },
            }
        )


def evaluate_one(row: dict, timeout: float) -> dict:
    code = extract_code(row.get("generated_code", ""))
    if not code:
        return {
            **row,
            "response_id": response_id(row),
            "passed": False,
            "failure_type": "generation_failure",
            "error": "empty output",
        }

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
        "response_id": response_id(row),
        "extracted_code": code,
        "passed": result["passed"],
        "failure_type": result.get("failure_type"),
        "error": result.get("error"),
        "safe_diagnostics": result.get("safe_diagnostics"),
        "private_diagnostics": result.get("private_diagnostics"),
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
