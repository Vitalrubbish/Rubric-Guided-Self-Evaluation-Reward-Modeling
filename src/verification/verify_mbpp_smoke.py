#!/usr/bin/env python3
"""Run coding benchmark tests for a generated-response JSONL."""

from __future__ import annotations

import argparse
import ast
import builtins
import contextlib
import concurrent.futures
import io
import json
import multiprocessing as mp
import os
import re
import sys
import tempfile
import unittest
import traceback
import types
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(0)


@contextlib.contextmanager
def redirect_stdin(target):
    old_stdin = sys.stdin
    try:
        sys.stdin = target
        yield target
    finally:
        sys.stdin = old_stdin


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


def extract_starter_from_prompt(prompt: str) -> str:
    marker = "\nStarting code:\n"
    end_marker = "\n\nDefine "
    if marker not in prompt:
        return ""
    tail = prompt.split(marker, 1)[1]
    if end_marker in tail:
        tail = tail.split(end_marker, 1)[0]
    elif "\n\nPython code:" in tail:
        tail = tail.split("\n\nPython code:", 1)[0]
    return tail.strip("\n\r") + "\n"


def bigcodebench_sources(row: dict, code: str) -> list[tuple[str, str]]:
    starter = row.get("code_prompt") or row.get("starter_code") or extract_starter_from_prompt(row.get("prompt", ""))
    sources = [("generated_only", code)]
    required = set(row.get("interface_names") or [])
    entry_point = row.get("entry_point")
    if entry_point:
        required.add(entry_point)
    defines_required = any(re.search(rf"\bdef\s+{re.escape(name)}\s*\(", code) for name in required)
    if starter and not defines_required and not code.lstrip().startswith(starter.lstrip()):
        sources.append(("starter_plus_completion", starter.rstrip() + "\n" + code))
    return sources


def parse_apps_input_output(row: dict) -> dict:
    raw = row.get("input_output") or {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def apps_sources(row: dict, code: str) -> list[tuple[str, str]]:
    starter = row.get("starter_code") or extract_starter_from_prompt(row.get("prompt", ""))
    sources = [("generated_only", code)]
    required = set(row.get("interface_names") or [])
    defines_required = any(re.search(rf"\bdef\s+{re.escape(name)}\s*\(", code) for name in required)
    defines_solution_class = bool(re.search(r"\bclass\s+Solution\b", code))
    if starter and required and not defines_required and not defines_solution_class and not code.lstrip().startswith(starter.lstrip()):
        sources.append(("starter_plus_completion", starter.rstrip() + "\n" + code))
    return sources


def apps_namespace() -> dict[str, object]:
    return {
        "__name__": "__apps_candidate__",
        "List": List,
        "Dict": Dict,
        "Set": Set,
        "Tuple": Tuple,
        "Optional": Optional,
        "Any": Any,
        "types": types,
    }


def normalize_stdout(text: object) -> list[str]:
    return str(text).strip().split()


def compare_apps_stdout(actual: str, expected: object) -> bool:
    expected_text = str(expected)
    return actual.strip() == expected_text.strip() or normalize_stdout(actual) == normalize_stdout(expected_text)


def normalize_apps_value(value: object) -> object:
    if isinstance(value, tuple):
        return [normalize_apps_value(item) for item in value]
    if isinstance(value, list):
        return [normalize_apps_value(item) for item in value]
    if isinstance(value, dict):
        return {key: normalize_apps_value(item) for key, item in value.items()}
    return value


def compare_apps_value(actual: object, expected: object, depth: int = 0) -> bool:
    if depth < 3 and isinstance(expected, list) and len(expected) == 1:
        if compare_apps_value(actual, expected[0], depth + 1):
            return True
    if isinstance(actual, float) or isinstance(expected, float):
        try:
            return abs(float(actual) - float(expected)) <= 1e-6
        except Exception:  # noqa: BLE001
            return False
    return normalize_apps_value(actual) == normalize_apps_value(expected)


def run_apps_function_cases(namespace: dict[str, object], io_spec: dict) -> None:
    fn_name = io_spec.get("fn_name")
    if not isinstance(fn_name, str) or not fn_name:
        raise NameError("APPS function-call input_output does not define fn_name")

    inputs = io_spec.get("inputs") or []
    outputs = io_spec.get("outputs") or []
    if len(inputs) != len(outputs):
        raise ValueError(f"APPS input/output length mismatch: {len(inputs)} != {len(outputs)}")

    solution_cls = namespace.get("Solution")
    free_fn = namespace.get(fn_name)
    for index, (args_value, expected) in enumerate(zip(inputs, outputs)):
        if isinstance(args_value, list):
            args = args_value
        elif isinstance(args_value, tuple):
            args = list(args_value)
        else:
            args = [args_value]

        if isinstance(solution_cls, type) and hasattr(solution_cls, fn_name):
            candidate = getattr(solution_cls(), fn_name)
        elif callable(free_fn):
            candidate = free_fn
        else:
            raise NameError(f"entry point not defined: Solution.{fn_name} or {fn_name}")

        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            actual = candidate(*args)
        if not compare_apps_value(actual, expected):
            raise AssertionError(
                f"case {index}: expected {safe_repr(expected, 160)}, got {safe_repr(actual, 160)}"
            )


def run_apps_stdin_cases(source: str, io_spec: dict) -> None:
    inputs = io_spec.get("inputs") or []
    outputs = io_spec.get("outputs") or []
    if len(inputs) != len(outputs):
        raise ValueError(f"APPS input/output length mismatch: {len(inputs)} != {len(outputs)}")

    for index, (stdin_text, expected) in enumerate(zip(inputs, outputs)):
        namespace = apps_namespace()
        namespace["__name__"] = "__main__"
        stdout = io.StringIO()
        try:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(io.StringIO()), redirect_stdin(io.StringIO(str(stdin_text))):
                exec(compile(source, "<generated_apps>", "exec"), namespace, namespace)
        except SystemExit as exc:
            if exc.code not in (None, 0):
                raise
        actual = stdout.getvalue()
        if not compare_apps_stdout(actual, expected):
            raise AssertionError(
                f"case {index}: expected stdout {safe_repr(expected, 160)}, got {safe_repr(actual, 160)}"
            )


def run_unittest_cases(namespace: dict[str, object]) -> None:
    case = namespace.get("TestCases")
    if not isinstance(case, type) or not issubclass(case, unittest.TestCase):
        raise NameError("BigCodeBench test does not define unittest.TestCase class TestCases")
    suite = unittest.TestLoader().loadTestsFromTestCase(case)
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=0).run(suite)
    if not result.wasSuccessful():
        first = (result.failures + result.errors)[0]
        exc_text = first[1].strip().splitlines()[-1] if first[1].strip() else "unittest failed"
        raise AssertionError(exc_text)


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
                source_name, exc, _ = next((item for item in errors if isinstance(item[1], AssertionError)), errors[-1])
                queue.put({"passed": False, "failure_type": "logic_error", "error": f"{source_name}: {exc}"})
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
        elif row.get("dataset") == "bigcodebench":
            errors = []
            old_cwd = os.getcwd()
            with tempfile.TemporaryDirectory(prefix="bigcodebench_verify_") as tmpdir:
                os.chdir(tmpdir)
                try:
                    os.environ.setdefault("MPLCONFIGDIR", os.path.join(tmpdir, "mplconfig"))
                    for source_name, candidate_source in bigcodebench_sources(row, code):
                        namespace = {"__name__": "__bigcodebench_candidate__"}
                        source = "\n".join(
                            part
                            for part in [
                                candidate_source,
                                row.get("test", ""),
                                "run_unittest_cases(globals())",
                            ]
                            if part
                        )
                        namespace["run_unittest_cases"] = run_unittest_cases
                        try:
                            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                                exec(compile(source, "<generated_bigcodebench>", "exec"), namespace, namespace)
                            queue.put({"passed": True, "failure_type": None, "error": None, "source_mode": source_name})
                            return
                        except Exception as exc:  # noqa: BLE001
                            errors.append((source_name, exc, traceback.format_exc(limit=2)))
                finally:
                    os.chdir(old_cwd)
            if any(isinstance(exc, AssertionError) for _, exc, _ in errors):
                source_name, exc, _ = next((item for item in errors if isinstance(item[1], AssertionError)), errors[-1])
                queue.put(
                    {
                        "passed": False,
                        "failure_type": "logic_error",
                        "error": f"{source_name}: {exc}",
                        "safe_diagnostics": {"diagnostic_kind": "wrong_output"},
                    }
                )
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
                        "safe_diagnostics": {
                            "diagnostic_kind": "runtime_error",
                            "exception_type": type(exc).__name__,
                        },
                    }
                )
            return
        elif row.get("dataset") == "apps":
            io_spec = parse_apps_input_output(row)
            fn_name = io_spec.get("fn_name")
            old_cwd = os.getcwd()
            errors = []
            with tempfile.TemporaryDirectory(prefix="apps_verify_") as tmpdir:
                os.chdir(tmpdir)
                try:
                    for source_name, candidate_source in apps_sources(row, code):
                        try:
                            if fn_name:
                                namespace = apps_namespace()
                                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                                    exec(compile(candidate_source, "<generated_apps>", "exec"), namespace, namespace)
                                run_apps_function_cases(namespace, io_spec)
                            else:
                                run_apps_stdin_cases(candidate_source, io_spec)
                            queue.put({"passed": True, "failure_type": None, "error": None, "source_mode": source_name})
                            return
                        except Exception as exc:  # noqa: BLE001
                            errors.append((source_name, exc, traceback.format_exc(limit=2)))
                finally:
                    os.chdir(old_cwd)

            if any(isinstance(exc, AssertionError) for _, exc, _ in errors):
                source_name, exc, _ = next((item for item in errors if isinstance(item[1], AssertionError)), errors[-1])
                queue.put(
                    {
                        "passed": False,
                        "failure_type": "logic_error",
                        "error": f"{source_name}: {exc}",
                        "safe_diagnostics": {"diagnostic_kind": "wrong_output"},
                    }
                )
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
                        "safe_diagnostics": {
                            "diagnostic_kind": "runtime_error",
                            "exception_type": type(exc).__name__,
                        },
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


def evaluate_one(row: dict, timeout: float, start_method: str = "spawn") -> dict:
    code = extract_code(row.get("generated_code", ""))
    if not code:
        return {
            **row,
            "response_id": response_id(row),
            "passed": False,
            "failure_type": "generation_failure",
            "error": "empty output",
        }

    ctx = mp.get_context(start_method)
    queue: mp.Queue = ctx.Queue()
    process = ctx.Process(target=run_code, args=(row, code, queue))
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
        "source_mode": result.get("source_mode"),
        "safe_diagnostics": result.get("safe_diagnostics"),
        "private_diagnostics": result.get("private_diagnostics"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/responses/vllm_smoke_responses.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/responses/vllm_smoke_labeled.jsonl"))
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--workers", type=int, default=1, help="Number of rows to verify concurrently.")
    parser.add_argument(
        "--process-start-method",
        choices=("spawn", "fork", "forkserver"),
        default="spawn",
        help="Subprocess start method for per-response execution. spawn avoids fork-from-thread deadlocks.",
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = list(read_jsonl(args.input))
    total = 0
    passed = 0
    with args.output.open("w", encoding="utf-8") as f:
        if args.workers <= 1:
            iterator = (evaluate_one(row, args.timeout, args.process_start_method) for row in rows)
            for result in iterator:
                total += 1
                passed += int(bool(result.get("passed")))
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = [executor.submit(evaluate_one, row, args.timeout, args.process_start_method) for row in rows]
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    total += 1
                    passed += int(bool(result.get("passed")))
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"evaluated {total} responses, passed={passed}, failed={total - passed}")
    print(f"wrote labels to {args.output}")


if __name__ == "__main__":
    main()
