#!/usr/bin/env python3
"""Shared utilities for executable-rubric probes."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable


PYTHON = sys.executable


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def parse_apps_input_output(row: dict[str, Any]) -> dict[str, Any]:
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


def normalize_json_value(value: Any) -> Any:
    if isinstance(value, tuple):
        return [normalize_json_value(item) for item in value]
    if isinstance(value, list):
        return [normalize_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): normalize_json_value(item) for key, item in value.items()}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def normalize_case(raw: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    args = raw.get("args")
    if args is None:
        args = raw.get("inputs")
    if args is None:
        args = raw.get("input")
    if args is None:
        return None
    if isinstance(args, tuple):
        args = list(args)
    if not isinstance(args, list):
        args = [args]
    if "expected" in raw:
        expected = raw["expected"]
    elif "output" in raw:
        expected = raw["output"]
    elif "outputs" in raw:
        expected = raw["outputs"]
    else:
        return None
    return {
        "args": normalize_json_value(args),
        "expected": normalize_json_value(expected),
    }


def case_key(case: dict[str, Any]) -> str:
    return json.dumps(case, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


RUNNER = r'''
import contextlib
import io
import json
import math
import sys
import types
from typing import Any, Dict, List, Optional, Set, Tuple

payload = json.loads(sys.argv[1])
code = payload["code"]
fn_name = payload["fn_name"]
cases = payload["cases"]

def normalize(value):
    if isinstance(value, tuple):
        return [normalize(item) for item in value]
    if isinstance(value, list):
        return [normalize(item) for item in value]
    if isinstance(value, dict):
        return {key: normalize(item) for key, item in value.items()}
    return value

def compare(actual, expected):
    if isinstance(actual, float) or isinstance(expected, float):
        try:
            return math.isclose(float(actual), float(expected), rel_tol=1e-9, abs_tol=1e-6)
        except Exception:
            return False
    return normalize(actual) == normalize(expected)

def safe_repr(value, limit=240):
    try:
        text = repr(value)
    except Exception as exc:
        text = f"<repr failed: {type(exc).__name__}: {exc}>"
    return text if len(text) <= limit else text[: limit - 3] + "..."

namespace = {
    "__name__": "__exec_rubric_candidate__",
    "Any": Any,
    "Dict": Dict,
    "List": List,
    "Optional": Optional,
    "Set": Set,
    "Tuple": Tuple,
    "types": types,
}
try:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        exec(compile(code, "<candidate>", "exec"), namespace, namespace)
except Exception as exc:
    print(json.dumps({"setup_error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False))
    sys.exit(0)

solution_cls = namespace.get("Solution")
free_fn = namespace.get(fn_name)
if isinstance(solution_cls, type) and hasattr(solution_cls, fn_name):
    candidate = getattr(solution_cls(), fn_name)
elif callable(free_fn):
    candidate = free_fn
else:
    print(json.dumps({"setup_error": f"entry point not defined: Solution.{fn_name} or {fn_name}"}, ensure_ascii=False))
    sys.exit(0)

results = []
for index, case in enumerate(cases):
    args = case.get("args") or []
    expected = case.get("expected")
    if not isinstance(args, list):
        args = [args]
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            actual = candidate(*args)
        ok = compare(actual, expected)
        results.append(
            {
                "test_index": index,
                "passed": bool(ok),
                "actual_repr": safe_repr(actual),
                "expected_repr": safe_repr(expected),
            }
        )
    except Exception as exc:
        results.append(
            {
                "test_index": index,
                "passed": False,
                "exception": f"{type(exc).__name__}: {exc}",
                "expected_repr": safe_repr(expected),
            }
        )
print(json.dumps({"results": results}, ensure_ascii=False))
'''


def execute_function_tests(
    code: str,
    fn_name: str,
    cases: list[dict[str, Any]],
    timeout: float = 10.0,
) -> dict[str, Any]:
    payload = json.dumps(
        {
            "code": code,
            "fn_name": fn_name,
            "cases": cases,
        },
        ensure_ascii=False,
    )
    runner_path = ""
    try:
        with tempfile.NamedTemporaryFile("w", suffix="_exec_rubric_runner.py", delete=False, encoding="utf-8") as handle:
            handle.write(RUNNER)
            runner_path = handle.name
        proc = subprocess.run(
            [PYTHON, runner_path, payload],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        for line in reversed(proc.stdout.strip().splitlines()):
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict) and ("results" in parsed or "setup_error" in parsed):
                return parsed
        return {"setup_error": f"runner produced no JSON result (stderr: {proc.stderr[-240:]})"}
    except subprocess.TimeoutExpired:
        return {"setup_error": "timeout"}
    finally:
        if runner_path:
            try:
                os.unlink(runner_path)
            except OSError:
                pass


def passed_count(result: dict[str, Any], total: int) -> int:
    if result.get("setup_error"):
        return 0
    return sum(1 for item in result.get("results") or [] if item.get("passed"))


def all_passed(result: dict[str, Any], total: int) -> bool:
    return not result.get("setup_error") and passed_count(result, total) == total
