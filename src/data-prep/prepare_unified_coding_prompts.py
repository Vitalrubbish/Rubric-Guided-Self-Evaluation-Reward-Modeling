#!/usr/bin/env python3
"""Build dataset-neutral coding prompts from multiple coding benchmarks."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Iterable

if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(0)


BUILTIN_CALLS = {
    "abs",
    "all",
    "any",
    "assert",
    "bool",
    "dict",
    "enumerate",
    "float",
    "int",
    "len",
    "list",
    "max",
    "min",
    "range",
    "reversed",
    "round",
    "set",
    "sorted",
    "str",
    "sum",
    "tuple",
    "zip",
}

PROMPT_MODE = "unified_coding_hidden_tests_v1"


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def clean_text(text: str | None) -> str:
    text = text or ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def remove_embedded_starter_hint(text: str | None) -> str:
    text = clean_text(text)
    text = re.sub(
        r"\n?You should write self-contained code starting with:\s*```.*?```\s*",
        "\n",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return clean_text(text)


def _format_args(args: ast.arguments) -> str:
    rendered: list[str] = []
    positional = list(args.posonlyargs) + list(args.args)
    defaults = [None] * (len(positional) - len(args.defaults)) + list(args.defaults)
    for arg, default in zip(positional, defaults):
        text = arg.arg
        if default is not None:
            text += f"={ast.unparse(default)}"
        rendered.append(text)
    if args.vararg:
        rendered.append(f"*{args.vararg.arg}")
    elif args.kwonlyargs:
        rendered.append("*")
    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        text = arg.arg
        if default is not None:
            text += f"={ast.unparse(default)}"
        rendered.append(text)
    if args.kwarg:
        rendered.append(f"**{args.kwarg.arg}")
    return ", ".join(rendered)


def callable_names_from_asserts(test_list: list[str]) -> list[str]:
    names: set[str] = set()
    for test in test_list:
        names.update(re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", test))
    return sorted(name for name in names if name not in BUILTIN_CALLS)


def interface_signatures_from_code(code: str, target_names: list[str] | None = None) -> list[str]:
    try:
        tree = ast.parse(code or "")
    except SyntaxError:
        return []
    targets = set(target_names or [])
    signatures: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and (not targets or node.name in targets):
            signatures.append(f"def {node.name}({_format_args(node.args)})")
        elif isinstance(node, ast.ClassDef):
            class_signatures: list[str] = []
            signatures.append(f"class {node.name}")
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and (
                    item.name == "__init__" or not targets or item.name in targets
                ):
                    class_signatures.append(f"  def {item.name}({_format_args(item.args)})")
            if not targets or node.name in targets or class_signatures:
                signatures.extend(class_signatures[:3])
            else:
                signatures.pop()
    return signatures


def stable_split(key: str, train_pct: int = 80, validation_pct: int = 10) -> str:
    bucket = int(hashlib.sha1(key.encode("utf-8")).hexdigest()[:8], 16) % 100
    if bucket < train_pct:
        return "train"
    if bucket < train_pct + validation_pct:
        return "validation"
    return "test"


def unified_prompt(
    task_text: str,
    interface_signatures: list[str] | None = None,
    interface_names: list[str] | None = None,
    starter_code: str | None = None,
    stdin_stdout: bool = False,
) -> str:
    sections = [
        "You are an expert Python programmer. Solve the following task.",
        "Return only valid Python code, with no Markdown fences and no explanation.",
        "",
        "Task:",
        clean_text(task_text),
    ]

    starter = clean_text(starter_code)
    if starter:
        sections.extend(["", "Starting code:", starter])

    signatures = interface_signatures or []
    names = interface_names or []
    if signatures:
        sections.extend(
            [
                "",
                "Define code matching this public interface:",
                *[f"- {signature}" for signature in signatures],
            ]
        )
    elif names:
        sections.extend(
            [
                "",
                "Define the callable name(s) expected by the evaluator:",
                ", ".join(names),
            ]
        )
    elif stdin_stdout:
        sections.extend(["", "Your program should read from standard input and write to standard output."])

    sections.extend(["", "Python code:"])
    return "\n".join(sections) + "\n"


def convert_mbpp(raw_dir: Path, splits: list[str]) -> Iterable[dict]:
    for split in splits:
        path = raw_dir / f"mbpp_{split}.jsonl"
        if not path.exists():
            continue
        for row in read_jsonl(path):
            task_id = str(row["task_id"])
            names = callable_names_from_asserts(row.get("test_list", []))
            signatures = interface_signatures_from_code(row.get("code", ""), names)
            yield {
                "id": f"mbpp/{split}/{task_id}",
                "dataset": "mbpp",
                "split": split,
                "prompt_mode": PROMPT_MODE,
                "prompt": unified_prompt(row.get("text", ""), signatures, names),
                "interface_names": names,
                "interface_signatures": signatures,
                "canonical_solution": row.get("code", ""),
                "test_list": row.get("test_list", []),
                "test_setup_code": row.get("test_setup_code", ""),
                "entry_point": None,
            }


def convert_humanevalplus(raw_dir: Path) -> Iterable[dict]:
    path = raw_dir / "humanevalplus_test.jsonl"
    if not path.exists():
        return
    for row in read_jsonl(path):
        task_id = str(row["task_id"])
        entry_point = row.get("entry_point")
        signatures = interface_signatures_from_code(row.get("prompt", ""), [entry_point] if entry_point else None)
        yield {
            "id": f"humanevalplus/test/{task_id}",
            "dataset": "humanevalplus",
            "split": "test",
            "prompt_mode": PROMPT_MODE,
            "prompt": unified_prompt(row.get("prompt", ""), signatures, [entry_point] if entry_point else []),
            "interface_names": [entry_point] if entry_point else [],
            "interface_signatures": signatures,
            "canonical_solution": row.get("canonical_solution", ""),
            "test": row.get("test", ""),
            "entry_point": entry_point,
        }


def read_bigcodebench_parquet(path: Path) -> Iterable[dict]:
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - depends on runtime env
        raise SystemExit("BigCodeBench parquet conversion requires pandas/pyarrow") from exc
    frame = pd.read_parquet(path)
    for item in frame.to_dict(orient="records"):
        yield item


def convert_bigcodebench(path: Path) -> Iterable[dict]:
    if not path.exists():
        return
    for row in read_bigcodebench_parquet(path):
        task_id = str(row.get("task_id"))
        entry_point = row.get("entry_point") or None
        code_prompt = clean_text(row.get("code_prompt"))
        names = [entry_point] if entry_point else []
        signatures = interface_signatures_from_code(code_prompt, names)
        yield {
            "id": f"bigcodebench/{task_id}",
            "dataset": "bigcodebench",
            "split": stable_split(f"bigcodebench/{task_id}"),
            "prompt_mode": PROMPT_MODE,
            "prompt": unified_prompt(
                remove_embedded_starter_hint(row.get("instruct_prompt") or row.get("complete_prompt") or ""),
                signatures,
                names,
                starter_code=code_prompt,
            ),
            "interface_names": names,
            "interface_signatures": signatures,
            "canonical_solution": row.get("canonical_solution", ""),
            "starter_code": code_prompt,
            "code_prompt": code_prompt,
            "test": row.get("test", ""),
            "entry_point": entry_point,
            "libs": row.get("libs"),
        }


def parse_apps_input_output(row: dict) -> dict:
    raw = row.get("input_output")
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def parse_apps_solutions(row: dict) -> list[str]:
    raw = row.get("solutions")
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, str) and item.strip()]
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [item for item in parsed if isinstance(item, str) and item.strip()]
        except json.JSONDecodeError:
            return [raw] if raw.strip() else []
    return []


def convert_apps(paths: list[Path]) -> Iterable[dict]:
    for path in paths:
        if not path.exists():
            continue
        split = "test" if "test" in path.name else "train"
        for index, row in enumerate(read_jsonl(path)):
            problem_id = str(row.get("problem_id") or row.get("id") or index)
            io_spec = parse_apps_input_output(row)
            fn_name = io_spec.get("fn_name")
            names = [fn_name] if isinstance(fn_name, str) and fn_name else []
            stdin_stdout = not bool(names)
            starter_code = row.get("starter_code") or ""
            signatures = interface_signatures_from_code(starter_code, names)
            solutions = parse_apps_solutions(row)
            yield {
                "id": f"apps/{split}/{problem_id}",
                "dataset": "apps",
                "split": split,
                "prompt_mode": PROMPT_MODE,
                "prompt": unified_prompt(
                    row.get("question", ""),
                    signatures,
                    names,
                    starter_code=starter_code,
                    stdin_stdout=stdin_stdout,
                ),
                "interface_names": names,
                "interface_signatures": signatures,
                "starter_code": starter_code,
                "input_output": row.get("input_output"),
                "difficulty": row.get("difficulty"),
                "io_mode": "stdin_stdout" if stdin_stdout else "function_call",
                "canonical_solution": solutions[0] if solutions else "",
                "canonical_solutions": solutions,
            }


def limited(rows: Iterable[dict], limit: int | None) -> Iterable[dict]:
    for index, row in enumerate(rows):
        if limit is not None and index >= limit:
            break
        yield row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/coding_prompts_unified.jsonl"))
    parser.add_argument("--include-mbpp", action="store_true")
    parser.add_argument("--include-humanevalplus", action="store_true")
    parser.add_argument("--include-bigcodebench", action="store_true")
    parser.add_argument("--include-apps", action="store_true")
    parser.add_argument("--all", action="store_true", help="Include every available supported dataset.")
    parser.add_argument(
        "--mbpp-splits",
        nargs="+",
        choices=("train", "validation", "test"),
        default=["train", "validation", "test"],
    )
    parser.add_argument(
        "--bigcodebench-file",
        type=Path,
        default=Path("data/raw/bigcodebench/v0.1.4.parquet"),
    )
    parser.add_argument(
        "--apps-glob",
        default="apps/*.jsonl",
        help="Glob under raw-dir for sampled APPS JSONL files.",
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    include_mbpp = args.all or args.include_mbpp
    include_humanevalplus = args.all or args.include_humanevalplus
    include_bigcodebench = args.all or args.include_bigcodebench
    include_apps = args.all or args.include_apps
    if not any([include_mbpp, include_humanevalplus, include_bigcodebench, include_apps]):
        include_mbpp = True

    rows: list[dict] = []
    if include_mbpp:
        rows.extend(convert_mbpp(args.raw_dir, args.mbpp_splits))
    if include_humanevalplus:
        rows.extend(convert_humanevalplus(args.raw_dir))
    if include_bigcodebench:
        rows.extend(convert_bigcodebench(args.bigcodebench_file))
    if include_apps:
        rows.extend(convert_apps(sorted(args.raw_dir.glob(args.apps_glob))))

    count = write_jsonl(args.output, limited(rows, args.limit))
    print(f"wrote {count} unified prompts to {args.output}")


if __name__ == "__main__":
    main()
