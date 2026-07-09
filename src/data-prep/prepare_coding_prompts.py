#!/usr/bin/env python3
"""Prepare coding prompts without leaking verifier asserts into model inputs."""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


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


def infer_callable_names(problem: dict) -> list[str]:
    names: set[str] = set()
    for test in problem.get("test_list", []):
        names.update(re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", test))
    return sorted(name for name in names if name not in BUILTIN_CALLS)


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


def infer_interface_signatures(problem: dict) -> list[str]:
    code = problem.get("code") or ""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    callables = set(infer_callable_names(problem))
    signatures: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and (not callables or node.name in callables):
            signatures.append(f"def {node.name}({_format_args(node.args)})")
        elif isinstance(node, ast.ClassDef) and (not callables or node.name in callables):
            signatures.append(f"class {node.name}")
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                    signatures.append(f"  def __init__({_format_args(item.args)})")
                    break
    return signatures


def mbpp_prompt(problem: dict, mode: str = "hidden_tests") -> str:
    interface_names = infer_callable_names(problem)
    interface_signatures = infer_interface_signatures(problem)
    interface_hint = ""
    if interface_signatures:
        interface_hint = (
            "Define code matching this public interface:\n"
            + "\n".join(f"- {signature}" for signature in interface_signatures)
            + "\n\n"
        )
    elif interface_names:
        interface_hint = (
            "Define the callable name(s) expected by the evaluator: "
            f"{', '.join(interface_names)}.\n\n"
        )

    if mode == "visible_tests":
        tests = "\n".join(problem.get("test_list", []))
        return (
            "You are an expert Python programmer. Solve the following task.\n"
            "Return only valid Python code, with no Markdown fences and no explanation.\n\n"
            f"Task: {problem['text']}\n\n"
            "Your code must define every function/class used by these tests:\n"
            f"{tests}\n\n"
            "Python code:\n"
        )

    return (
        "You are an expert Python programmer. Solve the following task.\n"
        "Return only valid Python code, with no Markdown fences and no explanation.\n\n"
        f"Task: {problem['text']}\n\n"
        f"{interface_hint}"
        "Python code:\n"
    )


def convert_mbpp(raw_dir: Path, splits: list[str], prompt_mode: str):
    for split in splits:
        path = raw_dir / f"mbpp_{split}.jsonl"
        for row in read_jsonl(path):
            task_id = row["task_id"]
            yield {
                "id": f"mbpp/{split}/{task_id}",
                "dataset": "mbpp",
                "split": split,
                "prompt": mbpp_prompt(row, prompt_mode),
                "prompt_mode": f"mbpp_{prompt_mode}",
                "interface_names": infer_callable_names(row),
                "interface_signatures": infer_interface_signatures(row),
                "canonical_solution": row.get("code", ""),
                "test_list": row.get("test_list", []),
                "test_setup_code": row.get("test_setup_code", ""),
                "entry_point": None,
            }


def convert_humanevalplus(raw_dir: Path):
    path = raw_dir / "humanevalplus_test.jsonl"
    for row in read_jsonl(path):
        task_id = row["task_id"]
        prompt = (
            "You are an expert Python programmer. Complete the function below.\n"
            "Return only valid Python code, with no Markdown fences and no explanation.\n\n"
            f"{row['prompt']}"
        )
        yield {
            "id": f"humanevalplus/{task_id}",
            "dataset": "humanevalplus",
            "split": "test",
            "prompt": prompt,
            "canonical_solution": row.get("canonical_solution", ""),
            "test": row.get("test", ""),
            "entry_point": row.get("entry_point"),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/coding_prompts.jsonl"))
    parser.add_argument(
        "--dataset",
        choices=("mbpp", "all"),
        default="mbpp",
        help="Dataset scope. Phase 1 uses MBPP only; 'all' keeps the legacy MBPP+HumanEval+ merge.",
    )
    parser.add_argument(
        "--mbpp-splits",
        nargs="+",
        choices=("train", "test", "validation"),
        default=["train", "test", "validation"],
        help="MBPP splits to include.",
    )
    parser.add_argument(
        "--mbpp-prompt-mode",
        choices=("hidden_tests", "visible_tests"),
        default="hidden_tests",
        help="Use hidden_tests to keep asserts out of prompts; visible_tests is the legacy leaked prompt.",
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    items = list(convert_mbpp(args.raw_dir, args.mbpp_splits, args.mbpp_prompt_mode))
    if args.dataset == "all":
        items.extend(convert_humanevalplus(args.raw_dir))

    count = 0
    with args.output.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            count += 1
            if args.limit is not None and count >= args.limit:
                break

    print(f"wrote {count} prompts to {args.output}")


if __name__ == "__main__":
    main()
