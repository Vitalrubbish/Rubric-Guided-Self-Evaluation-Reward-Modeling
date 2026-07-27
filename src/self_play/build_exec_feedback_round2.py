#!/usr/bin/env python3
"""Build the execution-feedback round-2 probe inputs.

Pipeline:
1. merge LLM-extracted public test cases (data/self_play/exec_feedback_tests/);
2. VALIDATE each case by executing it against the row's canonical solution —
   rows whose extracted expectations do not all pass the canonical solution
   are dropped (guarantees no fabricated expectations);
3. execute the model's previous (failed) repair code on the validated public
   cases and render a feedback text (expected vs actual / exception);
4. emit a round-2 generation input: original critic+repair prompt + the
   previous attempt + its public-example behavior, same output contract.

Leakage: public cases come only from the task text; input_output (hidden
suite) is never shown to the model — it is only used downstream by the
verifier for final scoring.
"""

from __future__ import annotations

import argparse
import ast
import glob
import json
import subprocess
import sys
import tempfile
from collections import Counter
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


RUNNER = r'''
import json, sys
payload = json.loads(sys.argv[1])
code = payload["code"]
fn = payload["fn_name"]
cases = payload["cases"]
ns = {}
try:
    exec(code, ns)
except Exception as exc:
    print(json.dumps({"setup_error": f"{type(exc).__name__}: {exc}"}))
    sys.exit(0)
f = ns.get(fn)
if f is None:
    # allow Solution class style
    sol = ns.get("Solution")
    if sol is not None and hasattr(sol(), fn):
        f = getattr(sol(), fn)
if f is None:
    print(json.dumps({"setup_error": f"callable {fn} not found"}))
    sys.exit(0)
results = []
for case in cases:
    try:
        expected = ast.literal_eval(case["expected"])
    except Exception:
        expected = case["expected"]
    try:
        actual = f(*case["args"])
        results.append({"args": case["args"], "expected": case["expected"],
                        "actual_repr": repr(actual), "ok": actual == expected})
    except Exception as exc:
        results.append({"args": case["args"], "expected": case["expected"],
                        "actual_repr": None, "ok": False,
                        "error": f"{type(exc).__name__}: {exc}"})
print(json.dumps({"results": results}))
'''


def run_cases(code: str, fn_name: str, cases: list[dict[str, Any]], timeout: int = 10) -> dict[str, Any]:
    payload = json.dumps({"code": code, "fn_name": fn_name, "cases": cases})
    with tempfile.NamedTemporaryFile("w", suffix="_runner.py", delete=False) as handle:
        handle.write("import ast\n" + RUNNER)
        runner_path = handle.name
    try:
        proc = subprocess.run(
            [PYTHON, runner_path, payload],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        for line in proc.stdout.strip().splitlines():
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict) and ("results" in parsed or "setup_error" in parsed):
                return parsed
            # ignore other json (e.g. values printed by the tested code)
        return {"setup_error": f"runner produced no json (stderr: {proc.stderr[-200:]})"}
    except subprocess.TimeoutExpired:
        return {"setup_error": "timeout"}


def render_feedback(fn_name: str, case_results: list[dict[str, Any]], max_show: int = 3) -> str:
    lines = []
    failures = 0
    for res in case_results:
        call = f"{fn_name}({', '.join(repr(a) for a in res['args'])})"
        if res["ok"]:
            lines.append(f"- {call} -> OK (matches expected {res['expected']})")
        else:
            failures += 1
            if res.get("error"):
                lines.append(f"- {call} -> raised {res['error']} (expected {res['expected']})")
            else:
                lines.append(f"- {call} -> expected {res['expected']}, but returned {res['actual_repr']}")
    shown = lines[: max_show + 5]
    hidden_note = ""
    if len(lines) > len(shown):
        hidden_note = f"\n(... {len(lines) - len(shown)} more cases omitted)"
    summary = f"{failures}/{len(case_results)} public cases FAIL"
    return summary + ":\n" + "\n".join(shown) + hidden_note


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, default=Path("data/self_play/exec_feedback_probe_round1_rows.jsonl"))
    parser.add_argument("--tests-dir", type=Path, default=Path("data/self_play/exec_feedback_tests"))
    parser.add_argument("--output", type=Path, default=Path("data/self_play/exec_feedback_probe_round2_input.jsonl"))
    parser.add_argument("--summary-output", type=Path, default=Path("data/self_play/exec_feedback_probe_round2_summary.json"))
    parser.add_argument("--response-prefix", default="Repair response:")
    args = parser.parse_args()

    rows = [r for r in read_jsonl(args.rows) if r.get("io_mode") == "function_call"]
    tests: dict[str, dict[str, Any]] = {}
    for part in sorted(glob.glob(str(args.tests_dir / "part*.jsonl"))):
        for rec in read_jsonl(Path(part)):
            tests[rec["gate_row_id"]] = rec

    out_rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for row in rows:
        gid = row["gate_row_id"]
        rec = tests.get(gid)
        if rec is None:
            counts["skip:no_extraction"] += 1
            continue
        cases = rec.get("cases") or []
        if not cases:
            counts["skip:no_public_cases"] += 1
            continue
        fn_name = rec.get("fn_name") or ""
        # 1. validate expectations against the canonical solution
        canon = run_cases(str(row.get("canonical_solution") or ""), fn_name, cases)
        if canon.get("setup_error"):
            counts["skip:canonical_setup_error"] += 1
            continue
        canon_results = canon.get("results") or []
        if not all(r.get("ok") for r in canon_results):
            counts["skip:extraction_invalid"] += 1
            continue
        # 2. run the model's previous repair on the validated public cases
        prev = run_cases(str(row.get("previous_repair_code") or ""), fn_name, cases)
        if prev.get("setup_error"):
            feedback = f"The previous repair attempt does not run: {prev['setup_error']}"
            n_fail = len(cases)
        else:
            prev_results = prev.get("results") or []
            n_fail = sum(1 for r in prev_results if not r.get("ok"))
            feedback = render_feedback(fn_name, prev_results)
        if n_fail == 0:
            counts["skip:previous_actually_passes_public"] += 1
            continue
        # 3. build round-2 prompt
        base_prompt = str(row["prompt"])
        trailer = "\n\nRepair response:"
        if trailer in base_prompt:
            base_prompt = base_prompt.rsplit(trailer, 1)[0]
        prev_code = str(row.get("previous_repair_code") or "").strip()
        prompt = (
            f"{base_prompt}\n\n"
            "Your previous repair attempt:\n"
            f"{prev_code}\n\n"
            "Its behavior when executed on the public examples:\n"
            f"{feedback}\n\n"
            "Using only this public execution evidence, identify the remaining concrete errors "
            "and write one corrected implementation.\n"
            f"{trailer}"
        )
        out_rows.append(
            {
                "id": f"{gid}__exec_feedback_round2",
                "problem_id": row["problem_id"],
                "dataset": "apps",
                "split": "validation",
                "task_type": "method2_self_play_critic_repair",
                "prompt": prompt,
                "completion": "",
                "source": "exec_feedback_probe_round2",
                "interface_names": row.get("interface_names") or [],
                "interface_signatures": row.get("interface_signatures") or [],
                "starter_code": row.get("starter_code"),
                "input_output": row.get("input_output"),
                "difficulty": row.get("difficulty"),
                "io_mode": row.get("io_mode"),
                "metadata": {
                    "problem_id": row["problem_id"],
                    "public_cases": len(cases),
                    "public_cases_failed_by_previous": n_fail,
                    "fn_name": fn_name,
                },
            }
        )
        counts["built"] += 1

    write_jsonl(args.output, out_rows)
    summary = {
        "rows_input": len(rows),
        "counts": dict(counts),
        "output": str(args.output),
        "built_rows": len(out_rows),
        "policy": {
            "public_cases": "extracted from task text only; validated by execution against the canonical solution",
            "leakage": "hidden input_output never shown to the model; used only by the verifier for final scoring",
            "previous_repair": "bare Coder-32B-AWQ round-1 repair that failed the hidden gate",
        },
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
