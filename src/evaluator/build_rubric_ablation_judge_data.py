#!/usr/bin/env python3
"""Build the rubric-ablation self-evaluation judge dataset.

Research question (Homework 3, Step 3): does the auto-discovered Phase 2
rubric add self-evaluation signal for the base model, beyond its intrinsic
judgment and beyond a structurally identical but semantically shuffled rubric?

Three arms are built for every sampled response:

- ``no_rubric``: judge sees only the public task and the submitted code;
- ``auto_rubric``: judge additionally sees the audited Phase 2 rubric
  (9 dimensions with definitions and checks);
- ``random_rubric``: same rubric structure and token budget, but dimension
  names are deranged across contents, destroying semantic validity.

The same responses appear in all three arms so arm comparisons are paired.
Gold labels are verifier pass/fail and are never shown to the judge.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


JUDGE_INSTRUCTION = (
    "You are the same Python model acting as a rubric-guided self-evaluator.\n"
    "Decide whether the submitted code should pass the task. Use only the public task, "
    "public interface, and submitted code. Do not assume hidden test results."
)

VERDICT_INSTRUCTION = (
    "Answer with exactly one line:\n"
    "Verdict: PASS or FAIL\n"
    "Use PASS only if the code is a correct and complete solution for ordinary valid inputs."
)


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


def clip_text(text: Any, limit: int) -> str:
    value = str(text or "").strip()
    if len(value) <= limit:
        return value
    return value[:limit].rstrip() + "\n...[clipped]"


def rubric_blocks(rubric: dict[str, Any]) -> list[dict[str, str]]:
    blocks: list[dict[str, str]] = []
    for dim in rubric.get("dimensions") or []:
        checks = [str(item).strip() for item in dim.get("what_to_check") or [] if str(item).strip()]
        blocks.append(
            {
                "id": str(dim.get("id") or ""),
                "name": str(dim.get("name") or dim.get("id") or ""),
                "definition": str(dim.get("definition") or "").strip(),
                "checks": " ".join(checks),
            }
        )
    if not blocks:
        raise ValueError("rubric contains no dimensions")
    return blocks


def render_rubric(blocks: list[dict[str, str]]) -> str:
    lines = ["Rubric dimensions to check before judging:"]
    for block in blocks:
        lines.append(f"- {block['name']}: {block['definition']} Check: {block['checks']}")
    return "\n".join(lines)


def deranged_permutation(size: int, rng: random.Random) -> list[int]:
    if size < 2:
        raise ValueError("need at least two rubric dimensions for a derangement")
    while True:
        perm = list(range(size))
        rng.shuffle(perm)
        if all(perm[i] != i for i in range(size)):
            return perm


def judge_prompt(row: dict[str, Any], rubric_text: str | None, task_chars: int, code_chars: int) -> str:
    code = clip_text(row.get("extracted_code") or row.get("generated_code"), code_chars)
    task = clip_text(row.get("prompt"), task_chars)
    parts = [JUDGE_INSTRUCTION]
    if rubric_text:
        parts.append(rubric_text)
    parts.append(VERDICT_INSTRUCTION)
    parts.append(f"Public task prompt:\n{task}")
    parts.append(f"Submitted code:\n```python\n{code}\n```")
    return "\n\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--rubric", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=300, help="total sampled responses; must be even per class")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task-chars", type=int, default=3000)
    parser.add_argument("--code-chars", type=int, default=3000)
    args = parser.parse_args()

    rows = read_jsonl(args.responses)
    passed = [row for row in rows if row.get("passed")]
    failed = [row for row in rows if not row.get("passed")]
    per_class = args.rows // 2
    if len(passed) < per_class or len(failed) < per_class:
        raise ValueError(f"not enough rows per class: pass={len(passed)}, fail={len(failed)}, need {per_class}")

    rng = random.Random(args.seed)
    rng.shuffle(passed)
    rng.shuffle(failed)
    sampled = passed[:per_class] + failed[:per_class]
    rng.shuffle(sampled)

    # Threshold-selection discipline: validation half selects, test half reports.
    split_assign: dict[str, str] = {}
    for cls_rows in (passed[:per_class], failed[:per_class]):
        half = per_class // 2
        for index, row in enumerate(cls_rows):
            split_assign[str(row.get("response_id"))] = "validation" if index < half else "test"

    rubric = json.loads(args.rubric.read_text(encoding="utf-8"))
    blocks = rubric_blocks(rubric)
    auto_text = render_rubric(blocks)
    perm = deranged_permutation(len(blocks), random.Random(args.seed + 1))
    random_blocks = []
    for index, block in enumerate(blocks):
        donor = blocks[perm[index]]
        random_blocks.append({**block, "name": donor["name"], "id": donor["id"]})
    random_text = render_rubric(random_blocks)
    short_text = "Rubric dimensions to check before judging:\n" + "\n".join(
        f"- {block['name']}" for block in blocks
    )

    arms = {
        "no_rubric": None,
        "auto_rubric": auto_text,
        "random_rubric": random_text,
        "short_rubric": short_text,
    }

    out_rows: list[dict[str, Any]] = []
    for row in sampled:
        response_id = str(row.get("response_id") or row.get("id"))
        for arm, rubric_text in arms.items():
            out_rows.append(
                {
                    "id": f"{response_id}__{arm}",
                    "response_id": response_id,
                    "arm": arm,
                    "task_type": "judge_single",
                    "split": split_assign[response_id],
                    "prompt": judge_prompt(row, rubric_text, args.task_chars, args.code_chars),
                    "gold_passed": bool(row.get("passed")),
                    "gold_failure_type": row.get("failure_type"),
                    "dataset": row.get("dataset"),
                    "difficulty": row.get("difficulty"),
                    "io_mode": row.get("io_mode"),
                }
            )

    out_rows.sort(key=lambda row: (row["response_id"], row["arm"]))
    write_jsonl(args.output, out_rows)

    summary = {
        "responses": str(args.responses),
        "rubric": str(args.rubric),
        "rubric_name": rubric.get("name"),
        "output": str(args.output),
        "seed": args.seed,
        "sampled_responses": len(sampled),
        "arms": sorted(arms),
        "rows": len(out_rows),
        "gold_counts": dict(Counter("pass" if row["gold_passed"] else "fail" for row in out_rows if row["arm"] == "no_rubric")),
        "split_counts": dict(Counter(row["split"] for row in out_rows if row["arm"] == "no_rubric")),
        "random_rubric_derangement": perm,
        "policy": {
            "leakage": "judge prompts contain only the public task prompt and submitted code; verifier labels are never shown",
            "in_distribution_caveat": "rubric dimensions were discovered from failures of this same response distribution, so auto_rubric absolute numbers are in-distribution; arm deltas remain internally valid because all arms see identical rows",
            "pairing": "identical responses appear in every arm; arm comparisons must use paired statistics",
        },
    }
    write_json(args.summary_output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
