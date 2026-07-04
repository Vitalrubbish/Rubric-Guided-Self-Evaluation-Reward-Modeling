#!/usr/bin/env python3
"""Compare original vs revised verification labels."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--revised", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/revision/revision_comparison_summary.json"))
    args = parser.parse_args()

    original = {row["id"]: row for row in read_jsonl(args.original)}
    revised = {row["id"]: row for row in read_jsonl(args.revised)}
    transitions = Counter()
    by_dataset = defaultdict(Counter)
    edited_transitions = Counter()
    examples = []

    for item_id, orig in original.items():
        rev = revised.get(item_id)
        if not rev:
            continue
        before = bool(orig.get("passed"))
        after = bool(rev.get("passed"))
        key = f"{'pass' if before else 'fail'}->{'pass' if after else 'fail'}"
        transitions[key] += 1
        by_dataset[orig.get("dataset")][key] += 1
        if rev.get("revision_edits"):
            edited_transitions[key] += 1
        if before != after and len(examples) < 20:
            examples.append(
                {
                    "id": item_id,
                    "dataset": orig.get("dataset"),
                    "transition": key,
                    "edits": rev.get("revision_edits"),
                    "before_error": orig.get("error"),
                    "after_error": rev.get("error"),
                }
            )

    total = sum(transitions.values())
    before_pass = transitions["pass->pass"] + transitions["pass->fail"]
    after_pass = transitions["pass->pass"] + transitions["fail->pass"]
    summary = {
        "total": total,
        "original_passed": before_pass,
        "revised_passed": after_pass,
        "original_pass_rate": round(before_pass / total, 6) if total else 0,
        "revised_pass_rate": round(after_pass / total, 6) if total else 0,
        "net_pass_delta": after_pass - before_pass,
        "transitions": dict(transitions),
        "edited_transitions": dict(edited_transitions),
        "by_dataset": {dataset: dict(counter) for dataset, counter in sorted(by_dataset.items())},
        "transition_examples": examples,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
