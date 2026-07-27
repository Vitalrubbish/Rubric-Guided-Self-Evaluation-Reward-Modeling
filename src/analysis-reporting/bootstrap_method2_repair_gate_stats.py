#!/usr/bin/env python3
"""Bootstrap uncertainty analysis for Method 2 repair gate pass rates.

The Method 2 held-out repair gate has only 38 rows, so single-point pass
rates like 24/38 vs 23/38 vs 22/38 are dominated by sampling noise. This
script quantifies that noise:

- Wilson 95% CI for each version's pass rate;
- paired bootstrap CI for pass-rate differences between versions
  (rows are resampled jointly so per-row pairing is preserved);
- exact McNemar p-value (binomial test on discordant pairs);
- timeout-flake sensitivity (P->F rows with identical code and a timeout
  failure are re-scored as pass);
- the sample size needed for a given CI half-width at the observed rate.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any

Z_95 = 1.959963984540054


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


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def code_sha_like(row: dict[str, Any]) -> str:
    import hashlib

    code = str(row.get("extracted_code") or row.get("generated_code") or "")
    return hashlib.sha256(code.encode("utf-8")).hexdigest()[:12]


def wilson_ci(passed: int, total: int) -> list[float]:
    if total == 0:
        return [0.0, 0.0]
    p = passed / total
    denom = 1.0 + Z_95**2 / total
    center = (p + Z_95**2 / (2 * total)) / denom
    half = (Z_95 / denom) * math.sqrt(p * (1 - p) / total + Z_95**2 / (4 * total**2))
    return [max(0.0, center - half), min(1.0, center + half)]


def mcnemar_exact_p(discordant_a_win: int, discordant_b_win: int) -> float:
    """Two-sided exact McNemar p-value under the null of equal discordance."""
    n = discordant_a_win + discordant_b_win
    if n == 0:
        return 1.0
    k = min(discordant_a_win, discordant_b_win)
    # P(X <= k) for Binomial(n, 0.5), doubled, capped at 1.
    cdf = sum(math.comb(n, i) for i in range(0, k + 1)) / (2**n)
    return min(1.0, 2.0 * cdf)


def paired_bootstrap_diff(
    base_flags: list[bool],
    cand_flags: list[bool],
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    rng = random.Random(seed)
    n = len(base_flags)
    diffs: list[float] = []
    for _ in range(iterations):
        b = 0
        c = 0
        for _ in range(n):
            idx = rng.randrange(n)
            b += 1 if base_flags[idx] else 0
            c += 1 if cand_flags[idx] else 0
        diffs.append(c / n - b / n)
    diffs.sort()
    lo = diffs[int(0.025 * iterations)]
    hi = diffs[min(iterations - 1, int(0.975 * iterations))]
    frac_le_zero = sum(1 for d in diffs if d <= 0.0) / iterations
    return {
        "bootstrap_iterations": iterations,
        "diff_ci95": [lo, hi],
        "diff_ci95_rows": [round(lo * n, 2), round(hi * n, 2)],
        "prob_candidate_not_better": round(frac_le_zero, 4),
    }


def n_for_half_width(p: float, half_width: float) -> int:
    """Approximate n for a normal-approximation CI half-width at rate p."""
    if half_width <= 0:
        raise ValueError("half_width must be positive")
    return math.ceil((Z_95**2) * p * (1 - p) / (half_width**2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True, help="baseline labeled JSONL (e.g. v0.3)")
    parser.add_argument(
        "--candidate",
        type=Path,
        required=True,
        action="append",
        help="candidate labeled JSONL; repeat for multiple candidates",
    )
    parser.add_argument("--iterations", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    baseline_rows = read_jsonl(args.baseline)
    baseline_by_id = {str(row.get("id")): row for row in baseline_rows}
    n = len(baseline_rows)
    baseline_passed = sum(1 for row in baseline_rows if row.get("passed"))

    summary: dict[str, Any] = {
        "baseline": str(args.baseline),
        "rows": n,
        "baseline_passed": baseline_passed,
        "baseline_pass_rate": baseline_passed / n,
        "baseline_wilson_ci95": wilson_ci(baseline_passed, n),
        "sample_size_reference": {
            "note": "approximate n for a 95% CI half-width at the baseline pass rate",
            "half_width_0.05": n_for_half_width(baseline_passed / n, 0.05),
            "half_width_0.10": n_for_half_width(baseline_passed / n, 0.10),
        },
        "comparisons": [],
    }

    for candidate_path in args.candidate:
        candidate_by_id = {str(row.get("id")): row for row in read_jsonl(candidate_path)}
        common_ids = sorted(set(baseline_by_id) & set(candidate_by_id))
        if not common_ids:
            raise SystemExit(f"no common sample ids with {candidate_path}")

        base_flags = [bool(baseline_by_id[i].get("passed")) for i in common_ids]
        cand_flags = [bool(candidate_by_id[i].get("passed")) for i in common_ids]

        cand_passed = sum(cand_flags)
        both_fail = sum(1 for b, c in zip(base_flags, cand_flags) if not b and not c)
        base_only = sum(1 for b, c in zip(base_flags, cand_flags) if b and not c)
        cand_only = sum(1 for b, c in zip(base_flags, cand_flags) if not b and c)

        # Timeout-flake sensitivity: re-score likely flakes as pass.
        flake_ids = []
        for i in common_ids:
            baseline = baseline_by_id[i]
            candidate = candidate_by_id[i]
            if (
                baseline.get("passed")
                and not candidate.get("passed")
                and candidate.get("failure_type") == "timeout"
                and code_sha_like(baseline) == code_sha_like(candidate)
            ):
                flake_ids.append(i)
        adjusted_flags = list(cand_flags)
        for idx, i in enumerate(common_ids):
            if i in flake_ids:
                adjusted_flags[idx] = True
        adjusted_passed = sum(adjusted_flags)

        comparison = {
            "candidate": str(candidate_path),
            "common_rows": len(common_ids),
            "candidate_passed": cand_passed,
            "candidate_pass_rate": cand_passed / len(common_ids),
            "candidate_wilson_ci95": wilson_ci(cand_passed, len(common_ids)),
            "diff_rows": cand_passed - baseline_passed,
            "transitions": dict(
                Counter(
                    f"{'P' if b else 'F'}->{'P' if c else 'F'}"
                    for b, c in zip(base_flags, cand_flags)
                )
            ),
            "mcnemar": {
                "baseline_only_pass": base_only,
                "candidate_only_pass": cand_only,
                "both_fail": both_fail,
                "exact_p_two_sided": round(mcnemar_exact_p(base_only, cand_only), 4),
            },
            "paired_bootstrap": paired_bootstrap_diff(base_flags, cand_flags, args.iterations, args.seed),
            "timeout_flake_sensitivity": {
                "likely_timeout_flakes": flake_ids,
                "adjusted_candidate_passed": adjusted_passed,
                "adjusted_candidate_pass_rate": adjusted_passed / len(common_ids),
                "adjusted_diff_rows": adjusted_passed - baseline_passed,
            },
        }
        summary["comparisons"].append(comparison)

    if args.output:
        write_json(args.output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
