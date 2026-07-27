# Phase 2 Dataset Decision

Date: 2026-07-12

## Current Decision

The active dataset has already moved from MBPP/BigCodeBench experiments to APPS simple:

```text
APPS official train
-> executable prompt conversion
-> difficulty=introductory
-> deterministic verifier-readiness filter
-> 2613 prompts
```

The selected file is:

```text
data/processed/apps_train_simple_executable_prompts_unified.jsonl
```

This replaces the previous dataset-expansion recommendation. APPS is no longer a future sample candidate; it is the current formal source for Phase 1 and Phase 2.

## Why This Dataset

The project needed a dataset large enough to support later evaluator/critic training for a 7B model, while still being executable and reportable.

The selected APPS subset is appropriate because:

- it is defined by official metadata, not arbitrary sampling;
- it stays within the requested 2k-3k scale;
- it uses only APPS official train;
- APPS official test remains held out;
- `introductory` difficulty avoids the severe truncation and complexity seen in harder APPS tasks;
- both function-call and stdin/stdout modes are supported by the current verifier adapter.

## Dataset Summary

```text
source input = data/processed/apps_train_executable_prompts_unified.jsonl
selected output = data/processed/apps_train_simple_executable_prompts_unified.jsonl
```

| Field | Count |
| --- | ---: |
| Input rows seen | 3771 |
| Selected rows | 2613 |
| function_call | 2489 |
| stdin_stdout | 124 |
| introductory | 2613 |

## Consequence For Older Expansion Plans

Older documents discussed HumanEval+, BigCodeBench, and APPS samples as possible expansion routes. Those are now archived exploration paths, not active Phase 1/2 inputs.

Current active source:

```text
APPS train introductory verifier-ready set
```

Current non-active sources:

```text
MBPP hidden-tests k=3/k=5
BigCodeBench smoke/compatible subsets
HumanEval+ transfer smoke
GSM8K/MATH transfer data
```

They may be useful later for out-of-domain evaluation, but they should not be mixed into the current APPS taxonomy/rubric report unless explicitly reintroduced with separate dataset sections and separate metrics.

## Open Risk

APPS simple still has nontrivial long-output behavior:

| Metric | Value |
| --- | ---: |
| Full responses | 2613 |
| `finish_reason=length` | 330 |
| Length rate | 12.63% |

The current mitigation is to:

- report length-finished responses separately;
- exclude length-finished responses from core taxonomy construction;
- keep the full labeled file available for generation-quality reporting.

## Current Status

The APPS simple dataset has already completed:

```text
generation
-> verification
-> non-length failure extraction
-> Phase 1 taxonomy discovery/consolidation/refinement
-> Phase 2 rubric generation
```

No further dataset expansion is required before the next Method 1 evaluator/critic step.
