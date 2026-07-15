# APPS DPO-v2 Canary Evaluation

- Rows: 160
- Base pass@1: 0.4750
- Candidate pass@1: 0.4562
- Net passes: -3
- Canary gate: FAIL

## Gates

- [ ] pass_at_1_not_lower
- [ ] syntax_errors_not_higher
- [ ] length_finishes_not_higher
- [x] positive_transition_present
- [ ] positive_transitions_not_lower_than_regressions
- [x] train_overlap_zero

## Full Summary

```json
{
  "rows": 160,
  "base_passed": 76,
  "base_pass_rate": 0.475,
  "candidate_passed": 73,
  "candidate_pass_rate": 0.45625,
  "net_pass_delta": -3,
  "transitions": {
    "base_fail->candidate_fail": 80,
    "base_pass->candidate_pass": 69,
    "base_pass->candidate_fail": 7,
    "base_fail->candidate_pass": 4
  },
  "positive_transitions": 4,
  "negative_transitions": 7,
  "base_failure_counts": {
    "syntax_error": 44,
    "passed": 76,
    "runtime_error": 14,
    "logic_error": 22,
    "timeout": 4
  },
  "candidate_failure_counts": {
    "syntax_error": 48,
    "runtime_error": 13,
    "logic_error": 23,
    "timeout": 3,
    "passed": 73
  },
  "base_length_finishes": 28,
  "candidate_length_finishes": 31,
  "training_overlap_count": 0,
  "decoding": {
    "temperature": 0.0,
    "top_p": 1.0,
    "repetition_penalty": 1.0,
    "max_tokens": 2048,
    "seed": 42
  },
  "paired_verification": {
    "paired_verification_run_id": "90f6d04e8fcc86f14330264e93c05ba32d0de91c5d88b9503cf2b86e5c96895d",
    "paired_verification_timeout": 30.0,
    "paired_verification_workers": 4
  },
  "gates": {
    "pass_at_1_not_lower": false,
    "syntax_errors_not_higher": false,
    "length_finishes_not_higher": false,
    "positive_transition_present": true,
    "positive_transitions_not_lower_than_regressions": false,
    "train_overlap_zero": true
  },
  "canary_passed": false,
  "policy": "train-derived DPO-dev; excluded from preference training and final 523 held-out"
}
```
