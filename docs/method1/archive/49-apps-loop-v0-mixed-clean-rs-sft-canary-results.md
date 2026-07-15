# APPS DPO-v2 Canary Evaluation

- Rows: 160
- Base pass@1: 0.4813
- Candidate pass@1: 0.4750
- Net passes: -1
- Canary gate: FAIL

## Gates

- [ ] pass_at_1_not_lower
- [ ] syntax_errors_not_higher
- [x] length_finishes_not_higher
- [x] positive_transition_present
- [ ] positive_transitions_not_lower_than_regressions
- [x] train_overlap_zero

## Full Summary

```json
{
  "rows": 160,
  "base_passed": 77,
  "base_pass_rate": 0.48125,
  "candidate_passed": 76,
  "candidate_pass_rate": 0.475,
  "net_pass_delta": -1,
  "transitions": {
    "base_fail->candidate_fail": 78,
    "base_pass->candidate_pass": 71,
    "base_pass->candidate_fail": 6,
    "base_fail->candidate_pass": 5
  },
  "positive_transitions": 5,
  "negative_transitions": 6,
  "base_failure_counts": {
    "syntax_error": 45,
    "passed": 77,
    "runtime_error": 14,
    "logic_error": 22,
    "timeout": 2
  },
  "candidate_failure_counts": {
    "syntax_error": 46,
    "passed": 76,
    "runtime_error": 15,
    "logic_error": 22,
    "timeout": 1
  },
  "base_length_finishes": 28,
  "candidate_length_finishes": 27,
  "training_overlap_count": 0,
  "decoding": {
    "temperature": 0.0,
    "top_p": 1.0,
    "repetition_penalty": 1.0,
    "max_tokens": 2048,
    "seed": 42
  },
  "paired_verification": {
    "paired_verification_run_id": "ad986c0501d6c8bfea09cc5ca9cf636bac1f4c2f28e2679094c5d3bfe1117de9",
    "paired_verification_timeout": 30.0,
    "paired_verification_workers": 4
  },
  "gates": {
    "pass_at_1_not_lower": false,
    "syntax_errors_not_higher": false,
    "length_finishes_not_higher": true,
    "positive_transition_present": true,
    "positive_transitions_not_lower_than_regressions": false,
    "train_overlap_zero": true
  },
  "canary_passed": false,
  "policy": "train-derived DPO-dev; excluded from preference training and final 523 held-out"
}
```
