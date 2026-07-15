# APPS DPO-v2 Canary Evaluation

- Rows: 160
- Base pass@1: 0.4813
- Candidate pass@1: 0.4688
- Net passes: -2
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
  "candidate_passed": 75,
  "candidate_pass_rate": 0.46875,
  "net_pass_delta": -2,
  "transitions": {
    "base_fail->candidate_fail": 77,
    "base_pass->candidate_pass": 69,
    "base_pass->candidate_fail": 8,
    "base_fail->candidate_pass": 6
  },
  "positive_transitions": 6,
  "negative_transitions": 8,
  "base_failure_counts": {
    "syntax_error": 45,
    "passed": 77,
    "runtime_error": 14,
    "logic_error": 22,
    "timeout": 2
  },
  "candidate_failure_counts": {
    "syntax_error": 47,
    "passed": 75,
    "runtime_error": 14,
    "logic_error": 23,
    "timeout": 1
  },
  "base_length_finishes": 28,
  "candidate_length_finishes": 25,
  "training_overlap_count": 0,
  "decoding": {
    "temperature": 0.0,
    "top_p": 1.0,
    "repetition_penalty": 1.0,
    "max_tokens": 2048,
    "seed": 42
  },
  "paired_verification": {
    "paired_verification_run_id": "7020b487eaef3bebd73943b321d71e7ea5c5c16023ff033477c70205da438775",
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
