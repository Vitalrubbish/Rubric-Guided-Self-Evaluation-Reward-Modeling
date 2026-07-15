# APPS DPO-v2 Canary Evaluation

- Rows: 160
- Base pass@1: 0.4813
- Candidate pass@1: 0.4875
- Net passes: +1
- Canary gate: FAIL

## Gates

- [x] pass_at_1_not_lower
- [ ] syntax_errors_not_higher
- [ ] length_finishes_not_higher
- [x] positive_transition_present
- [x] positive_transitions_not_lower_than_regressions
- [x] train_overlap_zero

## Full Summary

```json
{
  "rows": 160,
  "base_passed": 77,
  "base_pass_rate": 0.48125,
  "candidate_passed": 78,
  "candidate_pass_rate": 0.4875,
  "net_pass_delta": 1,
  "transitions": {
    "base_fail->candidate_fail": 78,
    "base_pass->candidate_pass": 73,
    "base_fail->candidate_pass": 5,
    "base_pass->candidate_fail": 4
  },
  "positive_transitions": 5,
  "negative_transitions": 4,
  "base_failure_counts": {
    "syntax_error": 45,
    "passed": 77,
    "runtime_error": 14,
    "logic_error": 22,
    "timeout": 2
  },
  "candidate_failure_counts": {
    "syntax_error": 47,
    "passed": 78,
    "runtime_error": 11,
    "logic_error": 23,
    "timeout": 1
  },
  "base_length_finishes": 28,
  "candidate_length_finishes": 29,
  "training_overlap_count": 0,
  "decoding": {
    "temperature": 0.0,
    "top_p": 1.0,
    "repetition_penalty": 1.0,
    "max_tokens": 2048,
    "seed": 42
  },
  "paired_verification": {
    "paired_verification_run_id": "7e164e7dee884d8c1289be5d99420d5abe11c3cb5653f064447b4f3ad16bfbad",
    "paired_verification_timeout": 30.0,
    "paired_verification_workers": 4
  },
  "gates": {
    "pass_at_1_not_lower": true,
    "syntax_errors_not_higher": false,
    "length_finishes_not_higher": false,
    "positive_transition_present": true,
    "positive_transitions_not_lower_than_regressions": true,
    "train_overlap_zero": true
  },
  "canary_passed": false,
  "policy": "train-derived DPO-dev; excluded from preference training and final 523 held-out"
}
```
