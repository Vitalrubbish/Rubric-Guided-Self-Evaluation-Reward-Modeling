# Logic Multi-Candidate Self-Play

## Summary

| Metric | Value |
| --- | ---: |
| Attempted tasks | 20 |
| Total candidates | 100 |
| Passed candidates | 13 |
| Repaired tasks | 7 |
| Preference pairs | 7 |
| Task repair rate | 35.00% |
| Candidate pass rate | 13.00% |
| Gate passed | True |

## Selected IDs

- `mbpp/train/610`
- `mbpp/train/612`
- `mbpp/train/622`
- `mbpp/train/631`
- `mbpp/train/648`
- `mbpp/train/661`
- `mbpp/train/670`

## Interpretation

This selects at most one passing candidate per original failed task. If the gate fails, do not merge these pairs into DPO; use the failures to improve the critic prompt or add stronger external feedback.

## Outputs

- `data/self_play/llm_critic_pairs_mbpp_train_logic_n20_k5.jsonl`
- `data/self_play/llm_critic_metrics_mbpp_train_logic_n20_k5.json`
