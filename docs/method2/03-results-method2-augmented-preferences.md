# LLM-Critic Augmented Preference Data

## Summary

| Field | Value |
| --- | ---: |
| Total pairs | 266 |
| Base input | `data/preferences/preference_pairs_qwen25_k1_mbpp_train_augmented.jsonl` |
| LLM critic input | `data/self_play/llm_critic_pairs_mbpp_train_n54_v1.jsonl` |
| Output | `data/preferences/preference_pairs_qwen25_k1_mbpp_train_augmented_llmcritic54.jsonl` |

## Chosen Sources

| Source | Count |
| --- | ---: |
| canonical_solution | 158 |
| llm_self_play_revised_passed | 54 |
| rule_revised_success_output | 54 |

## Leakage Check

Only `mbpp/train` rows are retained. Validation/test rows are skipped by construction.

## Skipped

| Reason | Count |
| --- | ---: |
