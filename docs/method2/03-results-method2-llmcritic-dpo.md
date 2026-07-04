# LLM-Critic DPO Results

## Status

| Artifact | Status |
| --- | --- |
| preference_data | done |
| llm_critic | done |
| dpo_training | done |
| validation | done |
| protected_validation | done |

## Key Metrics

| Metric | Value |
| --- | ---: |
| Preference pairs | 266 |
| LLM critic attempted | 54 |
| LLM critic repaired | 54 |
| LLM critic repair rate | 100.00% |
| DPO steps | 266 |
| DPO mean loss | 0.6464 |
| DPO preference accuracy | 79.70% |
| Validation passed | 43/90 |
| Validation pass@1 | 47.78% |
| Protected validation passed | 54/90 |
| Protected validation pass@1 | 60.00% |

## Caveat

This run uses only MBPP train preference pairs. Validation is untouched, but the run is still small; use it as an ablation rather than a final headline unless it is repeated or scaled.
