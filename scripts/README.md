# Script Layout

Scripts are grouped by project stage:

| Directory | Purpose |
| --- | --- |
| `phase1/` | Phase 1 error discovery and taxonomy pipeline. |
| `phase2/` | Phase 2 taxonomy-to-rubric generation and rubric judge runners. |
| `method1/` | Archived APPS Method 1 DPO/SFT/evaluator experiments. |
| `method2/` | Active APPS Method 2 self-play repair route. |

Run scripts from the repository root, for example:

```bash
GPU=2 scripts/phase1/run_phase1_pipeline.sh
GPU=2 scripts/phase2/run_phase2_rubric_generation.sh
GPU=1 scripts/method2/run_method2_apps_self_play_v0_4_iterative_full.sh
```
