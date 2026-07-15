# Method 2 APPS v0.4 Iterative Self-Generated Repairs

## Goal

v0.3 solved the format problem:

- repair gate: `24/38 = 63.2%`
- extraction: `38/38 ok`
- remaining failures: mostly `logic_error`

v0.4 implements the Method 2 iteration loop:

```text
v0.3 adapter -> generate multiple repairs -> extract -> verifier -> keep passed repairs -> train v0.4
```

The key change is data, not model hyperparameters.

## Implementation

New generator support:

- `src/generation/vllm_lora_generate.py`
  - adds `--n`
  - writes one JSONL row per sample
  - keeps old `n=1` behavior compatible

New v0.4 builder:

- `src/self_play/build_method2_iterative_sft.py`
  - reads base v0.3 SFT rows
  - reads verifier-labeled self-generated candidates
  - keeps only `passed == true` and `method2_extraction_status == ok`
  - adds generated rows to `train` only
  - preserves base validation rows unchanged
  - defaults to at most `1` generated repair per original failed prompt
  - rebuilds clean `ERROR_FINDINGS + REVISED_CODE` completions from accepted candidates

## Scripts

Generate and verify v0.3 train candidates:

```bash
GPU=1 scripts/method2/run_method2_apps_self_play_generate_train_candidates_v0_4.sh
```

Defaults:

- source adapter: `outputs/method2_apps_self_play_critic_repair_sft_lora_v0_3_no_end_marker`
- source data: `data/sft/method2_apps_self_play_critic_repair_v0_3_no_end_marker.jsonl`
- split: `train`
- `K=5`
- `temperature=0.7`
- `top_p=0.95`
- stop sequences:
  - `\nPublic task prompt:`
  - `\nPrevious failed code:`

Build v0.4 SFT:

```bash
scripts/method2/build_method2_apps_self_play_sft_v0_4_iterative.sh
```

Train v0.4:

```bash
GPU=1 scripts/method2/run_method2_apps_self_play_critic_repair_sft_v0_4_iterative.sh
```

Evaluate v0.4 on the same repair gate:

```bash
GPU=1 scripts/method2/run_method2_apps_self_play_repair_gate_v0_4_iterative.sh
```

## Expected Outputs

Candidate generation:

- `data/self_play/method2_apps_self_play_v0_4_train_candidates_generations.jsonl`
- `data/self_play/method2_apps_self_play_v0_4_train_candidates_extracted.jsonl`
- `data/self_play/method2_apps_self_play_v0_4_train_candidates_labeled.jsonl`
- `data/self_play/method2_apps_self_play_v0_4_train_candidates_summary.json`

SFT build:

- `data/sft/method2_apps_self_play_critic_repair_v0_4_iterative.jsonl`
- `data/self_play/method2_apps_self_play_v0_4_accepted_self_generated.jsonl`
- `data/self_play/method2_apps_self_play_v0_4_iterative_summary.json`

Training:

- `outputs/method2_apps_self_play_critic_repair_sft_lora_v0_4_iterative`

Repair gate:

- `data/self_play/method2_apps_self_play_v0_4_iterative_validation_repair_gate_summary.json`

## Acceptance Criteria

Compare against v0.3:

- v0.3 pass: `24/38 = 63.2%`
- v0.3 extraction: `38/38 ok`
- v0.3 logic failures: `12`

v0.4 should:

- keep extraction at or near `38/38 ok`;
- avoid `empty_revised_code`;
- reduce `logic_error`;
- improve or at least not regress pass rate from `24/38`.

If candidate generation yields very few passed repairs, increase `K` first before changing model hyperparameters.
