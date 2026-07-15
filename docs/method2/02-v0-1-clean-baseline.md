# Method 2 APPS v0.1 Repair Gate

## Setup

- Adapter: `outputs/method2_apps_self_play_critic_repair_sft_lora_v0_1_clean`
- Validation input: `data/self_play/method2_apps_self_play_v0_1_clean_validation_input.jsonl`
- Rows: 38 APPS validation repair prompts
- Data source: `data/sft/method2_apps_self_play_critic_repair_v0_1_clean.jsonl`

## Original Gate Output

- Passed: 19 / 38
- Pass rate: 0.5000
- Failure counts:
  - syntax_error: 12
  - logic_error: 5
  - runtime_error: 1
  - timeout: 1
- Extraction:
  - ok: 33
  - missing_revised_code_marker: 5
- Finish reasons:
  - stop: 23
  - length: 15

The initial gate failed because syntax rate was 12/38 and 5 generations did not
emit the required `REVISED_CODE` marker.

## Post-Extraction v2

After improving `src/self_play/extract_method2_revised_code.py` to trim
parseable code prefixes and remove function-call demo harnesses:

- Passed: 23 / 38
- Pass rate: 0.6053
- Failure counts:
  - logic_error: 8
  - runtime_error: 2
  - syntax_error: 5
- Extraction:
  - ok: 33
  - missing_revised_code_marker: 5
- Finish reasons:
  - stop: 23
  - length: 15

Four cases moved from syntax error to pass. No originally passing case regressed
under the improved extraction.

## Interpretation

This is the first positive APPS signal for Method 2: the critic+repair adapter
repairs 60.5% of held-out failed validation prompts after robust extraction.

The remaining blockers are format stability and long generations:

- 5 / 38 outputs miss `REVISED_CODE`.
- 15 / 38 hit the generation length limit.
- Length outputs are mixed: 8 / 15 pass, but length also contributes most
  missing-marker failures.

## Next Step

Build v0.2 with stricter format pressure and shorter targets:

- filter or downweight very long revised-code completions;
- add a final prompt cue immediately before generation, e.g.
  `Now output ERROR_FINDINGS followed by REVISED_CODE:`;
- consider lower `max_tokens` only after prompt/target cleanup, because some
  length-finished outputs currently pass.
