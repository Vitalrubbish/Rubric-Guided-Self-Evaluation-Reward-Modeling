# Method 2 APPS v0.3 No-End-Marker Results

## Summary

v0.3 fixes the v0.2 early-stop failure and is the best Method 2 repair gate result so far.

Repair gate:

- rows: `38`
- passed: `24`
- pass rate: `63.2%`
- gate passed: `true`
- extraction: `38/38 ok`
- finish: `22 stop / 16 length`
- failures: `12 logic_error`, `1 syntax_error`, `1 runtime_error`

Training:

- train rows: `330`
- validation rows: `38`
- global step: `42`
- train loss: `1.7479`
- eval loss: `2.1118`
- eval perplexity: `8.26`
- best checkpoint: `checkpoint-25`

## Comparison

| Version | Pass | Extraction | Length | Main Problem |
|---|---:|---:|---:|---|
| v0.1 postextract | `23/38 = 60.5%` | `33 ok / 5 missing` | `15/38` | marker misses and syntax tail cleanup |
| v0.2 format | `13/38 = 34.2%` | `25 ok / 9 empty / 4 missing` | `4/38` | early `END_REVISED_CODE` stop |
| v0.3 no-end-marker | `24/38 = 63.2%` | `38 ok` | `16/38` | logic repair quality |

Against v0.1 on the same 38 validation rows:

- kept pass: `19`
- improved: `5`
- regressed: `4`
- both failed: `10`

Against v0.2:

- kept pass: `9`
- improved: `15`
- regressed: `4`
- both failed: `10`

## Interpretation

The `END_REVISED_CODE` strategy should be rejected. It reduced length completions, but taught the model to emit the stop marker immediately.

v0.3 proves that a prompt-final answer delimiter, `Repair response:`, is enough to make extraction stable without an explicit generated end marker.

The remaining bottleneck is no longer format:

- no missing marker;
- no empty revised code;
- only one syntax failure;
- most failures are logic errors.

## Next Direction

Do not spend the next iteration on stricter format controls. The next useful work is logic-focused data improvement:

- preserve v0.3 prompt format;
- add verifier/public-example feedback into `ERROR_FINDINGS`;
- add more successful APPS repair pairs for the 14 failed validation-style cases;
- consider a second Method 2 self-play loop where v0.3 generates repairs, verifier keeps only successful repairs, and those are added back as high-value SFT examples.
