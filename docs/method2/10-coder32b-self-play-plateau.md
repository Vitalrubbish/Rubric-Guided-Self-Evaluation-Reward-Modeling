# Coder-32B Self-Play Loop: Plateau At The Factory Ceiling

Date: 2026-07-26

## Design

Exact replication of the 7B v0.3 -> v0.4 -> v0.6 self-play protocol at
32B scale, with only the model changed:

- model: Qwen2.5-Coder-32B-Instruct, QLoRA (NF4 + double quant, bf16
  compute; bitsandbytes 0.50.0) on a shared A800;
- r0: bootstrap SFT on the same 373 rows used by 7B v0.3;
- r1/r2: K=5 self-generated candidates on the same 335 train prompts,
  verifier-filtered, v0.4-recipe SFT (base 373 + accepted self-generated),
  retrained from base each round;
- acceptance: 200-row gate with paired bootstrap (pre-registered), plus
  gold-100 telemetry (frontier-LLM scored attribution).

QLoRA/bnb inference patches: `src/training/train_causallm_sft_lora.py`
(--load-in-4bit), `src/generation/vllm_lora_generate.py`
(--quantization/--load-format), gate/candidate scripts env passthrough.

## Results

Gate-200 trajectory (vs bare AWQ 32B measured on the same gate):

| model | gate200 | gold100 logic repair | attribution hit | repair\|hit |
| --- | --- | --- | --- | --- |
| bare (AWQ, no training) | 97/200 = 48.5% | 27% | 46 | 43.5% |
| r0 (bootstrap SFT) | 92/200 = 46.0% | 29% | 49 | 40.8% |
| r1 (self-play round 1) | 97/200 = 48.5% | 28% | 46 | 41.3% |
| r2 (self-play round 2) | 97/200 = 48.5% | 28% | — | — |

Paired comparisons:

- r1 vs r0: +5 rows, CI95 [0, +10], P(better)=96.6% (mirrors 7B round 1);
- r2 vs r1: +0, transitions 3 F->P / 3 P->F, McNemar p=1.0, CI95 [-5, +5];
- r1/r2 vs bare: identical pass count (97); row-level churn ~20 rows
  around a stable 84-row common core; no capability-core expansion.

Candidate pass rate (self-generated fuel): r1 round 83.7%, r2 round
83.9% — the model's output distribution on the 335 train prompts is
already saturated after round 1.

## Verdict

**The self-play loop at 32B does not compound.** Two rounds leave the
model exactly at the bare factory level: same pass count, same telemetry,
only borderline-case churn. Combined with 7B (36 -> 41 -> 36), the
evidence now spans two scales and five checkpoints:

> Verifier-filtered self-play SFT on a fixed prompt set does not change
> the model's stable capability core, at 7B or 32B. The loop has no
> information inflow beyond 1-bit verifier selection over the model's own
> already-saturated output distribution.

Notably, r0 showed a small (insignificant) dip below bare: the 7B-era
bootstrap data is below 32B's own level — training a stronger model on
weaker demonstrations is mildly counterproductive, and the self-play
rounds merely return to the factory ceiling without exceeding it.

## Conditions And Caveats

- All 32B numbers are QLoRA-NF4 (training) and AWQ (bare reference);
  quantization is a constant condition across rounds, so trajectory
  comparisons are internally valid. A bf16 confirmation run of r1/r2 vs
  bare is queued for when a full GPU frees.
- The 200-row gate has +-7pp Wilson half-width; nothing here rules out
  per-round effects smaller than ~2-3pp. The claim is "no compounding",
  not "zero effect at any granularity".

## Consequences (agreed next routes)

1. Execution-feedback probe: repair with public-test execution diffs in
   the prompt (N-bit signal instead of 1-bit), test-time first, then a
   feedback-distillation loop evaluated WITHOUT feedback. This replaces
   the exhausted information channel.
2. Self-generated curriculum: model generates new problem variants to
   escape the saturated 335-prompt pool.
3. v0.7 external-signal arm (gold diagnosis->repair demonstrations) as
   the controlled answer to "which errors need external signals".
4. 72B deprioritized: it would raise the plateau, not answer a new
   question.

## Artifacts

- adapters: `outputs/method2_coder32b_qlora_r{0,1,2}`
- SFT: `data/sft/method2_coder32b_qlora_r{1,2}_iterative.jsonl`
- gates: `data/self_play/coder32b_qlora_r{0,1,2}_gate200_*.jsonl`,
  `..._gold100_*.jsonl`
- telemetry scorings: `data/annotation/apps_simple_gold_scoring_{coder32b,qlora_r0,qlora_r1}_merged.jsonl`
- bootstrap stats: `data/self_play/coder32b_qlora_r1_vs_r0_bootstrap.json`

## Addendum: Doc-Faithful Method-2 DPO (2026-07-26)

The assignment text specifies preference-pair training `(A < B)`, while the
loop above used SFT. The faithful variant was run as a separate arm:

- pairs: 215 unique problems from r1/r2 accepted self-generated rows
  (chosen = model-written ERROR_FINDINGS + verifier-passing repair;
  rejected = placeholder finding + original failed code), filtered to 174
  after dropping pairs whose chosen findings claimed the failed code was
  already correct; training used 143 after token-length filters;
- QLoRA-DPO from base, beta 0.1, LR 5e-7, 1 epoch
  (`outputs/method2_coder32b_qlora_dpo_r1`, rewards/accuracies 0.87);
- data: `data/preferences/method2_coder32b_self_play_dpo_pairs_clean.jsonl`.

Result: **93/200 = 46.5%**, vs bare −4 rows, CI95 [−13, +5], McNemar
p=0.52; gold100 28/100 (flat). The negative-sample channel does not lift
the model above the factory ceiling either — DPO lands in the same
statistical band as every other variant (46-48.5%), marginally below it.

**Method 2 is therefore falsified at 32B in both forms**: SFT on positive
self-play data and DPO on (failed < repaired) preference pairs alike.
