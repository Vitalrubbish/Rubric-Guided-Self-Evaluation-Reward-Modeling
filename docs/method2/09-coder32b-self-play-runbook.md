# Coder-32B Self-Play Loop Runbook

Date: 2026-07-25

## Goal

Decisive self-evolution experiment at 32B scale: replicate the 7B
v0.3 -> v0.4 -> v0.6 protocol with Qwen2.5-Coder-32B-Instruct (bf16) and
measure whether the self-play trajectory compounds (climbs over rounds)
or plateaus/oscillates as it did at 7B (36 -> 41 -> 36).

Preconditions already measured (inference-only, AWQ): bare gate200 repair
48.5%, gold100 logic repair 27%, attribution hit 46%, repair|hit 43.5%.

## Hardware

- ONE nearly free A800 80GB per step (>= 74GB free). Do NOT run on shared
  GPUs below that: bf16 32B weights are ~65GB and vLLM LoRA inference +
  training are both single-GPU jobs. Steps are sequential.
- Training uses the existing single-GPU HF Trainer
  (`src/training/train_causallm_sft_lora.py`, bf16 + gradient
  checkpointing built in). AWQ cannot be used for training.

## Variables

```bash
M32=models/models--Qwen--Qwen2.5-Coder-32B-Instruct/snapshots/381fc969f78efac66bc87ff7ddeadb7e73c218a7
BASE_SFT=data/sft/method2_apps_self_play_critic_repair_v0_3_no_end_marker.jsonl   # 373 rows, same as 7B
GATE200=data/sft/method2_apps_self_play_critic_repair_gate200_validation.jsonl
GOLD100=data/self_play/method2_gold100_telemetry_input.jsonl
GPU=0   # pick the free one
```

## Round 0: bootstrap SFT (32B-r0)

```bash
MODEL=$M32 DATA=$BASE_SFT \
OUTPUT_DIR=outputs/method2_coder32b_critic_repair_sft_lora_r0 \
GPU=$GPU scripts/method2/run_method2_self_play_critic_repair_sft_v0.sh
```

## Round 0 evaluation: gate200 + gold100 (same commands every round, swap ADAPTER and tag)

```bash
R=r0
ADAPTER=outputs/method2_coder32b_critic_repair_sft_lora_$R

# gate200
MODEL=$M32 ADAPTER=$ADAPTER INPUT=$GATE200 \
VALIDATION_INPUT=data/self_play/coder32b_${R}_gate200_input.jsonl \
VALIDATION_INPUT_SUMMARY=data/self_play/coder32b_${R}_gate200_input_summary.json \
GENERATIONS=data/self_play/coder32b_${R}_gate200_generations.jsonl \
EXTRACTED=data/self_play/coder32b_${R}_gate200_extracted.jsonl \
EXTRACT_SUMMARY=data/self_play/coder32b_${R}_gate200_extract_summary.json \
LABELED=data/self_play/coder32b_${R}_gate200_labeled.jsonl \
SUMMARY=data/self_play/coder32b_${R}_gate200_summary.json \
MIN_VALIDATION_ROWS=200 GPU=$GPU GPU_MEMORY_UTILIZATION=0.9 PROMPT_BATCH_SIZE=8 \
scripts/method2/run_method2_apps_self_play_repair_gate_v0_3_no_end_marker.sh

# gold100 telemetry (generations + verifier; attribution grading is done
# separately by the frontier-LLM scoring pipeline, not in this runbook)
MODEL=$M32 ADAPTER=$ADAPTER INPUT=$GOLD100 \
VALIDATION_INPUT=data/self_play/coder32b_${R}_gold100_input.jsonl \
VALIDATION_INPUT_SUMMARY=data/self_play/coder32b_${R}_gold100_input_summary.json \
GENERATIONS=data/self_play/coder32b_${R}_gold100_generations.jsonl \
EXTRACTED=data/self_play/coder32b_${R}_gold100_extracted.jsonl \
EXTRACT_SUMMARY=data/self_play/coder32b_${R}_gold100_extract_summary.json \
LABELED=data/self_play/coder32b_${R}_gold100_labeled.jsonl \
SUMMARY=data/self_play/coder32b_${R}_gold100_summary.json \
MIN_VALIDATION_ROWS=100 GPU=$GPU GPU_MEMORY_UTILIZATION=0.9 PROMPT_BATCH_SIZE=8 \
scripts/method2/run_method2_apps_self_play_repair_gate_v0_3_no_end_marker.sh
```

## Round 1: self-play iteration (32B-r1)

```bash
# 1. candidates: r0 adapter, K=5 on the same 335 train prompts
MODEL=$M32 ADAPTER=outputs/method2_coder32b_critic_repair_sft_lora_r0 \
TRAIN_INPUT=data/self_play/coder32b_r1_train_input.jsonl \
TRAIN_INPUT_SUMMARY=data/self_play/coder32b_r1_train_input_summary.json \
GENERATIONS=data/self_play/coder32b_r1_train_candidates_generations.jsonl \
EXTRACTED=data/self_play/coder32b_r1_train_candidates_extracted.jsonl \
EXTRACT_SUMMARY=data/self_play/coder32b_r1_train_candidates_extract_summary.json \
LABELED=data/self_play/coder32b_r1_train_candidates_labeled.jsonl \
CANDIDATE_SUMMARY=data/self_play/coder32b_r1_train_candidates_summary.json \
GPU=$GPU GPU_MEMORY_UTILIZATION=0.9 PROMPT_BATCH_SIZE=8 \
scripts/method2/run_method2_apps_self_play_generate_train_candidates_v0_4.sh

# 2. build r1 SFT (same recipe as 7B v0.4)
GENERATED_LABELED=data/self_play/coder32b_r1_train_candidates_labeled.jsonl \
SFT_OUTPUT=data/sft/method2_coder32b_critic_repair_r1_iterative.jsonl \
ACCEPTED_OUTPUT=data/self_play/coder32b_r1_accepted_self_generated.jsonl \
SUMMARY_OUTPUT=data/self_play/coder32b_r1_iterative_summary.json \
SOURCE_TAG=method2_coder32b_r1_self_generated_pass \
scripts/method2/build_method2_apps_self_play_sft_v0_4_iterative.sh

# 3. train from base
MODEL=$M32 DATA=data/sft/method2_coder32b_critic_repair_r1_iterative.jsonl \
OUTPUT_DIR=outputs/method2_coder32b_critic_repair_sft_lora_r1 \
GPU=$GPU scripts/method2/run_method2_self_play_critic_repair_sft_v0.sh

# 4. evaluate (run the Round-0 eval block with R=r1)
```

## Round 2: repeat Round 1 with the r1 adapter (mirror of 7B v0.6)

Same three steps, substituting r1 -> r2 in all paths and using
`ADAPTER=outputs/method2_coder32b_critic_repair_sft_lora_r1` for candidate
generation.

## Acceptance (after each round)

```bash
python src/analysis-reporting/bootstrap_method2_repair_gate_stats.py \
  --baseline data/self_play/coder32b_r0_gate200_labeled.jsonl \
  --candidate data/self_play/coder32b_r1_gate200_labeled.jsonl \
  --output data/self_play/coder32b_r1_vs_r0_bootstrap.json
```

Pre-registered decision rule (same as 7B):

- bootstrap CI95 excludes 0 and is positive -> compounding: self-evolution
  trajectory holds at 32B; continue to round 3;
- CI includes 0 -> plateau: same information-conservation ceiling as 7B,
  at a higher capability level;
- CI excludes 0 and is negative -> self-consuming regression.

Report per round: gate200 pass rate + Wilson CI + paired bootstrap diff,
gold100 overall repair, and (via the frontier-LLM grading step) attribution
hit rate and repair|hit.

## Expected Durations (single A800, shared cluster)

| step | estimate |
| --- | --- |
| r0/r1/r2 training (~640 rows, 75 steps) | 2-4 h |
| candidate generation (335x5) | 1-2 h |
| gate200 eval | ~30 min |
| gold100 eval | ~15 min |

## Notes

- Keep LEARNING_RATE/EPOCHS at script defaults (5e-7, 1 epoch): same
  conservative protocol as 7B.
- bf16 vs AWQ: the 48.5%/43.5% preconditions were measured on AWQ; the
  loop trains and evaluates in bf16. Report any discrepancy vs those
  numbers as the quantization check.
- After gold100 labeled files exist for a round, hand them back for
  frontier-LLM attribution grading (files:
  `data/self_play/coder32b_<R>_gold100_extracted.jsonl` and
  `..._labeled.jsonl`).
