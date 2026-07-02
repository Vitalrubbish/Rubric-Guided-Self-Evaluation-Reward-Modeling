# Rubric-Guided Self-Evaluation and Reward Modeling

Training a model that can self-discover error patterns and automatically generate rubrics (scoring criteria) for self-evaluation during task completion, thereby improving its own outputs and training methodology. The core idea: the model does not rely on human-provided scoring standards; instead, it automatically induces rubrics from its own failure experiences and uses them as learnable reward signals to guide self-evolution.

## Core Approach

1. **Error Pattern Discovery** — Model generates responses on benchmarks (GSM8K, MT-Bench), external verifiers label failures, and the model clusters/attributes errors to output an error taxonomy.
2. **Automatic Rubric Generation** — Based on discovered error patterns, the model generates scoring rubrics with dimensions, 1-5 scoring criteria, and positive/negative examples.
3. **Self-Evaluation with Rubrics** — The model scores new responses using its own rubrics, validated against external judgments via Cohen's Kappa.
4. **Self-Evolving Loop** — Iterative DPO training using rubric-based rewards, self-play error discovery, and meta-learning to generalize self-evaluation across tasks.

## Evaluation Benchmarks

- MT-Bench (multi-turn, multi-dimension)
- AlpacaEval 2.0 (instruction following)
- GSM8K / MATH (error pattern discovery accuracy)

## Project Phases

### Phase 1 (Assignment 3): Error Discovery + Rubric Generation + Baseline Evaluation

- Step 1: Build error pattern discovery pipeline
- Step 2: Automatic rubric generation with comparison to human-written rubrics
- Step 3: Self-evaluation with auto-generated rubrics (Cohen's Kappa)

### Phase 2 (Assignment 4): Self-Evolving Loop

- Method 1: Error-Pattern → Rubric → RL closed loop (iterative DPO)
- Method 2: Self-Play Error Discovery (explicit error identification before improvement)
- Method 3: Meta-Learning to Self-Evaluate (cross-task generalization)

## Key Metrics

| Metric | Description |
|--------|-------------|
| Error Pattern Coverage | How well discovered patterns cover actual failures |
| Rubric Discriminability (AUC) | How well rubrics distinguish good vs. bad responses |
| Self-Evaluation Consistency | Alignment between self-scores and external judgments |
| Error Detection Rate | Precision/recall of model-identified errors |
| Cross-Task Generalization | Zero-shot rubric quality on unseen tasks |
