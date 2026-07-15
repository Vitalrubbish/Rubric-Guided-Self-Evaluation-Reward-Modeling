# Method 1 Corrected Route: Verifier-Gated Generative Self-Evaluator

## Why the route changes

The classification-head evaluator was useful as a diagnostic baseline, but it
is not the right final form for Method 1.

It showed that Qwen2.5-7B can learn visible-code evaluation signal:

```text
LLM sequence-classifier test AUC: 0.8409
static critic test AUC:          0.8284
no-gate rubric test AUC:         0.7198
```

However, the learned ability lives in a separate `score` head. That separates
evaluation from generation and weakens the core project claim: one model should
learn to generate solutions, critique its own outputs, apply rubrics, and then
improve from those signals.

Therefore the classification-head evaluator is retained only as an ablation and
calibration reference. The Method 1 mainline should use `AutoModelForCausalLM`
throughout.

## Corrected objective

Train the same CausalLM policy to perform three related behaviors:

```text
solve:
  generate executable Python code for a task

critique:
  explain whether a submitted solution satisfies the public task contract

judge:
  output a verifier-aligned PASS/FAIL or pairwise winner using rubric language
```

The external verifier remains a hard bootstrap anchor in this stage. It is not
the final reward model. Its role is to prevent early self-evaluation noise from
being amplified before the model has learned a stable judging format.

## Stage A: verifier-gated generative bootstrap

Stage A builds a mixed CausalLM SFT dataset:

```text
1. single-solution judge traces
   input: public task + public interface + submitted code
   output: rubric checklist + analysis + PASS/FAIL + primary error

2. pairwise judge traces
   input: public task + candidate A + candidate B
   output: rubric comparison + Winner: A/B
   source: verifier-confirmed passing repair > failed original

3. solve traces
   input: original coding prompt
   output: verifier-passing code
   source: base verified pass rows and verifier-passing repairs
```

This is not a separate reward model. It is instruction tuning for the generator
itself. The point is to teach the model the language and format of rubric-based
self-evaluation while preserving code generation behavior.

## Stage B: joint improvement

After Stage A produces a stable generative evaluator, Method 1 can re-enter
preference optimization:

```text
CausalLM joint training =
  code SFT on verified passing solutions
+ evaluator SFT on verifier-matched critique/judge traces
+ repair DPO on verifier-confirmed pass > fail pairs
```

Only after the generative evaluator passes held-out self-evaluation gates should
the pipeline allow model-generated judge decisions to construct new preference
pairs. Even then, verifier checks remain as audit and calibration, not as the
main inference-time scorer.

## What not to do

Do not use raw self-judgment as DPO truth:

```text
"model thinks A > B" -> DPO(A > B)
```

Current evidence shows that self-evaluation is not reliable enough for that.
The safer policy is:

```text
model-generated critique/judgment enters training only when verifier outcome,
parseability, and public-interface checks agree with it.
```

Do not optimize only an evaluator head. That can improve AUC but does not teach
the generator to produce better code or better textual self-critiques.

## Evaluation gates

The next generative evaluator should be evaluated on held-out rows with no
execution result in the prompt:

```text
binary self-evaluation:
  parse generated verdict
  compare PASS/FAIL to verifier label
  report AUC-like ranking if confidence is emitted
  report accuracy, balanced accuracy, Kappa
  report overacceptance and false rejection

format reliability:
  verdict parse rate
  required fields present
  no code execution diagnostics leaked into prompt

generation preservation:
  run code-generation canary after SFT
  pass@1 must not drop below the matching base/dev baseline
  syntax errors must not increase
```

The central gate is not just "can it judge"; it is:

```text
Can the same CausalLM improve self-evaluation without damaging code generation?
```

## Immediate implementation

The repository should keep:

```text
src/evaluator/build_generative_evaluator_sft_data.py
src/training/train_causallm_sft_lora.py
scripts/method1/build_method1_apps_generative_evaluator_sft_v1.sh
scripts/method1/run_method1_apps_generative_evaluator_sft_v1.sh
```

The classification-head route should be removed from active implementation and
kept only as historical result artifacts:

```text
data/evaluator/apps_simple_llm_self_evaluator_lora_v1_summary.json
outputs/apps_simple_method1_llm_self_evaluator_lora_v1/
```

Those artifacts are evidence for the route correction, not the continuing
training path.
