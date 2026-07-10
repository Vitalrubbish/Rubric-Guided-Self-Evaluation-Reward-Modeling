# Phase 2 Dataset Expansion Recommendation

Date: 2026-07-10

## Conclusion

A larger dataset is a reasonable follow-up extension, but it should not be the first Phase 2 step.

Phase 1 should remain frozen as:

```text
MBPP hidden-tests k=3/k=5 -> verifier -> safe failures -> LLM attribution -> clustering -> consolidated taxonomy -> rubric-operational refined taxonomy
```

The first Phase 2 step should be automatic rubric generation from the Phase 1 refined taxonomy, followed by an LLM rubric judge evaluation on held-out responses. Dataset expansion should come after the rubric baseline, as a Phase 2.5 or robustness extension.

## Why Expansion Is Reasonable Later

Phase 1 has already solved several key engineering issues:

- prompts contain no assertions;
- verifier data is separated from model input;
- safe and private artifacts are separated;
- alignment is performed at `response_id` level;
- LLM attribution and automatic clustering are working;
- k=3 and k=5 have broadly stable error distributions.

This means the pipeline has a basis for transfer. However, changing to a larger dataset before completing taxonomy -> rubric -> rubric judge would not answer the core Phase 2 question.

## Why Not Switch Directly To A Larger Dataset

Several constraints remain:

- the verifier is still lightweight multiprocessing, not a full sandbox;
- clustering can still have task/sample repetition bias;
- the taxonomy now has automated consolidated and refined versions, but the full rubric judge baseline must be validated first;
- larger datasets usually require stdin/stdout handling, multi-file tasks, dependency management, complex timeouts, and safer execution.

The expansion should therefore start with an adapter and smoke run, then move to a medium-scale run, and only then consider full-scale evaluation.

## Candidate Datasets

| Candidate | Role | Strengths | Risks | Recommendation |
| --- | --- | --- | --- | --- |
| HumanEval+ / EvalPlus | Cross-benchmark validation | Local `data/raw/humanevalplus_test.jsonl` already exists; engineering cost is low; tests are stricter | Small task count, so it is not a real scale-up | First priority as a pipeline-transfer smoke |
| BigCodeBench | More realistic code tasks | Closer to real function-call and complex-instruction tasks; official description reports 1140 software-engineering-oriented tasks | Requires prompt, evaluator, and dependency adaptation | Second priority and suitable for a Phase 2 main extension |
| LiveCodeBench | Newer and less contaminated | Continuously updated; release_v6 has 1055 code-generation problems | Online-style format and execution framework are more complex | Suitable for temporal generalization evaluation |
| APPS | True large scale | 10,000 problems across easy to competition-level tasks | stdin/stdout, sandboxing, time limits, and failure attribution are substantially more complex | Do not run full scale immediately; start with a 500-1000 task sample |

## Recommended Route

### Step 1: HumanEval+ transfer smoke

The goal is not to increase scale; it is to confirm that the Phase 1 pipeline can move across data formats.

Recommended output directories:

```text
data/responses/phase2_humanevalplus_*
data/analysis/phase2_humanevalplus_*
```

Acceptance criteria:

- prompts do not leak hidden tests;
- verifier execution works;
- safe failure artifacts contain no private fields;
- attribution and clustering complete successfully;
- taxonomy results are comparable with MBPP.

### Step 2: BigCodeBench small-scale run

Start with 200-300 tasks at k=3.

Required additions:

- `prepare_bigcodebench_prompts.py`
- `verify_bigcodebench.py` or an adapter around the official evaluator output
- dataset adapter support in `build_failure_artifacts.py`
- the same safe/private diagnostics separation used by Phase 1

Acceptance criteria:

- at least 500 failed responses are available for attribution;
- verifier failure types can map to syntax/runtime/logic/timeout/interface categories;
- the largest cluster is no more than 25%;
- both unique task count and response count are reported.

### Step 3: APPS sample, not full scale

If the project goal shifts toward reward-model or DPO training, then sample APPS.

Recommended first sample:

```text
APPS introductory/interview subset, 500-1000 tasks, k=3
```

Do not start with all 10,000 tasks, because execution safety, runtime, and attribution cost all increase sharply.

## Current Decision

The reasonable next sequence is:

1. Freeze Phase 1 documentation and refined taxonomy.
2. Generate the rubric automatically from the refined taxonomy.
3. Run the LLM rubric judge on held-out validation/test responses.
4. Because the first full LLM judge baseline is valid but lenient, rerun the repaired/fallback judge cases when a GPU is actually idle and use that as the stabilized Phase 2 baseline.
5. After the rubric baseline is stabilized, run the minimal HumanEval+ cross-benchmark transfer.
6. Design a BigCodeBench adapter and run 200-300 tasks first.
7. After the BigCodeBench smoke is stable, decide whether to run an APPS sample.

Do not convert Phase 1 into a large-dataset version now. Phase 1 should remain the stable baseline, and dataset expansion should proceed only after the rubric baseline is validated.
