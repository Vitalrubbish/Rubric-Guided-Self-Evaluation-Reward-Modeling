# 选题五：Rubric-Guided Self-Evaluation and Reward Modeling

## 核心问题

训练一个能够在完成任务的过程中自我发现错误模式，并自定义 rubric（评分标准）进行自我评估的模型，从而改进自身输出和训练方式。

核心思想是：模型不依赖人类提供的评分标准，而是通过自身的失败经验自动归纳出 rubric，并将其作为可学习的 reward signal，引导模型进行自我进化（self-evolving）。

## 建议实验设定

### 推荐评估基准

- MT-Bench：多轮、多维度对话评估
- AlpacaEval 2.0：指令遵循评估
- GSM8K：数学推理与错误模式发现
- MATH：更高难度数学推理与迁移评估

## 作业 3：错误模式发现、Rubric 自动生成与基线评估

### Step 1：构建错误模式发现 Pipeline

- 模型在 GSM8K / MT-Bench 上生成大量 responses（500+）。
- 使用外部 verifier（如正确答案、规则验证器或人工/LLM judge）标注哪些 responses 失败。
- 让模型自身对失败样本进行 clustering 和归因分析，输出错误类型 taxonomy。
- 示例错误模式：
  - 计算进位错误
  - 多步推理丢失前提
  - 答非所问

### Step 2：Rubric 自动生成

- 基于发现的错误模式，让模型自动生成对应的评分 rubric。
- 每条 rubric 需要包含：
  - 维度
  - 1-5 分评分标准
  - 正例
  - 反例
- 对比自动生成 rubric 与人类编写 rubric 的覆盖率和区分度。

### Step 3：基于自生成 Rubric 的 Self-Evaluation

- 模型使用自己生成的 rubric 对新的 responses 逐条打分。
- 将模型自评结果与外部评判对比。
- 计算自评准确性，例如 Cohen's Kappa。

### 作业 3 评估指标

| 方法 | 错误模式覆盖率 | Rubric 区分度（AUC） | 自评与外部一致性 |
| --- | --- | --- | --- |
| 人类编写 rubric（upper bound） | 待评估 | 待评估 | 待评估 |
| 模型自动发现 rubric | 待评估 | 待评估 | 待评估 |
| 随机 rubric（ablation） | 待评估 | 待评估 | 待评估 |

## 作业 4：Self-Evolving 循环——从错误中自我进化

### 方法 1：Error-Pattern -> Rubric -> RL 闭环

迭代流程：

1. 生成 response。
2. 自动发现错误模式。
3. 生成或更新 rubric。
4. 使用 rubric 作为 reward 进行 DPO。
5. 重新生成 response。

每轮迭代后追踪：

- Rubric 是否进化：新增、删除或细化维度。
- 分数是否提升。
- 是否出现 reward hacking。

关键实验：

- 允许 rubric 自我更新。
- 固定首轮 rubric。
- 对比两者，分析 rubric 进化对性能的贡献。

### 方法 2：Self-Play Error Discovery

流程：

1. 模型生成 response A。
2. 模型尝试找出 A 中的错误。
3. 模型生成改进版 B。
4. 使用 `(A < B)` 构造 preference pair 进行训练。

对比标准 Self-Rewarding：

- 标准方法直接打分。
- 本方法要求模型先显式识别错误，再进行改进。

需要追踪：

- 模型找错能力是否随迭代提升。
- 错误检出率。
- 误报率。
- 哪类错误模型能自我发现。
- 哪类错误需要外部信号。

### 方法 3：Meta-Learning to Self-Evaluate

实验思路：

- 在多个不同任务上进行 self-evolving 实验，观察模型是否学会“如何生成好的 rubric”的 meta-skill。
- 先在 GSM8K 上完成 self-evolving，再迁移到 MATH / 代码生成等新任务。

评估重点：

- Zero-shot rubric generation 质量。
- 模型能否在新任务上直接生成有效 rubric，而不需要先经历失败样本。

该方法验证 self-evolving 的泛化性：模型不仅学会“做对某个任务”，还学会“如何自我改进”。
