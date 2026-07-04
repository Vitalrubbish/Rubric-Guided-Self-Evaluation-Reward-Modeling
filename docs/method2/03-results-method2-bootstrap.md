# Self-Play Error Discovery Bootstrap

## 定位

这份产物对齐作业中的 Method 2：`response A -> 找出 A 的错误 -> 生成改进版 B -> 用 (A < B) 训练`。

重要 caveat：当前 critique 来自 verifier 失败信息和 protected rule revision 的编辑类型，不是单独由 LLM critic 生成。因此它是可训练的 self-play proxy/bootstrap，后续可以替换为真正的模型自找错版本。

## 核心指标

| 指标 | 数值 |
| --- | ---: |
| 原始失败样本 | 551 |
| 生成的 A<B pairs | 178 |
| failed rows with edits | 305 |
| detection coverage on failures | 55.35% |
| repair precision given edit | 58.36% |
| repair recall over all failures | 32.30% |
| pass preservation rate | 100.00% |
| harmful edit rate on initially passed | 0.00% |

## Transition Matrix

| Transition | Count |
| --- | ---: |
| pass_to_pass | 577 |
| pass_to_fail | 0 |
| fail_to_pass | 178 |
| fail_to_fail | 373 |

## Successful Repairs By Split

| Split | Count |
| --- | ---: |
| humanevalplus/test | 31 |
| mbpp/test | 81 |
| mbpp/train | 54 |
| mbpp/validation | 12 |

## Edit Success

| Edit | Attempts | Success | Success Rate |
| --- | ---: | ---: | ---: |
| drop_trailing_prose | 1 | 0 | 0.00% |
| remove_print_examples | 9 | 1 | 11.11% |
| truncate_duplicate_function_body | 295 | 177 | 60.00% |

## 哪些错误更容易自发现

在当前 proxy 里，最容易被发现并修复的是重复函数体、代码块后多余执行样例、代码后夹杂说明文字等格式/语法类错误。这类错误不需要深层语义判断，外部 verifier 确认后可直接形成高置信度 preference pair。

仍需要外部信号或更强 critic 的主要是逻辑错误、复杂 runtime 错误和 timeout。protected revision 后剩余失败样本仍以 `logic_wrong_output` 为主，说明真正的 Method 2 下一步应让模型显式解释语义错误，再生成 B。

## 输出文件

- `data/self_play/self_play_pairs_from_protected_revision.jsonl`
- `data/self_play/self_play_error_discovery_metrics.json`
