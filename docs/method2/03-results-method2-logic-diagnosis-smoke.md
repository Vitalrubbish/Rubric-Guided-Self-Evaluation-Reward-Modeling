# Logic Two-Stage Failure Diagnosis

## 输入

- Source labeled file: `data/self_play/llm_critic_mbpp_train_logic_n20_twostage_v1_labeled.jsonl`
- Diagnosed failures: 3

## 分类分布

| Diagnosis | Count |
| --- | ---: |
| `right_spec_wrong_algorithm` | 2 |
| `wrong_spec` | 1 |

## 置信度

| Confidence | Count |
| --- | ---: |
| `medium` | 3 |

## 样本明细

| ID | Diagnosis | Confidence | First failing assertion | Recommended fix |
| --- | --- | --- | --- | --- |
| `mbpp/train/603` | `right_spec_wrong_algorithm` | medium | assert get_ludic(10) == [1, 2, 3, 5, 7] actual=[] expected=[1, 2, 3, 5, 7] | Keep the two-stage split, but add an algorithm-sketch stage before writing code and ask the model to manually simulate the visible tests. |
| `mbpp/train/609` | `right_spec_wrong_algorithm` | medium | assert floor_Min(10,20,30) == 15 actual=20 expected=15 | Keep the two-stage split, but add an algorithm-sketch stage before writing code and ask the model to manually simulate the visible tests. |
| `mbpp/train/610` | `wrong_spec` | medium | assert remove_kth_element([1,1,2,3,4,4,5,1],3)==[1, 1, 3, 4, 4, 5, 1] actual=[1, 1, 2, 4, 4, 5, 1] expected=[1, 1, 3, 4, 4, 5, 1] | Strengthen Stage 1 to produce an explicit natural-language specification before code generation. |

## 代表样本

### `right_spec_wrong_algorithm`

- `mbpp/train/603` (medium): task: Write a function to get a lucid number smaller than or equal to n.; inferred_spec: Generate a list of ludic numbers up to and including the given limit n, where ludic numbers follow a specific filtering rule.; failed assertions: 3/3; first mismatch: assert get_ludic(10) == [1, 2, 3, 5, 7] actual=[] expected=[1, 2, 3, 5, 7]
  - Next: Keep the two-stage split, but add an algorithm-sketch stage before writing code and ask the model to manually simulate the visible tests.

- `mbpp/train/609` (medium): task: Write a python function to find minimum possible value for the given periodic function.; inferred_spec: Find the value that minimizes a periodic function over the range defined by the three input values, where the function's minimum may not necessarily be one of the inputs.; failed assertions: 3/3; first mismatch: assert floor_Min(10,20,30) == 15 actual=20 expected=15
  - Next: Keep the two-stage split, but add an algorithm-sketch stage before writing code and ask the model to manually simulate the visible tests.

### `wrong_spec`

- `mbpp/train/610` (medium): Stage 1 inferred_spec is missing or too short.; inferred_spec: 
  - Next: Strengthen Stage 1 to produce an explicit natural-language specification before code generation.

## Gate 决策

- Dominant diagnosis: `right_spec_wrong_algorithm` (2/3)
- 下一步：优先做 Stage 2 algorithm-sketch prompt，再写代码。
