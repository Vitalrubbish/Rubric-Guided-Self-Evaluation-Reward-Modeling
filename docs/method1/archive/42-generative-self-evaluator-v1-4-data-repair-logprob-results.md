# Generative Self-Evaluator Logprob Gate
## Summary
- Rows scored: `523`
- Adapter: `outputs/apps_simple_method1_generative_self_evaluator_sft_lora_v1_4_data_repair`
- Verdict prefix: `Verdict:`
- Pass completion: ` PASS
`
- Fail completion: ` FAIL
`
- Validation selected threshold: `1.251953`
- Test selected balanced accuracy: `0.6742`
- Test selected overacceptance: `0.2420`
- Test selected false rejection: `0.4095`
- Test AUC: `0.7839`
- Canary passed: `False`
## Gates
- [ ] test_balanced_accuracy_ge_min
- [x] test_overacceptance_le_max
- [x] validation_safe_threshold_exists
## Oracle Diagnostic
- Test oracle safe threshold balanced accuracy: `0.6853`
- Test oracle safe overacceptance: `0.2484`
## Full Summary
```json
{
  "rows_scored": 523,
  "model": "models/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28",
  "adapter": "outputs/apps_simple_method1_generative_self_evaluator_sft_lora_v1_4_data_repair",
  "input": "data/sft/apps_simple_method1_generative_self_evaluator_v1_4_data_repair.jsonl",
  "scores_file": "data/evaluator/apps_simple_method1_generative_self_eval_v1_4_data_repair_logprob_scores.jsonl",
  "splits": [
    "validation",
    "test"
  ],
  "prompt_format": "raw",
  "verdict_prefix": "Verdict:",
  "pass_completion": " PASS\n",
  "fail_completion": " FAIL\n",
  "split_counts": {
    "test": 262,
    "validation": 261
  },
  "gold_counts": {
    "fail": 302,
    "pass": 221
  },
  "token_counts": {
    "pass": {
      "2": 523
    },
    "fail": {
      "2": 523
    }
  },
  "validation_probability_metrics": {
    "auc": 0.765576694411415,
    "brier": 0.21744427321634527,
    "log_loss": 0.6703882847384819
  },
  "test_probability_metrics": {
    "auc": 0.7838641188959661,
    "brier": 0.216497645649127,
    "log_loss": 0.6757191902977575
  },
  "validation_zero_threshold": {
    "accuracy": 0.6896551724137931,
    "balanced_accuracy": 0.7094827586206897,
    "kappa": 0.4,
    "auc_binary_predictions": 0.7094827586206897,
    "predicted_pass_rate": 0.6551724137931034,
    "true_pass_rate": 0.4444444444444444,
    "overacceptance_rate": 0.4689655172413793,
    "false_rejection_rate": 0.11206896551724138,
    "precision_pass": 0.6023391812865497,
    "recall_pass": 0.8879310344827587,
    "specificity_fail": 0.5310344827586206,
    "confusion": {
      "tn": 77,
      "fp": 68,
      "fn": 13,
      "tp": 103
    },
    "threshold": 0.0
  },
  "test_zero_threshold": {
    "accuracy": 0.6908396946564885,
    "balanced_accuracy": 0.7246891113133151,
    "kappa": 0.41112159387313385,
    "auc_binary_predictions": 0.7246891113133151,
    "predicted_pass_rate": 0.6259541984732825,
    "true_pass_rate": 0.40076335877862596,
    "overacceptance_rate": 0.445859872611465,
    "false_rejection_rate": 0.10476190476190476,
    "precision_pass": 0.573170731707317,
    "recall_pass": 0.8952380952380953,
    "specificity_fail": 0.554140127388535,
    "confusion": {
      "tn": 87,
      "fp": 70,
      "fn": 11,
      "tp": 94
    },
    "threshold": 0.0
  },
  "validation_threshold_sweep": {
    "best_balanced_accuracy": {
      "accuracy": 0.7126436781609196,
      "balanced_accuracy": 0.721551724137931,
      "kappa": 0.4322960470984021,
      "auc_binary_predictions": 0.721551724137931,
      "predicted_pass_rate": 0.5555555555555556,
      "true_pass_rate": 0.4444444444444444,
      "overacceptance_rate": 0.3586206896551724,
      "false_rejection_rate": 0.19827586206896552,
      "precision_pass": 0.6413793103448275,
      "recall_pass": 0.8017241379310345,
      "specificity_fail": 0.6413793103448275,
      "confusion": {
        "tn": 93,
        "fp": 52,
        "fn": 23,
        "tp": 93
      },
      "threshold": 0.5615234375
    },
    "best_accuracy": {
      "accuracy": 0.7126436781609196,
      "balanced_accuracy": 0.721551724137931,
      "kappa": 0.4322960470984021,
      "auc_binary_predictions": 0.721551724137931,
      "predicted_pass_rate": 0.5555555555555556,
      "true_pass_rate": 0.4444444444444444,
      "overacceptance_rate": 0.3586206896551724,
      "false_rejection_rate": 0.19827586206896552,
      "precision_pass": 0.6413793103448275,
      "recall_pass": 0.8017241379310345,
      "specificity_fail": 0.6413793103448275,
      "confusion": {
        "tn": 93,
        "fp": 52,
        "fn": 23,
        "tp": 93
      },
      "threshold": 0.5615234375
    },
    "best_with_overacceptance_le_max": {
      "accuracy": 0.6781609195402298,
      "balanced_accuracy": 0.6681034482758621,
      "kappa": 0.34031413612565453,
      "auc_binary_predictions": 0.668103448275862,
      "predicted_pass_rate": 0.39080459770114945,
      "true_pass_rate": 0.4444444444444444,
      "overacceptance_rate": 0.2413793103448276,
      "false_rejection_rate": 0.4224137931034483,
      "precision_pass": 0.6568627450980392,
      "recall_pass": 0.5775862068965517,
      "specificity_fail": 0.7586206896551724,
      "confusion": {
        "tn": 110,
        "fp": 35,
        "fn": 49,
        "tp": 67
      },
      "threshold": 1.251953125
    },
    "selected_thresholds": [
      {
        "accuracy": 0.6015325670498084,
        "balanced_accuracy": 0.6396551724137931,
        "kappa": 0.2571428571428571,
        "auc_binary_predictions": 0.6396551724137931,
        "predicted_pass_rate": 0.8275862068965517,
        "true_pass_rate": 0.4444444444444444,
        "overacceptance_rate": 0.7034482758620689,
        "false_rejection_rate": 0.017241379310344827,
        "precision_pass": 0.5277777777777778,
        "recall_pass": 0.9827586206896551,
        "specificity_fail": 0.296551724137931,
        "confusion": {
          "tn": 43,
          "fp": 102,
          "fn": 2,
          "tp": 114
        },
        "threshold": -2.0
      },
      {
        "accuracy": 0.6360153256704981,
        "balanced_accuracy": 0.6689655172413793,
        "kappa": 0.3143544506816358,
        "auc_binary_predictions": 0.6689655172413793,
        "predicted_pass_rate": 0.7777777777777778,
        "true_pass_rate": 0.4444444444444444,
        "overacceptance_rate": 0.6275862068965518,
        "false_rejection_rate": 0.034482758620689655,
        "precision_pass": 0.5517241379310345,
        "recall_pass": 0.9655172413793104,
        "specificity_fail": 0.3724137931034483,
        "confusion": {
          "tn": 54,
          "fp": 91,
          "fn": 4,
          "tp": 112
        },
        "threshold": -1.0
      },
      {
        "accuracy": 0.6551724137931034,
        "balanced_accuracy": 0.6827586206896552,
        "kappa": 0.34359805510534847,
        "auc_binary_predictions": 0.6827586206896552,
        "predicted_pass_rate": 0.7279693486590039,
        "true_pass_rate": 0.4444444444444444,
        "overacceptance_rate": 0.5655172413793104,
        "false_rejection_rate": 0.06896551724137931,
        "precision_pass": 0.5684210526315789,
        "recall_pass": 0.9310344827586207,
        "specificity_fail": 0.43448275862068964,
        "confusion": {
          "tn": 63,
          "fp": 82,
          "fn": 8,
          "tp": 108
        },
        "threshold": -0.5
      },
      {
        "accuracy": 0.6896551724137931,
        "balanced_accuracy": 0.7094827586206897,
        "kappa": 0.4,
        "auc_binary_predictions": 0.7094827586206897,
        "predicted_pass_rate": 0.6551724137931034,
        "true_pass_rate": 0.4444444444444444,
        "overacceptance_rate": 0.4689655172413793,
        "false_rejection_rate": 0.11206896551724138,
        "precision_pass": 0.6023391812865497,
        "recall_pass": 0.8879310344827587,
        "specificity_fail": 0.5310344827586206,
        "confusion": {
          "tn": 77,
          "fp": 68,
          "fn": 13,
          "tp": 103
        },
        "threshold": 0.0
      },
      {
        "accuracy": 0.7049808429118773,
        "balanced_accuracy": 0.7155172413793103,
        "kappa": 0.4191114836546521,
        "auc_binary_predictions": 0.7155172413793103,
        "predicted_pass_rate": 0.5708812260536399,
        "true_pass_rate": 0.4444444444444444,
        "overacceptance_rate": 0.3793103448275862,
        "false_rejection_rate": 0.1896551724137931,
        "precision_pass": 0.6308724832214765,
        "recall_pass": 0.8103448275862069,
        "specificity_fail": 0.6206896551724138,
        "confusion": {
          "tn": 90,
          "fp": 55,
          "fn": 22,
          "tp": 94
        },
        "threshold": 0.5
      },
      {
        "accuracy": 0.7011494252873564,
        "balanced_accuracy": 0.7,
        "kappa": 0.3979416809605488,
        "auc_binary_predictions": 0.7,
        "predicted_pass_rate": 0.4674329501915709,
        "true_pass_rate": 0.4444444444444444,
        "overacceptance_rate": 0.2896551724137931,
        "false_rejection_rate": 0.3103448275862069,
        "precision_pass": 0.6557377049180327,
        "recall_pass": 0.6896551724137931,
        "specificity_fail": 0.7103448275862069,
        "confusion": {
          "tn": 103,
          "fp": 42,
          "fn": 36,
          "tp": 80
        },
        "threshold": 1.0
      },
      {
        "accuracy": 0.6475095785440613,
        "balanced_accuracy": 0.6155172413793104,
        "kappa": 0.24452554744525545,
        "auc_binary_predictions": 0.6155172413793103,
        "predicted_pass_rate": 0.19923371647509577,
        "true_pass_rate": 0.4444444444444444,
        "overacceptance_rate": 0.09655172413793103,
        "false_rejection_rate": 0.6724137931034483,
        "precision_pass": 0.7307692307692307,
        "recall_pass": 0.3275862068965517,
        "specificity_fail": 0.903448275862069,
        "confusion": {
          "tn": 131,
          "fp": 14,
          "fn": 78,
          "tp": 38
        },
        "threshold": 2.0
      }
    ]
  },
  "selected_threshold_policy": "validation best balanced accuracy among thresholds satisfying overacceptance <= max; fallback to validation best balanced accuracy",
  "selected_threshold": 1.251953125,
  "validation_selected_threshold": {
    "accuracy": 0.6781609195402298,
    "balanced_accuracy": 0.6681034482758621,
    "kappa": 0.34031413612565453,
    "auc_binary_predictions": 0.668103448275862,
    "predicted_pass_rate": 0.39080459770114945,
    "true_pass_rate": 0.4444444444444444,
    "overacceptance_rate": 0.2413793103448276,
    "false_rejection_rate": 0.4224137931034483,
    "precision_pass": 0.6568627450980392,
    "recall_pass": 0.5775862068965517,
    "specificity_fail": 0.7586206896551724,
    "confusion": {
      "tn": 110,
      "fp": 35,
      "fn": 49,
      "tp": 67
    },
    "threshold": 1.251953125
  },
  "test_selected_threshold": {
    "accuracy": 0.6908396946564885,
    "balanced_accuracy": 0.6742189869578405,
    "kappa": 0.35120758177927247,
    "auc_binary_predictions": 0.6742189869578405,
    "predicted_pass_rate": 0.3816793893129771,
    "true_pass_rate": 0.40076335877862596,
    "overacceptance_rate": 0.24203821656050956,
    "false_rejection_rate": 0.4095238095238095,
    "precision_pass": 0.62,
    "recall_pass": 0.5904761904761905,
    "specificity_fail": 0.7579617834394905,
    "confusion": {
      "tn": 119,
      "fp": 38,
      "fn": 43,
      "tp": 62
    },
    "threshold": 1.251953125
  },
  "test_oracle_threshold_sweep": {
    "best_balanced_accuracy": {
      "accuracy": 0.7404580152671756,
      "balanced_accuracy": 0.748741279951471,
      "kappa": 0.4793383599275235,
      "auc_binary_predictions": 0.748741279951471,
      "predicted_pass_rate": 0.49236641221374045,
      "true_pass_rate": 0.40076335877862596,
      "overacceptance_rate": 0.2929936305732484,
      "false_rejection_rate": 0.20952380952380953,
      "precision_pass": 0.6434108527131783,
      "recall_pass": 0.7904761904761904,
      "specificity_fail": 0.7070063694267515,
      "confusion": {
        "tn": 111,
        "fp": 46,
        "fn": 22,
        "tp": 83
      },
      "threshold": 0.7529296875
    },
    "best_accuracy": {
      "accuracy": 0.7404580152671756,
      "balanced_accuracy": 0.748741279951471,
      "kappa": 0.4793383599275235,
      "auc_binary_predictions": 0.748741279951471,
      "predicted_pass_rate": 0.49236641221374045,
      "true_pass_rate": 0.40076335877862596,
      "overacceptance_rate": 0.2929936305732484,
      "false_rejection_rate": 0.20952380952380953,
      "precision_pass": 0.6434108527131783,
      "recall_pass": 0.7904761904761904,
      "specificity_fail": 0.7070063694267515,
      "confusion": {
        "tn": 111,
        "fp": 46,
        "fn": 22,
        "tp": 83
      },
      "threshold": 0.7529296875
    },
    "best_with_overacceptance_le_max": {
      "accuracy": 0.6984732824427481,
      "balanced_accuracy": 0.6853199878677586,
      "kappa": 0.371225469348077,
      "auc_binary_predictions": 0.6853199878677586,
      "predicted_pass_rate": 0.3969465648854962,
      "true_pass_rate": 0.40076335877862596,
      "overacceptance_rate": 0.2484076433121019,
      "false_rejection_rate": 0.38095238095238093,
      "precision_pass": 0.625,
      "recall_pass": 0.6190476190476191,
      "specificity_fail": 0.7515923566878981,
      "confusion": {
        "tn": 118,
        "fp": 39,
        "fn": 40,
        "tp": 65
      },
      "threshold": 1.12890625
    },
    "selected_thresholds": [
      {
        "accuracy": 0.6221374045801527,
        "balanced_accuracy": 0.6799818016378526,
        "kappa": 0.3139184256467227,
        "auc_binary_predictions": 0.6799818016378526,
        "predicted_pass_rate": 0.7557251908396947,
        "true_pass_rate": 0.40076335877862596,
        "overacceptance_rate": 0.6114649681528662,
        "false_rejection_rate": 0.02857142857142857,
        "precision_pass": 0.5151515151515151,
        "recall_pass": 0.9714285714285714,
        "specificity_fail": 0.3885350318471338,
        "confusion": {
          "tn": 61,
          "fp": 96,
          "fn": 3,
          "tp": 102
        },
        "threshold": -2.0
      },
      {
        "accuracy": 0.6412213740458015,
        "balanced_accuracy": 0.6927509857446164,
        "kappa": 0.3404038780866677,
        "auc_binary_predictions": 0.6927509857446164,
        "predicted_pass_rate": 0.7213740458015268,
        "true_pass_rate": 0.40076335877862596,
        "overacceptance_rate": 0.5668789808917197,
        "false_rejection_rate": 0.047619047619047616,
        "precision_pass": 0.5291005291005291,
        "recall_pass": 0.9523809523809523,
        "specificity_fail": 0.43312101910828027,
        "confusion": {
          "tn": 68,
          "fp": 89,
          "fn": 5,
          "tp": 100
        },
        "threshold": -1.0
      },
      {
        "accuracy": 0.6755725190839694,
        "balanced_accuracy": 0.7182590233545647,
        "kappa": 0.39255905297037796,
        "auc_binary_predictions": 0.7182590233545647,
        "predicted_pass_rate": 0.6717557251908397,
        "true_pass_rate": 0.40076335877862596,
        "overacceptance_rate": 0.4968152866242038,
        "false_rejection_rate": 0.06666666666666667,
        "precision_pass": 0.5568181818181818,
        "recall_pass": 0.9333333333333333,
        "specificity_fail": 0.5031847133757962,
        "confusion": {
          "tn": 79,
          "fp": 78,
          "fn": 7,
          "tp": 98
        },
        "threshold": -0.5
      },
      {
        "accuracy": 0.6908396946564885,
        "balanced_accuracy": 0.7246891113133151,
        "kappa": 0.41112159387313385,
        "auc_binary_predictions": 0.7246891113133151,
        "predicted_pass_rate": 0.6259541984732825,
        "true_pass_rate": 0.40076335877862596,
        "overacceptance_rate": 0.445859872611465,
        "false_rejection_rate": 0.10476190476190476,
        "precision_pass": 0.573170731707317,
        "recall_pass": 0.8952380952380953,
        "specificity_fail": 0.554140127388535,
        "confusion": {
          "tn": 87,
          "fp": 70,
          "fn": 11,
          "tp": 94
        },
        "threshold": 0.0
      },
      {
        "accuracy": 0.7175572519083969,
        "balanced_accuracy": 0.740673339399454,
        "kappa": 0.4501106131941687,
        "auc_binary_predictions": 0.740673339399454,
        "predicted_pass_rate": 0.5687022900763359,
        "true_pass_rate": 0.40076335877862596,
        "overacceptance_rate": 0.37579617834394907,
        "false_rejection_rate": 0.14285714285714285,
        "precision_pass": 0.6040268456375839,
        "recall_pass": 0.8571428571428571,
        "specificity_fail": 0.6242038216560509,
        "confusion": {
          "tn": 98,
          "fp": 59,
          "fn": 15,
          "tp": 90
        },
        "threshold": 0.5
      },
      {
        "accuracy": 0.7175572519083969,
        "balanced_accuracy": 0.7122838944494996,
        "kappa": 0.4192775414844545,
        "auc_binary_predictions": 0.7122838944494996,
        "predicted_pass_rate": 0.4312977099236641,
        "true_pass_rate": 0.40076335877862596,
        "overacceptance_rate": 0.2611464968152866,
        "false_rejection_rate": 0.3142857142857143,
        "precision_pass": 0.6371681415929203,
        "recall_pass": 0.6857142857142857,
        "specificity_fail": 0.7388535031847133,
        "confusion": {
          "tn": 116,
          "fp": 41,
          "fn": 33,
          "tp": 72
        },
        "threshold": 1.0
      },
      {
        "accuracy": 0.6793893129770993,
        "balanced_accuracy": 0.6299666363360631,
        "kappa": 0.2802668585257374,
        "auc_binary_predictions": 0.6299666363360631,
        "predicted_pass_rate": 0.22519083969465647,
        "true_pass_rate": 0.40076335877862596,
        "overacceptance_rate": 0.12101910828025478,
        "false_rejection_rate": 0.6190476190476191,
        "precision_pass": 0.6779661016949152,
        "recall_pass": 0.38095238095238093,
        "specificity_fail": 0.8789808917197452,
        "confusion": {
          "tn": 138,
          "fp": 19,
          "fn": 65,
          "tp": 40
        },
        "threshold": 2.0
      }
    ]
  },
  "gates": {
    "test_balanced_accuracy_ge_min": false,
    "test_overacceptance_le_max": true,
    "validation_safe_threshold_exists": true
  },
  "canary_passed": false,
  "policy": "forced-choice verdict logprob scoring; threshold selected only on validation"
}
```