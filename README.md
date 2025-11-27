# Finetune_CODET5+
Fine-tune a single CodeLLM across four tasks (Code Search, Code Repair, Clone Dectection, and Test Generation) using unified preprocessing and multi-task learning.


for eval use Finetune_CODET5_plus/src/instruction_tuning_for_us_eval.py (TEST_ONLY mode!)
eval output sample:
Per-task performance:
  code_search:
    Samples: 200
    Accuracy: 0.000
    MRR: 0.000
  clone_detection:
    Samples: 277
    Precision: 0.000
    Recall: 0.000
    F1: 0.000
  code_repair:
    Samples: 200
    Pass@1: 0.240
    Pass@5: 0.750
    Pass@10: 0.940
    Plausible patches: 48
  test_generation:
    Samples: 554
    Pass@1: 0.491
    Pass@5: 0.966
    Pass@10: 0.999
    Plausible patches: 272

Detailed results saved to:
  JSON: ./output/inst_tuning_results_detailed.json
  Text: ./output/inst_tuning_results_summary.txt
  
### TO DO:
check please 
code_search:
    Samples: 200
    Accuracy: 0.000
    MRR: 0.000
  clone_detection:
    Samples: 277
    Precision: 0.000
    Recall: 0.000
    F1: 0.000

  AND continue ***training*** 
