Finetune_CODET5+

This project fine-tunes a single Code LLM across four software engineering tasks using unified preprocessing and multi-task instruction tuning:

Code Search

Code Repair

Clone Detection

Test Generation

The goal is to start from a pretrained CodeT5+ model and then jointly fine-tune it on all tasks, comparing the fine-tuned model with the pretrained baseline.

Model

Base model (pre-finetuning): Salesforce/codet5p-770m

Fine-tuned model: TBD

All results below are from the pretrained model only (no task-specific fine-tuning).

Data & Tasks

Expected directory structure:

data/
  codesearch/
    train.jsonl
    val.jsonl
    test.jsonl
  clone/
    train.jsonl
    val.jsonl
    test.jsonl
  repair/
    train.jsonl
    val.jsonl
    test.jsonl
  test_gen/
    train.jsonl
    val.jsonl
    test.jsonl


Unified JSON format:

{
  "task": "<task_name>",
  "input": "<text prompt>",
  "output": "<target text>"
}


Conversion logic is implemented in:

src/instruction_tuning_for_us_eval_v2.py

Task Descriptions

Code Search

Input: natural-language query + 3 code candidates

Output: index (0, 1, or 2)

Metrics: Accuracy, MRR

Clone Detection

Input: two code snippets

Output: 0 (not clone) or 1 (clone)

Metrics: Precision, Recall, F1

Code Repair

Input: buggy code

Output: fixed code

Metrics: Pass@K, number of plausible patches

Test Generation

Input: code under test

Output: generated unit tests

Metrics: Pass@K, number of plausible patches

Multi-Task Instruction Tuning Script

Main script:

src/instruction_tuning_for_us_eval_v2.py


Key configuration:

TEST_ONLY = True
TEST_MODEL_SOURCE = "pretrained"  # or "finetuned"
MODEL_NAME = "Salesforce/codet5p-770m"
OUTPUT_DIR = "./fine_tuned_multitask_model"
DATA_ROOT = "<project_root>/data"


Evaluation task list:

EVAL_TASKS = [
  "code_search",
  "clone_detection",
  "code_repair",
  "test_generation",
]

How to Run
1. Evaluate pretrained model (baseline)

Set:

TEST_ONLY = True
TEST_MODEL_SOURCE = "pretrained"
MODEL_NAME = "Salesforce/codet5p-770m"


Run:

cd src
python instruction_tuning_for_us_eval_v2.py


Outputs:

./output/inst_tuning_results_detailed.json
./output/inst_tuning_results_summary.txt
results_table.tex
radar_plot.png

2. Fine-tune the model

Set:

TEST_ONLY = False
MODEL_NAME = "Salesforce/codet5p-770m"
OUTPUT_DIR = "./fine_tuned_multitask_model"
NUM_EPOCHS = 1
TRAIN_BATCH_SIZE = 4
LEARNING_RATE = 2e-5


Run:

cd src
python instruction_tuning_for_us_eval_v2.py


To evaluate the fine-tuned model:

TEST_ONLY = True
TEST_MODEL_SOURCE = "finetuned"
OUTPUT_DIR = "./fine_tuned_multitask_model"

Baseline Results (Pre-Finetuning, CodeT5p-770M)

Baseline evaluation on 800 samples (200 per task):

============================================================
TESTING MULTI-TASK MODEL
============================================================
Testing with 800 samples
Test samples by task:
  code_search: 200
  clone_detection: 200
  code_repair: 200
  test_generation: 200
============================================================

Evaluation Summary
Overall Success Rate: 11.8%
Total samples tested: 800
Tasks evaluated: 4/4

Per-task Performance

Code Search

Accuracy: 0.325

MRR: 0.325

Clone Detection

Precision: 0.250

Recall: 0.071

F1: 0.111

Code Repair

Pass@1: 0.010

Pass@5: 0.049

Pass@10: 0.098

Plausible patches: 2

Test Generation

Pass@1: 0.025

Pass@5: 0.120

Pass@10: 0.228

Plausible patches: 5

Detailed outputs:

./output/inst_tuning_results_detailed.json
./output/inst_tuning_results_summary.txt

Artifacts Generated

inst_tuning_results_detailed.json — per-sample prediction logs

inst_tuning_results_summary.txt — readable summary

results_table.tex — LaTeX table for reports

radar_plot.png — radar visualization of task metrics

TODO

Fine-tune Salesforce/codet5p-770m on all four tasks

Re-run evaluation using the fine-tuned model

Update README with new results and checkpoint path
