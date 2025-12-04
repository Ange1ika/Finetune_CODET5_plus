Finetune_CODET5+<br>

This project fine-tunes a single Code LLM across four software engineering tasks using unified preprocessing and multi-task instruction tuning:<br>

Code Search<br>

Code Repair<br>

Clone Detection<br>

Test Generation<br>

The goal is to start from a pretrained CodeT5+ model and then jointly fine-tune it on all tasks, comparing the fine-tuned model with the pretrained baseline.<br>

Model<br>

Base model (pre-finetuning): Salesforce/codet5p-770m<br>

Fine-tuned model: TBD<br>

All results below are from the pretrained model only (no fine-tuning).<br>

Data & Tasks<br>

Expected directory structure:<br>

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
```<br>

Unified JSON format:<br>



{
"task": "<task_name>",
"input": "<text prompt>",
"output": "<target text>"
}


Conversion logic is implemented in:<br>
`src/instruction_tuning_for_us_eval_v2.py`<br>

### Task Descriptions<br>

**Code Search**<br>
Input: natural-language query + 3 code candidates<br>
Output: index (0, 1, or 2)<br>
Metrics: Accuracy, MRR<br>

**Clone Detection**<br>
Input: two code snippets<br>
Output: 0 or 1<br>
Metrics: Precision, Recall, F1<br>

**Code Repair**<br>
Input: buggy code<br>
Output: fixed code<br>
Metrics: Pass@K, plausible patches<br>

**Test Generation**<br>
Input: code under test<br>
Output: generated unit tests<br>
Metrics: Pass@K, plausible patches<br>

---

## Multi-Task Instruction Tuning Script<br>

Main script:<br>
`src/instruction_tuning_for_us_eval_v2.py`<br>

Key configuration:<br>



TEST_ONLY = True
TEST_MODEL_SOURCE = "pretrained"
MODEL_NAME = "Salesforce/codet5p-770m"
OUTPUT_DIR = "./fine_tuned_multitask_model"
DATA_ROOT = "<project_root>/data"
EVAL_TASKS = ["code_search", "clone_detection", "code_repair", "test_generation"]


---

## How to Run<br>

### 1. Evaluate pretrained model (baseline)<br>

Set:<br>



TEST_ONLY = True
TEST_MODEL_SOURCE = "pretrained"
MODEL_NAME = "Salesforce/codet5p-770m"


Run:<br>



cd src
python instruction_tuning_for_us_eval_v2.py


Outputs:<br>



./output/inst_tuning_results_detailed.json
./output/inst_tuning_results_summary.txt
results_table.tex
radar_plot.png


---

### 2. Fine-tune the model<br>

Set:<br>



TEST_ONLY = False
MODEL_NAME = "Salesforce/codet5p-770m"
OUTPUT_DIR = "./fine_tuned_multitask_model"
NUM_EPOCHS = 1
TRAIN_BATCH_SIZE = 4
LEARNING_RATE = 2e-5


Run:<br>



cd src
python instruction_tuning_for_us_eval_v2.py


To evaluate the fine-tuned model later:<br>



TEST_ONLY = True
TEST_MODEL_SOURCE = "finetuned"
OUTPUT_DIR = "./fine_tuned_multitask_model"


---

## Baseline Results (Pre-Finetuning, CodeT5p-770M)<br>

Baseline evaluation on **800 samples** (200 per task):<br>


============================================================
TESTING MULTI-TASK MODEL
Testing with 800 samples
Test samples by task:
code_search: 200
clone_detection: 200
code_repair: 200
test_generation: 200

### Evaluation Summary<br>



Overall Success Rate: 11.8%
Total samples tested: 800
Tasks evaluated: 4/4


### Per-task Performance<br>

**Code Search**<br>
Accuracy: 0.325<br>
MRR: 0.325<br>

**Clone Detection**<br>
Precision: 0.250<br>
Recall: 0.071<br>
F1: 0.111<br>

**Code Repair**<br>
Pass@1: 0.010<br>
Pass@5: 0.049<br>
Pass@10: 0.098<br>
Plausible patches: 2<br>

**Test Generation**<br>
Pass@1: 0.025<br>
Pass@5: 0.120<br>
Pass@10: 0.228<br>
Plausible patches: 5<br>

Detailed outputs:<br>



./output/inst_tuning_results_detailed.json
./output/inst_tuning_results_summary.txt


---

## Artifacts Generated<br>

- `inst_tuning_results_detailed.json` — per-sample prediction logs<br>
- `inst_tuning_results_summary.txt` — readable summary<br>
- `results_table.tex` — LaTeX table<br>
- `radar_plot.png` — radar visualization<br>

---

## TODO<br>

- Fine-tune `Salesforce/codet5p-770m` on all four tasks<br>
- Re-run evaluation using the fine-tuned model<br>
- Update README with new results and checkpoint path<br>
