"""
Multi-Task Instruction Tuning for CodeT5+

CONFIGURATION:
- Edit DATA_PATH to point to your dataset
- Adjust training parameters (epochs, batch size, learning rate)
- Set TEST_ONLY = True to test an existing model without training
- All paths and parameters can be customized below
"""

import json
import os
import random
import signal
from datetime import datetime
from torch.utils.data import Dataset
import re
from collections import Counter

import torch
import numpy as np
from transformers import (
    AutoTokenizer, 
    T5ForConditionalGeneration, 
    Trainer, 
    TrainingArguments,
    EarlyStoppingCallback,
)
from generate_radar_plot import generate_radar_plot
# Execution Mode
TEST_ONLY = False             # Set to True to only test existing model (no training)

# Evaluation Settings
# List of tasks to evaluate. Empty list = evaluate all tasks, or you can specify tasks like the following.
EVAL_TASKS = [
    "code_search",
    "clone_detection",
    "code_repair",
    "test_generation"
]


# ========================================
# INSTRUCTION PREFIXES FOR EACH TASK
# ========================================
TASK_PREFIXES = {
    # Code Search: query + candidate code snippets -> index (0/1/2)
    "code_search": "code search: choose the correct code snippet index for the given query.",

    # Clone Detection: (source code, target code) -> 0/1
    "clone_detection": "clone detection: decide whether the two code snippets are semantically equivalent. Answer 1 if they are clones, otherwise 0.",

    # Code Repair: buggy code -> fixed code
    "code_repair": "fix a bug: return ONLY the fixed code without any explanation.",

    # Test Generation: code under test -> test code
    "test_generation": "generate tests: write unit tests for the given code."
}

# ========================================
# CONFIGURATION - EDIT THESE VALUES
# ========================================

# Data and Model Settings
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(BASE_DIR, "..", "data")


OUTPUT_DIR = "./fine_tuned_multitask_model"    # Where to save the trained model
TEST_OUTPUT_DIR = "./output"                   # Where to save test results
MODEL_NAME = "Salesforce/codet5p-220m"         # Base model to fine-tune
RANDOM_SEED = 42                               # For reproducible results

# Training Settings
TRAIN_BATCH_SIZE = 4          # Number of samples per training batch (reduce if out of memory)
EVAL_BATCH_SIZE = 4            # Number of samples per validation batch
LEARNING_RATE = 2e-5           # How fast the model learns (lower = more stable)
NUM_EPOCHS = 20                # How many times to go through the entire dataset
MAX_INPUT_LENGTH = 512        # Maximum tokens for input text (code/descriptions)
MAX_TARGET_LENGTH = 256        # Maximum tokens for output text

# Validation and Early Stopping
EARLY_STOPPING_PATIENCE = 3   # Stop training if no improvement for this many evaluations
EVAL_STEPS = 300              # Evaluate model performance every N training steps
SAVE_STEPS = 300              # Save model checkpoint every N steps
VALIDATION_SIZE = 500         # Number of samples to use for validation (was 0, needs >0 for eval)

# Text Generation Settings (for inference/testing)
GENERATION_MAX_LENGTH = 512
GENERATION_MIN_LENGTH = 10
GENERATION_NUM_BEAMS = 4
GENERATION_LENGTH_PENALTY = 0.8
GENERATION_REPETITION_PENALTY = 1.1
GENERATION_NO_REPEAT_NGRAM_SIZE = 3

def parse_index_from_text(text, valid_indices=None):
    """
    Parse an integer index (e.g., 0/1/2) from a free-form model output.
    Example:
        "The answer is 1." -> 1
    If no valid index is found, returns None.
    """
    if valid_indices is None:
        valid_indices = {0, 1, 2}

    # Try to extract a standalone integer token
    m = re.search(r"\b([0-9]+)\b", text)
    if m:
        idx = int(m.group(1))
        if idx in valid_indices:
            return idx

    # Fallback: check if any valid index appears in the text
    for i in sorted(valid_indices):
        if str(i) in text:
            return i

    return None


def evaluate_code_search(results):
    """
    Evaluate Code Search with:
    - Primary metric: accuracy
    - Secondary metric: MRR (here effectively top-1, so similar to accuracy)
    The model output is free-form text; we parse an index (0/1/2) from it.
    """
    correct = 0
    reciprocal_ranks = []

    for r in results:
        pred_text = r["model_output"].strip()
        gold_text = r["expected_output"].strip()

        # Parse gold index
        try:
            gold_idx = int(gold_text)
        except ValueError:
            reciprocal_ranks.append(0.0)
            continue

        # Parse predicted index from model output
        pred_idx = parse_index_from_text(pred_text, valid_indices={0, 1, 2})
        if pred_idx is None:
            reciprocal_ranks.append(0.0)
            continue

        if pred_idx == gold_idx:
            correct += 1
            reciprocal_ranks.append(1.0)
        else:
            reciprocal_ranks.append(0.0)

    if len(results) == 0:
        return 0.0, 0.0

    accuracy = correct / len(results)
    mrr = sum(reciprocal_ranks) / len(results)

    return accuracy, mrr


def evaluate_clone_detection(results):
    TP = FP = FN = 0

    for r in results:
        pred = r["model_output"].strip()
        gold = r["expected_output"].strip()


        try:
            gold_label = int(gold)
        except ValueError:
            continue

        pred_label = parse_index_from_text(pred, valid_indices={0, 1})
        if pred_label is None:
            continue


        if pred_label == 1 and gold_label == 1: TP += 1
        if pred_label == 1 and gold_label == 0: FP += 1
        if pred_label == 0 and gold_label == 1: FN += 1

    precision = TP / (TP + FP) if (TP + FP) else 0
    recall = TP / (TP + FN) if (TP + FN) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    return precision, recall, f1

def read_jsonl(path):
    """
    Read a .jsonl file and yield one JSON object per line.

    Each line in the file is a valid JSON object.
    This function ignores empty lines.
    """
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)
            
            
def build_raw_examples_for_split(data_root: str, split: str):
    """
    Build unified raw examples for ALL tasks (code_search, clone_detection,
    code_repair, test_generation) for a given split (train/val/test).

    Each returned example has the unified schema:
        {
            "task":   <task_name>,           # e.g., "code_search"
            "input":  <string prompt>,       # full natural language input to the model
            "output": <string target>,       # expected output text
        }

    This function is the ONLY place that knows about the original JSON structure
    of codesearch/clone/repair/test_gen.
    """
    examples = []

    # -------------------------
    # 1) CODE SEARCH
    # -------------------------
    cs_dir = os.path.join(data_root, "codesearch")
    cs_path = os.path.join(cs_dir, f"{split}.jsonl")
    if os.path.exists(cs_path):
        for item in read_jsonl(cs_path):
            # Example structure (given by you):
            # {
            #   "task": "SEARCH",
            #   "input": "<natural language query>",
            #   "output": "<gold code snippet>",  # not strictly needed for index prediction
            #   "choices": [code0, code1, code2],
            #   "answer": 1,                      # correct index
            #   ...
            # }

            query = item["input"]
            choices = item["choices"]
            answer_idx = int(item["answer"])

            # Build a text prompt that includes query + candidate code snippets.
            # The model is asked to output the index (0, 1 or 2).
            choice_lines = []
            for i, code in enumerate(choices):
                choice_lines.append(f"[{i}] {code}")

            prompt = (
                f"{TASK_PREFIXES['code_search']}\n\n"
                f"Query:\n{query}\n\n"
                "Candidate code snippets:\n"
                + "\n\n".join(choice_lines)
                + "\n\nAnswer with the index (0, 1, or 2):"
            )

            examples.append(
                {
                    "task": "code_search",
                    "input": prompt,
                    "output": str(answer_idx),
                }
            )

    # -------------------------
    # 2) CLONE DETECTION
    # -------------------------
    clone_dir = os.path.join(data_root, "clone")
    clone_path = os.path.join(clone_dir, f"{split}.jsonl")
    if os.path.exists(clone_path):
        for item in read_jsonl(clone_path):
            # Example structure (given by you):
            # {
            #   "task": "CLONE",
            #   "source": "<source code>",
            #   "target": "<target code>",
            #   "label": 0 or 1,
            #   ...
            # }

            src_code = item["source"]
            tgt_code = item["target"]
            label = int(item["label"])  # 0 or 1

            prompt = (
                f"{TASK_PREFIXES['clone_detection']}\n\n"
                "SOURCE CODE:\n"
                f"{src_code}\n\n"
                "TARGET CODE:\n"
                f"{tgt_code}\n\n"
                "Answer with 1 (clone) or 0 (not clone)."
            )

            examples.append(
                {
                    "task": "clone_detection",
                    "input": prompt,
                    "output": str(label),
                }
            )

    # -------------------------
    # 3) CODE REPAIR
    # -------------------------
    repair_dir = os.path.join(data_root, "repair")
    repair_path = os.path.join(repair_dir, f"{split}.jsonl")
    # NOTE: right now you have "all.jsonl". You can create train/val/test splits,
    # or temporarily symlink/copy "all.jsonl" to "train.jsonl".
    if os.path.exists(repair_path):
        for item in read_jsonl(repair_path):
            # Example structure (given by you):
            # {
            #   "task": "repair",
            #   "input": "<buggy code>",
            #   "output": "<fixed code>",
            #   ...
            # }

            buggy_code = item["input"]
            fixed_code = item["output"]

            prompt = f"{TASK_PREFIXES['code_repair']} {buggy_code}"

            examples.append(
                {
                    "task": "code_repair",
                    "input": prompt,
                    "output": fixed_code,
                }
            )

    # -------------------------
    # 4) TEST GENERATION
    # -------------------------
    tg_dir = os.path.join(data_root, "test_gen")
    tg_path = os.path.join(tg_dir, f"{split}.jsonl")
    if os.path.exists(tg_path):
        for item in read_jsonl(tg_path):
            # Example structure (given by you):
            # {
            #   "task": "TEST_GENERATION",
            #   "source": "<code under test>",
            #   "target": "<test code>",
            #   ...
            # }

            code_under_test = item["source"]
            test_code = item["target"]

            prompt = (
                f"{TASK_PREFIXES['test_generation']}\n\n"
                "CODE UNDER TEST:\n"
                f"{code_under_test}\n\n"
                "Write unit tests:"
            )

            examples.append(
                {
                    "task": "test_generation",
                    "input": prompt,
                    "output": test_code,
                }
            )

    return examples

import math

def pass_at_k(n, c, k):
    if n < k or n == 0:
        return 0.0
    return 1 - (math.comb(n - c, k) / math.comb(n, k))

def evaluate_pass_at_k(results, k=1):
    total = len(results)
    correct = sum(r["correct"] for r in results)
    return pass_at_k(total, correct, k)





def wilson_confidence_interval(successes, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)

    p_hat = successes / n
    denominator = 1 + z**2 / n
    centre = p_hat + z**2 / (2 * n)
    margin = z * math.sqrt((p_hat*(1-p_hat) + z**2/(4*n)) / n)

    lower = (centre - margin) / denominator
    upper = (centre + margin) / denominator
    return round(lower, 3), round(upper, 3)
def load_all_test_data_from_folders(data_root):
    """
    Automatically loads test data from:
    - data/codesearch/test.jsonl
    - data/clone/test.jsonl
    - data/repair/test.jsonl
    - data/test_gen/test.jsonl
    """

    print("Loading test data from task folders...")

    all_samples = []

    # -------------------------
    # CODE SEARCH
    # -------------------------
    cs_path = os.path.join(data_root, "codesearch", "test.jsonl")
    if os.path.exists(cs_path):
        for item in read_jsonl(cs_path):
            all_samples.append({
                "task": "code_search",
                "input": (
                    f"{TASK_PREFIXES['code_search']}\n\n"
                    f"Query:\n{item['input']}\n\n"
                    "Candidate code snippets:\n"
                    + "\n\n".join([f"[{i}] {c}" for i, c in enumerate(item["choices"])])
                    + "\n\nAnswer with the index (0, 1, or 2):"
                ),
                "expected_output": str(item["answer"])
            })

    # -------------------------
    # CLONE DETECTION
    # -------------------------
    clone_path = os.path.join(data_root, "clone", "test.jsonl")
    if os.path.exists(clone_path):
        for item in read_jsonl(clone_path):
            all_samples.append({
                "task": "clone_detection",
                "input": (
                    f"{TASK_PREFIXES['clone_detection']}\n\n"
                    f"SOURCE CODE:\n{item['source']}\n\n"
                    f"TARGET CODE:\n{item['target']}\n\n"
                    "Answer with 1 (clone) or 0 (not clone)."
                ),
                "expected_output": str(item["label"])
            })

    # -------------------------
    # CODE REPAIR
    # -------------------------
    repair_path = os.path.join(data_root, "repair", "test.jsonl")
    if os.path.exists(repair_path):
        for item in read_jsonl(repair_path):
            all_samples.append({
                "task": "code_repair",
                "input": f"{TASK_PREFIXES['code_repair']} {item['input']}",
                "expected_output": item["output"]
            })

    # -------------------------
    # TEST GENERATION
    # -------------------------
    testgen_path = os.path.join(data_root, "test_gen", "test.jsonl")
    if os.path.exists(testgen_path):
        for item in read_jsonl(testgen_path):
            all_samples.append({
                "task": "test_generation",
                "input": (
                    f"{TASK_PREFIXES['test_generation']}\n\n"
                    f"CODE UNDER TEST:\n{item['source']}\n\n"
                    "Write unit tests:"
                ),
                "expected_output": item["target"]
            })

    print(f"Loaded {len(all_samples)} total test samples from all tasks.")
    return all_samples



class MultiTaskDataset(Dataset):
    """
    Dataset class for multi-task instruction tuning.
    
    Creates training examples for all 5 tasks from each data item:
    - code_search
    - clone_detection
    - code_repair
    - test_generation
    
    Each element of `data` is a dict:
        {"task": <task_name>, "input": <input_text>, "output": <target_text>}
    """
    
    
    
    def __init__(self, data, tokenizer, max_input_length=512, max_target_length=512):
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        
        self.examples = data
        
        print(f"Created {len(self.examples)} training examples from {len(data)} data items")
        
    
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Tokenize input
        input_encoding = self.tokenizer(
            example['input'],
            max_length=self.max_input_length,
            padding=False,  # Let collate_fn handle padding
            truncation=True,
            return_tensors=None
        )
        
        # Tokenize target
        target_encoding = self.tokenizer(
            example['output'],
            max_length=self.max_target_length,
            padding=False,  # Let collate_fn handle padding
            truncation=True,
            return_tensors=None
        )
        
        return {
            "task": example["task"], 
            'input_ids': input_encoding['input_ids'],
            'attention_mask': input_encoding['attention_mask'],
            'labels': target_encoding['input_ids']
        }

def collate_fn(batch):
    """Custom collate function to handle variable length sequences
    
    This function:
    - Pads input_ids and attention_mask with 0
    - Pads labels with -100 (so that they are ignored by the loss)
    - Optionally keeps the list of task names for analysis
    """
    input_ids = [item['input_ids'] for item in batch]
    attention_masks = [item['attention_mask'] for item in batch]
    labels = [item['labels'] for item in batch]
    
    # Convert to tensors and pad
    input_ids = [torch.tensor(ids) if not isinstance(ids, torch.Tensor) else ids for ids in input_ids]
    attention_masks = [torch.tensor(mask) if not isinstance(mask, torch.Tensor) else mask for mask in attention_masks]
    labels = [torch.tensor(lbls) if not isinstance(lbls, torch.Tensor) else lbls for lbls in labels]
    
    # Pad sequences to the same length within the batch
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_masks = torch.nn.utils.rnn.pad_sequence(attention_masks, batch_first=True, padding_value=0)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_masks,
        'labels': labels,       
    }

def prepare_datasets(tokenizer, validation_size=VALIDATION_SIZE, seed=42):   
    """
    Build raw multi-task examples for 'train' and 'val' splits,
    then wrap them into MultiTaskDataset objects.

    NOTE:
    - We rely on build_raw_examples_for_split() to mix tasks.
    - Here we can still subsample for validation if needed.
    """
    
    # Build raw examples per split
    train_raw = build_raw_examples_for_split(DATA_ROOT, "train")
    val_raw = build_raw_examples_for_split(DATA_ROOT, "val")    
    
    random.seed(seed)
    random.shuffle(train_raw)
    random.shuffle(val_raw)
    
    # Optionally cut validation size if you want smaller val set
    if 0 < validation_size < len(val_raw):
        val_raw = val_raw[:validation_size]
        
    # Create dataset
    train_dataset = MultiTaskDataset(train_raw, tokenizer, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH)
    eval_dataset = MultiTaskDataset(val_raw, tokenizer, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH)

    print(f"Dataset split: {len(train_dataset)} training, {len(eval_dataset)} validation samples")
   
    return train_dataset, eval_dataset

def setup_model_and_tokenizer(model_name="Salesforce/codet5p-220m"):
    """Load and configure the model and tokenizer"""
    print(f"Loading model and tokenizer: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name, trust_remote_code=True)
    
    # Add special tokens that might be useful for code tasks
    special_tokens = {
        "additional_special_tokens": [
            "<code>", "</code>", 
            "<bug>", "</bug>",
            "<fix>", "</fix>",
            "<summary>", "</summary>"
        ]
    }
    
    num_added = tokenizer.add_special_tokens(special_tokens)
    if num_added > 0:
        model.resize_token_embeddings(len(tokenizer))
        print(f"Added {num_added} special tokens to vocabulary")
    
    # FOR OPT
    model.config.use_cache = False
    
    return model, tokenizer

def train_model(
    model, 
    tokenizer, 
    train_dataset, 
    eval_dataset, 
    device, 
    output_dir="./fine_tuned_multitask_model",
    num_epochs=NUM_EPOCHS,
    batch_size=TRAIN_BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    eval_steps=EVAL_STEPS,
    save_steps=SAVE_STEPS,
    early_stopping_patience=EARLY_STOPPING_PATIENCE
):
    """Train the multi-task model using Hugging Face Trainer"""

    # Updated args FOR OPT
    cuda_available = torch.cuda.is_available()
    num_workers = 2 if cuda_available else 0
    print(f"Data loading optimization: num_workers={num_workers}, pin_memory={cuda_available}")
    
    # Check if we have validation data
    has_eval = eval_dataset is not None and len(eval_dataset) > 0
    
    # Configure training parameters
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="steps" if has_eval else "no",
        eval_steps=eval_steps if has_eval else None,
        save_steps=save_steps,
        save_strategy="steps",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        warmup_steps=300,
        logging_dir=f"{output_dir}/logs",
        logging_steps=100,
        load_best_model_at_end=has_eval,  # Only if we have eval data
        metric_for_best_model="eval_loss" if has_eval else None,
        greater_is_better=False,
        save_total_limit=3,
        dataloader_pin_memory=cuda_available,
        dataloader_num_workers=num_workers,
        dataloader_prefetch_factor=2 if num_workers > 0 else None,
        ddp_find_unused_parameters=False if cuda_available else None,
        remove_unused_columns=False,
        prediction_loss_only=True,
        fp16=cuda_available,
        fp16_full_eval=cuda_available,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        group_by_length=False,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        auto_find_batch_size=False,
    )

    
    # Setup early stopping callback
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=early_stopping_patience,
        early_stopping_threshold=0.001
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
        callbacks=[early_stopping],
    )
    
    print("\n" + "="*60)
    print("STARTING MULTI-TASK TRAINING")
    print("="*60)
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(eval_dataset)}")
    print(f"Max epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print("="*60)
    
    # Train the model
    train_result = trainer.train()
    
    # Save the final model
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"\nTraining completed! Model saved to: {output_dir}")
    print(f"Final training loss: {train_result.training_loss:.4f}")
    
    return trainer, train_result

def generate_response(text, tokenizer, model, device, task_type="general"):
    """Generate response for given input text"""
    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=MAX_INPUT_LENGTH
    )
    # FOR OPT
    inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
    
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_length=GENERATION_MAX_LENGTH,
            min_length=GENERATION_MIN_LENGTH,
            num_beams=GENERATION_NUM_BEAMS,
            length_penalty=GENERATION_LENGTH_PENALTY,
            repetition_penalty=GENERATION_REPETITION_PENALTY,
            no_repeat_ngram_size=GENERATION_NO_REPEAT_NGRAM_SIZE,
            early_stopping=True,
            do_sample=False,  # Use beam search for better quality
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.strip()

def timeout_handler(signum, frame):
    """Handler for timeout signal."""
    raise TimeoutError("Operation timed out")

def execute_code_safely(code, timeout=5):
    """Execute Python code safely with timeout protection and return success status"""
    try:
        # Set up timeout signal (2 seconds for consistency with code_generation_finetuning.py)
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(2)  # 2 second timeout
        
        try:
            # Create a clean namespace for execution
            namespace = {}
            
            # Execute the code in the namespace
            exec(code, namespace)
            
            return True, "", ""
        finally:
            # Always clean up the alarm
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
            
    except TimeoutError:
        return False, "", "Code execution timeout (possible infinite loop)"
    except Exception as e:
        return False, "", str(e)

def evaluate_task_results(task_name, results):
    """
    Aggregate evaluation metrics for a specific task using the per-sample
    correctness and scores computed in evaluate_single_result().
    """
    n = len(results)
    successes = sum(r["correct"] for r in results)

    ci_low, ci_high = wilson_confidence_interval(successes, n)

    # -----------------------
    # Code Search
    # -----------------------
    if task_name == "code_search":
        acc, mrr = evaluate_code_search(results)
        return {
            "samples": n,
            "accuracy": round(acc, 3),
            "mrr": round(mrr, 3),
            "ci_95": [ci_low, ci_high],
            "target": "> 0.75"
        }

    # -----------------------
    # Clone Detection
    # -----------------------
    if task_name == "clone_detection":
        p, r, f1 = evaluate_clone_detection(results)
        return {
            "samples": n,
            "precision": round(p, 3),
            "recall": round(r, 3),
            "f1": round(f1, 3),
            "ci_95": [ci_low, ci_high],
            "target": "> 0.65"
        }

    # -----------------------
    # Code Repair
    #   Primary: Pass@K
    #   Secondary: number of plausible patches
    # -----------------------
    if task_name == "code_repair":
        pass_1 = evaluate_pass_at_k(results, 1)
        pass_5 = evaluate_pass_at_k(results, 5)
        pass_10 = evaluate_pass_at_k(results, 10)

        plausible = sum(
            1 for r in results
            if r["model_output"].strip() and r["correct"]
        )

        return {
            "samples": n,
            "pass@1": round(pass_1, 3),
            "pass@5": round(pass_5, 3),
            "pass@10": round(pass_10, 3),
            "plausible_patches": plausible,
            "ci_95": [ci_low, ci_high],
            "target": "> 0.30"
        }

    # -----------------------
    # Test Generation
    #   Primary: BLEU
    #   Secondary: correctness / coverage
    # -----------------------
    if task_name == "test_generation":
        pass_1 = evaluate_pass_at_k(results, 1)
        pass_5 = evaluate_pass_at_k(results, 5)
        pass_10 = evaluate_pass_at_k(results, 10)

        plausible = sum(
            1 for r in results 
            if r["model_output"].strip() and r["correct"]
        )

        return {
            "samples": n,
            "pass@1": round(pass_1, 3),
            "pass@5": round(pass_5, 3),
            "pass@10": round(pass_10, 3),
            "plausible_patches": plausible,
            "ci_95": [ci_low, ci_high],
            "target": "> 0.30"
        }


def generate_latex_table(metrics, output_path="results_table.tex"):
    with open(output_path, "w") as f:
        f.write("\\begin{table}[t]\n\\centering\n")
        f.write("\\begin{tabular}{lcccc}\n\\hline\n")
        f.write("Task & Metric & Score & 95\\% CI & Target \\\\\n\\hline\n")

        for task, m in metrics.items():
            if task == "code_search":
                f.write(f"Code Search & Accuracy & {m['accuracy']} & [{m['ci_95'][0]}, {m['ci_95'][1]}] & {m['target']} \\\\\n")
                f.write(f"& MRR & {m['mrr']} & & \\\\\n")

            elif task == "clone_detection":
                f.write(f"Clone Detection & F1 & {m['f1']} & [{m['ci_95'][0]}, {m['ci_95'][1]}] & {m['target']} \\\\\n")

            elif task in ["code_repair", "test_generation"]:
                f.write(f"{task.replace('_',' ').title()} & Pass@1 & {m['pass@1']} & [{m['ci_95'][0]}, {m['ci_95'][1]}] & {m['target']} \\\\\n")

        f.write("\\hline\\end{tabular}\n")
        f.write("\\caption{Multi-task evaluation results with 95\\% confidence intervals.}\n")
        f.write("\\end{table}\n")

    print(f"LaTeX table saved to {output_path}")

def evaluate_single_result(task_name, result):
    """
    Evaluate a single (task, sample) pair and return:
      - is_correct: boolean
      - score: float (used for aggregation in evaluate_task_results)
    
    Metrics per task (as defined in your table):
      - code_search: Accuracy (primary), MRR (secondary)
      - clone_detection: F1 (primary), Accuracy / Precision / Recall (secondary)
      - code_repair: Pass@K (primary), number of plausible patches (secondary)
      - test_generation: BLEU (primary), correctness / coverage (secondary)
    """
    pred = result["model_output"].strip()
    gold = result["expected_output"].strip()

    # 1) CODE SEARCH  (Primary: accuracy, Secondary: MRR)
    if task_name == "code_search":
        try:
            gold_idx = int(gold)
        except ValueError:
            return False, 0.0

        pred_idx = parse_index_from_text(pred, valid_indices={0, 1, 2})
        if pred_idx is None:
            return False, 0.0

        is_correct = (pred_idx == gold_idx)
        # Score is simply 0 or 1; MRR is computed at the aggregate level
        return is_correct, 1.0 if is_correct else 0.0

    # 2) CLONE DETECTION (Primary: F1)
    if task_name == "clone_detection":
        try:
            gold_label = int(gold)
        except ValueError:
            return False, 0.0

        pred_label = parse_index_from_text(pred, valid_indices={0, 1})
        if pred_label is None:
            return False, 0.0

        is_correct = (pred_label == gold_label)
        return is_correct, 1.0 if is_correct else 0.0

    # 3) CODE REPAIR (Primary: Pass@K, Secondary: plausible patches)
    if task_name == "code_repair":
        # We only care whether the patch compiles and executes successfully.
        try:
            compile(pred, "<string>", "exec")
            exec_ok, _, _ = execute_code_safely(pred)
            is_correct = exec_ok
            return is_correct, 1.0 if is_correct else 0.0
        except Exception:
            return False, 0.0

    # 4) TEST GENERATION (Primary: BLEU, Secondary: correctness/coverage)
    if task_name == "test_generation":
        try:
            compile(pred, "<string>", "exec")
            exec_ok, _, _ = execute_code_safely(pred)
            is_correct = exec_ok
            return is_correct, 1.0 if is_correct else 0.0
        except Exception:
            return False, 0.0

    # Fallback for unknown tasks
    return False, 0.0



def test_multitask_model(model, tokenizer, device, test_samples=None):
    """Test the trained model on all test samples"""
    if test_samples is None:
        test_samples = load_all_test_data_from_folders(DATA_ROOT)

    if not test_samples:
        print("ERROR: No test samples were loaded.")
        return
    
    print("\n" + "="*60)
    print("TESTING MULTI-TASK MODEL")
    print("="*60)
    print(f"Testing with {len(test_samples)} samples")
    
    # Count samples by task
    task_counts = {}
    for sample in test_samples:
        task = sample['task']
        task_counts[task] = task_counts.get(task, 0) + 1
    
    print("Test samples by task:")
    for task, count in task_counts.items():
        print(f"  {task}: {count} samples")
    print("="*60)
    
    model.eval()
    
    # Prepare detailed results for file
    detailed_results = {
        'metadata': {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_dir': OUTPUT_DIR,
            'total_samples': len(test_samples),
            'task_distribution': task_counts
        },
        'results_by_task': {},
        'evaluation_metrics': {}
    }
    
    # Test all samples and collect results
    task_summaries = {}
    
    # Determine which tasks to evaluate
    tasks_to_evaluate = EVAL_TASKS if EVAL_TASKS else list(TASK_PREFIXES.keys())
    
    # Validate task names
    invalid_tasks = [task for task in tasks_to_evaluate if task not in TASK_PREFIXES]
    if invalid_tasks:
        print(f"ERROR: Invalid task names in EVAL_TASKS: {invalid_tasks}")
        print(f"Available tasks: {list(TASK_PREFIXES.keys())}")
        return
    
    print(f"Evaluating tasks: {tasks_to_evaluate}")
    if EVAL_TASKS:
        print(f"Note: Only evaluating {len(EVAL_TASKS)} out of {len(TASK_PREFIXES)} available tasks")
    
    for task_name in tasks_to_evaluate:
        task_samples = [s for s in test_samples if s['task'] == task_name]
        if task_samples:
            print(f"Testing {task_name}: {len(task_samples)} samples...", end="", flush=True)
            
            task_results = []
            
            for i, sample in enumerate(task_samples):
                response = generate_response(sample['input'], tokenizer, model, device, sample['task'])
                
                result = {
                    'sample_id': i + 1,
                    'input': sample['input'],
                    'expected_output': sample.get('expected_output', ''),
                    'model_output': response,
                    'input_length': len(sample['input']),
                    'output_length': len(response)
                }
                
                # Add correctness evaluation
                is_correct, score = evaluate_single_result(task_name, result)
                result['correct'] = bool(is_correct)
                result['score'] = round(score, 3)
                
                task_results.append(result)
                
                # Progress indicator
                if (i + 1) % 10 == 0:
                    print(f" {i + 1}", end="", flush=True)
            
            print(" ✓")
            
            # Store detailed results
            detailed_results['results_by_task'][task_name] = task_results
            
            # Evaluate task performance using appropriate metrics
            print(f"  Evaluating {task_name}...", end="", flush=True)
            task_metrics = evaluate_task_results(task_name, task_results)
            print(" ✓")
            
            # Store evaluation metrics
            task_summaries[task_name] = task_metrics
            detailed_results['evaluation_metrics'][task_name] = task_metrics

    # Calculate overall success rate across all tasks
    overall_success_rate = 0.0
    total_samples_evaluated = 0
    weighted_success = 0.0
    
    for task_name, metrics in task_summaries.items():
        task_samples = metrics["samples"]

        if "accuracy" in metrics:
            task_success = metrics["accuracy"]
        elif "f1" in metrics:
            task_success = metrics["f1"]
        elif "pass@1" in metrics:
            task_success = metrics["pass@1"]
        else:
            task_success = 0.0

        weighted_success += task_success * task_samples
        total_samples_evaluated += task_samples

    
    if total_samples_evaluated > 0:
        overall_success_rate = weighted_success / total_samples_evaluated
    
    # Add overall metrics to results
    detailed_results['evaluation_metrics']['overall'] = {
        'overall_success_rate': round(overall_success_rate, 3),
        'total_samples_evaluated': total_samples_evaluated,
        'tasks_evaluated': len(task_summaries)
    }
    
    # Save detailed results to file
    results_file = os.path.join(TEST_OUTPUT_DIR, 'inst_tuning_results_detailed.json')
    results_summary_file = os.path.join(TEST_OUTPUT_DIR, 'inst_tuning_results_summary.txt')
    os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)

    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, indent=2, ensure_ascii=False)
    
    # Create human-readable summary file
    with open(results_summary_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("MULTI-TASK MODEL TEST RESULTS SUMMARY\n")
        f.write("="*80 + "\n")
        f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model Directory: {OUTPUT_DIR}\n")
        f.write(f"Total samples tested: {len(test_samples)}\n")
        f.write(f"Tasks tested: {len(task_counts)}\n")
        f.write(f"Overall Success Rate: {overall_success_rate:.1%}\n\n")
        
        f.write("Task Distribution:\n")
        for task, count in task_counts.items():
            f.write(f"  {task}: {count} samples\n")
        f.write("\n")
        
        f.write("EVALUATION METRICS BY TASK:\n")
        f.write("="*80 + "\n")
        for task_name, metrics in task_summaries.items():
            f.write(f"\n{task_name.upper()}:\n")
            f.write(f"  Samples tested: {metrics['samples']}\n")

            if "accuracy" in metrics:
                # code_search
                f.write(f"  Accuracy: {metrics['accuracy']:.3f}\n")
                f.write(f"  MRR: {metrics['mrr']:.3f}\n")

            elif "f1" in metrics:
                # clone_detection
                f.write(f"  Precision: {metrics['precision']:.3f}\n")
                f.write(f"  Recall: {metrics['recall']:.3f}\n")
                f.write(f"  F1: {metrics['f1']:.3f}\n")

            elif "pass@1" in metrics:
                # code_repair / test_generation
                f.write(f"  Pass@1: {metrics['pass@1']:.3f}\n")
                f.write(f"  Pass@5: {metrics['pass@5']:.3f}\n")
                f.write(f"  Pass@10: {metrics['pass@10']:.3f}\n")
                f.write(f"  Plausible patches: {metrics['plausible_patches']}\n")

        f.write("\n" + "="*80 + "\n")
        f.write("SAMPLE OUTPUTS (First 3 per task)\n")
        f.write("="*80 + "\n")
                
        for task_name in tasks_to_evaluate:
            if task_name in detailed_results['results_by_task']:
                task_results = detailed_results['results_by_task'][task_name]
                f.write(f"\n{task_name.upper()}:\n")
                f.write("-" * 40 + "\n")
                
                for i, result in enumerate(task_results[:3]):  # Show first 3 samples
                    f.write(f"\nSample {i+1}:\n")
                    f.write(f"Input: {result['input'][:200]}...\n")
                    f.write(f"Expected: {result['expected_output'][:200]}...\n")
                    f.write(f"Output: {result['model_output'][:200]}...\n")
                    f.write("-" * 20 + "\n")

    generate_latex_table(task_summaries)
    generate_radar_plot(task_summaries)

    # Show summary on screen
    print(f"\n" + "="*60)
    print("EVALUATION RESULTS SUMMARY")
    print("="*60)
    print(f"Overall Success Rate: {overall_success_rate:.1%}")
    print(f"Total samples tested: {len(test_samples)}")
    print(f"Tasks evaluated: {len(tasks_to_evaluate)}/{len(TASK_PREFIXES)} ({', '.join(tasks_to_evaluate)})")
    if EVAL_TASKS:
        excluded_tasks = [task for task in TASK_PREFIXES.keys() if task not in tasks_to_evaluate]
        if excluded_tasks:
            print(f"Tasks excluded: {', '.join(excluded_tasks)}")
    print()
    print("Per-task performance:")
    for task_name, metrics in task_summaries.items():
        print(f"  {task_name}:")
        print(f"    Samples: {metrics['samples']}")

        if "accuracy" in metrics:
            print(f"    Accuracy: {metrics['accuracy']:.3f}")
            print(f"    MRR: {metrics['mrr']:.3f}")

        elif "f1" in metrics:
            print(f"    Precision: {metrics['precision']:.3f}")
            print(f"    Recall: {metrics['recall']:.3f}")
            print(f"    F1: {metrics['f1']:.3f}")

        elif "pass@1" in metrics:
            print(f"    Pass@1: {metrics['pass@1']:.3f}")
            print(f"    Pass@5: {metrics['pass@5']:.3f}")
            print(f"    Pass@10: {metrics['pass@10']:.3f}")
            print(f"    Plausible patches: {metrics['plausible_patches']}")

    print(f"\nDetailed results saved to:")
    print(f"  JSON: {results_file}")
    print(f"  Text: {results_summary_file}")
    print("="*60)


def save_training_config(output_dir, config):
    """Save training configuration for future reference"""
    config_path = os.path.join(output_dir, 'training_config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)
    print(f"Training configuration saved to: {config_path}")

def main():
    torch.cuda.empty_cache()

    """
    Main function - runs the multi-task instruction tuning.
    Edit the configuration constants at the top of the file to customize training.
    """
    print("="*60)
    print("MULTI-TASK INSTRUCTION TUNING FOR CODET5+")
    print("="*60)
    print(f"Data root: {DATA_ROOT}  (uses codesearch/clone/repair/test_gen/**.jsonl)")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Model name: {MODEL_NAME}")
    print(f"Number of epochs: {NUM_EPOCHS}")
    print(f"Batch size: {TRAIN_BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Validation size: {VALIDATION_SIZE}")
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Test only mode: {TEST_ONLY}")
    print("="*60)
    
    # Set random seeds for reproducibility
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)
    
    # Determine device to use
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA GPU for training")
        # FOR OPT
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        print("✓ GPU optimizations enabled (TF32, cuDNN benchmark)")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon MPS for training")
    else:
        device = torch.device("cpu")
        print("Using CPU for training")
    
    # Load model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(MODEL_NAME)
    # FOR OPT
    if torch.cuda.is_available():
        model = model.to(device, non_blocking=True)
        print("✓ Model moved to GPU with non-blocking transfer")
    else:
        model = model.to(device)
    
    if TEST_ONLY:
        # Only test existing model
        print("Test-only mode: Loading existing model")
        if os.path.exists(OUTPUT_DIR):
            model = T5ForConditionalGeneration.from_pretrained(OUTPUT_DIR)
            tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)
            # FOR OPT
            if torch.cuda.is_available():
                model = model.to(device, non_blocking=True)
                print("✓ Model moved to GPU with non-blocking transfer")
            else:
                model = model.to(device)
            test_multitask_model(model, tokenizer, device)
        else:
            print(f"Error: Model directory {OUTPUT_DIR} not found")
        return
    
    
    
    # Load and prepare training data
    train_dataset, eval_dataset = prepare_datasets(tokenizer, VALIDATION_SIZE, RANDOM_SEED)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Train the model
    trainer, train_result = train_model(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        device=device,
        output_dir=OUTPUT_DIR,
        num_epochs=NUM_EPOCHS,
        batch_size=TRAIN_BATCH_SIZE,
        learning_rate=LEARNING_RATE
    )
    
    # Save training configuration for reference
    config = {
        'model_name': MODEL_NAME,
        'data_path': DATA_ROOT,
        'num_epochs': NUM_EPOCHS,
        'batch_size': TRAIN_BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'validation_size': VALIDATION_SIZE,
        'seed': RANDOM_SEED,
        'final_training_loss': train_result.training_loss,
        'task_prefixes': TASK_PREFIXES,
        'total_training_samples': len(train_dataset),
        'total_validation_samples': len(eval_dataset)
    }
    save_training_config(OUTPUT_DIR, config)
    
    # Test the trained model
    # print("\nTesting the trained multi-task model...")
    # test_multitask_model(model, tokenizer, device)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"Model saved to: {OUTPUT_DIR}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(eval_dataset)}")
    print(f"Final training loss: {train_result.training_loss:.4f}")
    print("="*60)

if __name__ == "__main__":
    main()
