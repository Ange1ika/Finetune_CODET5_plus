"""
Finetune CodeT5+ models on any Seq2Seq LM tasks
You can customize your own training data by following the HF dataset format to cache it to args.cache_data
Author: Yue Wang
Date: June 2023
"""


"""python "CodeT5+/tune_codet5p_seq2seq.py" \
  --data-file better_dataset.jsonl \
  --load Salesforce/codet5p-220m \
  --save-dir saved_models/clone_detection \
  --epochs 3 \
  --batch-size-per-replica 4 \
  --fp16
"""
"""
Finetune CodeT5+ models on any Seq2Seq LM tasks with W&B hyperparameter search
You can customize your own training data by following the HF dataset format to cache it to args.cache_data
Author: Yue Wang
Date: June 2023
"""

import os
import pprint
import argparse
import numpy as np
import wandb
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(eval_pred):
    """Вычисляет accuracy, precision, recall, F1 для clone detection"""
    predictions, labels = eval_pred
    
    # Декодируем предсказания (берем argmax по vocab)
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    
    # predictions shape: (batch_size, seq_len, vocab_size)
    decoded_preds = np.argmax(predictions, axis=-1)
    
    # Заменяем -100 на pad_token_id для корректного декодирования
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
    # Декодируем в строки
    decoded_preds = tokenizer.batch_decode(decoded_preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Преобразуем в бинарные метки (0/1)
    try:
        pred_labels = [int(p.strip()) if p.strip() in ['0', '1'] else 0 for p in decoded_preds]
        true_labels = [int(l.strip()) for l in decoded_labels]
    except:
        # Fallback если что-то пошло не так
        pred_labels = [0] * len(decoded_preds)
        true_labels = [int(l.strip()) if l.strip() in ['0', '1'] else 0 for l in decoded_labels]
    
    # Вычисляем метрики
    accuracy = accuracy_score(true_labels, pred_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average='binary', zero_division=0
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

def run_training(args, model, train_data, eval_data):
    print("Starting main loop")

    training_args = TrainingArguments(
        report_to=["wandb", "tensorboard"],
        output_dir=args.save_dir,
        overwrite_output_dir=False,

        do_train=True,
        do_eval=True,  # ✅ Включаем валидацию
        save_strategy='epoch',
        eval_strategy='epoch',  # ✅ Валидируем каждую эпоху
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size_per_replica,
        per_device_eval_batch_size=args.batch_size_per_replica * 2,  # ✅ Больше batch для eval
        gradient_accumulation_steps=args.grad_acc_steps,

        learning_rate=args.lr,
        weight_decay=0.05,
        warmup_steps=args.lr_warmup_steps,

        logging_dir=args.save_dir,
        logging_first_step=True,
        logging_steps=args.log_freq,
        save_total_limit=1,
        
        load_best_model_at_end=True,  # ✅ Загружаем лучшую модель в конце
        metric_for_best_model='f1',  # ✅ Отбираем по F1
        greater_is_better=True,

        dataloader_drop_last=True,
        dataloader_num_workers=4,

        local_rank=args.local_rank,
        deepspeed=args.deepspeed,
        fp16=args.fp16,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,  # ✅ Добавляем eval датасет
        compute_metrics=compute_metrics,  # ✅ Подключаем метрики
    )
    
    trainer.train()

    if args.local_rank in [0, -1]:
        final_metrics = trainer.evaluate()
        wandb.log({
            "final_f1": final_metrics.get("eval_f1", 0.0),
            "final_accuracy": final_metrics.get("eval_accuracy", 0.0),
            "final_precision": final_metrics.get("eval_precision", 0.0),
            "final_recall": final_metrics.get("eval_recall", 0.0)
        })

        wandb.log({"final_f1": final_metrics["eval_f1"]})
        
        if "eval_f1" in final_metrics:
            wandb.log({
                "final_f1": final_metrics["eval_f1"],
                "final_accuracy": final_metrics["eval_accuracy"],
                "final_precision": final_metrics["eval_precision"],
                "final_recall": final_metrics["eval_recall"]
            })
            wandb.run.summary["final_f1"] = final_metrics["eval_f1"]
            
        print("\n" + "="*50)
        print("📊 FINAL VALIDATION METRICS:")
        print("="*50)
        for key, value in final_metrics.items():
            print(f"{key}: {value:.4f}")
        print("="*50 + "\n")
        
        final_checkpoint_dir = os.path.join(args.save_dir, "final_checkpoint")
        model.save_pretrained(final_checkpoint_dir)
        print(f"✅ Saved model to {final_checkpoint_dir}")

def load_tokenize_data(args):
    """Загружает JSONL и токенизирует, разделяет на train/val"""
    if os.path.exists(args.cache_data):
        dataset = load_from_disk(args.cache_data)
        print(f"==> Loaded {len(dataset)} cached samples")
    else:
        # === Загружаем JSONL ===
        dataset = load_dataset("json", data_files=args.data_file, split="train")
        global tokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.load)

        def preprocess_function(examples):
            inputs = [
                f"detect clones: {src} [SEP] {tgt}"
                for src, tgt in zip(examples["source"], examples["target"])
            ]
            labels = [str(l) for l in examples["label"]]

            model_inputs = tokenizer(
                inputs,
                max_length=args.max_source_len,
                padding="max_length",
                truncation=True,
            )
            label_inputs = tokenizer(
                labels,
                max_length=args.max_target_len,
                padding="max_length",
                truncation=True,
            )

            model_inputs["labels"] = label_inputs["input_ids"]
            model_inputs["labels"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label]
                for label in model_inputs["labels"]
            ]
            return model_inputs

        print("Tokenizing dataset...")
        dataset = dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=dataset.column_names,
            num_proc=4,
            load_from_cache_file=False,
        )

        dataset.save_to_disk(args.cache_data)
        print(f"✅ Saved preprocessed dataset to {args.cache_data}")
    
    # ✅ Разделяем на train/validation
    if args.val_split > 0:
        split_dataset = dataset.train_test_split(
            test_size=args.val_split,
            seed=42
        )
        train_data = split_dataset['train']
        eval_data = split_dataset['test']
        print(f"✅ Split: {len(train_data)} train, {len(eval_data)} validation")
    else:
        train_data = dataset
        eval_data = None
        print(f"⚠️  No validation split - using all {len(train_data)} samples for training")
    
    return train_data, eval_data

def main(args):
    argsdict = vars(args)
    print(pprint.pformat(argsdict))
    os.makedirs(args.save_dir, exist_ok=True)

    with open(os.path.join(args.save_dir, "command.txt"), "w") as f:
        f.write(pprint.pformat(argsdict))

    # ✅ Получаем train и validation датасеты
    train_data, eval_data = load_tokenize_data(args)
    
    if args.data_num != -1:
        train_data = train_data.select(range(args.data_num))
        if eval_data is not None:
            eval_data = eval_data.select(range(min(args.data_num // 5, len(eval_data))))

    # Глобальный tokenizer для compute_metrics
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.load)
    
    model = AutoModelForSeq2SeqLM.from_pretrained(args.load)
    print(f"==> Loaded model {args.load}, parameters: {model.num_parameters()}")

    run_training(args, model, train_data, eval_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CodeT5+ finetuning for Clone Detection")
    parser.add_argument('--data-file', default='better_dataset.jsonl', type=str)
    parser.add_argument('--data-num', default=-1, type=int)
    parser.add_argument('--max-source-len', default=512, type=int)
    parser.add_argument('--max-target-len', default=8, type=int)
    parser.add_argument('--cache-data', default='cache_data/clone_detection', type=str)
    parser.add_argument('--load', default='Salesforce/codet5p-220m', type=str)
    
    # ✅ Добавляем параметр для validation split
    parser.add_argument('--val-split', default=0.1, type=float, 
                        help='Доля данных для валидации (0.1 = 10%)')

    # Training
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--lr', default=5e-5, type=float)
    parser.add_argument('--lr-warmup-steps', default=200, type=int)
    parser.add_argument('--batch-size-per-replica', default=8, type=int)
    parser.add_argument('--grad-acc-steps', default=4, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--deepspeed', default=None, type=str)
    parser.add_argument('--fp16', default=True, action='store_true')

    parser.add_argument('--save-dir', default='saved_models/clone_detection', type=str)
    parser.add_argument('--log-freq', default=50, type=int)
    parser.add_argument('--save-freq', default=500, type=int)
    args = parser.parse_args()

    main(args)