import torch.multiprocessing
torch.multiprocessing.set_start_method("spawn", force=True)

import os
import torch
import wandb
from tune_codet5p_seq2seq import main
import argparse
import traceback
import sys
import psutil

def print_sys_info(stage=""):
    print(f"\n=== [DEBUG] {stage} ===")
    print(f"🧠 CPU memory used: {psutil.virtual_memory().percent}%")
    if torch.cuda.is_available():
        print(f"🎮 GPU total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"🎮 GPU allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"🎮 GPU reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    print("=========================\n")

def train():
    wandb.init(project="codet5-clone-detection")
    config = wandb.config

    try:
        print_sys_info("Before torch.cuda.empty_cache()")
        torch.cuda.empty_cache()

        print_sys_info("After empty_cache()")

        args = argparse.Namespace(
            data_file="/home/anzhelika/Desktop/CodeT5/better_dataset.jsonl",
            epochs=5,
            lr=config.learning_rate if "learning_rate" in config else 5e-5,
            batch_size_per_replica=config.batch_size if "batch_size" in config else 2,
            grad_acc_steps=config.grad_acc_steps if "grad_acc_steps" in config else 2,
            lr_warmup_steps=200,
            val_split=0.2,
            save_dir=f"saved_models/run_{wandb.run.name}",
            load=getattr(config, "models_variant", "Salesforce/codet5p-220m"),
            fp16=True,
            local_rank=-1,
            deepspeed=None,
            data_num=-1,
            max_source_len=getattr(config, "max_source_len", 512),
            max_target_len=8,
            cache_data="cache_data/clone_detection",
            log_freq=50,
            save_freq=500,
        )

        print_sys_info("Before main(args)")
        print(f"🚀 MODEL: {args.load}, MAX_LEN={args.max_source_len}, BATCH={args.batch_size_per_replica}")
        print(f"Save dir: {args.save_dir}")

        main(args)

        print_sys_info("After main(args)")

    except Exception as e:
        print("\n❌ Exception caught:")
        print(traceback.format_exc())
        print_sys_info("After exception")
        sys.exit(1)

if __name__ == "__main__":
    print("=== W&B DEBUG START ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
        print(f"VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print("=======================")

    wandb.agent("8fw5ee4l", function=train, project="codet5-clone-detection")
