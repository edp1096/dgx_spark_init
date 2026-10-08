#
# Based on NVIDIA dgx-spark-playbooks pytorch-fine-tune
# Adapted from Llama3_8B_LoRA_finetuning.py -> Llama 3.2 3B
#

import os
import torch
import argparse
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType


def is_distributed():
    return int(os.environ.get("WORLD_SIZE", 1)) > 1


# Define prompt templates
ALPACA_PROMPT_TEMPLATE = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction: {}

### Input: {}

### Response: {}"""

def get_alpaca_dataset(eos_token, dataset_size=512):
    def preprocess(x):
        texts = [
            ALPACA_PROMPT_TEMPLATE.format(instruction, input, output) + eos_token
            for instruction, input, output in zip(x["instruction"], x["input"], x["output"])
        ]
        return {"text": texts}

    dataset = load_dataset("tatsu-lab/alpaca", split="train").select(range(dataset_size)).shuffle(seed=42)
    return dataset.map(preprocess, remove_columns=dataset.column_names, batched=True)


def main(args):
    distributed = is_distributed()

    print(f"Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=args.dtype,
        device_map="auto" if not distributed else None,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Configure LoRA
    peft_config = LoraConfig(
        r=args.lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, peft_config)

    # FSDP requires uniform dtype - cast LoRA params (float32) to match base model (bfloat16)
    if distributed:
        model = model.to(torch.bfloat16)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")

    # Load dataset
    print(f"Loading dataset with {args.dataset_size} samples...")
    dataset = get_alpaca_dataset(tokenizer.eos_token, args.dataset_size)

    # SFT config
    config = {
        "per_device_train_batch_size": args.batch_size,
        "num_train_epochs": 0.05,  # Warmup epoch
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "optim": "adamw_torch",
        "save_strategy": 'no',
        "remove_unused_columns": False,
        "seed": 42,
        "dataset_text_field": "text",
        "packing": False,
        "max_length": args.seq_length,
        "report_to": "none",
        "logging_dir": args.log_dir,
        "logging_steps": args.logging_steps,
        "gradient_checkpointing": args.gradient_checkpointing,
    }

    # torch.compile + warmup only for single node (incompatible with FSDP)
    if not distributed:
        print("Compiling model with torch.compile()...")
        model = torch.compile(model)

        print("Running warmup for torch.compile()...")
        SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=dataset,
            args=SFTConfig(**config),
        ).train()
    else:
        print("Distributed mode: skipping torch.compile() and warmup")

    # Train
    print(f"\nStarting LoRA fine-tuning for {args.num_epochs} epoch(s)...")
    config["num_train_epochs"] = args.num_epochs
    config["report_to"] = "tensorboard"

    if args.output_dir:
        config["output_dir"] = args.output_dir

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(**config),
    )

    trainer_stats = trainer.train()

    # Print stats
    print(f"\n{'='*60}")
    print("TRAINING COMPLETED")
    print(f"{'='*60}")
    print(f"Training runtime: {trainer_stats.metrics['train_runtime']:.2f} seconds")
    print(f"Samples per second: {trainer_stats.metrics['train_samples_per_second']:.2f}")
    print(f"Steps per second: {trainer_stats.metrics['train_steps_per_second']:.2f}")
    print(f"Train loss: {trainer_stats.metrics['train_loss']:.4f}")
    print(f"{'='*60}\n")

    # Save model (all ranks participate in FSDP gather, rank 0 writes to disk)
    if args.output_dir:
        trainer.save_model(args.output_dir)
        if int(os.environ.get("RANK", 0)) == 0:
            tokenizer.save_pretrained(args.output_dir)
            print(f"LoRA adapter saved to {args.output_dir}")

    # Clean up distributed process group
    if distributed and torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


def parse_arguments():
    parser = argparse.ArgumentParser(description="Llama 3.2 3B Fine-tuning with LoRA")

    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-3B-Instruct",
                        help="Model name or path")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float32", "float16", "bfloat16"],
                        help="Model dtype")

    parser.add_argument("--batch_size", type=int, default=8,
                        help="Per device training batch size")
    parser.add_argument("--seq_length", type=int, default=2048,
                        help="Maximum sequence length")
    parser.add_argument("--num_epochs", type=int, default=1,
                        help="Number of training epochs")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Enable gradient checkpointing to save memory")

    parser.add_argument("--lora_rank", type=int, default=8,
                        help="LoRA rank")

    parser.add_argument("--dataset_size", type=int, default=512,
                        help="Number of samples to use from dataset")

    parser.add_argument("--logging_steps", type=int, default=1,
                        help="Log every N steps")
    parser.add_argument("--log_dir", type=str, default="logs",
                        help="Directory for logs")

    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save the LoRA adapter")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    print(f"\n{'='*60}")
    print("LLAMA 3.2 3B LoRA FINE-TUNING CONFIGURATION")
    print(f"{'='*60}")
    print(f"Model: {args.model_name}")
    print(f"Training mode: LoRA")
    print(f"Batch size: {args.batch_size}")
    print(f"Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"Sequence length: {args.seq_length}")
    print(f"Number of epochs: {args.num_epochs}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"LoRA rank: {args.lora_rank}")
    print(f"Dataset size: {args.dataset_size}")
    print(f"Gradient checkpointing: {args.gradient_checkpointing}")
    print(f"Output directory: {args.output_dir or 'Not saving'}")
    print(f"{'='*60}\n")

    main(args)
