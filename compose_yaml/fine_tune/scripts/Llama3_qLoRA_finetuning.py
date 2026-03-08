#
# Based on NVIDIA dgx-spark-playbooks pytorch-fine-tune
# Adapted from Llama3_70B_qLoRA_finetuning.py -> Llama 3.2 3B
#

import torch
import argparse
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training


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
    print(f"Loading model: {args.model_name}")
    print(f"Training mode: QLoRA (4-bit quantization)")

    # Configure 4-bit quantization
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=args.dtype,
        bnb_4bit_quant_storage=args.dtype
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=quantization_config,
        dtype=args.dtype,
        device_map="cuda",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Prepare model for QLoRA
    print(f"Preparing model for QLoRA (4-bit) with rank {args.lora_rank}...")
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=args.lora_rank,
        target_modules="all-linear",
        lora_alpha=16,
        lora_dropout=0,
        task_type=TaskType.CAUSAL_LM
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")

    # Load dataset
    print(f"Loading dataset with {args.dataset_size} samples...")
    dataset = get_alpaca_dataset(tokenizer.eos_token, args.dataset_size)

    # SFT config - QLoRA는 torch.compile 없이 바로 학습
    config = {
        "per_device_train_batch_size": args.batch_size,
        "num_train_epochs": args.num_epochs,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "optim": "adamw_torch",
        "save_strategy": 'no',
        "remove_unused_columns": False,
        "seed": 42,
        "dataset_text_field": "text",
        "packing": False,
        "max_length": args.seq_length,
        "report_to": "tensorboard",
        "logging_dir": args.log_dir,
        "logging_steps": args.logging_steps,
        "gradient_checkpointing": args.gradient_checkpointing,
    }

    if args.output_dir:
        config["output_dir"] = args.output_dir

    # Train
    print(f"\nStarting QLoRA fine-tuning for {args.num_epochs} epoch(s)...")
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(**config),
        peft_config=peft_config,
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

    # Save model
    if args.output_dir:
        print(f"Saving QLoRA adapter to {args.output_dir}...")
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print(f"QLoRA adapter saved to {args.output_dir}")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Llama 3.2 3B Fine-tuning with QLoRA")

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
                        help="Directory to save the QLoRA adapter")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    print(f"\n{'='*60}")
    print("LLAMA 3.2 3B QLoRA FINE-TUNING CONFIGURATION")
    print(f"{'='*60}")
    print(f"Model: {args.model_name}")
    print(f"Training mode: QLoRA (4-bit quantization)")
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
