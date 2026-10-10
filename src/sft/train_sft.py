"""

SFT phase using TRL SFTTrainer with LoRA on Llama 3.2 3B Instruct.
"""
from __future__ import annotations
import argparse
import yaml
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main(cfg: dict) -> None:
    print(f"=== SFT: {cfg.get('name', 'untitled')} ===")
    print(f"Base model: {cfg['model_name']}")
    print(f"Data:       {cfg['data_path']}")
    print(f"Output:     {cfg['output_dir']}")

    # ─── tokenizer + model ───
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # SFT uses right padding (different from GRPO)

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        torch_dtype=torch.bfloat16,
        attn_implementation=cfg.get("attn_implementation", "sdpa"),
    )

    # ─── LoRA config ───
    peft_config = LoraConfig(
        r=cfg.get("lora_r", 16),
        lora_alpha=cfg.get("lora_alpha", 32),
        lora_dropout=cfg.get("lora_dropout", 0.05),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=cfg.get("lora_target_modules", "all-linear"),
    )

    # ─── dataset ───
    ds = load_dataset("json", data_files=cfg["data_path"], split="train")
    print(f"Dataset size: {len(ds)} rows")

    # ─── SFTConfig ───
    sft_args = SFTConfig(
        output_dir=cfg["output_dir"],
        max_length=cfg.get("max_length", 3072),
        packing=cfg.get("packing", False),
        completion_only_loss=True,  # only learn on completion, not prompt
        per_device_train_batch_size=cfg.get("per_device_train_batch_size", 2),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 16),
        learning_rate=cfg.get("learning_rate", 2e-5),
        num_train_epochs=cfg.get("num_train_epochs", 1),
        warmup_ratio=cfg.get("warmup_ratio", 0.03),
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        save_steps=cfg.get("save_steps", 500),
        save_total_limit=cfg.get("save_total_limit", 2),
        report_to=cfg.get("report_to", []),
        seed=cfg.get("seed", 42),
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    print("Starting SFT training...")
    trainer.train()
    trainer.save_model(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])
    print(f"SFT complete. Adapter saved to {cfg['output_dir']}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--data_path", default=None, help="Override data_path")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.data_path is not None:
        cfg["data_path"] = args.data_path
    main(cfg)
