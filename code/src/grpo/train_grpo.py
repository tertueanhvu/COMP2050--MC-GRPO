"""
GRPO training entry point. Supports traditional GRPO advanatge and the MC-GRPO variant.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import yaml
import torch
from datasets import load_dataset
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

# Make `src` importable when running directly
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.grpo.reward_paper import paper_reward
from src.grpo.trainer_mc_grpo import MCGRPOTrainer


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main(cfg: dict) -> None:
    algorithm = cfg["algorithm"]              # "grpo" | "mc_grpo"
    batching = cfg.get("batching", "iii")     # "ii" (random) | "iii" (matched)
    is_mc = algorithm == "mc_grpo"

    # ── tokenizer + model (continue from SFT adapter) ──
    print(f"Loading tokenizer + model from SFT adapter: {cfg['sft_adapter']}")
    tokenizer = AutoTokenizer.from_pretrained(cfg["sft_adapter"])
    tokenizer.padding_side = "left"           # required for generation

    model = AutoPeftModelForCausalLM.from_pretrained(
        cfg["sft_adapter"],
        is_trainable=True,
        dtype=torch.bfloat16,
    )

    # ── dataset ──
    print(f"Loading GRPO dataset: {cfg['data_path']}")
    ds = load_dataset("json", data_files=cfg["data_path"], split="train")
    print(f"  Dataset size: {len(ds)} rows | columns: {ds.column_names}")

    scale_rewards = "group"

    shuffle_dataset = batching != "iii"
    print(f"  algorithm={algorithm} batching={batching} "
          f"-> scale_rewards={scale_rewards}, shuffle_dataset={shuffle_dataset}")

    # ── GRPOConfig (TRL 1.4.0) ──
    args = GRPOConfig(
        # Loss
        loss_type="dapo",
        epsilon=0.2,
        epsilon_high=0.28,
        beta=0.0,
        scale_rewards=scale_rewards,
        mask_truncated_completions=True,

        # Generation
        num_generations=cfg.get("num_generations", 4),
        temperature=cfg.get("temperature", 1.0),
        top_p=cfg.get("top_p", 0.95),
        max_completion_length=cfg.get("max_completion_length", 2048),

        # Data ordering 
        shuffle_dataset=shuffle_dataset,

        # vLLM
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=cfg.get("vllm_gpu_memory_utilization", 0.5),
        vllm_enable_sleep_mode=cfg.get("vllm_enable_sleep_mode", True),
        vllm_importance_sampling_correction=cfg.get("vllm_is_correction", True),
        vllm_max_model_length=cfg.get("vllm_max_model_length", 4096),

        # Training
        learning_rate=cfg.get("learning_rate", 5e-7),
        max_steps=cfg.get("max_steps", 1800),
        per_device_train_batch_size=cfg.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 32),
        warmup_ratio=cfg.get("warmup_ratio", 0.05),
        bf16=True,
        gradient_checkpointing=True,
        log_completions=True,
        logging_steps=cfg.get("logging_steps", 10),
        save_steps=cfg.get("save_steps", 200),
        save_total_limit=cfg.get("save_total_limit", 3),
        seed=cfg["seed"],
        output_dir=cfg["output_dir"],
        report_to=cfg.get("report_to", []),
        project=cfg.get("wandb_project", "mode-grpo-trl"),
    )

    # ── trainer (variant or vanilla) ──
    Trainer = MCGRPOTrainer if is_mc else GRPOTrainer
    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=ds,
        reward_funcs=[paper_reward],
        args=args,
    )

    print(f"Starting training ({cfg['condition_id']}, seed={cfg['seed']}, "
          f"max_steps={args.max_steps}, trainer={Trainer.__name__})...")
    trainer.train()
    print(f"Training complete. Output saved to {cfg['output_dir']}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Path to GRPO config YAML")
    p.add_argument("--seed", type=int, default=None, help="Override seed in config")
    p.add_argument("--max_steps", type=int, default=None, help="Override max_steps")
    p.add_argument("--output_dir", default=None, help="Override output_dir")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)

    if args.seed is not None:
        cfg["seed"] = args.seed
        if args.output_dir is None and "_seed" in cfg.get("output_dir", ""):
            base = cfg["output_dir"].rsplit("_seed", 1)[0]
            cfg["output_dir"] = f"{base}_seed{args.seed}"
    if args.max_steps is not None:
        cfg["max_steps"] = args.max_steps
    if args.output_dir is not None:
        cfg["output_dir"] = args.output_dir

    main(cfg)
