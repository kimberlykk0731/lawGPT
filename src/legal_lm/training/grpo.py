from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from trl import GRPOConfig, GRPOTrainer

from legal_lm.rewards import build_reward_functions
from legal_lm.training.common import (
    attach_processing_class,
    build_interval_kwargs,
    build_lora_config,
    ensure_dataset_columns,
    ensure_output_dir,
    instantiate_supported,
    load_json_dataset,
    load_model_and_tokenizer,
    load_run_config,
    save_resolved_config,
    trainer_supports_eval,
)


def train_grpo(config_path: str | Path):
    config = load_run_config(config_path)
    run_cfg = config["run"]
    trainer_cfg = config["trainer"]
    output_dir = ensure_output_dir(config)
    train_dataset, eval_dataset = load_json_dataset(run_cfg["train_file"], run_cfg.get("validation_file"))
    ensure_dataset_columns(
        train_dataset,
        ["prompt", "answer", "reasoning", "citations", "facts", "issues", "metadata_json"],
        "GRPO train dataset",
    )
    ensure_dataset_columns(
        eval_dataset,
        ["prompt", "answer", "reasoning", "citations", "facts", "issues", "metadata_json"],
        "GRPO eval dataset",
    )
    model, tokenizer = load_model_and_tokenizer(config)
    peft_config = build_lora_config(config)
    save_resolved_config(config, output_dir)
    reward_funcs = build_reward_functions(run_cfg.get("reward_profile", "criminal"))

    config_kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "learning_rate": float(trainer_cfg.get("learning_rate", 1e-6)),
        "num_train_epochs": float(trainer_cfg.get("num_train_epochs", 1)),
        "per_device_train_batch_size": int(trainer_cfg.get("per_device_train_batch_size", 1)),
        "gradient_accumulation_steps": int(trainer_cfg.get("gradient_accumulation_steps", 1)),
        "bf16": bool(trainer_cfg.get("bf16", True)),
        "fp16": bool(trainer_cfg.get("fp16", False)),
        "gradient_checkpointing": bool(trainer_cfg.get("gradient_checkpointing", True)),
        "beta": float(trainer_cfg.get("beta", 0.0)),
        "num_generations": int(trainer_cfg.get("num_generations", 4)),
        "temperature": float(trainer_cfg.get("temperature", 0.7)),
        "scale_rewards": trainer_cfg.get("scale_rewards", "batch"),
        "max_prompt_length": int(run_cfg.get("max_prompt_length", 1536)),
        "max_completion_length": int(run_cfg.get("max_completion_length", 768)),
        "use_vllm": bool(run_cfg.get("use_vllm", False)),
        "report_to": [],
        **build_interval_kwargs(
            has_eval=eval_dataset is not None,
            logging_steps=int(trainer_cfg.get("logging_steps", 5)),
            save_steps=int(trainer_cfg.get("save_steps", 100)),
            eval_steps=int(trainer_cfg.get("eval_steps", trainer_cfg.get("save_steps", 100))),
        ),
    }
    args = instantiate_supported(GRPOConfig, config_kwargs)

    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "reward_funcs": reward_funcs,
        "args": args,
        "train_dataset": train_dataset,
        "peft_config": peft_config,
    }
    if eval_dataset is not None and trainer_supports_eval(GRPOTrainer):
        trainer_kwargs["eval_dataset"] = eval_dataset
    attach_processing_class(trainer_kwargs, GRPOTrainer, tokenizer)
    trainer = GRPOTrainer(**trainer_kwargs)
    resume_from_checkpoint = run_cfg.get("resume_from_checkpoint")
    if resume_from_checkpoint:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    else:
        trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    return trainer


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GRPO model for Chinese legal reasoning.")
    parser.add_argument("--config", default="configs/grpo/qwen25_7b_lora.yaml")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    train_grpo(args.config)


if __name__ == "__main__":
    main()
