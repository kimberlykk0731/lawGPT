from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from datasets import Dataset
from trl import SFTConfig, SFTTrainer

from legal_lm.data.build_sft import build_sft_rows
from legal_lm.data.registry import load_samples
from legal_lm.training.common import (
    attach_processing_class,
    build_interval_kwargs,
    build_lora_config,
    ensure_dataset_columns,
    ensure_output_dir,
    instantiate_supported,
    load_model_and_tokenizer,
    load_run_config,
    save_resolved_config,
)


def build_sft_datasets(config: dict[str, Any]) -> tuple[Dataset, Dataset | None]:
    run_cfg = config["run"]
    dataset_config_path = config["dataset_config_path"]
    dataset_name = run_cfg["dataset_name"]
    system_prompt = config["prompt"]["system_prompt"]
    mode = run_cfg["sft_mode"]

    train_samples = load_samples(dataset_config_path, dataset_name, run_cfg.get("split", "train"))
    train_rows = build_sft_rows(train_samples, mode=mode, system_prompt=system_prompt)
    train_dataset = Dataset.from_list(train_rows)

    eval_dataset = None
    validation_split = run_cfg.get("validation_split")
    if validation_split:
        eval_samples = load_samples(dataset_config_path, dataset_name, validation_split)
        eval_rows = build_sft_rows(eval_samples, mode=mode, system_prompt=system_prompt)
        eval_dataset = Dataset.from_list(eval_rows)

    return train_dataset, eval_dataset


def train_sft(config_path: str | Path) -> SFTTrainer:
    config = load_run_config(config_path)
    train_dataset, eval_dataset = build_sft_datasets(config)
    ensure_dataset_columns(train_dataset, ["text"], "SFT train dataset")
    ensure_dataset_columns(eval_dataset, ["text"], "SFT eval dataset")
    model, tokenizer = load_model_and_tokenizer(config)
    peft_config = build_lora_config(config)
    output_dir = ensure_output_dir(config)
    save_resolved_config(config, output_dir)

    trainer_cfg = config["trainer"]
    run_cfg = config["run"]
    config_kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "dataset_text_field": "text",
        "max_seq_length": int(run_cfg.get("max_seq_length", 2048)),
        "max_length": int(run_cfg.get("max_seq_length", 2048)),
        "packing": bool(run_cfg.get("packing", False)),
        "learning_rate": float(trainer_cfg.get("learning_rate", 2e-5)),
        "num_train_epochs": float(trainer_cfg.get("num_train_epochs", 1)),
        "per_device_train_batch_size": int(trainer_cfg.get("per_device_train_batch_size", 1)),
        "per_device_eval_batch_size": int(trainer_cfg.get("per_device_eval_batch_size", 1)),
        "gradient_accumulation_steps": int(trainer_cfg.get("gradient_accumulation_steps", 1)),
        "warmup_ratio": float(trainer_cfg.get("warmup_ratio", 0.03)),
        "lr_scheduler_type": str(trainer_cfg.get("lr_scheduler_type", "cosine")),
        "weight_decay": float(trainer_cfg.get("weight_decay", 0.0)),
        "bf16": bool(trainer_cfg.get("bf16", True)),
        "fp16": bool(trainer_cfg.get("fp16", False)),
        "gradient_checkpointing": bool(trainer_cfg.get("gradient_checkpointing", True)),
        "report_to": [],
        **build_interval_kwargs(
            has_eval=eval_dataset is not None,
            logging_steps=int(run_cfg.get("logging_steps", 10)),
            save_steps=int(run_cfg.get("save_steps", 200)),
            eval_steps=int(run_cfg.get("eval_steps", 200)),
        ),
    }
    args = instantiate_supported(SFTConfig, config_kwargs)

    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "peft_config": peft_config,
    }
    attach_processing_class(trainer_kwargs, SFTTrainer, tokenizer)
    trainer = SFTTrainer(**trainer_kwargs)
    resume_from_checkpoint = run_cfg.get("resume_from_checkpoint")
    if resume_from_checkpoint:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    else:
        trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    return trainer


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SFT model for Chinese legal reasoning.")
    parser.add_argument("--config", default="configs/sft/qwen25_7b_lora.yaml")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    train_sft(args.config)


if __name__ == "__main__":
    main()
