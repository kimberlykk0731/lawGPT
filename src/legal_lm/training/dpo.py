from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from trl import DPOConfig, DPOTrainer

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


def train_alignment(config_path: str | Path):
    config = load_run_config(config_path)
    run_cfg = config["run"]
    trainer_cfg = config["trainer"]
    output_dir = ensure_output_dir(config)
    train_dataset, eval_dataset = load_json_dataset(run_cfg["train_file"], run_cfg.get("validation_file"))
    ensure_dataset_columns(train_dataset, ["prompt", "chosen", "rejected"], "Alignment train dataset")
    ensure_dataset_columns(eval_dataset, ["prompt", "chosen", "rejected"], "Alignment eval dataset")
    model, tokenizer = load_model_and_tokenizer(config)
    peft_config = build_lora_config(config)
    save_resolved_config(config, output_dir)
    method = str(run_cfg.get("alignment_method", "dpo")).lower()

    if method == "orpo":
        from trl.experimental.orpo import ORPOConfig, ORPOTrainer

        config_kwargs: dict[str, Any] = {
            "output_dir": str(output_dir),
            "learning_rate": float(trainer_cfg.get("learning_rate", 5e-6)),
            "num_train_epochs": float(trainer_cfg.get("num_train_epochs", 1)),
            "per_device_train_batch_size": int(trainer_cfg.get("per_device_train_batch_size", 1)),
            "per_device_eval_batch_size": int(trainer_cfg.get("per_device_eval_batch_size", 1)),
            "gradient_accumulation_steps": int(trainer_cfg.get("gradient_accumulation_steps", 1)),
            "warmup_ratio": float(trainer_cfg.get("warmup_ratio", 0.03)),
            "lr_scheduler_type": str(trainer_cfg.get("lr_scheduler_type", "cosine")),
            "bf16": bool(trainer_cfg.get("bf16", True)),
            "fp16": bool(trainer_cfg.get("fp16", False)),
            "gradient_checkpointing": bool(trainer_cfg.get("gradient_checkpointing", True)),
            "beta": float(trainer_cfg.get("beta", 0.1)),
            "max_length": int(run_cfg.get("max_length", 2048)),
            "max_prompt_length": int(run_cfg.get("max_prompt_length", 1536)),
            "report_to": [],
            **build_interval_kwargs(
                has_eval=eval_dataset is not None,
                logging_steps=int(trainer_cfg.get("logging_steps", 10)),
                save_steps=int(trainer_cfg.get("save_steps", 100)),
                eval_steps=int(trainer_cfg.get("eval_steps", 100)),
            ),
        }
        args = instantiate_supported(ORPOConfig, config_kwargs)
        trainer_kwargs: dict[str, Any] = {
            "model": model,
            "args": args,
            "train_dataset": train_dataset,
            "peft_config": peft_config,
        }
        if eval_dataset is not None and trainer_supports_eval(ORPOTrainer):
            trainer_kwargs["eval_dataset"] = eval_dataset
        attach_processing_class(trainer_kwargs, ORPOTrainer, tokenizer)
        trainer = ORPOTrainer(**trainer_kwargs)
    else:
        config_kwargs = {
            "output_dir": str(output_dir),
            "learning_rate": float(trainer_cfg.get("learning_rate", 5e-6)),
            "num_train_epochs": float(trainer_cfg.get("num_train_epochs", 1)),
            "per_device_train_batch_size": int(trainer_cfg.get("per_device_train_batch_size", 1)),
            "per_device_eval_batch_size": int(trainer_cfg.get("per_device_eval_batch_size", 1)),
            "gradient_accumulation_steps": int(trainer_cfg.get("gradient_accumulation_steps", 1)),
            "warmup_ratio": float(trainer_cfg.get("warmup_ratio", 0.03)),
            "lr_scheduler_type": str(trainer_cfg.get("lr_scheduler_type", "cosine")),
            "bf16": bool(trainer_cfg.get("bf16", True)),
            "fp16": bool(trainer_cfg.get("fp16", False)),
            "gradient_checkpointing": bool(trainer_cfg.get("gradient_checkpointing", True)),
            "beta": float(trainer_cfg.get("beta", 0.1)),
            "loss_type": str(trainer_cfg.get("loss_type", "sigmoid")),
            "max_length": int(run_cfg.get("max_length", 2048)),
            "max_prompt_length": int(run_cfg.get("max_prompt_length", 1536)),
            "remove_unused_columns": False,
            "report_to": [],
            **build_interval_kwargs(
                has_eval=eval_dataset is not None,
                logging_steps=int(trainer_cfg.get("logging_steps", 10)),
                save_steps=int(trainer_cfg.get("save_steps", 100)),
                eval_steps=int(trainer_cfg.get("eval_steps", 100)),
            ),
        }
        args = instantiate_supported(DPOConfig, config_kwargs)
        trainer_kwargs = {
            "model": model,
            "ref_model": None,
            "args": args,
            "train_dataset": train_dataset,
            "peft_config": peft_config,
        }
        if eval_dataset is not None and trainer_supports_eval(DPOTrainer):
            trainer_kwargs["eval_dataset"] = eval_dataset
        attach_processing_class(trainer_kwargs, DPOTrainer, tokenizer)
        trainer = DPOTrainer(**trainer_kwargs)

    resume_from_checkpoint = run_cfg.get("resume_from_checkpoint")
    if resume_from_checkpoint:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    else:
        trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    return trainer


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DPO/ORPO model for Chinese legal alignment.")
    parser.add_argument("--config", default="configs/dpo/qwen25_7b_lora.yaml")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    train_alignment(args.config)


if __name__ == "__main__":
    main()
