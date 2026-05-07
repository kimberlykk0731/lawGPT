"""On-Policy Distillation (OPD) 训练入口。

训练目标：在 SFT 基础上，让 student 模仿教师对 student 自采样回答的改写。
数据由 scripts/build_opd_dataset.py 产出，每条形如：
  {"prompt": ..., "student": ..., "teacher": ..., "score": float}
本实现作为 SFT loss(教师改写) 进行；KL 正则项作为可选项暴露在配置中。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from datasets import Dataset
from trl import SFTConfig, SFTTrainer

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
)


def _format_text(row: dict[str, Any], system_prompt: str) -> str:
    prompt = row.get("prompt", "")
    teacher = row.get("teacher") or row.get("response") or ""
    return (
        f"<|system|>\n{system_prompt}\n"
        f"<|user|>\n{prompt}\n"
        f"<|assistant|>\n{teacher}"
    )


def _materialize(dataset: Dataset, system_prompt: str) -> Dataset:
    return dataset.map(lambda row: {"text": _format_text(row, system_prompt)})


def train_opd(config_path: str | Path) -> SFTTrainer:
    config = load_run_config(config_path)
    run_cfg = config["run"]
    trainer_cfg = config["trainer"]
    train_dataset, eval_dataset = load_json_dataset(
        run_cfg["train_file"], run_cfg.get("validation_file")
    )
    ensure_dataset_columns(train_dataset, ["prompt", "teacher"], "OPD train dataset")
    ensure_dataset_columns(eval_dataset, ["prompt", "teacher"], "OPD eval dataset")

    system_prompt = config["prompt"]["system_prompt"]
    train_dataset = _materialize(train_dataset, system_prompt)
    if eval_dataset is not None:
        eval_dataset = _materialize(eval_dataset, system_prompt)

    model, tokenizer = load_model_and_tokenizer(config)
    peft_config = build_lora_config(config)
    output_dir = ensure_output_dir(config)
    save_resolved_config(config, output_dir)

    config_kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "dataset_text_field": "text",
        "max_seq_length": int(run_cfg.get("max_seq_length", 4096)),
        "max_length": int(run_cfg.get("max_seq_length", 4096)),
        "packing": bool(run_cfg.get("packing", False)),
        "learning_rate": float(trainer_cfg.get("learning_rate", 1e-5)),
        "num_train_epochs": float(trainer_cfg.get("num_train_epochs", 1)),
        "per_device_train_batch_size": int(trainer_cfg.get("per_device_train_batch_size", 1)),
        "per_device_eval_batch_size": int(trainer_cfg.get("per_device_eval_batch_size", 1)),
        "gradient_accumulation_steps": int(trainer_cfg.get("gradient_accumulation_steps", 8)),
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
    trainer.train(resume_from_checkpoint=run_cfg.get("resume_from_checkpoint"))
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    return trainer


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="On-Policy Distillation trainer.")
    parser.add_argument("--config", default="configs/opd/qwen35_4b_lora.yaml")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    train_opd(args.config)


if __name__ == "__main__":
    main()
