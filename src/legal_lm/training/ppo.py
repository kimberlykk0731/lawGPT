"""PPO 训练入口（学习型 RM + 可选规则奖励叠加）。

实现细节：
- 走 TRL 的 PPOTrainer / PPOConfig，rollout 可由 vLLM 加速（use_vllm=true）
- 奖励 = w1 * RM(score) + w2 * sum(rule_rewards)，权重在配置 `reward.composite` 中
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from trl import PPOConfig, PPOTrainer

from legal_lm.rewards.composite import CompositeReward
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


def _build_reward_callable(config: dict[str, Any]):
    reward_cfg = config.get("reward", {})
    composite_cfg = reward_cfg.get("composite", {})
    rm_path = reward_cfg.get("reward_model_path") or config["run"].get("reward_model_path")
    rule_profile = reward_cfg.get("rule_profile", "general")
    weights = composite_cfg.get("weights", {"rule": 0.3, "learned": 0.7})
    return CompositeReward.from_paths(
        rm_path=rm_path,
        rule_profile=rule_profile,
        weights=weights,
    )


def train_ppo(config_path: str | Path) -> PPOTrainer:
    config = load_run_config(config_path)
    run_cfg = config["run"]
    trainer_cfg = config["trainer"]
    output_dir = ensure_output_dir(config)
    train_dataset, eval_dataset = load_json_dataset(
        run_cfg["train_file"], run_cfg.get("validation_file")
    )
    ensure_dataset_columns(train_dataset, ["prompt"], "PPO train dataset")
    ensure_dataset_columns(eval_dataset, ["prompt"], "PPO eval dataset")

    model, tokenizer = load_model_and_tokenizer(config)
    peft_config = build_lora_config(config)
    save_resolved_config(config, output_dir)
    composite = _build_reward_callable(config)

    config_kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "learning_rate": float(trainer_cfg.get("learning_rate", 1e-6)),
        "num_train_epochs": float(trainer_cfg.get("num_train_epochs", 1)),
        "per_device_train_batch_size": int(trainer_cfg.get("per_device_train_batch_size", 1)),
        "gradient_accumulation_steps": int(trainer_cfg.get("gradient_accumulation_steps", 8)),
        "bf16": bool(trainer_cfg.get("bf16", True)),
        "fp16": bool(trainer_cfg.get("fp16", False)),
        "gradient_checkpointing": bool(trainer_cfg.get("gradient_checkpointing", True)),
        "kl_coef": float(trainer_cfg.get("kl_coef", 0.05)),
        "cliprange": float(trainer_cfg.get("cliprange", 0.2)),
        "cliprange_value": float(trainer_cfg.get("cliprange_value", 0.2)),
        "vf_coef": float(trainer_cfg.get("vf_coef", 0.1)),
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
    args = instantiate_supported(PPOConfig, config_kwargs)

    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": args,
        "train_dataset": train_dataset,
        "peft_config": peft_config,
        "reward_funcs": [composite.as_callable()],
    }
    attach_processing_class(trainer_kwargs, PPOTrainer, tokenizer)
    trainer = PPOTrainer(**trainer_kwargs)
    trainer.train(resume_from_checkpoint=run_cfg.get("resume_from_checkpoint"))
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    return trainer


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PPO trainer with composite reward.")
    parser.add_argument("--config", default="configs/ppo/qwen35_4b_lora.yaml")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    train_ppo(args.config)


if __name__ == "__main__":
    main()
