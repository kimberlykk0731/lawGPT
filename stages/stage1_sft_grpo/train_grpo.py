"""Qwen3-8B GRPO with verifiable rewards on legal tasks.

Layout: 8× A800 → 7 GPUs train + 1 GPU vLLM rollout (or 4× H100 → 3+1).
Resumes from the SFT checkpoint produced by train_sft.py.

Defensive against trl version skew: introspects GRPOConfig at runtime and
only passes kwargs that the installed trl version actually supports. Logs
which optional features are dropped so you can match your story to what
actually trained.

Known version dependencies:
- trl ≥ 0.13: introduced GRPOConfig + GRPOTrainer
- trl 0.13/0.14: vLLM via colocate-mode, supports `vllm_device`
- trl 0.14+: `temperature` keyword
- trl 0.15+: DAPO Clip-Higher (`epsilon`, `epsilon_high`); dropped `vllm_device`
- trl 0.16+: vLLM via server-mode (run `trl vllm-serve` separately)
"""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path

import trl
from datasets import load_from_disk
from transformers import AutoTokenizer, TrainerCallback
from trl import GRPOConfig, GRPOTrainer

from stages.rewards.rlvr import flush_rlvr_metrics, legal_reward_fn


class SwanLabLegalMetricsCallback(TrainerCallback):
    """Flushes per-task F1 / parse-failure / format-reward stats to swanlab at log time.

    Goes through swanlab's wandb shim so the call shape stays `wandb.log({...})`.
    """

    def on_log(self, args, state, control, logs=None, **kwargs):
        try:
            from swanlab.integration.wandb import wandb
        except ImportError:
            return
        if wandb.run is None:
            return
        extra = flush_rlvr_metrics()
        if extra:
            extra["train/global_step"] = state.global_step
            wandb.log(extra, step=state.global_step)


def _filter_kwargs_to_signature(kwargs: dict, sig_params: dict, label: str) -> dict:
    """Return a copy of kwargs containing only keys present in sig_params; print drops."""
    accepted, dropped = {}, []
    for k, v in kwargs.items():
        if k in sig_params:
            accepted[k] = v
        else:
            dropped.append(k)
    if dropped:
        print(f"[{label}] trl=={trl.__version__} dropped unsupported args: {dropped}")
    return accepted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Injected by deepspeed launcher; not user-set.")
    parser.add_argument("--model_path", default="ckpts/legalgpt-8b-sft",
                        help="Must be the SFT checkpoint, not the base model")
    parser.add_argument("--dataset_path", type=Path, required=True,
                        help="Output of stages/data_prep.py (rlvr_demo or rlvr_demo_filtered)")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--deepspeed", default="stages/stage1_sft_grpo/configs/ds_zero3_bf16.json")
    parser.add_argument("--use_vllm", action="store_true", default=True)
    parser.add_argument("--vllm_device", default="cuda:7",
                        help="trl 0.13/0.14 only — ignored on trl 0.15+ (colocate).")
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--vllm_max_model_len", type=int, default=4096)
    parser.add_argument("--num_generations", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=float, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.04)
    parser.add_argument("--max_prompt_length", type=int, default=2048)
    parser.add_argument("--max_completion_length", type=int, default=1024)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--save_total_limit", type=int, default=3)
    # DAPO Clip-Higher: asymmetric PPO clip; trl 0.15+ supports it natively.
    # On 0.13/0.14 these are dropped by _filter_kwargs_to_signature.
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--epsilon_high", type=float, default=0.28)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--resume_from_checkpoint", default=None,
                        help='Checkpoint dir or "auto" to pick latest under --output_dir.')
    parser.add_argument("--swanlab_project", default="legalgpt-2026")
    parser.add_argument("--swanlab_run_name", default="stage1-grpo-v1")
    args = parser.parse_args()

    import os
    os.environ.setdefault("SWANLAB_PROJECT", args.swanlab_project)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    dataset = load_from_disk(str(args.dataset_path))

    def format_prompt(example: dict) -> dict:
        example["prompt"] = tokenizer.apply_chat_template(
            example["prompt"],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        return example

    dataset = dataset.map(format_prompt)

    # Build kwargs and let trl filter — version-defensive.
    grpo_sig = inspect.signature(GRPOConfig.__init__).parameters
    grpo_kwargs = _filter_kwargs_to_signature({
        # Core training (universal across trl 0.13+)
        "output_dir": str(args.output_dir),
        "num_generations": args.num_generations,
        "max_prompt_length": args.max_prompt_length,
        "max_completion_length": args.max_completion_length,
        "learning_rate": args.learning_rate,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.05,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_train_epochs": args.num_train_epochs,
        "max_grad_norm": 0.5,
        "beta": args.beta,
        "bf16": True,
        "gradient_checkpointing": True,
        "deepspeed": args.deepspeed,
        "logging_steps": args.logging_steps,
        "save_strategy": "steps",
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "report_to": ["swanlab"],
        "run_name": args.swanlab_run_name,
        # Sampling (trl 0.14+ exposes temperature)
        "temperature": args.temperature,
        # DAPO Clip-Higher (trl 0.15+)
        "epsilon": args.epsilon,
        "epsilon_high": args.epsilon_high,
        # vLLM rollout (varies by trl version)
        "use_vllm": args.use_vllm,
        "vllm_device": args.vllm_device,
        "vllm_gpu_memory_utilization": args.vllm_gpu_memory_utilization,
        "vllm_max_model_len": args.vllm_max_model_len,
    }, grpo_sig, label="train_grpo")

    config = GRPOConfig(**grpo_kwargs)

    trainer = GRPOTrainer(
        model=args.model_path,
        reward_funcs=legal_reward_fn,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=[SwanLabLegalMetricsCallback()],
    )
    resume = args.resume_from_checkpoint
    if resume == "auto":
        resume = True
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(str(args.output_dir))


if __name__ == "__main__":
    main()
