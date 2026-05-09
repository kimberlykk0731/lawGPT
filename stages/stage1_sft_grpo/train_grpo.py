"""Qwen3-8B GRPO with verifiable rewards on legal tasks.

Layout: 4× H100 → 3 GPUs train + 1 GPU vLLM rollout (3× rollout speedup).
Resumes from the SFT checkpoint produced by train_sft.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Injected by deepspeed launcher; not user-set.")
    parser.add_argument("--model_path", default="ckpts/legalgpt-8b-sft",
                        help="Must be the SFT checkpoint, not the base model")
    parser.add_argument("--dataset_path", type=Path, required=True,
                        help="Output of build_rlvr_dataset.py")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--deepspeed", default="stages/stage1_sft_grpo/configs/ds_zero3_bf16.json")
    parser.add_argument("--use_vllm", action="store_true", default=True)
    parser.add_argument("--vllm_device", default="cuda:3")
    parser.add_argument("--num_generations", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=float, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.04)
    parser.add_argument("--max_prompt_length", type=int, default=2048)
    parser.add_argument("--max_completion_length", type=int, default=1024)
    # DAPO Clip-Higher: asymmetric PPO clip prevents premature policy collapse.
    # epsilon = lower bound (default 0.2), epsilon_high = upper bound (DAPO recommends 0.28).
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--epsilon_high", type=float, default=0.28)
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

    config = GRPOConfig(
        output_dir=str(args.output_dir),
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        temperature=1.0,
        top_p=0.95,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=args.num_train_epochs,
        max_grad_norm=0.5,
        beta=args.beta,
        epsilon=args.epsilon,
        epsilon_high=args.epsilon_high,
        use_vllm=args.use_vllm,
        vllm_device=args.vllm_device,
        vllm_gpu_memory_utilization=0.85,
        vllm_max_model_len=4096,
        bf16=True,
        gradient_checkpointing=True,
        deepspeed=args.deepspeed,
        logging_steps=5,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        report_to=["swanlab"],
        run_name=args.swanlab_run_name,
    )

    trainer = GRPOTrainer(
        model=args.model_path,
        reward_funcs=legal_reward_fn,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=[SwanLabLegalMetricsCallback()],
    )
    trainer.train()
    trainer.save_model(str(args.output_dir))


if __name__ == "__main__":
    main()
