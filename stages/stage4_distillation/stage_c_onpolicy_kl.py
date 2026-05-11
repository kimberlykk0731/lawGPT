"""Stage 4-C: On-policy KL distillation.

Loop:
  1. Student rolls out a completion (sample, not greedy — greedy collapses).
  2. Both teacher and student forward on the student's rollout.
  3. forward KL(student || teacher) over the *generated* tokens only.

Memory note: teacher (16GB Qwen3-8B bf16) + student (3.4GB Qwen3-1.7B) +
optimizer (~13GB) ≈ 33GB peak — fits on a single H100, no ZeRO-3 needed.

Curriculum: ramp max_new_tokens 128 → 256 → 512 across epochs to avoid
training on student-noise dominated trajectories early on.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


class PromptDataset(Dataset):
    def __init__(self, path: Path, tokenizer, max_length: int):
        rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        text = self.tokenizer.apply_chat_template(
            self.rows[idx]["messages"],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        encoded = self.tokenizer(text, truncation=True, max_length=self.max_length,
                                 return_tensors="pt", padding="max_length")
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }


def curriculum_max_new_tokens(epoch: int, schedule: list[int]) -> int:
    return schedule[min(epoch, len(schedule) - 1)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_model", default="ckpts/legalgpt-8b-grpo")
    parser.add_argument("--student_model", default="ckpts/student-logits")
    parser.add_argument("--prompts_jsonl", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--prompt_max_length", type=int, default=1024)
    parser.add_argument("--curriculum", type=int, nargs="+", default=[128, 256, 512])
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--per_device_batch_size", type=int, default=2)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--swanlab_project", default="legalgpt-2026")
    parser.add_argument("--swanlab_run_name", default="stage4c-onpolicy-kl")
    args = parser.parse_args()

    try:
        import swanlab  # noqa: F401
        from stages._swanlab_shim import wandb
        wandb.init(
            project=args.swanlab_project,
            name=args.swanlab_run_name,
            tags=["stage4", "distill", "onpolicy"],
            config=vars(args),
        )
    except ImportError:
        wandb = None  # type: ignore[assignment]

    tokenizer = AutoTokenizer.from_pretrained(args.student_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher_model, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).cuda().eval()
    student = AutoModelForCausalLM.from_pretrained(
        args.student_model, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).cuda()
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.learning_rate)

    dataset = PromptDataset(args.prompts_jsonl, tokenizer, args.prompt_max_length)
    loader = DataLoader(dataset, batch_size=args.per_device_batch_size, shuffle=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    for epoch in range(args.num_train_epochs):
        max_new_tokens = curriculum_max_new_tokens(epoch, args.curriculum)
        print(f"[epoch {epoch}] max_new_tokens={max_new_tokens}")

        for batch in loader:
            prompts = batch["input_ids"].cuda()
            prompt_len = prompts.size(1)

            student.eval()
            with torch.no_grad():
                rollout = student.generate(
                    prompts,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=0.95,
                    pad_token_id=tokenizer.pad_token_id,
                )
            student.train()

            student_logits = student(rollout).logits
            with torch.no_grad():
                teacher_logits = teacher(rollout).logits

            student_gen = student_logits[:, prompt_len - 1:-1, :]
            teacher_gen = teacher_logits[:, prompt_len - 1:-1, :]

            T = args.temperature
            student_log_probs = F.log_softmax(student_gen / T, dim=-1)
            teacher_probs = F.softmax(teacher_gen / T, dim=-1)
            loss = F.kl_div(
                student_log_probs, teacher_probs, reduction="batchmean",
            ) * (T ** 2)

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1
            if global_step % args.logging_steps == 0:
                with torch.no_grad():
                    rollout_lengths = (rollout != tokenizer.pad_token_id).sum(-1).float()
                    teacher_full = F.softmax(teacher_gen.float(), dim=-1)
                    student_full = F.softmax(student_gen.float(), dim=-1)
                    teacher_entropy = -(teacher_full * teacher_full.clamp(min=1e-12).log()).sum(-1).mean()
                    student_entropy = -(student_full * student_full.clamp(min=1e-12).log()).sum(-1).mean()
                    teacher_logprob = F.log_softmax(teacher_gen.float(), dim=-1).gather(
                        -1, rollout[:, prompt_len:].unsqueeze(-1)
                    ).mean()
                    agreement = (student_gen.argmax(-1) == teacher_gen.argmax(-1)).float().mean()

                metrics = {
                    "loss/forward_kl": loss.item(),
                    "rollout/length_mean": rollout_lengths.mean().item(),
                    "rollout/length_max": rollout_lengths.max().item(),
                    "rollout/curriculum_max": float(max_new_tokens),
                    "rollout/teacher_logprob_mean": teacher_logprob.item(),
                    "kd/teacher_entropy_on_rollout": teacher_entropy.item(),
                    "kd/student_entropy_on_rollout": student_entropy.item(),
                    "kd/agreement_top1": agreement.item(),
                    "actor/grad_norm": grad_norm.item(),
                    "train/global_step": global_step,
                }
                print(f"step {global_step} loss={loss.item():.4f} agree={agreement.item():.3f}")
                if wandb is not None and wandb.run is not None:
                    wandb.log(metrics, step=global_step)

        student.save_pretrained(str(args.output_dir / f"epoch_{epoch}"))

    student.save_pretrained(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))


if __name__ == "__main__":
    main()
