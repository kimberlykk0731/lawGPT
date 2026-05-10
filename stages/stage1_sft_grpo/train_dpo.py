"""DPO baseline for the GRPO comparison ablation.

Project goal claims: GRPO 较 DPO +4.2pt on crime-prediction. To produce that
number you need a DPO checkpoint trained from the same SFT init on a preference
dataset derived from the same RLVR samples (sample two completions per prompt;
the higher-F1 one wins).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import Dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

from stages.rewards.rlvr import legal_reward_fn


def build_preference_pairs(
    rlvr_dataset_path: Path,
    sft_model_path: str,
    output_path: Path,
    n_per_prompt: int = 4,
) -> Path:
    """Sample N completions per RLVR prompt with the SFT model, score with the RLVR
    reward, and pair best vs. worst as (chosen, rejected). Persists JSONL."""
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(sft_model_path, trust_remote_code=True)
    base = load_from_disk(str(rlvr_dataset_path))
    prompts_text = [
        tokenizer.apply_chat_template(row["prompt"], tokenize=False, add_generation_prompt=True,
                                      enable_thinking=False)
        for row in base
    ]
    llm = LLM(model=sft_model_path, dtype="bfloat16", gpu_memory_utilization=0.85)
    sampling = SamplingParams(temperature=1.0, top_p=0.95, max_tokens=1024, n=n_per_prompt)
    completions = llm.generate(prompts_text, sampling)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for row, prompt_text, gen in zip(base, prompts_text, completions):
            texts = [out.text for out in gen.outputs]
            scores = legal_reward_fn(
                prompts=[prompt_text] * len(texts),
                completions=[[{"role": "assistant", "content": t}] for t in texts],
                task=[row["task"]] * len(texts),
                ground_truth=[row["ground_truth"]] * len(texts),
            )
            best_i, worst_i = max(range(len(scores)), key=scores.__getitem__), \
                              min(range(len(scores)), key=scores.__getitem__)
            if scores[best_i] - scores[worst_i] < 0.1:
                continue
            handle.write(json.dumps({
                "prompt": prompt_text,
                "chosen": texts[best_i],
                "rejected": texts[worst_i],
            }, ensure_ascii=False) + "\n")
            written += 1
    print(f"[build_preference] wrote {written} pairs -> {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Injected by deepspeed launcher; not user-set.")
    parser.add_argument("--build-preferences", dest="build_preferences", action="store_true",
                        help="Pre-step: sample N completions per RLVR prompt and emit "
                             "a (chosen, rejected) preference JSONL. Then exit.")
    parser.add_argument("--rlvr_dataset", type=Path, default=None,
                        help="Required when --build-preferences is set; output of "
                             "stages/data_prep.py (rlvr_demo).")
    parser.add_argument("--n_per_prompt", type=int, default=4,
                        help="--build-preferences only: rollouts per prompt for ranking.")
    parser.add_argument("--model_path", default="ckpts/legalgpt-8b-sft")
    parser.add_argument("--preference_jsonl", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=None,
                        help="Required for training (skipped when --build-preferences).")
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--learning_rate", type=float, default=5e-7)
    parser.add_argument("--num_train_epochs", type=float, default=3)
    parser.add_argument("--deepspeed", default="stages/stage1_sft_grpo/configs/ds_zero3_bf16.json")
    parser.add_argument("--resume_from_checkpoint", default=None,
                        help='Checkpoint dir or "auto" to pick latest under --output_dir.')
    parser.add_argument("--swanlab_project", default="legalgpt-2026")
    parser.add_argument("--swanlab_run_name", default="stage1-dpo-baseline")
    args = parser.parse_args()

    if args.build_preferences:
        if args.rlvr_dataset is None:
            parser.error("--rlvr_dataset is required when --build-preferences is set")
        build_preference_pairs(
            rlvr_dataset_path=args.rlvr_dataset,
            sft_model_path=args.model_path,
            output_path=args.preference_jsonl,
            n_per_prompt=args.n_per_prompt,
        )
        return

    if args.output_dir is None:
        parser.error("--output_dir is required when training (omit only with --build-preferences)")

    import os
    os.environ.setdefault("SWANLAB_PROJECT", args.swanlab_project)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, trust_remote_code=True, torch_dtype="bfloat16",
    )

    rows = [json.loads(line) for line in args.preference_jsonl.open(encoding="utf-8") if line.strip()]
    dataset = Dataset.from_list(rows)

    config = DPOConfig(
        output_dir=str(args.output_dir),
        beta=args.beta,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        bf16=True,
        gradient_checkpointing=True,
        deepspeed=args.deepspeed,
        logging_steps=5,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        report_to=["swanlab"],
        run_name=args.swanlab_run_name,
    )

    trainer = DPOTrainer(
        model=model,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    resume = args.resume_from_checkpoint
    if resume == "auto":
        resume = True
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(str(args.output_dir))


if __name__ == "__main__":
    main()
