"""DAPO Dynamic Sampling — offline pre-filter for RLVR dataset.

For each prompt, sample N completions from the SFT model via vLLM and compute
RLVR rewards. Drop prompts where the spread of rewards is below a threshold —
under GRPO group baseline these have advantage ≈ 0 and waste compute. Keeping
only "informative" prompts is the cherry-picked half of DAPO that fits this task
without rewriting the trainer.

Run once *after* SFT and *before* GRPO. train_grpo.py points to the filtered
dataset.
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

from datasets import Dataset, load_from_disk
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from stages.rewards.rlvr import legal_reward_fn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="ckpts/legalgpt-8b-sft",
                        help="Initial policy used to estimate per-prompt reward spread")
    parser.add_argument("--dataset_path", type=Path, required=True,
                        help="Output of stages/data_prep.py (rlvr_demo)")
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--n_rollouts", type=int, default=8,
                        help="Should match GRPOConfig.num_generations")
    parser.add_argument("--min_spread", type=float, default=0.1,
                        help="Drop prompts where max(reward) - min(reward) < this")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--report_path", type=Path, default=None,
                        help="Optional JSON dump of per-prompt spread for diagnosis")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    dataset = load_from_disk(str(args.dataset_path))

    rendered = [
        tokenizer.apply_chat_template(
            row["prompt"], tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        )
        for row in dataset
    ]
    llm = LLM(
        model=args.model_path,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=4096,
    )
    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        n=args.n_rollouts,
    )
    outputs = llm.generate(rendered, sampling)

    kept_rows: list[dict] = []
    spread_stats: list[dict] = []
    for row, prompt_text, gen in zip(dataset, rendered, outputs):
        completion_texts = [out.text for out in gen.outputs]
        rewards = legal_reward_fn(
            prompts=[prompt_text] * len(completion_texts),
            completions=[[{"role": "assistant", "content": t}] for t in completion_texts],
            task=[row["task"]] * len(completion_texts),
            ground_truth=[row["ground_truth"]] * len(completion_texts),
        )
        spread = max(rewards) - min(rewards)
        spread_stats.append({
            "task": row["task"],
            "spread": spread,
            "min": min(rewards),
            "max": max(rewards),
            "mean": sum(rewards) / len(rewards),
        })
        if spread >= args.min_spread:
            kept_rows.append(dict(row))

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(kept_rows).save_to_disk(str(args.output_path))

    avg_spread = statistics.mean(s["spread"] for s in spread_stats) if spread_stats else 0.0
    pct_kept = 100 * len(kept_rows) / max(len(dataset), 1)
    by_task: dict[str, list[float]] = {}
    for stat in spread_stats:
        by_task.setdefault(stat["task"], []).append(stat["spread"])

    print(f"[dyn_sampling] kept {len(kept_rows)}/{len(dataset)} ({pct_kept:.1f}%); "
          f"avg spread {avg_spread:.3f}; threshold {args.min_spread}")
    for task, spreads in by_task.items():
        kept_task = sum(1 for s in spreads if s >= args.min_spread)
        print(f"  {task}: kept {kept_task}/{len(spreads)} "
              f"({100 * kept_task / len(spreads):.1f}%) avg_spread={statistics.mean(spreads):.3f}")

    if args.report_path:
        import json
        args.report_path.parent.mkdir(parents=True, exist_ok=True)
        args.report_path.write_text(json.dumps({
            "config": {
                "model_path": args.model_path,
                "n_rollouts": args.n_rollouts,
                "min_spread": args.min_spread,
            },
            "summary": {
                "kept": len(kept_rows),
                "total": len(dataset),
                "kept_pct": pct_kept,
                "avg_spread": avg_spread,
            },
            "by_task": {
                task: {
                    "n": len(spreads),
                    "kept": sum(1 for s in spreads if s >= args.min_spread),
                    "avg_spread": statistics.mean(spreads),
                } for task, spreads in by_task.items()
            },
        }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
