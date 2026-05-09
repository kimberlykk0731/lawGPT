"""Cross-stage evaluation runner for the legal RLVR test set.

Loads each model via vLLM, generates one completion per prompt at temperature 0,
scores via the RLVR reward function (set F1 per task), and dumps a JSON report.

Used by:
- Stage 1: compare SFT / GRPO / DPO checkpoints (the +Δpt headline)
- Stage 4: verify student retains 92%+ of teacher (`--retain_baseline teacher`)
- Stage 5: ablation final scores

Convenience: pass `--retain_baseline <model_path>` to print + swanlab-log
`retain_pct = student_score / teacher_score` directly. Without the flag,
the first model in `--models` is treated as the baseline.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from datasets import load_from_disk
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from stages.rewards.rlvr import legal_reward_fn


def evaluate_model(
    model_path: str,
    dataset_path: Path,
    max_tokens: int = 1024,
    gpu_memory_utilization: float = 0.85,
) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    dataset = load_from_disk(str(dataset_path))

    rendered = [
        tokenizer.apply_chat_template(
            row["prompt"], tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        )
        for row in dataset
    ]
    llm = LLM(
        model=model_path, dtype="bfloat16",
        gpu_memory_utilization=gpu_memory_utilization, max_model_len=4096,
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    outputs = llm.generate(rendered, sampling)

    rewards_per_task: dict[str, list[float]] = defaultdict(list)
    for row, prompt_text, output in zip(dataset, rendered, outputs):
        completion = output.outputs[0].text
        rewards = legal_reward_fn(
            prompts=[prompt_text],
            completions=[[{"role": "assistant", "content": completion}]],
            task=[row["task"]],
            ground_truth=[row["ground_truth"]],
        )
        rewards_per_task[row["task"]].append(rewards[0])

    summary: dict[str, float] = {}
    for task, scores in rewards_per_task.items():
        summary[f"{task}_mean"] = statistics.mean(scores)
        summary[f"{task}_n"] = len(scores)
    summary["overall_mean"] = statistics.mean(
        score for scores in rewards_per_task.values() for score in scores
    )
    return {"model": model_path, "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True,
                        help="Models to evaluate. First one is baseline unless "
                             "--retain_baseline is given.")
    parser.add_argument("--eval_dataset", type=Path, required=True,
                        help="Held-out RLVR dataset (datasets.save_to_disk format)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--retain_baseline", default=None,
                        help="Optional: explicit baseline model path. Defaults to --models[0].")
    parser.add_argument("--swanlab_project", default="legalgpt-2026")
    parser.add_argument("--swanlab_run_name", default=None,
                        help="If set, mirrors the comparison to swanlab.")
    args = parser.parse_args()

    wandb = None
    if args.swanlab_run_name:
        try:
            from swanlab.integration.wandb import wandb as _wandb
            _wandb.init(
                project=args.swanlab_project,
                name=args.swanlab_run_name,
                tags=["eval", "compare"],
                config=vars(args),
            )
            wandb = _wandb
        except ImportError:
            print("[eval] swanlab not installed; comparison printed to stdout only")

    reports: list[dict] = []
    for model_path in args.models:
        print(f"\n[eval] {model_path}")
        report = evaluate_model(model_path, args.eval_dataset, args.max_tokens)
        for key, value in report["summary"].items():
            print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
        reports.append(report)
        if wandb is not None:
            tag = Path(model_path).name
            for k, v in report["summary"].items():
                if isinstance(v, float):
                    wandb.log({f"final/{tag}/{k}": v})

    args.output.parent.mkdir(parents=True, exist_ok=True)

    baseline_path = args.retain_baseline or reports[0]["model"]
    baseline = next((r for r in reports if r["model"] == baseline_path), reports[0])

    comparison: list[dict] = []
    print(f"\n[eval] baseline = {baseline['model']}: "
          f"overall_mean = {baseline['summary']['overall_mean']:.4f}")
    for r in reports:
        if r["model"] == baseline["model"]:
            continue
        b_overall = baseline["summary"]["overall_mean"] or 1e-9
        delta = r["summary"]["overall_mean"] - baseline["summary"]["overall_mean"]
        retain_pct = r["summary"]["overall_mean"] / b_overall

        per_task_retain: dict[str, float] = {}
        for key in r["summary"]:
            if not key.endswith("_mean") or key == "overall_mean":
                continue
            b_v = baseline["summary"].get(key)
            if isinstance(b_v, float) and b_v > 1e-9:
                per_task_retain[key] = r["summary"][key] / b_v

        record = {
            "model": r["model"],
            "delta_pt": delta * 100,
            "retain_pct": retain_pct,
            "per_task_retain": per_task_retain,
        }
        comparison.append(record)
        print(f"  {r['model']}:")
        print(f"    overall_mean = {r['summary']['overall_mean']:.4f}  "
              f"Δ {delta * 100:+.2f}pt  retain = {retain_pct * 100:.2f}%")
        for k, v in per_task_retain.items():
            print(f"    {k}: retain = {v * 100:.2f}%")

        if wandb is not None:
            tag = Path(r["model"]).name
            wandb.log({
                f"compare/{tag}/delta_pt": delta * 100,
                f"compare/{tag}/retain_pct": retain_pct * 100,
            })
            for k, v in per_task_retain.items():
                wandb.log({f"compare/{tag}/retain_{k}": v * 100})

    args.output.write_text(json.dumps({
        "reports": reports,
        "baseline": baseline["model"],
        "comparison": comparison,
    }, ensure_ascii=False, indent=2))
    print(f"\n[eval] wrote {args.output}")
    if wandb is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
