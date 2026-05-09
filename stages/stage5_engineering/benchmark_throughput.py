"""vLLM throughput benchmark.

Used to verify: 蒸馏后的 1.7B 学生在单 A10 上吞吐相对 8B 教师 6×。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from vllm import LLM, SamplingParams


def load_test_prompts(path: Path, n: int) -> list[str]:
    rows: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "prompt" in row:
                rows.append(row["prompt"] if isinstance(row["prompt"], str) else json.dumps(row["prompt"]))
            elif "messages" in row:
                user_turns = [m["content"] for m in row["messages"] if m["role"] == "user"]
                if user_turns:
                    rows.append(user_turns[-1])
            if len(rows) >= n:
                break
    return rows[:n]


def benchmark(model_path: str, prompts: list[str], batch_size: int = 64,
              max_tokens: int = 256) -> dict:
    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        max_model_len=2048,
        dtype="bfloat16",
        max_num_seqs=batch_size,
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=max_tokens)

    start = time.time()
    outputs = llm.generate(prompts, sampling)
    elapsed = time.time() - start

    total_output_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    return {
        "model": model_path,
        "n_prompts": len(prompts),
        "elapsed_s": elapsed,
        "tokens_per_s": total_output_tokens / elapsed,
        "req_per_s": len(prompts) / elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True,
                        help="One or more model paths to benchmark sequentially")
    parser.add_argument("--prompts_jsonl", type=Path, required=True)
    parser.add_argument("--n_prompts", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_tokens", type=int, default=256)
    args = parser.parse_args()

    prompts = load_test_prompts(args.prompts_jsonl, args.n_prompts)
    print(f"[bench] loaded {len(prompts)} prompts")

    results: list[dict] = []
    for model in args.models:
        print(f"\n[bench] starting {model}")
        result = benchmark(model, prompts, args.batch_size, args.max_tokens)
        results.append(result)
        print(f"  tokens/s: {result['tokens_per_s']:.1f}  req/s: {result['req_per_s']:.2f}")

    if len(results) >= 2:
        baseline = results[0]
        print("\n[bench] speedup vs first model:")
        for r in results[1:]:
            speedup = r["tokens_per_s"] / baseline["tokens_per_s"]
            print(f"  {r['model']}: {speedup:.2f}x")


if __name__ == "__main__":
    main()
