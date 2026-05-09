"""Stage 4-A: Black-box distillation via teacher-generated SFT data.

Why first: pure logit KL is hard to align when student is far from teacher;
warm up the student on teacher *text* first, then do KL.

Pipeline:
  1. Teacher (legalgpt-8b-grpo) generates one completion per prompt via vLLM.
  2. Student (Qwen3-1.7B) SFTs on the (prompt, teacher-completion) corpus.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_model", default="ckpts/legalgpt-8b-grpo")
    parser.add_argument("--prompts_jsonl", type=Path, required=True,
                        help='SFT prompts JSONL with {"messages": [...]} rows')
    parser.add_argument("--output_jsonl", type=Path, required=True)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model, trust_remote_code=True)
    teacher = LLM(
        model=args.teacher_model,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=4096,
        dtype="bfloat16",
    )
    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        n=1,
    )

    prompts: list[dict] = []
    with args.prompts_jsonl.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))

    rendered = [
        tokenizer.apply_chat_template(
            row["messages"][:-1] if row["messages"][-1]["role"] == "assistant" else row["messages"],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for row in prompts
    ]
    outputs = teacher.generate(rendered, sampling)

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for row, output in zip(prompts, outputs):
            messages = row["messages"]
            if messages and messages[-1]["role"] == "assistant":
                messages = messages[:-1]
            handle.write(json.dumps({
                "messages": messages + [
                    {"role": "assistant", "content": output.outputs[0].text}
                ],
            }, ensure_ascii=False) + "\n")
            written += 1
    print(f"[blackbox] wrote {written} teacher completions -> {args.output_jsonl}")
    print("Next: run train_sft.py from stage1 with this JSONL on the Qwen3-1.7B student.")


if __name__ == "__main__":
    main()
