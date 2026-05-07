"""OPD 数据构建：student 自采样 + teacher 改写打分。

子命令：
- sample: 用 SFT/合并后的 student 模型对 prompt 自采样若干回答
- refine: 用强模型对 student 候选改写并保留 top-1 作为 teacher 答案

输出：
  {"prompt": ..., "student": ..., "teacher": ..., "score": float, "domain": ...}
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from legal_lm.data.distill import API_CONFIGS, DistillClient


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _sample_student(args: argparse.Namespace) -> None:
    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise ImportError("vllm is required for OPD sampling: pip install vllm") from exc

    rows = _read_jsonl(Path(args.input))
    rows = rows if args.max_samples is None else rows[: args.max_samples]
    prompts = [row.get("prompt") or row.get("instruction") or "" for row in rows]

    llm = LLM(model=args.model_path, tensor_parallel_size=args.tensor_parallel_size, dtype="bfloat16")
    sp = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
        n=args.num_samples_per_prompt,
        top_p=0.95,
    )
    outputs = llm.generate(prompts, sp)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row, output in zip(rows, outputs):
            for sample_output in output.outputs:
                handle.write(
                    json.dumps(
                        {
                            "prompt": row.get("prompt") or row.get("instruction") or "",
                            "student": sample_output.text,
                            "domain": row.get("domain", "general"),
                            "metadata": row.get("metadata", {}),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    print(f"[build_opd_dataset.sample] {len(rows)} prompts × {args.num_samples_per_prompt} samples -> {out_path}")


REFINE_PROMPT = """你是一名资深中国法律专家。下面是另一位法律 AI 对一道法律问题给出的回答草稿，
请你在保留正确部分的基础上：
1) 修正错误结论与错误法条引用
2) 补全缺失的法律推理与风险提示
3) 输出最终改写版本，保持结构化（【结论】【法律依据】【推理摘要】【风险提示】）

【问题】
{prompt}

【AI 草稿】
{student}

请给出改写后的最终回答，并在最后一行输出 SCORE=<0~1 的整体质量分>。
"""


def _refine(args: argparse.Namespace) -> None:
    client = DistillClient(api=args.api, model=args.model, temperature=0.3)
    rows = _read_jsonl(Path(args.input))

    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["prompt"], []).append(row)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    handle = out_path.open("w", encoding="utf-8")

    def _process(prompt: str, group: list[dict]) -> list[dict]:
        results: list[dict] = []
        for row in group:
            text = client.generate(REFINE_PROMPT.format(prompt=prompt, student=row["student"]))
            if not text:
                continue
            score = _extract_score(text)
            results.append(
                {
                    "prompt": prompt,
                    "student": row["student"],
                    "teacher": _strip_score_line(text),
                    "score": score,
                    "domain": row.get("domain", "general"),
                }
            )
        results.sort(key=lambda r: r["score"], reverse=True)
        return results[: args.keep_top]

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_process, prompt, group): prompt for prompt, group in grouped.items()}
            kept = 0
            for fut in as_completed(futures):
                for row in fut.result():
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    kept += 1
                handle.flush()
            print(f"[build_opd_dataset.refine] prompts={len(grouped)} kept={kept} -> {out_path}")
    finally:
        handle.close()


def _extract_score(text: str) -> float:
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.upper().startswith("SCORE="):
            try:
                return float(line.split("=", 1)[1])
            except ValueError:
                continue
    return 0.5


def _strip_score_line(text: str) -> str:
    lines = text.splitlines()
    while lines and lines[-1].strip().upper().startswith("SCORE="):
        lines.pop()
    return "\n".join(lines).rstrip()


def main() -> None:
    parser = argparse.ArgumentParser(description="OPD 数据构建")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_sample = sub.add_parser("sample", help="student 自采样")
    p_sample.add_argument("--model-path", required=True)
    p_sample.add_argument("--input", required=True)
    p_sample.add_argument("--output", required=True)
    p_sample.add_argument("--temperature", type=float, default=0.8)
    p_sample.add_argument("--num-samples-per-prompt", type=int, default=4)
    p_sample.add_argument("--max-new-tokens", type=int, default=768)
    p_sample.add_argument("--tensor-parallel-size", type=int, default=1)
    p_sample.add_argument("--max-samples", type=int, default=None)
    p_sample.set_defaults(func=_sample_student)

    p_refine = sub.add_parser("refine", help="teacher 改写 + 打分")
    p_refine.add_argument("--api", choices=list(API_CONFIGS), default="deepseek")
    p_refine.add_argument("--model", default=None)
    p_refine.add_argument("--input", required=True)
    p_refine.add_argument("--output", required=True)
    p_refine.add_argument("--workers", type=int, default=4)
    p_refine.add_argument("--keep-top", type=int, default=1)
    p_refine.set_defaults(func=_refine)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
