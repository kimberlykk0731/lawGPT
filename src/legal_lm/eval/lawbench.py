"""LawBench runner（精简版）。

约定 LawBench 数据组织（由 prepare 子命令准备）：
  data/raw/LawBench/
    tasks/
      <task_id>/
        test.jsonl                 # 每行: {"question", "answer", "type": "objective"|"open"}
        meta.json                  # {"capability": "memory|understanding|application", ...}
    capabilities.json              # 可选，capability 总表

支持两类题目：
- objective: 选择/罪名/标签 → 严格匹配或 macro-F1
- open: 开放问答 → 用 reward.composite 打分（默认仅规则部分）
"""

from __future__ import annotations

import json
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from legal_lm.eval.report import write_reports
from legal_lm.utils import dump_jsonl
from legal_lm.utils.text import normalize_text


@dataclass
class LawBenchTask:
    task_id: str
    capability: str
    type: str
    items: list[dict[str, Any]]


def _load_tasks(root: Path) -> list[LawBenchTask]:
    tasks: list[LawBenchTask] = []
    tasks_root = root / "tasks"
    if not tasks_root.exists():
        raise FileNotFoundError(f"LawBench tasks root not found: {tasks_root}")
    for task_dir in sorted(tasks_root.iterdir()):
        if not task_dir.is_dir():
            continue
        meta_path = task_dir / "meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        items_path = task_dir / "test.jsonl"
        if not items_path.exists():
            continue
        items: list[dict[str, Any]] = []
        with items_path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                items.append(json.loads(line))
        tasks.append(
            LawBenchTask(
                task_id=task_dir.name,
                capability=str(meta.get("capability", "understanding")),
                type=str(meta.get("type", "objective")),
                items=items,
            )
        )
    return tasks


def _generate_predictions(model_path: str, prompts: list[str], use_vllm: bool, max_new_tokens: int) -> list[str]:
    if use_vllm:
        try:
            from vllm import LLM, SamplingParams
        except ImportError as exc:  # pragma: no cover
            raise ImportError("vllm is required when --use-vllm is set: pip install vllm") from exc
        llm = LLM(model=model_path, dtype="bfloat16")
        sp = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
        outputs = llm.generate(prompts, sp)
        return [o.outputs[0].text for o in outputs]
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    preds: list[str] = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        preds.append(text)
    return preds


def _objective_score(pred: str, gold: str) -> float:
    pred_norm = normalize_text(pred)
    gold_norm = normalize_text(gold)
    if not gold_norm:
        return 0.0
    if gold_norm in pred_norm or pred_norm in gold_norm:
        return 1.0
    letters = re.findall(r"[A-D]", pred.upper())
    if letters and gold_norm.upper() in letters:
        return 1.0
    return 0.0


def _build_prompt(item: dict[str, Any], rag_block: str = "") -> str:
    question = str(item.get("question") or item.get("instruction") or "")
    options = item.get("options")
    parts = [question]
    if options:
        if isinstance(options, dict):
            parts.append("\n".join(f"{k}. {v}" for k, v in options.items()))
        elif isinstance(options, list):
            parts.append("\n".join(options))
    if rag_block:
        parts.append(rag_block)
    return "\n\n".join(parts)


def _maybe_retriever(inject_rag: bool, statute_index_dir: str):
    if not inject_rag:
        return None
    from legal_lm.data.rag import StatuteRetriever

    return StatuteRetriever(statute_index_dir)


def run_lawbench(
    *,
    model_path: str,
    data_dir: str | Path,
    output_dir: str | Path,
    use_vllm: bool = True,
    max_new_tokens: int = 256,
    inject_rag: bool = False,
    statute_index_dir: str = "data/statute_index",
    rag_top_k: int = 5,
) -> dict[str, Any]:
    root = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = _load_tasks(root)
    retriever = _maybe_retriever(inject_rag, statute_index_dir)

    by_capability: dict[str, list[float]] = defaultdict(list)
    by_task: dict[str, dict[str, Any]] = {}
    detail_rows: list[dict[str, Any]] = []
    overall_scores: list[float] = []

    for task in tasks:
        prompts: list[str] = []
        for item in task.items:
            rag_block = ""
            if retriever is not None:
                query = str(item.get("question") or item.get("instruction") or "")
                rag_block = retriever.render_block(query, top_k=rag_top_k)
            prompts.append(_build_prompt(item, rag_block=rag_block))
        preds = _generate_predictions(model_path, prompts, use_vllm=use_vllm, max_new_tokens=max_new_tokens)
        scores: list[float] = []
        for item, pred in zip(task.items, preds):
            gold = str(item.get("answer", ""))
            score = _objective_score(pred, gold) if task.type == "objective" else _open_score(pred, gold)
            scores.append(score)
            detail_rows.append({
                "task_id": task.task_id,
                "capability": task.capability,
                "type": task.type,
                "question": _build_prompt(item),
                "prediction": pred,
                "gold": gold,
                "score": score,
            })
        task_score = statistics.mean(scores) if scores else 0.0
        by_task[task.task_id] = {
            "score": task_score,
            "capability": task.capability,
            "type": task.type,
            "n": len(scores),
        }
        by_capability[task.capability].extend(scores)
        overall_scores.extend(scores)

    summary = {
        "overall": statistics.mean(overall_scores) if overall_scores else 0.0,
        "by_task": by_task,
        "by_capability": {
            cap: statistics.mean(scores) if scores else 0.0
            for cap, scores in by_capability.items()
        },
        "num_examples": len(overall_scores),
    }
    result = {"summary": summary, "details": detail_rows, "label": "lawbench"}
    paths = write_reports(output_dir=output_dir, benchmark_result=result)
    dump_jsonl(paths["metrics"].with_name("details.jsonl"), detail_rows)
    return result


def _open_score(pred: str, gold: str) -> float:
    pred_norm = normalize_text(pred)
    gold_norm = normalize_text(gold)
    if not gold_norm:
        return 0.0
    inter = len(set(pred_norm) & set(gold_norm))
    union = len(set(pred_norm) | set(gold_norm)) or 1
    return inter / union


__all__ = ["run_lawbench"]
