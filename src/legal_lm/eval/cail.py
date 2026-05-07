"""CAIL2018 三件套评测：罪名预测 / 法条推荐 / 量刑预测。

数据约定（每条 JSON）：
  {
    "fact": "...",
    "meta": {
      "accusation": ["盗窃罪"],
      "relevant_articles": [264],
      "term_of_imprisonment": {"imprisonment": 24, "death_penalty": false, "life_imprisonment": false}
    }
  }

输出：
  charge_macro_f1, article_macro_f1, sentencing_mae（月）
"""

from __future__ import annotations

import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from legal_lm.eval.report import write_reports
from legal_lm.utils import dump_jsonl


_CHARGE_PROMPT = "请判断被告人在以下案件事实中的罪名，仅输出罪名（多个用顿号分隔）。\n\n{fact}\n\n罪名："
_ARTICLE_PROMPT = "请列出以下案件适用的《刑法》法条编号（数字，多个用逗号分隔）。\n\n{fact}\n\n法条："
_SENTENCE_PROMPT = (
    "请预测以下案件被告人的量刑（如有期徒刑给出月数；若死刑或无期徒刑请直接说明）。\n\n{fact}\n\n量刑："
)


def _read_data(data_dir: Path, split: str) -> list[dict[str, Any]]:
    candidates = [data_dir / f"{split}.json", data_dir / f"{split}.jsonl"]
    for path in candidates:
        if path.exists():
            rows: list[dict[str, Any]] = []
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    rows.append(json.loads(line))
            return rows
    raise FileNotFoundError(f"CAIL split missing under {data_dir}: {split}")


def _generate(model_path: str, prompts: list[str], use_vllm: bool, max_new_tokens: int) -> list[str]:
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
        preds.append(tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
    return preds


def _normalize_charges(text: str) -> set[str]:
    pieces = re.split(r"[、,，;；\n]", text)
    return {p.strip().rstrip("罪") + "罪" for p in pieces if p.strip()}


def _normalize_articles(text: str) -> set[int]:
    nums = re.findall(r"\d+", text)
    return {int(n) for n in nums if 1 <= int(n) <= 600}


def _parse_sentence_months(text: str) -> tuple[float | None, str | None]:
    if "死刑" in text:
        return None, "death"
    if "无期" in text:
        return None, "life"
    months_match = re.search(r"(\d+)\s*(?:个)?\s*月", text)
    years_match = re.search(r"(\d+)\s*年", text)
    months = 0
    matched = False
    if years_match:
        months += int(years_match.group(1)) * 12
        matched = True
    if months_match:
        months += int(months_match.group(1))
        matched = True
    return (float(months), None) if matched else (None, None)


def _macro_f1(preds: Iterable[set], golds: Iterable[set]) -> float:
    by_label: dict[Any, list[int]] = defaultdict(lambda: [0, 0, 0])  # tp, fp, fn
    for pred, gold in zip(preds, golds):
        for label in pred & gold:
            by_label[label][0] += 1
        for label in pred - gold:
            by_label[label][1] += 1
        for label in gold - pred:
            by_label[label][2] += 1
    f1s = []
    for tp, fp, fn in by_label.values():
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        f1s.append(f1)
    return statistics.mean(f1s) if f1s else 0.0


def run_cail(
    *,
    model_path: str,
    data_dir: str | Path,
    output_dir: str | Path,
    split: str = "test",
    use_vllm: bool = True,
    max_new_tokens: int = 64,
    max_samples: int | None = None,
) -> dict[str, Any]:
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_data(data_dir, split)
    if max_samples:
        rows = rows[:max_samples]

    fact_texts = [str(r.get("fact", "")) for r in rows]

    charge_prompts = [_CHARGE_PROMPT.format(fact=f) for f in fact_texts]
    article_prompts = [_ARTICLE_PROMPT.format(fact=f) for f in fact_texts]
    sentence_prompts = [_SENTENCE_PROMPT.format(fact=f) for f in fact_texts]

    charge_preds = _generate(model_path, charge_prompts, use_vllm, max_new_tokens)
    article_preds = _generate(model_path, article_prompts, use_vllm, max_new_tokens)
    sentence_preds = _generate(model_path, sentence_prompts, use_vllm, max_new_tokens)

    pred_charges = [_normalize_charges(p) for p in charge_preds]
    pred_articles = [_normalize_articles(p) for p in article_preds]

    gold_charges = []
    gold_articles = []
    gold_months: list[float | None] = []
    gold_special: list[str | None] = []
    for row in rows:
        meta = row.get("meta", {})
        gold_charges.append({str(c).strip().rstrip("罪") + "罪" for c in meta.get("accusation", [])})
        gold_articles.append({int(a) for a in meta.get("relevant_articles", []) if isinstance(a, int)})
        sent = meta.get("term_of_imprisonment", {})
        if sent.get("death_penalty"):
            gold_months.append(None)
            gold_special.append("death")
        elif sent.get("life_imprisonment"):
            gold_months.append(None)
            gold_special.append("life")
        else:
            gold_months.append(float(sent.get("imprisonment", 0)))
            gold_special.append(None)

    pred_months: list[float | None] = []
    pred_special: list[str | None] = []
    for text in sentence_preds:
        m, s = _parse_sentence_months(text)
        pred_months.append(m)
        pred_special.append(s)

    abs_errors: list[float] = []
    for pm, gm, ps, gs in zip(pred_months, gold_months, pred_special, gold_special):
        if gs is not None or ps is not None:
            abs_errors.append(0.0 if gs == ps else 60.0)
            continue
        if pm is None or gm is None:
            abs_errors.append(60.0)
            continue
        abs_errors.append(abs(pm - gm))

    summary = {
        "charge_macro_f1": _macro_f1(pred_charges, gold_charges),
        "article_macro_f1": _macro_f1(pred_articles, gold_articles),
        "sentencing_mae": statistics.mean(abs_errors) if abs_errors else 0.0,
        "num_examples": len(rows),
    }

    detail_rows = [
        {
            "fact": fact,
            "pred_charge": list(pred_charges[i]),
            "gold_charge": list(gold_charges[i]),
            "pred_article": list(pred_articles[i]),
            "gold_article": list(gold_articles[i]),
            "pred_sentence": sentence_preds[i],
            "gold_months": gold_months[i],
            "gold_special": gold_special[i],
        }
        for i, fact in enumerate(fact_texts)
    ]

    result = {"summary": summary, "details": detail_rows, "label": "cail2018"}
    paths = write_reports(output_dir=output_dir, benchmark_result=result)
    dump_jsonl(paths["metrics"].with_name("details.jsonl"), detail_rows)
    return result


__all__ = ["run_cail"]
