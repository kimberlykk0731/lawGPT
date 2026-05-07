from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer

from legal_lm.data.prompting import DEFAULT_SYSTEM_PROMPT, build_messages
from legal_lm.data.registry import load_samples
from legal_lm.data.schema import LegalSample
from legal_lm.utils import dump_jsonl, extract_articles, token_overlap_ratio


def _safe_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _structure_score(text: str) -> float:
    headings = ["【结论】", "【法律依据】", "【推理摘要】", "【风险提示】"]
    return sum(1 for item in headings if item in text) / len(headings)


def _citation_score(prediction: str, sample: LegalSample) -> float:
    pred_articles = set(extract_articles(prediction))
    gold_articles = set(extract_articles(" ".join(sample.citations)))
    if not gold_articles:
        gold_articles = set(extract_articles(sample.gold_answer))
    if not pred_articles and not gold_articles:
        return 1.0
    if not pred_articles or not gold_articles:
        return 0.0
    return len(pred_articles & gold_articles) / len(pred_articles | gold_articles)


def evaluate_prediction(sample: LegalSample, prediction: str) -> dict[str, Any]:
    answer_overlap = token_overlap_ratio(prediction, sample.gold_answer)
    reasoning_overlap = token_overlap_ratio(prediction, sample.brief_reasoning or sample.gold_reasoning)
    citation_score = _citation_score(prediction, sample)
    structure_score = _structure_score(prediction)
    overall = (
        0.35 * answer_overlap
        + 0.25 * reasoning_overlap
        + 0.25 * citation_score
        + 0.15 * structure_score
    )
    return {
        "id": sample.sample_id,
        "domain": sample.domain,
        "task_type": sample.task_type,
        "prediction": prediction,
        "gold_answer": sample.gold_answer,
        "gold_reasoning": sample.brief_reasoning or sample.gold_reasoning,
        "metrics": {
            "answer_overlap": answer_overlap,
            "reasoning_overlap": reasoning_overlap,
            "citation_score": citation_score,
            "structure_score": structure_score,
            "overall": overall,
        },
    }


def aggregate_results(details: list[dict[str, Any]]) -> dict[str, Any]:
    metric_names = ["answer_overlap", "reasoning_overlap", "citation_score", "structure_score", "overall"]
    overall = {
        name: _safe_mean([item["metrics"][name] for item in details])
        for name in metric_names
    }
    by_domain: dict[str, dict[str, float]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in details:
        grouped[item["domain"]].append(item)
    for domain, rows in grouped.items():
        by_domain[domain] = {
            name: _safe_mean([item["metrics"][name] for item in rows])
            for name in metric_names
        }
    return {"overall": overall, "by_domain": by_domain, "num_examples": len(details)}


def load_prediction_map(predictions_file: str | Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    path = Path(predictions_file)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            import json

            row = json.loads(line)
            prediction = row.get("prediction") or row.get("response") or row.get("output") or ""
            mapping[str(row.get("id") or row.get("sample_id"))] = str(prediction)
    return mapping


def _load_inference_model(model_name_or_path: str):
    try:
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
        )
        tokenizer_source = model.peft_config["default"].base_model_name_or_path
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16)
        tokenizer_source = model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


def generate_predictions(
    samples: list[LegalSample],
    model_name_or_path: str,
    max_new_tokens: int = 768,
    temperature: float = 0.1,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> list[dict[str, str]]:
    model, tokenizer = _load_inference_model(model_name_or_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    rows: list[dict[str, str]] = []

    for sample in samples:
        messages = build_messages(sample, mode="brief_reasoning", system_prompt=system_prompt)[:-1]
        if hasattr(tokenizer, "apply_chat_template"):
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt_text = f"{system_prompt}\n\n{messages[-1]['content']}\n\n请给出分析。"

        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0.0,
                temperature=max(temperature, 1e-5),
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        output_ids = generated[0][inputs["input_ids"].shape[1] :]
        prediction = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        rows.append({"id": sample.sample_id, "prediction": prediction})
    return rows


def run_benchmark(
    dataset_config_path: str | Path,
    dataset_name: str,
    split: str,
    predictions_file: str | Path | None = None,
    model_name_or_path: str | None = None,
    max_new_tokens: int = 768,
    temperature: float = 0.1,
    output_predictions_file: str | Path | None = None,
) -> dict[str, Any]:
    samples = load_samples(dataset_config_path, dataset_name, split)
    if predictions_file:
        prediction_map = load_prediction_map(predictions_file)
        prediction_rows = [{"id": sample.sample_id, "prediction": prediction_map.get(sample.sample_id, "")} for sample in samples]
    elif model_name_or_path:
        prediction_rows = generate_predictions(
            samples=samples,
            model_name_or_path=model_name_or_path,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        if output_predictions_file:
            dump_jsonl(output_predictions_file, prediction_rows)
    else:
        raise ValueError("Either predictions_file or model_name_or_path must be provided.")

    prediction_map = {row["id"]: row["prediction"] for row in prediction_rows}
    details = [evaluate_prediction(sample, prediction_map.get(sample.sample_id, "")) for sample in samples]
    return {"summary": aggregate_results(details), "details": details}
