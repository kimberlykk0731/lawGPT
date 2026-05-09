"""Build RLVR (verifiable-reward) GRPO dataset.

Produces (prompt, task, ground_truth) triples for two tasks:
- crime_prediction: predict the set of charges from the case facts
- contract_review:  predict the set of risk labels from a contract clause

Both ground_truth fields are list[str], scored programmatically via JSON F1.
20k samples (1:1 mix) is the sweet spot — beyond that GRPO starts overfitting
the reward function rather than learning the underlying skill.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable

from datasets import Dataset

CRIME_SYSTEM = (
    "你是法律专家。请根据案情描述给出涉及的罪名，"
    '用 JSON 数组格式输出，例如 ["盗窃罪", "故意伤害罪"]。'
)
CONTRACT_SYSTEM = (
    "你是合同审查专家。请识别下面合同条款中的风险点，"
    '用 JSON 数组格式输出每个风险点的类型，例如 ["违约责任过轻", "管辖约定不明"]。'
)


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_crime_samples(rows: Iterable[dict]) -> list[dict]:
    out: list[dict] = []
    for row in rows:
        charges = row.get("accusations") or row.get("charges") or []
        fact = row.get("fact") or row.get("fact_description")
        if not fact or not charges:
            continue
        out.append({
            "prompt": [
                {"role": "system", "content": CRIME_SYSTEM},
                {"role": "user", "content": fact},
            ],
            "task": "crime_prediction",
            "ground_truth": {"crimes": list(map(str, charges))},
        })
    return out


def build_contract_samples(rows: Iterable[dict]) -> list[dict]:
    out: list[dict] = []
    for row in rows:
        clause = row.get("clause_text") or row.get("clause")
        risks = row.get("risk_labels") or row.get("risks") or []
        if not clause or not risks:
            continue
        out.append({
            "prompt": [
                {"role": "system", "content": CONTRACT_SYSTEM},
                {"role": "user", "content": clause},
            ],
            "task": "contract_review",
            "ground_truth": {"risks": list(map(str, risks))},
        })
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--crime-jsonl", type=Path, required=True,
                        help="Cleaned criminal cases JSONL (fact + accusations)")
    parser.add_argument("--contract-jsonl", type=Path, required=True,
                        help="Cleaned contract clauses JSONL (clause_text + risk_labels)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--total", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    crime = build_crime_samples(_read_jsonl(args.crime_jsonl))
    contract = build_contract_samples(_read_jsonl(args.contract_jsonl))
    samples = crime + contract
    random.shuffle(samples)
    samples = samples[: args.total]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(samples).save_to_disk(str(args.output_dir))
    print(f"[build_rlvr] crime={len(crime)} contract={len(contract)} "
          f"kept={len(samples)} -> {args.output_dir}")


if __name__ == "__main__":
    main()
