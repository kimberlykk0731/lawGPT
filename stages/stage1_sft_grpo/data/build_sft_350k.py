"""Build the 350k multi-task SFT corpus.

Three steps, each callable independently:
  1. clean_cail:        filter raw CAIL by length / dedup / label sanity
  2. synthesize_multitask: weave cleaned cases into multi-task instruction templates
  3. rebalance:         square-root smoothed inverse-frequency rebalancing for long tail

Final mix targets ~350k:
  公开数据 (清洗后)      150k
  多任务联合合成         100k
  长尾重采样              50k
  外部指令 (DISC-Law)     50k
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Iterable

# === Step 1: clean public corpora =====================================================

def clean_cail(raw_path: Path) -> list[dict]:
    samples: list[dict] = []
    seen: set[str] = set()
    with raw_path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                case = json.loads(line)
            except json.JSONDecodeError:
                continue
            fact = case.get("fact", "")
            if not (50 <= len(fact) <= 2000):
                continue
            digest = hashlib.md5(fact[:200].encode("utf-8")).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            meta = case.get("meta", {})
            accusations = meta.get("accusation") or []
            if not accusations or len(accusations) > 5:
                continue
            samples.append({
                "fact": fact,
                "accusations": accusations,
                "articles": meta.get("relevant_articles", []),
                "imprisonment": meta.get("term_of_imprisonment", {}),
            })
    return samples


# === Step 2: multi-task instruction synthesis ========================================
#
# The templates below leave reasoning slots empty by design — per project goal,
# SFT data ships without CoT (model distillation is the "smart-text" pathway).
# If you later want CoT augmentation, hook in a teacher-side rewriter at the
# {fact_summary}/{reasoning} slots before formatting.

MULTI_TASK_TEMPLATES = [
    {
        "name": "triple_prediction",
        "system": "你是资深刑事法官，请按以下步骤分析案情。",
        "user": "案情描述：\n{fact}\n\n请输出：\n1. 涉及罪名（JSON 数组）\n2. 适用法条（《刑法》第X条）\n3. 量刑建议（月数）",
        "assistant": "1. 涉及罪名：{accusations_json}\n2. 适用法条：{articles_str}\n3. 量刑建议：{imprisonment_months} 个月",
    },
    {
        "name": "syllogism_cot",
        "system": "你是法律专家，请用三段论分析案情。",
        "user": "案情：{fact}\n请进行三段论分析。",
        "assistant": "**大前提**：{articles_str}\n**小前提**：{fact_summary}\n**结论**：{accusations_json}",
    },
    {
        "name": "consult_dialog",
        "system": "你是法律咨询助手。",
        "user": "我朋友涉嫌这种行为：{fact}\n会构成什么罪？大概会判多久？",
        "assistant": "根据您描述的情况，可能涉及：\n- 罪名：{accusations_natural}\n- 量刑参考：{imprisonment_natural}\n建议尽快咨询专业律师。",
    },
]


def _imprisonment_months(case: dict) -> int:
    impr = case.get("imprisonment") or {}
    if isinstance(impr, dict):
        return int(impr.get("imprisonment", 0))
    return int(impr or 0)


def _format_template(case: dict, template: dict) -> dict:
    accusations = case["accusations"]
    articles = case.get("articles", [])
    months = _imprisonment_months(case)
    fields = {
        "fact": case["fact"],
        "accusations_json": json.dumps(accusations, ensure_ascii=False),
        "accusations_natural": "、".join(accusations),
        "articles_str": "、".join(f"《刑法》第{a}条" for a in articles) or "（未提供）",
        "fact_summary": case["fact"][:120].rstrip() + "...",
        "imprisonment_months": months,
        "imprisonment_natural": f"约 {months} 个月" if months else "需结合情节认定",
    }
    return {
        "messages": [
            {"role": "system", "content": template["system"]},
            {"role": "user", "content": template["user"].format(**fields)},
            {"role": "assistant", "content": template["assistant"].format(**fields)},
        ],
        "template": template["name"],
    }


def synthesize_multitask(cases: Iterable[dict], n_target: int, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    cases_list = list(cases)
    out: list[dict] = []
    while len(out) < n_target and cases_list:
        case = rng.choice(cases_list)
        template = rng.choice(MULTI_TASK_TEMPLATES)
        out.append(_format_template(case, template))
    return out


# === Step 3: rebalance long tail =====================================================

def rebalance_by_inverse_frequency(
    cases: list[dict],
    n_target: int,
    alpha: float = 0.5,
    augment: Callable[[dict], dict] | None = None,
    seed: int = 42,
) -> list[dict]:
    """alpha=0 keeps original distribution, alpha=1 fully balances; 0.5 = sqrt smoothing."""
    rng = random.Random(seed)
    counts = Counter(case["accusations"][0] for case in cases if case.get("accusations"))
    weights = {crime: (1.0 / count) ** alpha for crime, count in counts.items()}
    z = sum(weights.values()) or 1.0
    target = {crime: max(1, int(n_target * w / z)) for crime, w in weights.items()}

    by_crime: dict[str, list[dict]] = defaultdict(list)
    for case in cases:
        if case.get("accusations"):
            by_crime[case["accusations"][0]].append(case)

    rebalanced: list[dict] = []
    for crime, n in target.items():
        pool = by_crime.get(crime, [])
        if not pool:
            continue
        if len(pool) >= n:
            rebalanced.extend(rng.sample(pool, n))
            continue
        rebalanced.extend(pool)
        for _ in range(n - len(pool)):
            base = rng.choice(pool)
            rebalanced.append(augment(base) if augment else base)
    return rebalanced


# === CLI =============================================================================

def _write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cail-raw", type=Path, required=True)
    parser.add_argument("--external-instruction-jsonl", type=Path, default=None,
                        help="Optional external instruction set (e.g. DISC-Law) appended verbatim")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-clean", type=int, default=150_000)
    parser.add_argument("--n-multitask", type=int, default=100_000)
    parser.add_argument("--n-rebalance", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    cleaned = clean_cail(args.cail_raw)
    rng.shuffle(cleaned)
    print(f"[step1] cleaned CAIL: {len(cleaned)}")

    clean_subset = cleaned[: args.n_clean]
    multitask = synthesize_multitask(cleaned, args.n_multitask, seed=args.seed)
    rebalanced = rebalance_by_inverse_frequency(cleaned, args.n_rebalance, seed=args.seed)

    output_dir = args.output_dir
    n_clean = _write_jsonl(output_dir / "clean.jsonl", clean_subset)
    n_multi = _write_jsonl(output_dir / "multitask.jsonl", multitask)
    n_rebal = _write_jsonl(output_dir / "rebalanced.jsonl", rebalanced)

    n_ext = 0
    if args.external_instruction_jsonl:
        with args.external_instruction_jsonl.open(encoding="utf-8") as src, \
             (output_dir / "external.jsonl").open("w", encoding="utf-8") as dst:
            for line in src:
                if line.strip():
                    dst.write(line)
                    n_ext += 1

    print(f"[done] clean={n_clean} multitask={n_multi} rebalanced={n_rebal} "
          f"external={n_ext} total={n_clean + n_multi + n_rebal + n_ext}")


if __name__ == "__main__":
    main()
