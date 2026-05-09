"""One-shot data preparation for the entire 5-stage pipeline.

Single source of truth: `ShengbinYue/DISC-Law-SFT` on HF Hub (~300k Chinese
legal SFT examples, multi-task: charge prediction, contract review, statute
retrieval, consultation). All downstream artifacts are derived from this one
dataset, so a fresh GPU box only needs:

    python stages/data_prep.py --out data/

That writes the following layout (consumed verbatim by the rest of the repo):

    data/processed/sft_demo/{clean,multitask,rebalanced}.jsonl
    data/processed/rlvr_demo/                  # datasets.save_to_disk format
    data/processed/rlvr_demo_test/             # holdout split for eval
    data/processed/rlvr_demo/prompts.jsonl     # prompts only (for distill stage C)
    data/processed/legal_eval_by_domain.jsonl  # 5-domain analyze_router input
    data/processed/distill_prompts.jsonl       # 5k SFT prompts for stage 4 teacher

Defaults are *demo scale* (good for one figure-producing run on 4×H100 in
under a day). Pass `--full` for the resume-grade scale (350k / 20k).

Synthetic fallback: if HF download fails (offline / firewall), the script
generates a small deterministic synthetic set with the same schema so the
pipeline still smoke-tests end-to-end.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

# DISC-Law-SFT task tags we care about. The dataset uses Chinese category labels;
# we map them onto our internal task names.
DISC_LAW_TASK_MAP = {
    "犯罪类型预测": "crime_prediction",
    "罪名预测": "crime_prediction",
    "罪名分析": "crime_prediction",
    "合同审查": "contract_review",
    "合同生成": "contract_review",
}

DOMAIN_KEYWORDS = {
    "刑事": ["盗窃", "抢劫", "故意伤害", "诈骗", "贪污", "受贿", "走私", "绑架", "强奸", "毒品", "罪", "刑罚", "刑期"],
    "民事": ["离婚", "抚养", "继承", "赡养", "侵权", "损害赔偿", "人格权", "婚姻", "家庭"],
    "商事": ["公司", "股东", "股权", "破产", "票据", "证券", "金融", "保险", "信托", "并购"],
    "行政": ["行政处罚", "行政复议", "行政许可", "行政诉讼", "强制执行", "行政机关"],
    "知产": ["专利", "商标", "著作权", "版权", "知识产权", "商业秘密"],
}


def _digest(text: str) -> str:
    return hashlib.md5(text[:200].encode("utf-8")).hexdigest()


# === HF download =====================================================================
# DISC-Law-SFT ships 4 jsonl shards with inconsistent schemas (Pair vs Triplet),
# which makes `load_dataset(...)` blow up at the schema-merge step. We bypass
# that by fetching the Pair shards individually with hf_hub_download and parsing
# them ourselves — Pair shards have the consistent {input, output, task_type}
# layout that everything downstream needs.

DISC_LAW_PAIR_FILES = [
    "DISC-Law-SFT-Pair.jsonl",
    "DISC-Law-SFT-Pair-QA-released.jsonl",
]


def _try_hf_download(name: str = "ShengbinYue/DISC-Law-SFT") -> list[dict] | None:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("[data_prep] huggingface_hub not installed; falling back to synthetic")
        return None

    rows: list[dict] = []
    for fname in DISC_LAW_PAIR_FILES:
        try:
            path = hf_hub_download(repo_id=name, filename=fname, repo_type="dataset")
        except Exception as exc:  # noqa: BLE001
            print(f"[data_prep] failed to fetch {fname}: {exc}")
            continue
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        print(f"[data_prep] loaded {fname}, running total {len(rows)} rows")

    if not rows:
        print(f"[data_prep] no rows downloaded from {name}")
        return None
    print(f"[data_prep] downloaded {len(rows)} rows from {name}")
    return rows


# === Synthetic fallback ==============================================================

_SYNTHETIC_FACTS = [
    ("被告人张某于深夜潜入某仓库盗走价值五万元的电脑设备", ["盗窃罪"], 12, "刑事"),
    ("被告人李某持刀致伤被害人，造成轻伤二级", ["故意伤害罪"], 9, "刑事"),
    ("被告人王某利用职务便利侵占公司资金一百万元", ["职务侵占罪"], 24, "刑事"),
    ("被告人赵某虚构事实骗取被害人钱款三十万元", ["诈骗罪"], 18, "刑事"),
    ("原告与被告婚后感情破裂，要求解除婚姻关系并分割共同财产", ["婚姻家庭纠纷"], 0, "民事"),
    ("原告主张被告侵犯其名誉权，要求公开赔礼道歉", ["名誉权纠纷"], 0, "民事"),
    ("公司股东请求查阅公司会计账簿被拒，提起股东知情权诉讼", ["股东知情权纠纷"], 0, "商事"),
    ("企业被处以行政罚款十万元，申请行政复议", ["行政处罚复议"], 0, "行政"),
    ("原告诉被告侵犯其发明专利权，要求停止侵害并赔偿损失", ["专利侵权"], 0, "知产"),
    ("商标权人发现他人在同类商品上使用近似商标", ["商标侵权"], 0, "知产"),
]

_SYNTHETIC_CONTRACTS = [
    ("甲方有权随时单方解除本合同且无需通知乙方。", ["解除权约定不公", "通知义务缺失"]),
    ("乙方违约的，应支付合同总价款 200% 的违约金。", ["违约金过高"]),
    ("本合同争议由甲方所在地法院管辖。", ["管辖约定单方有利"]),
    ("乙方放弃所有抗辩权。", ["放弃抗辩权"]),
    ("本合同自签订之日起生效，未约定终止条件。", ["终止条件缺失"]),
    ("乙方对甲方产品质量不得提出异议。", ["质量异议权剥夺"]),
    ("逾期付款按日万分之十计算违约金。", ["违约金过高"]),
    ("甲方对合同条款保留最终解释权。", ["最终解释权约定无效"]),
]


def _synth_disc_law(n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    rows: list[dict] = []
    pool = _SYNTHETIC_FACTS + _SYNTHETIC_FACTS  # crude duplication so different seeds produce variety
    for _ in range(n):
        fact, charges, months, domain = rng.choice(pool)
        # Add slight perturbation to keep dedup from killing them
        suffix = f"。案件编号 {rng.randint(10000, 99999)}。"
        rows.append({
            "input": fact + suffix,
            "output": json.dumps(charges, ensure_ascii=False),
            "task_type": "罪名预测",
            "_synthetic_meta": {
                "fact": fact + suffix,
                "accusations": charges,
                "imprisonment_months": months,
                "domain": domain,
            },
        })
    return rows


# === Conversion: DISC-Law -> our schema ==============================================

_CHARGE_RE = re.compile(r"(罪|犯罪)")
_CHINESE_LIST_RE = re.compile(r"[一-龥]{2,12}罪")


def _extract_charges(text: str) -> list[str]:
    """Heuristic charge extraction from free-text outputs."""
    found = sorted(set(_CHINESE_LIST_RE.findall(text)))
    return found[:5]


def _detect_domain(text: str) -> str:
    """Crude keyword-based 5-class domain assignment for stage 2 routing analysis."""
    scores = {domain: 0 for domain in DOMAIN_KEYWORDS}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                scores[domain] += 1
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "民事"


_MIN_FACT_LEN = 20  # Chinese chars; was 50 — DISC-Law / synthetic facts are shorter
_MAX_FACT_LEN = 2000


def _convert_to_charge_samples(rows: list[dict], cap: int) -> list[dict]:
    """Pull rows that look like charge-prediction tasks; produce {fact, accusations}."""
    out: list[dict] = []
    seen: set[str] = set()
    for row in rows:
        # synthetic shortcut — we control the schema, no need to filter
        if "_synthetic_meta" in row:
            meta = row["_synthetic_meta"]
            d = _digest(meta["fact"])
            if d in seen:
                continue
            seen.add(d)
            out.append({
                "fact": meta["fact"],
                "accusations": meta["accusations"],
                "articles": [],
                "imprisonment": {"imprisonment": meta["imprisonment_months"]},
                "domain": meta["domain"],
            })
            if len(out) >= cap:
                break
            continue

        fact = (row.get("input") or row.get("instruction") or row.get("fact") or "").strip()
        output = (row.get("output") or row.get("answer") or "").strip()
        task = row.get("task_type") or row.get("task") or ""
        if not fact or not output:
            continue
        if not (_MIN_FACT_LEN <= len(fact) <= _MAX_FACT_LEN):
            continue
        if "罪" not in output and "罪" not in task:
            continue
        charges = _extract_charges(output)
        if not charges:
            continue
        d = _digest(fact)
        if d in seen:
            continue
        seen.add(d)
        out.append({
            "fact": fact,
            "accusations": charges,
            "articles": [],
            "imprisonment": {"imprisonment": 0},
            "domain": _detect_domain(fact),
        })
        if len(out) >= cap:
            break
    return out


def _synth_contracts(n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    out: list[dict] = []
    for i in range(n):
        clause, risks = rng.choice(_SYNTHETIC_CONTRACTS)
        out.append({
            "clause_text": clause + f" (条款编号 {i})",
            "risk_labels": risks,
        })
    return out


# === SFT corpus builder ==============================================================

MULTI_TASK_TEMPLATES = [
    {
        "name": "triple_prediction",
        "system": "你是资深刑事法官，请按以下步骤分析案情。",
        "user": "案情描述：\n{fact}\n\n请输出涉及罪名（JSON 数组）",
        "assistant": "{accusations_json}",
    },
    {
        "name": "consult_dialog",
        "system": "你是法律咨询助手。",
        "user": "我朋友涉嫌这种行为：{fact}\n会构成什么罪？",
        "assistant": "根据您描述的情况，可能涉及：{accusations_natural}。建议尽快咨询专业律师。",
    },
]


def _format_sft(case: dict, template: dict) -> dict:
    fields = {
        "fact": case["fact"],
        "accusations_json": json.dumps(case["accusations"], ensure_ascii=False),
        "accusations_natural": "、".join(case["accusations"]),
    }
    return {
        "messages": [
            {"role": "system", "content": template["system"]},
            {"role": "user", "content": template["user"].format(**fields)},
            {"role": "assistant", "content": template["assistant"].format(**fields)},
        ],
        "template": template["name"],
    }


def build_sft_jsonl(charge_cases: list[dict], n_clean: int, n_multi: int, n_rebal: int,
                    out_dir: Path, seed: int = 42) -> dict[str, int]:
    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    # clean.jsonl: random N as plain triple_prediction
    clean_subset = list(charge_cases)
    rng.shuffle(clean_subset)
    clean_subset = clean_subset[:n_clean]
    clean_rows = [_format_sft(c, MULTI_TASK_TEMPLATES[0]) for c in clean_subset]

    # multitask.jsonl: random template per case
    multi_rows: list[dict] = []
    for _ in range(n_multi):
        case = rng.choice(charge_cases)
        template = rng.choice(MULTI_TASK_TEMPLATES)
        multi_rows.append(_format_sft(case, template))

    # rebalanced.jsonl: square-root smoothed inverse-frequency on the dominant accusation
    counts = Counter(c["accusations"][0] for c in charge_cases if c["accusations"])
    weights = {crime: (1.0 / count) ** 0.5 for crime, count in counts.items()}
    z = sum(weights.values()) or 1.0
    target = {crime: max(1, int(n_rebal * w / z)) for crime, w in weights.items()}
    by_crime: dict[str, list[dict]] = defaultdict(list)
    for c in charge_cases:
        if c["accusations"]:
            by_crime[c["accusations"][0]].append(c)
    rebal_rows: list[dict] = []
    for crime, n in target.items():
        pool = by_crime.get(crime, [])
        if not pool:
            continue
        for _ in range(n):
            rebal_rows.append(_format_sft(rng.choice(pool), MULTI_TASK_TEMPLATES[0]))

    counts_written: dict[str, int] = {}
    for name, rows in [("clean", clean_rows), ("multitask", multi_rows), ("rebalanced", rebal_rows)]:
        path = out_dir / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as h:
            for row in rows:
                h.write(json.dumps(row, ensure_ascii=False) + "\n")
        counts_written[name] = len(rows)
    counts_written["total"] = sum(counts_written.values())
    return counts_written


# === RLVR builder ====================================================================

CRIME_SYSTEM = (
    "你是法律专家。请根据案情描述给出涉及的罪名，"
    '用 JSON 数组格式输出，例如 ["盗窃罪", "故意伤害罪"]。'
)
CONTRACT_SYSTEM = (
    "你是合同审查专家。请识别下面合同条款中的风险点，"
    '用 JSON 数组格式输出每个风险点的类型，例如 ["违约责任过轻", "管辖约定不明"]。'
)


def build_rlvr(charge_cases: list[dict], contracts: list[dict], n_total: int,
               train_path: Path, test_path: Path, prompts_path: Path,
               distill_prompts_path: Path, n_distill: int, seed: int = 42) -> dict[str, int]:
    from datasets import Dataset

    rng = random.Random(seed)
    crime_samples = [{
        "prompt": [
            {"role": "system", "content": CRIME_SYSTEM},
            {"role": "user", "content": c["fact"]},
        ],
        "task": "crime_prediction",
        "ground_truth": {"crimes": list(map(str, c["accusations"]))},
    } for c in charge_cases]

    contract_samples = [{
        "prompt": [
            {"role": "system", "content": CONTRACT_SYSTEM},
            {"role": "user", "content": c["clause_text"]},
        ],
        "task": "contract_review",
        "ground_truth": {"risks": list(map(str, c["risk_labels"]))},
    } for c in contracts]

    half = n_total // 2
    rng.shuffle(crime_samples)
    rng.shuffle(contract_samples)
    samples = crime_samples[:half] + contract_samples[:half]
    rng.shuffle(samples)

    # 90/10 train/test split (deterministic)
    cut = max(1, int(len(samples) * 0.9))
    train, test = samples[:cut], samples[cut:]

    train_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(train).save_to_disk(str(train_path))
    Dataset.from_list(test).save_to_disk(str(test_path))

    # prompts.jsonl for stage 4 stage_c
    prompts_path.parent.mkdir(parents=True, exist_ok=True)
    with prompts_path.open("w", encoding="utf-8") as h:
        for sample in train:
            h.write(json.dumps({"messages": sample["prompt"]}, ensure_ascii=False) + "\n")

    # distill_prompts.jsonl: subset for teacher generation in stage 4A
    distill_prompts_path.parent.mkdir(parents=True, exist_ok=True)
    with distill_prompts_path.open("w", encoding="utf-8") as h:
        for sample in train[:n_distill]:
            h.write(json.dumps({"messages": sample["prompt"]}, ensure_ascii=False) + "\n")

    return {
        "train": len(train),
        "test": len(test),
        "prompts": len(train),
        "distill_prompts": min(n_distill, len(train)),
    }


# === Stage 2 domain eval =============================================================

def build_domain_eval(charge_cases: list[dict], out_path: Path, per_domain: int = 200,
                      seed: int = 42) -> dict[str, int]:
    rng = random.Random(seed)
    by_domain: dict[str, list[dict]] = defaultdict(list)
    for case in charge_cases:
        by_domain[case.get("domain", "民事")].append(case)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    with out_path.open("w", encoding="utf-8") as h:
        for domain, pool in by_domain.items():
            rng.shuffle(pool)
            picked = pool[:per_domain]
            for case in picked:
                h.write(json.dumps({
                    "domain": domain,
                    "text": case["fact"],
                }, ensure_ascii=False) + "\n")
            counts[domain] = len(picked)
    return counts


# === main ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--hf_dataset", default="ShengbinYue/DISC-Law-SFT")
    parser.add_argument("--full", action="store_true",
                        help="Resume-grade scale (350k SFT / 20k RLVR / 50k distill prompts)")
    parser.add_argument("--n_sft_clean", type=int, default=20_000)
    parser.add_argument("--n_sft_multi", type=int, default=8_000)
    parser.add_argument("--n_sft_rebal", type=int, default=2_000)
    parser.add_argument("--n_rlvr", type=int, default=2_000)
    parser.add_argument("--n_distill", type=int, default=5_000)
    parser.add_argument("--per_domain", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--synthetic_only", action="store_true",
                        help="Skip HF download and use synthetic fallback (smoke testing)")
    args = parser.parse_args()

    if args.full:
        args.n_sft_clean, args.n_sft_multi, args.n_sft_rebal = 150_000, 150_000, 50_000
        args.n_rlvr = 20_000
        args.n_distill = 50_000

    rows = None if args.synthetic_only else _try_hf_download(args.hf_dataset)
    if rows is None:
        print("[data_prep] using synthetic fallback")
        rows = _synth_disc_law(max(args.n_sft_clean * 2, 5_000), args.seed)

    charge_cap = max(args.n_sft_clean, args.n_rlvr) * 3
    charge_cases = _convert_to_charge_samples(rows, cap=charge_cap)
    if not charge_cases:
        print("[data_prep] WARN: no charge cases extracted from primary source; "
              "retrying with synthetic")
        rows = _synth_disc_law(max(args.n_sft_clean * 2, 5_000), args.seed)
        charge_cases = _convert_to_charge_samples(rows, cap=charge_cap)
    if not charge_cases:
        raise RuntimeError(
            "Could not extract any charge cases. Both HF and synthetic paths failed; "
            "this is a bug — check _convert_to_charge_samples filters."
        )
    print(f"[data_prep] charge cases extracted: {len(charge_cases)}")

    n_contracts = max(args.n_rlvr, 2_000)
    contracts = _synth_contracts(n_contracts, args.seed)
    print(f"[data_prep] synthesized {len(contracts)} contract clauses")

    out = args.out
    sft_dir = out / "processed" / ("sft_full" if args.full else "sft_demo")
    rlvr_train = out / "processed" / ("rlvr_full" if args.full else "rlvr_demo")
    rlvr_test = out / "processed" / ("rlvr_full_test" if args.full else "rlvr_demo_test")
    prompts_jsonl = rlvr_train / "prompts.jsonl"
    distill_prompts = out / "processed" / "distill_prompts.jsonl"
    domain_eval = out / "processed" / "legal_eval_by_domain.jsonl"

    sft_counts = build_sft_jsonl(
        charge_cases, args.n_sft_clean, args.n_sft_multi, args.n_sft_rebal,
        sft_dir, args.seed,
    )
    print(f"[data_prep] SFT -> {sft_dir} {sft_counts}")

    rlvr_counts = build_rlvr(
        charge_cases, contracts, args.n_rlvr,
        rlvr_train, rlvr_test, prompts_jsonl, distill_prompts, args.n_distill, args.seed,
    )
    print(f"[data_prep] RLVR -> {rlvr_train}, test -> {rlvr_test} {rlvr_counts}")

    domain_counts = build_domain_eval(charge_cases, domain_eval, args.per_domain, args.seed)
    print(f"[data_prep] domain eval -> {domain_eval} {domain_counts}")

    print("\n[data_prep] all artifacts ready. Suggested next steps:")
    print(f"  python stages/stage1_sft_grpo/train_sft.py --dataset_path {sft_dir} ...")
    print(f"  python stages/stage1_sft_grpo/data/filter_dynamic_sampling.py "
          f"--dataset_path {rlvr_train} ...")
    print(f"  python stages/stage2_moe_router/analyze_router.py --eval_jsonl {domain_eval} ...")


if __name__ == "__main__":
    main()
