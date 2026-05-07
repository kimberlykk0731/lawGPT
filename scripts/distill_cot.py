"""离线 CoT 蒸馏：调用强模型为 SFT 样本生成 <think> 推理。

读入 SFT 样本（统一 schema 或 instruction/input/output），输出蒸馏 JSONL：
  {"question": ..., "answer": "<think>...</think>...", "domain": ..., "source": api}

可选 --enforce-structure 丢弃未通过结构校验的样本。
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from legal_lm.data.distill import (
    API_CONFIGS,
    DistillClient,
    passes_structure_check,
)


def _question_from_record(record: dict) -> str:
    parts: list[str] = []
    if record.get("instruction"):
        parts.append(str(record["instruction"]))
    if record.get("facts"):
        parts.append(f"【案件事实】{record['facts']}")
    if record.get("input"):
        parts.append(str(record["input"]))
    if record.get("issues"):
        issues = record["issues"]
        if isinstance(issues, list):
            parts.append("【争点】" + "；".join(map(str, issues)))
    if not parts:
        parts.append(str(record.get("prompt") or record.get("question") or ""))
    return "\n".join(p for p in parts if p).strip()


def _read_records(path: Path) -> list[dict]:
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


def _existing_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    keys: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                rec = json.loads(line)
                keys.add(rec.get("question", ""))
            except json.JSONDecodeError:
                continue
    return keys


def main() -> None:
    parser = argparse.ArgumentParser(description="离线 CoT 蒸馏")
    parser.add_argument("--api", choices=list(API_CONFIGS), default="deepseek")
    parser.add_argument("--model", default=None, help="若不指定，使用 API_CONFIGS 默认")
    parser.add_argument("--input", required=True, help="SFT 样本 JSONL")
    parser.add_argument("--output", required=True, help="输出蒸馏 JSONL")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--enforce-structure", action="store_true")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records = _read_records(in_path)
    if args.max_samples:
        records = records[: args.max_samples]
    seen = _existing_keys(out_path)
    pending: list[dict] = []
    for rec in records:
        question = _question_from_record(rec)
        if not question or question in seen:
            continue
        pending.append({"question": question, "domain": rec.get("domain", "general")})
    print(f"[distill_cot] todo={len(pending)} api={args.api}")

    client = DistillClient(api=args.api, model=args.model, temperature=args.temperature)

    success, fail, dropped = 0, 0, 0
    handle = out_path.open("a", encoding="utf-8")
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(client.generate, item["question"]): item for item in pending}
            for fut in as_completed(futures):
                item = futures[fut]
                answer = fut.result()
                if not answer:
                    fail += 1
                    continue
                if args.enforce_structure and not passes_structure_check(answer):
                    dropped += 1
                    continue
                row = {
                    "question": item["question"],
                    "domain": item["domain"],
                    "answer": answer,
                    "source": args.api,
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                success += 1
                if (success + fail + dropped) % 50 == 0:
                    print(f"[distill_cot] ok={success} fail={fail} dropped={dropped}")
    finally:
        handle.close()
    print(f"[distill_cot] done ok={success} fail={fail} dropped={dropped} -> {out_path}")


if __name__ == "__main__":
    main()
