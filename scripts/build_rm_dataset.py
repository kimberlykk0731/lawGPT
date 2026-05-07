"""偏好对 → RM 训练数据。

输入：build_preference_dataset.py 产出的 {prompt, chosen, rejected} JSONL
输出：去重 + 字段重命名后的 RM 训练 JSONL
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _convert(line: str) -> dict | None:
    try:
        rec = json.loads(line)
    except json.JSONDecodeError:
        return None
    prompt = rec.get("prompt", "")
    chosen = rec.get("chosen", "")
    rejected = rec.get("rejected", "")
    if not (prompt and chosen and rejected):
        return None
    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}


def _process_split(in_path: Path, out_path: Path) -> int:
    seen: set[str] = set()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    with in_path.open(encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            rec = _convert(line)
            if rec is None:
                continue
            key = hashlib.md5(
                (rec["prompt"] + rec["chosen"] + rec["rejected"]).encode("utf-8")
            ).hexdigest()
            if key in seen:
                continue
            seen.add(key)
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            kept += 1
    return kept


def main() -> None:
    parser = argparse.ArgumentParser(description="偏好对 → RM 训练数据")
    parser.add_argument("--preference-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "validation"])
    args = parser.parse_args()

    pref_dir = Path(args.preference_dir)
    out_dir = Path(args.output_dir)
    for split in args.splits:
        in_path = pref_dir / f"{split}.jsonl"
        out_path = out_dir / f"{split}.jsonl"
        if not in_path.exists():
            print(f"[build_rm_dataset] skip missing split: {in_path}")
            continue
        kept = _process_split(in_path, out_path)
        print(f"[build_rm_dataset] {split}: kept={kept} -> {out_path}")


if __name__ == "__main__":
    main()
