"""PPO prompt 数据构建。

PPO 不需要 chosen/rejected，只需要 prompt + 元信息（供奖励组合使用）。
本脚本从统一 schema 数据中抽取 prompt 与必要的元字段输出 JSONL。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from legal_lm.data.prompting import DEFAULT_SYSTEM_PROMPT, render_user_prompt
from legal_lm.data.registry import load_samples


def main() -> None:
    parser = argparse.ArgumentParser(description="构建 PPO prompt 数据")
    parser.add_argument("--dataset-config", default="configs/datasets.yaml")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "validation"])
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        samples = load_samples(args.dataset_config, args.dataset_name, split)
        out_path = out_dir / f"{split}.jsonl"
        kept = 0
        with out_path.open("w", encoding="utf-8") as handle:
            for sample in samples:
                row = {
                    "prompt": render_user_prompt(sample),
                    "system_prompt": DEFAULT_SYSTEM_PROMPT,
                    "domain": sample.domain,
                    "facts": sample.facts,
                    "issues": list(sample.issues or []),
                    "citations": list(sample.citations or []),
                    "answer": sample.gold_answer,
                    "metadata_json": json.dumps(sample.metadata or {}, ensure_ascii=False),
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                kept += 1
        print(f"[build_ppo_dataset] {split}: kept={kept} -> {out_path}")


if __name__ == "__main__":
    main()
