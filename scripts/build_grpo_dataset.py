"""GRPO 可验证子任务数据构建。

针对 cail2018 等带结构化标签的数据集，按任务类型 (charge / article / sentencing)
拆出可程序化校验的子集，每条形如：
  {"prompt", "answer", "reasoning", "citations", "facts", "issues", "metadata_json", "task_type"}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from legal_lm.data.prompting import DEFAULT_SYSTEM_PROMPT, render_user_prompt
from legal_lm.data.registry import load_samples


_TASK_PROMPTS = {
    "charge": "请仅给出被告人的罪名（多个罪名用顿号分隔）。",
    "article": "请仅给出适用的具体法条编号（如《刑法》第二百六十四条）。",
    "sentencing": "请仅给出量刑结果（数值或死刑/无期徒刑等术语）。",
}


def _row_for(sample, task_type: str) -> dict | None:
    metadata = sample.metadata or {}
    if task_type == "charge" and not metadata.get("charges") and not sample.gold_answer:
        return None
    if task_type == "article" and not metadata.get("articles") and not sample.citations:
        return None
    if task_type == "sentencing":
        imprisonment = metadata.get("imprisonment", {}) if isinstance(metadata.get("imprisonment"), dict) else {}
        if not imprisonment and not sample.gold_answer:
            return None
    instruction = _TASK_PROMPTS[task_type]
    sample_with_inst = sample
    sample_with_inst.instruction = instruction
    return {
        "prompt": render_user_prompt(sample_with_inst),
        "answer": sample.gold_answer,
        "reasoning": sample.brief_reasoning or sample.gold_reasoning,
        "citations": list(sample.citations or []),
        "facts": sample.facts,
        "issues": list(sample.issues or []),
        "metadata_json": json.dumps(metadata, ensure_ascii=False),
        "task_type": task_type,
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="构建 GRPO 可验证子任务数据")
    parser.add_argument("--dataset-config", default="configs/datasets.yaml")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "validation"])
    parser.add_argument("--task-types", default="charge,article,sentencing",
                        help="逗号分隔的子任务，可选 charge/article/sentencing")
    args = parser.parse_args()

    task_types = [t.strip() for t in args.task_types.split(",") if t.strip()]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        samples = load_samples(args.dataset_config, args.dataset_name, split)
        out_path = out_dir / f"{split}.jsonl"
        kept = 0
        with out_path.open("w", encoding="utf-8") as handle:
            for sample in samples:
                for task_type in task_types:
                    row = _row_for(sample, task_type)
                    if row is None:
                        continue
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    kept += 1
        print(f"[build_grpo_dataset] {split}: kept={kept} -> {out_path}")


if __name__ == "__main__":
    main()
