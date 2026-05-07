from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from legal_lm.data.prompting import DEFAULT_SYSTEM_PROMPT, render_assistant_target, render_user_prompt
from legal_lm.data.registry import load_samples
from legal_lm.data.schema import LegalSample, SFTMode
from legal_lm.utils import dump_jsonl, ensure_dir


def synthesize_rejected_answer(sample: LegalSample) -> str:
    weak_answer = sample.metadata.get("weak_answer")
    if weak_answer:
        return str(weak_answer)

    conclusion = sample.gold_answer or "目前无法判断。"
    return (
        f"【结论】\n{conclusion}\n\n"
        "【法律依据】\n未明确援引具体法条。\n\n"
        "【推理摘要】\n仅依据一般经验作出判断，未充分展开争点分析。"
    )


def build_preference_rows(samples: list[LegalSample]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample in samples:
        prompt = render_user_prompt(sample)
        chosen = sample.chosen or render_assistant_target(sample, SFTMode.BRIEF_REASONING.value)
        rejected = sample.rejected or synthesize_rejected_answer(sample)
        rows.append(
            {
                "id": sample.sample_id,
                "domain": sample.domain,
                "task_type": sample.task_type,
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "metadata": sample.metadata,
            }
        )
    return rows


def build_grpo_rows(samples: list[LegalSample]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample in samples:
        rows.append(
            {
                "id": sample.sample_id,
                "domain": sample.domain,
                "task_type": sample.task_type,
                "prompt": render_user_prompt(sample),
                "answer": sample.gold_answer,
                "reasoning": sample.brief_reasoning or sample.gold_reasoning,
                "citations": sample.citations,
                "facts": sample.facts,
                "issues": sample.issues,
                "metadata_json": json.dumps(sample.metadata, ensure_ascii=False),
            }
        )
    return rows


def build_preference_dataset(
    dataset_config_path: str | Path,
    dataset_name: str,
    preference_output_dir: str | Path,
    grpo_output_dir: str | Path | None,
    splits: list[str],
) -> dict[str, dict[str, Path]]:
    pref_dir = ensure_dir(preference_output_dir)
    grpo_dir = ensure_dir(grpo_output_dir) if grpo_output_dir else None
    results: dict[str, dict[str, Path]] = {"preference": {}, "grpo": {}}

    for split in splits:
        samples = load_samples(dataset_config_path, dataset_name, split)
        pref_path = pref_dir / f"{split}.jsonl"
        dump_jsonl(pref_path, build_preference_rows(samples))
        results["preference"][split] = pref_path

        if grpo_dir:
            grpo_path = grpo_dir / f"{split}.jsonl"
            dump_jsonl(grpo_path, build_grpo_rows(samples))
            results["grpo"][split] = grpo_path
    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build preference and GRPO datasets.")
    parser.add_argument("--dataset-config", default="configs/datasets.yaml")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--preference-output-dir", required=True)
    parser.add_argument("--grpo-output-dir", default="data/processed/grpo")
    parser.add_argument("--splits", nargs="+", default=["train", "validation"])
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    build_preference_dataset(
        dataset_config_path=args.dataset_config,
        dataset_name=args.dataset_name,
        preference_output_dir=args.preference_output_dir,
        grpo_output_dir=args.grpo_output_dir,
        splits=args.splits,
    )


if __name__ == "__main__":
    main()
