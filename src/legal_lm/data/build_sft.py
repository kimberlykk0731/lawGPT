from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from legal_lm.data.prompting import (
    DEFAULT_SYSTEM_PROMPT,
    build_messages,
    build_rag_query,
    render_assistant_target,
    render_user_prompt,
)
from legal_lm.data.registry import load_samples
from legal_lm.data.schema import LegalSample, SFTMode
from legal_lm.utils import dump_jsonl, ensure_dir


def _maybe_load_retriever(index_dir: str | Path | None):
    if not index_dir:
        return None
    from legal_lm.data.rag import StatuteRetriever

    return StatuteRetriever(index_dir)


def build_sft_rows(
    samples: list[LegalSample],
    mode: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    retriever=None,
    rag_top_k: int = 5,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample in samples:
        rag_block = ""
        if retriever is not None:
            query = build_rag_query(sample)
            if query:
                rag_block = retriever.render_block(query, top_k=rag_top_k)
        prompt = render_user_prompt(sample, rag_block=rag_block)
        target = render_assistant_target(sample, mode)
        rows.append(
            {
                "id": sample.sample_id,
                "domain": sample.domain,
                "task_type": sample.task_type,
                "mode": mode,
                "prompt": prompt,
                "response": target,
                "messages": build_messages(
                    sample, mode, system_prompt=system_prompt, rag_block=rag_block
                ),
                "text": (
                    f"<|system|>\n{system_prompt}\n<|user|>\n{prompt}\n<|assistant|>\n{target}"
                ),
                "metadata": sample.metadata,
            }
        )
    return rows


def build_sft_dataset(
    dataset_config_path: str | Path,
    dataset_name: str,
    output_dir: str | Path,
    splits: list[str],
    mode: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    statute_index_dir: str | Path | None = None,
    rag_top_k: int = 5,
) -> dict[str, Path]:
    if mode not in {item.value for item in SFTMode}:
        raise ValueError(f"Unsupported SFT mode: {mode}")

    target_dir = ensure_dir(output_dir)
    retriever = _maybe_load_retriever(statute_index_dir)
    outputs: dict[str, Path] = {}
    for split in splits:
        samples = load_samples(dataset_config_path, dataset_name, split)
        rows = build_sft_rows(
            samples, mode=mode, system_prompt=system_prompt,
            retriever=retriever, rag_top_k=rag_top_k,
        )
        output_path = target_dir / f"{split}.jsonl"
        dump_jsonl(output_path, rows)
        outputs[split] = output_path
    return outputs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SFT dataset from unified legal schema.")
    parser.add_argument("--dataset-config", default="configs/datasets.yaml")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", default=SFTMode.BRIEF_REASONING.value)
    parser.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--inject-rag", action="store_true",
                        help="构造 prompt 时注入法条 RAG 段")
    parser.add_argument("--statute-index-dir", default="data/statute_index")
    parser.add_argument("--rag-top-k", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    build_sft_dataset(
        dataset_config_path=args.dataset_config,
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        splits=args.splits,
        mode=args.mode,
        system_prompt=args.system_prompt,
        statute_index_dir=args.statute_index_dir if args.inject_rag else None,
        rag_top_k=args.rag_top_k,
    )


if __name__ == "__main__":
    main()
