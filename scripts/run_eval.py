"""评测调度脚本。

子命令：
- prepare    拉取/准备 benchmark 数据
- run        跑评测（lawbench / cail2018）；保留 legacy 自定义 benchmark 路径
- compare    汇总多次 run 的指标到一个对比报告
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from legal_lm.eval import run_benchmark, summarize_cot_ablation_from_files, write_reports
from legal_lm.eval.cail import run_cail
from legal_lm.eval.cot_ablation import load_result
from legal_lm.eval.lawbench import run_lawbench
from legal_lm.utils import dump_jsonl


# ---------------------------------------------------------------- prepare ---


def _prepare(args: argparse.Namespace) -> None:
    target = Path(args.output_dir)
    target.mkdir(parents=True, exist_ok=True)
    if args.benchmark == "lawbench":
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:  # pragma: no cover
            raise ImportError("huggingface_hub is required: pip install huggingface_hub") from exc
        snapshot_download(repo_id="open-compass/LawBench", repo_type="dataset", local_dir=str(target))
        print(f"[run_eval.prepare] LawBench downloaded to {target}")
    elif args.benchmark == "cail2018":
        print(
            "[run_eval.prepare] CAIL2018 需手动准备：将 train.json / valid.json / test.json 放入 "
            f"{target}/。也可使用 datasets.load_dataset('thunlp/cail2018')。"
        )
    else:
        raise ValueError(f"Unsupported benchmark: {args.benchmark}")


# -------------------------------------------------------------------- run ---


def _run(args: argparse.Namespace) -> None:
    if args.benchmark == "lawbench":
        run_lawbench(
            model_path=args.model_path,
            data_dir=args.data_dir or "data/raw/LawBench",
            output_dir=args.output_dir,
            use_vllm=args.use_vllm,
            max_new_tokens=args.max_new_tokens,
            inject_rag=args.inject_rag,
            statute_index_dir=args.statute_index_dir,
            rag_top_k=args.rag_top_k,
        )
        return
    if args.benchmark == "cail2018":
        run_cail(
            model_path=args.model_path,
            data_dir=args.data_dir or "data/raw/CAIL2018",
            output_dir=args.output_dir,
            use_vllm=args.use_vllm,
            max_new_tokens=args.max_new_tokens,
            split=args.split,
            max_samples=args.max_samples,
        )
        return

    # legacy 自定义 benchmark 路径
    if not args.dataset_name:
        raise ValueError("--dataset-name is required for legacy benchmark.")
    benchmark_result = run_benchmark(
        dataset_config_path=args.dataset_config,
        dataset_name=args.dataset_name,
        split=args.split,
        predictions_file=args.predictions_file,
        model_name_or_path=args.model_path,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        output_predictions_file=args.output_predictions_file,
    )
    benchmark_result.setdefault("label", args.label or f"{args.dataset_name}-{args.split}")
    cot_ablation = (
        summarize_cot_ablation_from_files(args.cot_result_files) if args.cot_result_files else None
    )
    paths = write_reports(args.output_dir, benchmark_result, cot_ablation)
    if "details" in benchmark_result:
        dump_jsonl(paths["metrics"].with_name("details.jsonl"), benchmark_result["details"])


# ------------------------------------------------------------------ compare


def _flatten_summary(summary: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in summary.items():
        if key in {"by_task", "by_domain", "by_capability"}:
            continue
        if isinstance(value, dict):
            for k2, v2 in value.items():
                if isinstance(v2, (int, float)):
                    flat[f"{key}.{k2}"] = v2
            continue
        if isinstance(value, (int, float)):
            flat[key] = value
    return flat


def _compare(args: argparse.Namespace) -> None:
    target_dir = Path(args.output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    table: list[dict[str, Any]] = []
    for run_dir in args.runs:
        run_path = Path(run_dir)
        metrics_path = run_path / "metrics.json"
        if not metrics_path.exists():
            print(f"[compare] skip missing {metrics_path}")
            continue
        result = json.loads(metrics_path.read_text(encoding="utf-8"))
        flat = _flatten_summary(result.get("summary", {}))
        flat["label"] = result.get("label", run_path.name)
        flat["run_dir"] = str(run_path)
        table.append(flat)

    out_jsonl = target_dir / "compare.jsonl"
    dump_jsonl(out_jsonl, table)

    keys: list[str] = sorted({k for row in table for k in row if k not in {"label", "run_dir"}})
    md_lines = ["# 对比报告", "", "| Run | " + " | ".join(keys) + " |",
                "|---|" + "|".join(["---"] * len(keys)) + "|"]
    for row in table:
        cells = [row.get("label", "-")] + [
            (f"{row[k]:.4f}" if isinstance(row.get(k), float) else str(row.get(k, "-")))
            for k in keys
        ]
        md_lines.append("| " + " | ".join(cells) + " |")
    (target_dir / "compare.md").write_text("\n".join(md_lines), encoding="utf-8")
    print(f"[compare] wrote {out_jsonl} and compare.md")


# ----------------------------------------------------------------- main ---


def main() -> None:
    parser = argparse.ArgumentParser(description="评测调度")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_prep = sub.add_parser("prepare", help="准备 benchmark 数据")
    p_prep.add_argument("--benchmark", choices=["lawbench", "cail2018"], required=True)
    p_prep.add_argument("--output-dir", required=True)
    p_prep.set_defaults(func=_prepare)

    p_run = sub.add_parser("run", help="跑评测")
    p_run.add_argument("--benchmark", choices=["lawbench", "cail2018", "legacy"], default="legacy")
    p_run.add_argument("--model-path", required=True)
    p_run.add_argument("--data-dir", default=None)
    p_run.add_argument("--output-dir", required=True)
    p_run.add_argument("--use-vllm", action="store_true")
    p_run.add_argument("--max-new-tokens", type=int, default=256)
    p_run.add_argument("--split", default="test")
    p_run.add_argument("--max-samples", type=int, default=None)
    p_run.add_argument("--inject-rag", action="store_true")
    p_run.add_argument("--statute-index-dir", default="data/statute_index")
    p_run.add_argument("--rag-top-k", type=int, default=5)
    p_run.add_argument("--dataset-config", default="configs/datasets.yaml")
    p_run.add_argument("--dataset-name", default=None)
    p_run.add_argument("--predictions-file", default=None)
    p_run.add_argument("--output-predictions-file", default=None)
    p_run.add_argument("--temperature", type=float, default=0.1)
    p_run.add_argument("--label", default=None)
    p_run.add_argument("--cot-result-files", nargs="*", default=[])
    p_run.set_defaults(func=_run)

    p_cmp = sub.add_parser("compare", help="对比多次 run 的 metrics.json")
    p_cmp.add_argument("--runs", nargs="+", required=True)
    p_cmp.add_argument("--output-dir", required=True)
    p_cmp.set_defaults(func=_compare)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
