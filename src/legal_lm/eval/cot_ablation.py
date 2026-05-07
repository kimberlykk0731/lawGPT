from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_result(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def summarize_cot_ablation(results: list[dict[str, Any]]) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    for result in results:
        label = result.get("label") or result.get("mode") or result.get("name") or "unknown"
        summary = result.get("summary", {})
        overall = summary.get("overall", {})
        runs.append(
            {
                "label": label,
                "overall": overall.get("overall", 0.0),
                "answer_overlap": overall.get("answer_overlap", 0.0),
                "reasoning_overlap": overall.get("reasoning_overlap", 0.0),
                "citation_score": overall.get("citation_score", 0.0),
                "structure_score": overall.get("structure_score", 0.0),
            }
        )

    runs = sorted(runs, key=lambda item: item["overall"], reverse=True)
    best_mode = runs[0]["label"] if runs else None
    return {"runs": runs, "best_mode": best_mode}


def summarize_cot_ablation_from_files(result_files: list[str | Path]) -> dict[str, Any]:
    results = []
    for path in result_files:
        result = load_result(path)
        if "label" not in result:
            result["label"] = Path(path).stem
        results.append(result)
    return summarize_cot_ablation(results)
