from __future__ import annotations

from legal_lm.data.loaders.base import BaseDatasetLoader
from legal_lm.data.schema import LegalDomain, LegalSample, TaskType
from legal_lm.utils.io import load_json, load_jsonl


class CAILLoader(BaseDatasetLoader):
    def load_split(self, split: str) -> list[LegalSample]:
        path = self._resolve_split_file(split)
        rows = load_jsonl(path) if path.suffix == ".jsonl" else load_json(path)
        if isinstance(rows, dict):
            rows = rows.get("data", [])

        samples: list[LegalSample] = []
        for index, row in enumerate(rows):
            accusations = row.get("meta", {}).get("accusation") or row.get("accusation") or []
            articles = row.get("meta", {}).get("relevant_articles") or row.get("articles") or []
            term = row.get("meta", {}).get("term_of_imprisonment") or row.get("imprisonment") or {}
            imprisonment = _format_imprisonment(term)
            answer = f"罪名：{', '.join(map(str, accusations)) or '未给出'}；法条：{', '.join(map(str, articles)) or '未给出'}；刑期：{imprisonment}"
            reasoning = row.get("reason") or row.get("interpretation") or answer
            citations = [f"第{item}条" for item in articles]

            sample = LegalSample(
                sample_id=str(row.get("id") or f"{self.dataset_name}-{split}-{index}"),
                source_dataset=self.dataset_name,
                domain=LegalDomain.CRIMINAL.value,
                task_type=TaskType.JUDGMENT_PREDICTION.value,
                split=split,
                instruction="请根据案件事实预测可能罪名、相关法条和量刑区间，并说明理由。",
                facts=str(row.get("fact") or ""),
                issues=["罪名认定", "法条适用", "量刑判断"],
                statutes=[f"第{item}条" for item in articles],
                gold_answer=answer,
                gold_reasoning=str(reasoning),
                brief_reasoning=str(reasoning),
                distilled_cot=str(reasoning),
                citations=citations,
                metadata={
                    "charges": list(accusations),
                    "articles": list(articles),
                    "imprisonment": term,
                },
            ).normalize()
            samples.append(sample)
        return samples


def _format_imprisonment(term: dict | str | int | None) -> str:
    if isinstance(term, dict):
        if term.get("death_penalty"):
            return "死刑"
        if term.get("life_imprisonment"):
            return "无期徒刑"
        if "imprisonment" in term:
            return f"{term['imprisonment']}个月"
    if term is None:
        return "未给出"
    return str(term)
