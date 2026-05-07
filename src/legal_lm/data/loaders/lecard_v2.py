from __future__ import annotations

from pathlib import Path

from legal_lm.data.loaders.base import BaseDatasetLoader
from legal_lm.data.schema import LegalDomain, LegalSample, TaskType
from legal_lm.utils.io import iter_jsonl, load_jsonl


class LeCaRDv2Loader(BaseDatasetLoader):
    def load_split(self, split: str) -> list[LegalSample]:
        query_path = self._resolve_split_file(split)
        queries = load_jsonl(query_path)
        qrels = self._load_qrels(self.root / self.dataset_config.get("qrels_file", "relevence.trec"))

        samples: list[LegalSample] = []
        for index, row in enumerate(queries):
            qid = str(row.get("id") or row.get("qid") or index)
            relevant = qrels.get(qid, [])
            answer = (
                f"建议优先检索的类案编号：{', '.join(relevant[:10])}" if relevant else "需要基于案情检索相似裁判。"
            )
            reasoning = "优先匹配罪名、量刑情节、程序节点相近的案件，再比较裁判理由。"
            sample = LegalSample(
                sample_id=str(qid),
                source_dataset=self.dataset_name,
                domain=self.dataset_config.get("domain", LegalDomain.CRIMINAL.value),
                task_type=TaskType.RETRIEVAL.value,
                split=split,
                instruction="请根据案件事实检索最相关的类案，并说明匹配理由。",
                facts=str(row.get("fact") or row.get("query") or ""),
                issues=["类案检索", "罪名与量刑要素匹配"],
                statutes=[str(item) for item in row.get("article", [])],
                gold_answer=answer,
                gold_reasoning=reasoning,
                brief_reasoning=reasoning,
                distilled_cot=reasoning,
                citations=[str(item) for item in row.get("article", [])],
                metadata={"relevant_case_ids": relevant},
            ).normalize()
            samples.append(sample)
        return samples

    @staticmethod
    def _load_qrels(path: Path) -> dict[str, list[str]]:
        if not path.exists():
            return {}
        mapping: dict[str, list[str]] = {}
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.strip().split()
                if len(parts) != 4:
                    continue
                qid, _, pid, label = parts
                if int(label) <= 0:
                    continue
                mapping.setdefault(qid, []).append(pid)
        return mapping
