from __future__ import annotations

from legal_lm.data.loaders.base import BaseDatasetLoader
from legal_lm.data.schema import LegalDomain, LegalSample, TaskType
from legal_lm.utils.io import load_json, load_jsonl


class JECQALoader(BaseDatasetLoader):
    def load_split(self, split: str) -> list[LegalSample]:
        path = self._resolve_split_file(split)
        rows = load_jsonl(path) if path.suffix == ".jsonl" else load_json(path)
        if isinstance(rows, dict):
            rows = rows.get("data", [])

        samples: list[LegalSample] = []
        for index, row in enumerate(rows):
            options = row.get("options") or row.get("choices") or []
            option_lines = [f"{chr(65 + idx)}. {text}" for idx, text in enumerate(options)]
            instruction = "请回答以下中国法律考试或法律问答题，并给出简要理由。"
            question = str(row.get("question") or row.get("stem") or "")
            if option_lines:
                instruction = f"{instruction}\n\n题目：{question}\n选项：\n" + "\n".join(option_lines)
            answer = row.get("answer") or row.get("label") or row.get("target") or ""
            reasoning = row.get("analysis") or row.get("reasoning") or f"根据题干与法条，答案为 {answer}。"
            statutes = row.get("articles") or row.get("references") or []
            sample = LegalSample(
                sample_id=str(row.get("id") or row.get("qid") or f"{self.dataset_name}-{split}-{index}"),
                source_dataset=self.dataset_name,
                domain=self.dataset_config.get("domain", LegalDomain.GENERAL.value),
                task_type=TaskType.QA.value,
                split=split,
                instruction=instruction,
                facts=question,
                issues=["法律问答", "法条适用"],
                statutes=[str(item) for item in statutes],
                gold_answer=str(answer),
                gold_reasoning=str(reasoning),
                brief_reasoning=str(reasoning),
                distilled_cot=str(reasoning),
                citations=[str(item) for item in statutes],
                metadata={"options": options},
            ).normalize()
            samples.append(sample)
        return samples
