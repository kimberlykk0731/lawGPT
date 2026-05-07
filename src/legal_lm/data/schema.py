from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from legal_lm.utils.text import normalize_text


class LegalDomain(str, Enum):
    GENERAL = "general"
    CRIMINAL = "criminal"
    TORT = "tort"
    ADMINISTRATIVE = "administrative"


class TaskType(str, Enum):
    CASE_REASONING = "case_reasoning"
    QA = "qa"
    JUDGMENT_PREDICTION = "judgment_prediction"
    RETRIEVAL = "retrieval"
    PREFERENCE = "preference"


class SFTMode(str, Enum):
    ANSWER_ONLY = "answer_only"
    BRIEF_REASONING = "brief_reasoning"
    DISTILLED_COT = "distilled_cot"


@dataclass
class Citation:
    law_name: str = ""
    article: str = ""
    text: str = ""

    def to_text(self) -> str:
        segments = [self.law_name, f"第{self.article}条" if self.article else "", self.text]
        return " ".join(part for part in segments if part).strip()


@dataclass
class LegalSample:
    sample_id: str
    source_dataset: str
    domain: str
    task_type: str
    split: str
    instruction: str = ""
    facts: str = ""
    issues: list[str] = field(default_factory=list)
    statutes: list[str] = field(default_factory=list)
    gold_answer: str = ""
    gold_reasoning: str = ""
    brief_reasoning: str = ""
    distilled_cot: str = ""
    citations: list[str] = field(default_factory=list)
    chosen: str = ""
    rejected: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def normalize(self) -> "LegalSample":
        self.instruction = normalize_text(self.instruction)
        self.facts = normalize_text(self.facts)
        self.issues = [normalize_text(issue) for issue in self.issues if normalize_text(issue)]
        self.statutes = [normalize_text(item) for item in self.statutes if normalize_text(item)]
        self.gold_answer = normalize_text(self.gold_answer)
        self.gold_reasoning = normalize_text(self.gold_reasoning)
        self.brief_reasoning = normalize_text(self.brief_reasoning or self.gold_reasoning)
        self.distilled_cot = normalize_text(self.distilled_cot or self.gold_reasoning)
        self.citations = [normalize_text(item) for item in self.citations if normalize_text(item)]
        self.chosen = normalize_text(self.chosen)
        self.rejected = normalize_text(self.rejected)
        return self

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["issues"] = list(self.issues)
        payload["statutes"] = list(self.statutes)
        payload["citations"] = list(self.citations)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LegalSample":
        sample = cls(
            sample_id=str(payload.get("sample_id") or payload.get("id") or payload.get("qid") or ""),
            source_dataset=str(payload.get("source_dataset") or payload.get("dataset") or "local_jsonl"),
            domain=str(payload.get("domain") or LegalDomain.GENERAL.value),
            task_type=str(payload.get("task_type") or TaskType.CASE_REASONING.value),
            split=str(payload.get("split") or "train"),
            instruction=str(payload.get("instruction") or payload.get("question") or ""),
            facts=str(payload.get("facts") or payload.get("fact") or payload.get("query") or ""),
            issues=list(payload.get("issues") or []),
            statutes=list(payload.get("statutes") or payload.get("articles") or []),
            gold_answer=str(payload.get("gold_answer") or payload.get("answer") or payload.get("target") or ""),
            gold_reasoning=str(payload.get("gold_reasoning") or payload.get("reasoning") or ""),
            brief_reasoning=str(payload.get("brief_reasoning") or ""),
            distilled_cot=str(payload.get("distilled_cot") or payload.get("cot") or ""),
            citations=list(payload.get("citations") or []),
            chosen=str(payload.get("chosen") or ""),
            rejected=str(payload.get("rejected") or ""),
            metadata=dict(payload.get("metadata") or {}),
        )
        return sample.normalize()

    def require_identifier(self, fallback_prefix: str) -> "LegalSample":
        if not self.sample_id:
            self.sample_id = f"{fallback_prefix}-{abs(hash((self.facts, self.gold_answer))) % 10_000_000}"
        return self


def flatten_citations(citations: list[Citation] | list[str]) -> list[str]:
    flattened: list[str] = []
    for item in citations:
        if isinstance(item, Citation):
            text = item.to_text()
        else:
            text = str(item)
        text = normalize_text(text)
        if text:
            flattened.append(text)
    return flattened
