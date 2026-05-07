"""法律文本 PII 脱敏。

- 当事人姓名 → [当事人A]/[当事人B]/...
- 身份证号、手机号、银行卡号 → 全部 mask
- 案号细节 → 保留前缀年份与法院前缀
- 详细住址 → 保留省/市/区县级
"""

from __future__ import annotations

import re
from dataclasses import dataclass

ID_CARD_PATTERN = re.compile(r"\b\d{17}[\dXx]\b")
PHONE_PATTERN = re.compile(r"(?<!\d)(1[3-9]\d{9})(?!\d)")
BANK_CARD_PATTERN = re.compile(r"(?<!\d)(\d{16,19})(?!\d)")
CASE_NUMBER_PATTERN = re.compile(r"(\(?[（(]?\s*\d{4}\s*[）)]\s*[一-龥]{1,8}[一-龥\d]+号)")
ADDRESS_DETAIL_PATTERN = re.compile(
    r"((?:[一-龥]{2,8}(?:省|市|区|县|州))+)"
    r"(?:[一-龥\w]+(?:街道|乡|镇|路|街|村|社区|号|楼|室|单元))+"
)
NAME_PATTERN = re.compile(
    r"(被告人?|原告|被告|上诉人|被上诉人|申请人|被申请人|当事人)"
    r"[（(]?([一-龥]{2,4})[）)]?"
)


@dataclass
class RedactionStats:
    names: int = 0
    id_cards: int = 0
    phones: int = 0
    bank_cards: int = 0
    case_numbers: int = 0
    addresses: int = 0


class Redactor:
    def __init__(self) -> None:
        self.stats = RedactionStats()
        self._name_map: dict[str, str] = {}

    def _alloc_alias(self, name: str) -> str:
        if name in self._name_map:
            return self._name_map[name]
        alias = f"[当事人{chr(ord('A') + len(self._name_map) % 26)}]"
        self._name_map[name] = alias
        return alias

    def _redact_names(self, text: str) -> str:
        def repl(match: re.Match[str]) -> str:
            self.stats.names += 1
            return f"{match.group(1)}{self._alloc_alias(match.group(2))}"

        return NAME_PATTERN.sub(repl, text)

    def _redact_ids(self, text: str) -> str:
        def repl(_: re.Match[str]) -> str:
            self.stats.id_cards += 1
            return "[身份证]"

        return ID_CARD_PATTERN.sub(repl, text)

    def _redact_phones(self, text: str) -> str:
        def repl(_: re.Match[str]) -> str:
            self.stats.phones += 1
            return "[手机号]"

        return PHONE_PATTERN.sub(repl, text)

    def _redact_bank(self, text: str) -> str:
        def repl(_: re.Match[str]) -> str:
            self.stats.bank_cards += 1
            return "[银行卡号]"

        return BANK_CARD_PATTERN.sub(repl, text)

    def _redact_case_numbers(self, text: str) -> str:
        def repl(match: re.Match[str]) -> str:
            self.stats.case_numbers += 1
            raw = match.group(1)
            year_match = re.search(r"\d{4}", raw)
            year = year_match.group(0) if year_match else "----"
            return f"({year})某号"

        return CASE_NUMBER_PATTERN.sub(repl, text)

    def _redact_addresses(self, text: str) -> str:
        def repl(match: re.Match[str]) -> str:
            self.stats.addresses += 1
            return match.group(1) + "[详细地址]"

        return ADDRESS_DETAIL_PATTERN.sub(repl, text)

    def redact(self, text: str) -> str:
        if not text:
            return text
        text = self._redact_ids(text)
        text = self._redact_phones(text)
        text = self._redact_bank(text)
        text = self._redact_case_numbers(text)
        text = self._redact_addresses(text)
        text = self._redact_names(text)
        return text


def redact(text: str) -> str:
    """一次性脱敏（每段独立 alias 命名）。"""
    return Redactor().redact(text)


def has_unredacted_pii(text: str) -> bool:
    """简易检测：是否仍含明显 PII，用于推理护栏。"""
    if not text:
        return False
    if ID_CARD_PATTERN.search(text):
        return True
    if PHONE_PATTERN.search(text):
        return True
    if BANK_CARD_PATTERN.search(text):
        return True
    return False


__all__ = ["Redactor", "RedactionStats", "redact", "has_unredacted_pii"]
