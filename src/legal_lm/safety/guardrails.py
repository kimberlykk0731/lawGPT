"""推理护栏：System prompt 强制声明、结构化拒答、输出后处理。

被 demo/openai_api.py、demo/inference.py 调用。三层防护：
1) 输入侧：检测明显未脱敏 PII，拒答或脱敏后再交给模型
2) 高风险结论：死刑/无期/重大刑期 → 转人工
3) 输出侧：风险提示缺失则自动追加
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from legal_lm.data.redact import has_unredacted_pii, redact

DEFAULT_SAFETY_DECLARATION = (
    "（提示：本回答仅供参考，不构成正式法律意见，具体事项请咨询执业律师。）"
)

HIGH_RISK_KEYWORDS = ("死刑", "无期徒刑", "终身监禁")
RECENT_LEGISLATION_PATTERN = re.compile(r"(20\d{2})年\s*(?:新|最新)?(?:司法解释|修正案|立法)")


@dataclass
class GuardrailConfig:
    safety_declaration: str = DEFAULT_SAFETY_DECLARATION
    refuse_high_risk: bool = True
    refuse_unredacted_pii: bool = True
    auto_append_caution: bool = True
    risk_keywords: tuple[str, ...] = HIGH_RISK_KEYWORDS


@dataclass
class GuardrailDecision:
    allow: bool = True
    text: str = ""
    reasons: list[str] = field(default_factory=list)


class Guardrail:
    def __init__(self, config: GuardrailConfig | None = None) -> None:
        self.config = config or GuardrailConfig()

    def pre_input(self, user_text: str) -> GuardrailDecision:
        decision = GuardrailDecision(allow=True, text=user_text)
        if self.config.refuse_unredacted_pii and has_unredacted_pii(user_text):
            decision.text = redact(user_text)
            decision.reasons.append("input_pii_redacted")
        return decision

    def post_output(self, model_text: str) -> GuardrailDecision:
        decision = GuardrailDecision(allow=True, text=model_text)
        if self.config.refuse_high_risk and any(
            kw in model_text for kw in self.config.risk_keywords
        ):
            decision.allow = False
            decision.reasons.append("high_risk_conclusion")
            decision.text = (
                "本案涉及重大刑事量刑结论（如死刑/无期），建议转交人类法律专家审核。\n"
                + self.config.safety_declaration
            )
            return decision
        if RECENT_LEGISLATION_PATTERN.search(model_text):
            decision.reasons.append("recent_legislation_warning")
        if self.config.auto_append_caution:
            if "【风险提示】" not in model_text and self.config.safety_declaration not in model_text:
                decision.text = model_text.rstrip() + "\n\n" + self.config.safety_declaration
                decision.reasons.append("appended_caution")
        return decision


__all__ = ["DEFAULT_SAFETY_DECLARATION", "Guardrail", "GuardrailConfig", "GuardrailDecision"]
