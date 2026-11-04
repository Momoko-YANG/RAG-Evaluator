"""Policy compliance judge — LLM-based policy adherence evaluation.

Assesses whether the AI response complies with policy guidelines extracted
from the record's ``metadata.policy`` field or from ``evaluator_options``.

**Required dataset fields**: ``question``, ``response``
"""

from __future__ import annotations

from rag_eval_framework.evaluators.common.judge_prompts import (
    POLICY_COMPLIANCE_SYSTEM,
    POLICY_COMPLIANCE_USER,
)
from rag_eval_framework.evaluators.common.llm_judge import BaseLLMJudge


class PolicyComplianceJudge(BaseLLMJudge):
    """LLM judge that evaluates compliance with organisational policies."""

    @property
    def name(self) -> str:
        return "policy_compliance_judge"

    @property
    def description(self) -> str:
        return "LLM-based assessment of policy compliance."

    @property
    def required_fields(self) -> list[str]:
        return ["question", "response"]

    @property
    def system_prompt(self) -> str:
        return POLICY_COMPLIANCE_SYSTEM

    @property
    def user_prompt_template(self) -> str:
        return POLICY_COMPLIANCE_USER
