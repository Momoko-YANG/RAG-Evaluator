"""Citation judge — LLM-based evidence adherence evaluation.

Assesses whether the AI response properly references evidence from the
context passages.

**Required dataset fields**: ``question``, ``response``, ``contexts``
"""

from __future__ import annotations

from rag_eval_framework.evaluators.common.judge_prompts import CITATION_SYSTEM, CITATION_USER
from rag_eval_framework.evaluators.common.llm_judge import BaseLLMJudge


class CitationJudge(BaseLLMJudge):
    """LLM judge that evaluates citation / evidence adherence."""

    @property
    def name(self) -> str:
        return "citation_judge"

    @property
    def description(self) -> str:
        return "LLM-based assessment of citation and evidence adherence."

    @property
    def required_fields(self) -> list[str]:
        return ["question", "response", "contexts"]

    @property
    def system_prompt(self) -> str:
        return CITATION_SYSTEM

    @property
    def user_prompt_template(self) -> str:
        return CITATION_USER
