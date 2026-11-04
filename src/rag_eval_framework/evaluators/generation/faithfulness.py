"""Faithfulness — is the answer supported by the retrieved chunks?

* ``groundedness`` — Azure AI Evaluation SDK ``GroundednessEvaluator`` (1–5 → 0–1).
* ``hallucination_judge`` — LLM judge that looks for claims the chunks do not support.

**Required dataset fields**: ``question``, ``response``, ``contexts``
"""

from __future__ import annotations

from typing import Any

from rag_eval_framework.core.schemas import EvaluationRecord
from rag_eval_framework.evaluators.base import EvaluatorResult, EvaluatorStatus
from rag_eval_framework.evaluators.common.azure_sdk import AzureEvaluatorBase
from rag_eval_framework.evaluators.common.judge_prompts import (
    HALLUCINATION_SYSTEM,
    HALLUCINATION_USER,
)
from rag_eval_framework.evaluators.common.llm_judge import BaseLLMJudge
from rag_eval_framework.utils.normalisation import normalise_1_5


class GroundednessEvaluator(AzureEvaluatorBase):
    """Determines whether the response is grounded in the retrieved contexts."""

    @property
    def name(self) -> str:
        return "groundedness"

    @property
    def description(self) -> str:
        return "Checks whether the response is grounded in the retrieved contexts."

    @property
    def required_fields(self) -> list[str]:
        return ["question", "response", "contexts"]

    @property
    def _sdk_class_name(self) -> str:
        return "azure.ai.evaluation.GroundednessEvaluator"

    def _build_sdk_input(self, record: EvaluationRecord) -> dict[str, Any]:
        return {
            "query": record.question,
            "response": record.response,
            "context": "\n\n".join(record.contexts),
        }

    def _normalise_sdk_output(self, sdk_result: dict[str, Any]) -> EvaluatorResult:
        raw_score = sdk_result.get("groundedness", sdk_result.get("gpt_groundedness"))
        if raw_score is None:
            return EvaluatorResult(
                score=0.0,
                status=EvaluatorStatus.ERROR,
                reason="SDK did not return a groundedness score.",
                raw_output=sdk_result,
            )

        # Azure SDK typically returns 1-5; normalise to 0.0-1.0
        score = normalise_1_5(float(raw_score))
        reason = sdk_result.get(
            "groundedness_reason",
            sdk_result.get("gpt_groundedness_reason", ""),
        )
        return EvaluatorResult(
            score=score,
            reason=str(reason),
            metadata={"raw_score": raw_score},
            raw_output=sdk_result,
        )


class HallucinationJudge(BaseLLMJudge):
    """LLM judge that detects hallucinated or unsupported claims."""

    @property
    def name(self) -> str:
        return "hallucination_judge"

    @property
    def description(self) -> str:
        return "LLM-based detection of hallucinated or unsupported claims."

    @property
    def required_fields(self) -> list[str]:
        return ["question", "response", "contexts"]

    @property
    def system_prompt(self) -> str:
        return HALLUCINATION_SYSTEM

    @property
    def user_prompt_template(self) -> str:
        return HALLUCINATION_USER
