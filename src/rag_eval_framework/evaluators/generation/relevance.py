"""Relevance evaluator — wraps Azure AI Evaluation SDK ``RelevanceEvaluator``.

Checks whether the response is relevant to the user's question.

**Required dataset fields**: ``question``, ``response``
"""

from __future__ import annotations

from typing import Any

from rag_eval_framework.core.schemas import EvaluationRecord
from rag_eval_framework.evaluators.base import EvaluatorResult, EvaluatorStatus
from rag_eval_framework.evaluators.common.azure_sdk import AzureEvaluatorBase
from rag_eval_framework.utils.normalisation import normalise_1_5


class RelevanceEvaluator(AzureEvaluatorBase):
    """Determines whether the response is relevant to the question."""

    @property
    def name(self) -> str:
        return "relevance"

    @property
    def description(self) -> str:
        return "Checks whether the response is relevant to the user's question."

    @property
    def required_fields(self) -> list[str]:
        return ["question", "response"]

    @property
    def _sdk_class_name(self) -> str:
        return "azure.ai.evaluation.RelevanceEvaluator"

    def _build_sdk_input(self, record: EvaluationRecord) -> dict[str, Any]:
        return {
            "query": record.question,
            "response": record.response,
        }

    def _normalise_sdk_output(self, sdk_result: dict[str, Any]) -> EvaluatorResult:
        raw_score = sdk_result.get("relevance", sdk_result.get("gpt_relevance"))
        if raw_score is None:
            return EvaluatorResult(
                score=0.0,
                status=EvaluatorStatus.ERROR,
                reason="SDK did not return a relevance score.",
                raw_output=sdk_result,
            )

        score = normalise_1_5(float(raw_score))
        reason = sdk_result.get(
            "relevance_reason",
            sdk_result.get("gpt_relevance_reason", ""),
        )
        return EvaluatorResult(
            score=score,
            reason=str(reason),
            metadata={"raw_score": raw_score},
            raw_output=sdk_result,
        )
