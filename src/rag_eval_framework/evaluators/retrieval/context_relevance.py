"""Retrieval evaluator — wraps Azure AI Evaluation SDK ``RetrievalEvaluator``.

Checks whether the retrieved contexts are useful for answering the question.

**Required dataset fields**: ``question``, ``contexts`` (the answer is not used,
so retrieval-only runs can be scored too)
"""

from __future__ import annotations

from typing import Any

from rag_eval_framework.core.schemas import EvaluationRecord
from rag_eval_framework.evaluators.base import EvaluatorResult, EvaluatorStatus
from rag_eval_framework.evaluators.common.azure_sdk import AzureEvaluatorBase
from rag_eval_framework.utils.normalisation import normalise_1_5


class RetrievalEvaluator(AzureEvaluatorBase):
    """Determines whether the retrieved contexts are useful for answering."""

    @property
    def name(self) -> str:
        return "retrieval"

    @property
    def description(self) -> str:
        return "Checks whether retrieved contexts are useful for answering the question."

    @property
    def required_fields(self) -> list[str]:
        return ["question", "contexts"]

    @property
    def _sdk_class_name(self) -> str:
        return "azure.ai.evaluation.RetrievalEvaluator"

    def _build_sdk_input(self, record: EvaluationRecord) -> dict[str, Any]:
        # The SDK signature is RetrievalEvaluator(query=..., context=...).
        return {
            "query": record.question,
            "context": "\n\n".join(record.contexts),
        }

    def _normalise_sdk_output(self, sdk_result: dict[str, Any]) -> EvaluatorResult:
        raw_score = sdk_result.get("retrieval", sdk_result.get("gpt_retrieval"))
        if raw_score is None:
            return EvaluatorResult(
                score=0.0,
                status=EvaluatorStatus.ERROR,
                reason="SDK did not return a retrieval score.",
                raw_output=sdk_result,
            )

        score = normalise_1_5(float(raw_score))
        reason = sdk_result.get(
            "retrieval_reason",
            sdk_result.get("gpt_retrieval_reason", ""),
        )
        return EvaluatorResult(
            score=score,
            reason=str(reason),
            metadata={"raw_score": raw_score},
            raw_output=sdk_result,
        )
