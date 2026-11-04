"""Response completeness evaluator — wraps Azure AI Evaluation SDK.

Checks whether the response fully addresses the question given the
ground-truth answer.

**Required dataset fields**: ``question``, ``response``, ``ground_truth_answer``
"""

from __future__ import annotations

from typing import Any

from rag_eval_framework.core.schemas import EvaluationRecord
from rag_eval_framework.evaluators.base import EvaluatorResult, EvaluatorStatus
from rag_eval_framework.evaluators.common.azure_sdk import AzureEvaluatorBase
from rag_eval_framework.utils.normalisation import normalise_1_5


class ResponseCompletenessEvaluator(AzureEvaluatorBase):
    """Determines whether the response completely answers the question."""

    @property
    def name(self) -> str:
        return "response_completeness"

    @property
    def description(self) -> str:
        return "Checks whether the response completely answers the question."

    @property
    def required_fields(self) -> list[str]:
        return ["question", "response", "ground_truth_answer"]

    @property
    def _sdk_class_name(self) -> str:
        return "azure.ai.evaluation.ResponseCompletenessEvaluator"

    def _build_sdk_input(self, record: EvaluationRecord) -> dict[str, Any]:
        return {
            "query": record.question,
            "response": record.response,
            "ground_truth": record.ground_truth_answer,
        }

    def _normalise_sdk_output(self, sdk_result: dict[str, Any]) -> EvaluatorResult:
        raw_score = sdk_result.get(
            "response_completeness",
            sdk_result.get("gpt_response_completeness"),
        )
        if raw_score is None:
            return EvaluatorResult(
                score=0.0,
                status=EvaluatorStatus.ERROR,
                reason="SDK did not return a response_completeness score.",
                raw_output=sdk_result,
            )

        score = normalise_1_5(float(raw_score))
        reason = sdk_result.get(
            "response_completeness_reason",
            sdk_result.get("gpt_response_completeness_reason", ""),
        )
        return EvaluatorResult(
            score=score,
            reason=str(reason),
            metadata={"raw_score": raw_score},
            raw_output=sdk_result,
        )
