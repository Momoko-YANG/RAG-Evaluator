"""Shared helpers of the end-to-end evaluators."""

from __future__ import annotations

from typing import Any

from rag_eval_framework.core.results import EvaluatorResult, EvaluatorStatus
from rag_eval_framework.core.schemas import EvaluationRecord
from rag_eval_framework.evaluators.base import BaseEvaluator
from rag_eval_framework.utils.normalisation import normalise_for_match

# Phrases that mean "I cannot answer from the documents" (normalised substring match).
# The first is the wording the default RAG system prompt (adapters/rag_eval_api/prompts.py)
# asks for; the Japanese variants cover Japanese-language apps.
DEFAULT_ABSTAIN_PATTERNS = (
    "no information found",
    "情報が見当たりません",
    "見当たりません",
    "見つかりませんでした",
    "見つかりません",
    "わかりません",
    "分かりません",
    "記載がありません",
    "記載されていません",
    "情報がありません",
    "回答できません",
    "お答えできません",
    "i don't know",
    "i do not know",
    "cannot answer",
    "could not find",
    "no relevant information",
)


def skip(reason: str, **metadata: Any) -> EvaluatorResult:
    return EvaluatorResult(
        score=0.0,
        status=EvaluatorStatus.SKIPPED,
        reason=reason,
        metadata={"excluded_reason": reason, **metadata},
    )


def is_retrieval_only(record: EvaluationRecord) -> bool:
    return record.is_retrieval_only()


def detect_abstention(answer: str, patterns: tuple[str, ...] | list[str]) -> str | None:
    """The first abstention phrase found in *answer*, if any."""
    text = normalise_for_match(answer)
    return next((p for p in patterns if normalise_for_match(p) in text), None)


class E2EEvaluatorBase(BaseEvaluator):
    """End-to-end evaluators read the record (and its trace); they never call the system."""

    @property
    def required_fields(self) -> list[str]:
        return ["question"]

    def _options(self, config: Any) -> dict[str, Any]:
        options: dict[str, Any] = getattr(config, "evaluator_options", {}).get(self.name, {})
        return options
