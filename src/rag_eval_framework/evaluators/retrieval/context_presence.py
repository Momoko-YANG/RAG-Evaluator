"""Context presence evaluator — checks whether retrieved contexts are provided.

This is a **Phase 1 scaffold evaluator**.  It performs a simple deterministic
check and will be supplemented by Azure AI Evaluation SDK retrieval-quality
evaluators in Phase 2.
"""

from __future__ import annotations

from rag_eval_framework.core.schemas import EvaluationRecord
from rag_eval_framework.evaluators.base import BaseEvaluator, EvaluatorResult


class ContextPresenceEvaluator(BaseEvaluator):
    """Deterministic check: were any non-empty context passages provided?

    Scoring
    -------
    * **1.0** — at least one non-empty context string is present.
    * **0.0** — no contexts, or all contexts are empty / whitespace.
    """

    @property
    def name(self) -> str:
        return "context_presence"

    @property
    def description(self) -> str:
        return "Checks whether at least one non-empty context passage was retrieved."

    @property
    def required_fields(self) -> list[str]:
        # Retrieval only: no response needed (also scores retrieval-only runs).
        return ["question"]

    def evaluate(self, record: EvaluationRecord) -> EvaluatorResult:
        non_empty = [c for c in record.contexts if c.strip()]
        if non_empty:
            return EvaluatorResult(
                score=1.0,
                reason=f"{len(non_empty)} non-empty context(s) present.",
            )
        return EvaluatorResult(
            score=0.0,
            reason="No non-empty contexts provided.",
        )
