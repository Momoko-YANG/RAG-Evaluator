"""Hit@k: 1.0 when a relevant chunk is in the top k.

Relevance follows ``relevance_policy`` (default ``evidence_match``).
"""

from __future__ import annotations

from rag_eval_framework.evaluators.retrieval.base import RankEvaluatorBase


class RetrievalHitRateEvaluator(RankEvaluatorBase):
    """Hit@k: 1.0 if a relevant document is within the top k, else 0.0."""

    @property
    def name(self) -> str:
        return "retrieval_hit_rate"

    @property
    def description(self) -> str:
        return "Checks a labelled relevant document is among the top-k retrieved (Hit@k)."

    def _score(self, rank: int | None) -> float:
        return 1.0 if rank is not None else 0.0
