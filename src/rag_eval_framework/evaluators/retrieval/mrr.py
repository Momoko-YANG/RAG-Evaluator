"""MRR@k: reciprocal rank of the first relevant chunk in the top k.

Relevance follows ``relevance_policy`` (default ``evidence_match``).
"""

from __future__ import annotations

from rag_eval_framework.evaluators.retrieval.base import RankEvaluatorBase


class RetrievalMRREvaluator(RankEvaluatorBase):
    """Reciprocal rank of the first relevant document (MRR@k when averaged)."""

    @property
    def name(self) -> str:
        return "retrieval_mrr"

    @property
    def description(self) -> str:
        return "Reciprocal rank of the first labelled relevant retrieved document (MRR@k)."

    def _score(self, rank: int | None) -> float:
        return round(1.0 / rank, 4) if rank is not None else 0.0
