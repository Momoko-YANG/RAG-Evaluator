"""Recall@k against chunk-level relevance judgments (qrels).

    Recall@k(q) = |distinct relevant chunks in the top k| / |relevant chunks|

Relevance comes from the ``qrels`` policy only (:mod:`.relevance.judgments`): Recall
needs a fixed relevant set, which the open ``evidence_match`` policy does not have.
A chunk is relevant when its grade reaches ``relevance_threshold`` (default 2).

Configuration::

    local_mode:
      evaluators: [retrieval_recall]
      evaluator_options:
        retrieval_recall:
          qrels_path: datasets/my-project/qrels/single-gold-v2.jsonl
          k: [3, 10, 50]          # the first k is the evaluator's score
          relevance_threshold: 2
          require_reviewed: false  # true: only annotation_status "reviewed"
          annotation_version: null # set to reject entries of another version

Nothing is silently scored 0:

* a question without qrels, or without a chunk at the threshold, is skipped;
* Recall@k is ``not_available`` when fewer than k results were stored because of
  the collection depth (``metadata.retrieval_depth``, else ``metadata.rag_params.topK``)
  — e.g. ``topK=3, knn=50`` only supports k = 3.  When the search returned fewer
  results than were requested, the list is complete and Recall@k is computed.

Per-k values and the judgment kind (``single_gold_proxy``, ``pooled_partial``,
``human_reviewed``) are kept in the result metadata (``per_k``);
:func:`rag_eval_framework.core.aggregation.summarize_per_k` aggregates them.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any

from rag_eval_framework.core.schemas import EvaluationRecord
from rag_eval_framework.evaluators.base import EvaluatorResult, EvaluatorStatus
from rag_eval_framework.evaluators.retrieval.base import (
    AVAILABLE,
    NOT_AVAILABLE,
    RetrievalEvaluatorBase,
    collection_depth,
    depth_check,
    parse_ks,
)
from rag_eval_framework.evaluators.retrieval.relevance.judgments import QRELS, RankedJudgments

NAME = "retrieval_recall"


def recall_at_k(retrieved_ids: Sequence[str], relevant_ids: set[str], k: int) -> float | None:
    """Recall@k of one ranked list; ``None`` when there is no relevant id."""
    if k <= 0:
        raise ValueError("k must be positive")
    if not relevant_ids:
        return None
    return len(set(retrieved_ids[:k]) & relevant_ids) / len(relevant_ids)


class RetrievalRecallEvaluator(RetrievalEvaluatorBase):
    """Recall@k over distinct relevant chunks of a qrels file."""

    default_policy = QRELS
    allowed_policies = (QRELS,)

    def __init__(self) -> None:
        super().__init__()
        self._ks: list[int] = [3]

    @property
    def name(self) -> str:
        return NAME

    @property
    def description(self) -> str:
        return "Recall@k: share of the judged relevant chunks (qrels) retrieved in the top k."

    def _setup_options(self, options: dict[str, Any]) -> None:
        self._ks = parse_ks(self.name, options.get("k", 3))

    def definition(self) -> dict[str, Any]:
        return {**super().definition(), "ks": self._ks}

    def evaluate(self, record: EvaluationRecord) -> EvaluatorResult:
        judged = self._judge(record)
        if isinstance(judged, EvaluatorResult):
            judged.metadata["ks"] = self._ks
            return judged
        depth = collection_depth(record)
        per_k = {k: self._recall_at(k, judged, depth) for k in self._ks}
        primary = per_k[self._ks[0]]
        relevant = judged.relevant_ids or frozenset()
        metadata = {
            "ks": self._ks,
            "relevance_policy": judged.policy,
            "relevance_threshold": self.relevance.threshold,
            "judgment_kind": judged.kind,
            **(judged.annotation or {}),
            "relevant_count": len(relevant),
            "retrieved": len(judged.relevant),
            "collection_depth": depth,
            "per_k": {str(k): v for k, v in per_k.items()},
        }
        if primary["status"] != AVAILABLE:
            return EvaluatorResult(
                score=0.0,
                status=EvaluatorStatus.SKIPPED,
                reason=f"Recall@{self._ks[0]} not available: {primary['reason']}",
                metadata=metadata,
            )
        return EvaluatorResult(
            score=round(primary["value"], 4),
            reason=(
                f"{primary['relevant_retrieved']} of {len(relevant)} relevant chunks "
                f"in top {self._ks[0]} ({judged.kind})."
            ),
            metadata=metadata,
        )

    @staticmethod
    def _recall_at(k: int, judged: RankedJudgments, depth: int | None) -> dict[str, Any]:
        ok, note = depth_check(k, len(judged.ids), depth)
        if not ok:
            return {"status": NOT_AVAILABLE, "value": None, "reason": note}
        top = judged.ids[:k]
        relevant = set(judged.relevant_ids or ())
        value = recall_at_k(top, relevant, k)
        return {
            "status": AVAILABLE,
            "value": round(value or 0.0, 4),
            "relevant_retrieved": len(set(top) & relevant),
            "unjudged_in_top_k": sum(g is None for g in judged.grades[:k]),
            "duplicate_results": len(top) - len(set(top)),
            "matched_by": dict(Counter(m for m in judged.matched_by[:k] if m)),
            "reason": note,
        }
