"""Shared parts of the ranking evaluators (Hit, MRR, Recall, nDCG).

* relevance comes from one :class:`RelevancePolicy` (``relevance_policy`` option);
* ``k`` options are validated the same way;
* a k deeper than the collected results is ``not_available``, never scored 0.
"""

from __future__ import annotations

from typing import Any

from rag_eval_framework.core.schemas import EvaluationRecord
from rag_eval_framework.evaluators.base import BaseEvaluator, EvaluatorResult, EvaluatorStatus
from rag_eval_framework.evaluators.retrieval.relevance.judgments import (
    EVIDENCE_MATCH,
    POLICIES,
    RankedJudgments,
    RelevancePolicy,
)

AVAILABLE = "ok"
NOT_AVAILABLE = "not_available"


def collection_depth(record: EvaluationRecord) -> int | None:
    """Number of results requested when the record was collected, if known.

    ``metadata.retrieval_depth`` (set on truncated copies) wins over
    ``metadata.rag_params.topK``.  ``knn`` is not a depth.
    """
    metadata = record.metadata or {}
    for value in (metadata.get("retrieval_depth"), (metadata.get("rag_params") or {}).get("topK")):
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return None


def depth_check(k: int, stored: int, depth: int | None) -> tuple[bool, str]:
    """Whether a metric at k is defined for a list of *stored* results, and a note."""
    if stored >= k:
        return True, ""
    if depth is None:
        return False, f"only {stored} results stored and the collection depth is unknown"
    if depth < k:
        return False, f"collection depth topK={depth} < k={k}"
    return True, f"search returned {stored} of {depth} requested results"


def parse_ks(evaluator: str, value: Any) -> list[int]:
    ks = value if isinstance(value, list) else [value]
    if not ks or any(isinstance(k, bool) or not isinstance(k, int) or k < 1 for k in ks):
        raise ValueError(
            f"evaluator_options.{evaluator}.k must be a positive integer or a list of them."
        )
    return list(dict.fromkeys(ks))


class RetrievalEvaluatorBase(BaseEvaluator):
    """Relevance policy handling shared by all ranking evaluators."""

    default_policy = EVIDENCE_MATCH
    allowed_policies: tuple[str, ...] = POLICIES

    def __init__(self) -> None:
        self._relevance: RelevancePolicy | None = None

    @property
    def required_fields(self) -> list[str]:
        # No response needed: this scores retrieval only.  Labels are checked by the policy.
        return ["question"]

    @property
    def relevance(self) -> RelevancePolicy:
        if self._relevance is None:  # evaluate() without setup(): default options
            self._relevance = RelevancePolicy(
                self.name, {}, default=self.default_policy, allowed=self.allowed_policies
            )
        return self._relevance

    def setup(self, config: Any) -> None:
        options = getattr(config, "evaluator_options", {}).get(self.name, {})
        self._relevance = RelevancePolicy(
            self.name, options, default=self.default_policy, allowed=self.allowed_policies
        )
        self._setup_options(options)

    def _setup_options(self, options: dict[str, Any]) -> None:  # noqa: B027
        """Evaluator-specific options (e.g. ``k``)."""

    def definition(self) -> dict[str, Any]:
        return {**super().definition(), **self.relevance.describe()}

    def _judge(self, record: EvaluationRecord) -> RankedJudgments | EvaluatorResult:
        judged = self.relevance.judge(record)
        if isinstance(judged, str):
            return EvaluatorResult(
                score=0.0,
                status=EvaluatorStatus.SKIPPED,
                reason=judged,
                metadata={"excluded_reason": judged, "relevance_policy": self.relevance.name},
            )
        return judged


class RankEvaluatorBase(RetrievalEvaluatorBase):
    """Hit@k / MRR@k: the rank of the first relevant result within the top k."""

    def __init__(self) -> None:
        super().__init__()
        self._k: int | None = None

    def _setup_options(self, options: dict[str, Any]) -> None:
        k = options.get("k")
        if k is not None and (isinstance(k, bool) or not isinstance(k, int) or k < 1):
            raise ValueError(f"evaluator_options.{self.name}.k must be a positive integer.")
        self._k = k

    def definition(self) -> dict[str, Any]:
        return {**super().definition(), "k": self._k}

    def _score(self, rank: int | None) -> float:
        raise NotImplementedError

    def evaluate(self, record: EvaluationRecord) -> EvaluatorResult:
        judged = self._judge(record)
        if isinstance(judged, EvaluatorResult):
            return judged
        retrieved = len(judged.relevant)
        rank = judged.first_relevant_rank(self._k)
        scope = f"top {self._k}" if self._k is not None else "all"
        if rank is None:
            reason = f"No relevant document in {scope} of {retrieved} retrieved."
        else:
            reason = f"First relevant document at rank {rank} of {retrieved} retrieved."
        return EvaluatorResult(
            score=self._score(rank),
            reason=reason,
            metadata={
                "rank": rank,
                "k": self._k,
                "retrieved": retrieved,
                "relevance_policy": judged.policy,
                "judgment_kind": judged.kind,
            },
        )
