"""nDCG@k evaluators.

``retrieval_ndcg`` (pure Python, no SDK)
    nDCG@k for every k in ``k: [3, 10]`` under a relevance policy
    (:mod:`.relevance.judgments`):

    * ``qrels`` (default) — standard graded nDCG.  Gain ``2**grade - 1`` per judged
      chunk (unjudged: 0); IDCG@k from the best k judged grades of the question, so
      the value does not depend on how many results were collected.
    * ``evidence_match`` — binary labels derived from the top k results only, plus an
      unretrieved relevant document when none is found (nDCG = 0).  This is the
      ``document_retrieval`` definition and gives the same nDCG@3.

    Repeated results are dropped before ranking (as the SDK does).  A k beyond the
    collected depth is ``not_available``.

``document_retrieval`` (Azure AI Evaluation SDK ``DocumentRetrievalEvaluator``, no LLM)
    nDCG@3 (the score), XDCG@3, fidelity, top-1 / top-3 max relevance and holes.
    Labels: a chunk in the **top 3** matching ``ground_truth_documents`` /
    ``ground_truth_evidence`` is relevant (4), the others are not (0); when none is
    relevant, an unretrieved relevant document is added (nDCG = 0).  Chunks are
    identified by a hash of their normalised text; identical chunks count once.

    Version 2.0: labels come from the top 3 only.  Version 1.0 labelled every stored
    result, so relevant chunks at ranks 4-10 entered IDCG@3 and the same ranking
    scored lower from a Top-10 file than from its top-3 copy.  Top-3 inputs (all
    reported experiments) give identical values in both versions.

**Required dataset fields**: ``retrieved_documents`` (or ``contexts`` for
evidence-only labels), ``ground_truth_documents`` and/or ``ground_truth_evidence``.
"""

from __future__ import annotations

import hashlib
import logging
import math
from typing import Any

from rag_eval_framework.core.schemas import EvaluationRecord
from rag_eval_framework.evaluators.base import BaseEvaluator, EvaluatorResult, EvaluatorStatus
from rag_eval_framework.evaluators.retrieval.base import (
    AVAILABLE,
    NOT_AVAILABLE,
    RetrievalEvaluatorBase,
    collection_depth,
    depth_check,
    parse_ks,
)
from rag_eval_framework.evaluators.retrieval.relevance.judgments import (
    EVIDENCE_MATCH,
    QRELS,
    RankedJudgments,
)
from rag_eval_framework.evaluators.retrieval.relevance.matcher import (
    is_relevant_document,
    ranked_documents,
)
from rag_eval_framework.utils.normalisation import normalise_for_match

logger = logging.getLogger(__name__)

SDK_K = 3  # DocumentRetrievalEvaluator's default k (nDCG@3)
RELEVANT = 4
NOT_RELEVANT = 0
UNRETRIEVED_RELEVANT_ID = "__unretrieved_relevant__"

# Numeric outputs of the SDK kept in the result metadata.
_METRICS = (
    "ndcg@3",
    "xdcg@3",
    "fidelity",
    "top1_relevance",
    "top3_max_relevance",
    "holes",
    "holes_ratio",
    "total_retrieved_documents",
    "total_ground_truth_documents",
)


def _document_id(doc: dict[str, Any]) -> str:
    text = normalise_for_match(f"{doc.get('title', '')}\n{doc.get('content', '')}")
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def build_sdk_inputs(
    record: EvaluationRecord,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return ``(retrieval_ground_truth, retrieved_documents)`` for the SDK.

    Only the top :data:`SDK_K` results are labelled, so nDCG@3 does not depend on
    how many results the record stores.  ``relevance_score`` is derived from the rank
    so the SDK ranks documents exactly as the retriever returned them.
    """
    documents = ranked_documents(record)[:SDK_K]
    ground_truth: list[dict[str, Any]] = []
    retrieved: list[dict[str, Any]] = []
    seen: set[str] = set()
    for rank, doc in enumerate(documents):
        doc_id = _document_id(doc)
        if doc_id in seen:
            continue
        seen.add(doc_id)
        retrieved.append({"document_id": doc_id, "relevance_score": float(len(documents) - rank)})
        label = RELEVANT if is_relevant_document(record, doc) else NOT_RELEVANT
        ground_truth.append({"document_id": doc_id, "query_relevance_label": label})
    if not any(g["query_relevance_label"] == RELEVANT for g in ground_truth):
        ground_truth.append(
            {"document_id": UNRETRIEVED_RELEVANT_ID, "query_relevance_label": RELEVANT}
        )
    return ground_truth, retrieved


class AzureDocumentRetrievalEvaluator(BaseEvaluator):
    """nDCG@3 and related ranking metrics via the Azure AI Evaluation SDK (no LLM)."""

    def __init__(self) -> None:
        self._sdk_evaluator: Any = None

    @property
    def name(self) -> str:
        return "document_retrieval"

    @property
    def description(self) -> str:
        return "Azure SDK — nDCG@3 and ranking metrics from labelled relevant chunks (no LLM)."

    @property
    def version(self) -> str:
        return "2.0"

    def definition(self) -> dict[str, Any]:
        try:
            from importlib.metadata import version

            sdk: str | None = version("azure-ai-evaluation")
        except Exception:  # not installed
            sdk = None
        return {
            **super().definition(),
            "relevance_policy": EVIDENCE_MATCH,
            "labels_from": f"top {SDK_K} results",
            "k": SDK_K,
            "backend": "azure-ai-evaluation DocumentRetrievalEvaluator",
            "azure_ai_evaluation_version": sdk,
        }

    @property
    def required_fields(self) -> list[str]:
        # Labels are checked in evaluate(): either of the two label fields is enough.
        return ["question"]

    def setup(self, config: Any) -> None:
        try:
            from azure.ai.evaluation import (  # type: ignore[attr-defined,unused-ignore]
                DocumentRetrievalEvaluator,
            )
        except ImportError:
            logger.warning(
                "Evaluator '%s' disabled: install the Azure SDK with "
                "pip install 'rag-eval-framework[azure]'.",
                self.name,
            )
            return
        self._sdk_evaluator = DocumentRetrievalEvaluator(
            ground_truth_label_min=NOT_RELEVANT, ground_truth_label_max=RELEVANT
        )

    def evaluate(self, record: EvaluationRecord) -> EvaluatorResult:
        if not record.ground_truth_documents and not record.ground_truth_evidence:
            return EvaluatorResult(
                score=0.0,
                status=EvaluatorStatus.SKIPPED,
                reason="No ground_truth_documents or ground_truth_evidence label.",
            )
        if record.ground_truth_documents and not record.retrieved_documents and record.contexts:
            return EvaluatorResult(
                score=0.0,
                status=EvaluatorStatus.SKIPPED,
                reason=(
                    "ground_truth_documents needs retrieved_documents with titles; "
                    "only contexts are available."
                ),
            )
        if self._sdk_evaluator is None:
            return EvaluatorResult(
                score=0.0,
                status=EvaluatorStatus.ERROR,
                reason=(
                    f"Evaluator '{self.name}' is not configured: "
                    "pip install 'rag-eval-framework[azure]'."
                ),
            )

        ground_truth, retrieved = build_sdk_inputs(record)
        if not retrieved:
            return EvaluatorResult(score=0.0, reason="Nothing was retrieved.")

        try:
            result = self._sdk_evaluator(
                retrieval_ground_truth=ground_truth, retrieved_documents=retrieved
            )
        except Exception as exc:
            logger.exception("SDK evaluator '%s' raised an error.", self.name)
            return EvaluatorResult(
                score=0.0, status=EvaluatorStatus.ERROR, reason=f"SDK error: {exc}"
            )

        # Newer SDKs nest the metrics under "document_retrieval_properties".
        props = result.get("document_retrieval_properties") or result
        metrics = {key: props[key] for key in _METRICS if key in props}
        ndcg = max(0.0, min(1.0, float(props.get("ndcg@3", 0.0))))
        return EvaluatorResult(
            score=round(ndcg, 4),
            reason=(
                f"nDCG@3 {ndcg:.2f}; top-1 relevance {metrics.get('top1_relevance', '-')}, "
                f"fidelity {metrics.get('fidelity', '-')}."
            ),
            metadata=metrics,
            raw_output=dict(result),
        )


def ndcg_at_k(gains: list[float], ideal_gains: list[float], k: int) -> float:
    """DCG@k / IDCG@k with the log2(rank + 1) discount; 0 when IDCG is 0."""

    def dcg(values: list[float]) -> float:
        return sum(g / math.log2(rank + 1) for rank, g in enumerate(values[:k], start=1))

    ideal = dcg(sorted(ideal_gains, reverse=True))
    return dcg(gains) / ideal if ideal > 0 else 0.0


class RetrievalNDCGEvaluator(RetrievalEvaluatorBase):
    """nDCG@k under an explicit relevance policy, without the Azure SDK."""

    default_policy = QRELS

    def __init__(self) -> None:
        super().__init__()
        self._ks: list[int] = [3]

    @property
    def name(self) -> str:
        return "retrieval_ndcg"

    @property
    def description(self) -> str:
        return "nDCG@k from qrels grades (or top-k evidence labels), pure Python."

    def _setup_options(self, options: dict[str, Any]) -> None:
        self._ks = parse_ks(self.name, options.get("k", 3))

    def definition(self) -> dict[str, Any]:
        gain = "2**grade - 1" if self.relevance.name == QRELS else "binary, labels from top k"
        return {
            **super().definition(),
            "ks": self._ks,
            "gain": gain,
            "discount": "log2(rank + 1)",
            "duplicates": "dropped before ranking",
        }

    def evaluate(self, record: EvaluationRecord) -> EvaluatorResult:
        judged = self._judge(record)
        if isinstance(judged, EvaluatorResult):
            judged.metadata["ks"] = self._ks
            return judged
        depth = collection_depth(record)
        per_k = {k: self._ndcg_at(k, judged, depth) for k in self._ks}
        primary = per_k[self._ks[0]]
        metadata = {
            "ks": self._ks,
            "relevance_policy": judged.policy,
            "judgment_kind": judged.kind,
            **(judged.annotation or {}),
            "retrieved": len(judged.ids),
            "collection_depth": depth,
            "per_k": {str(k): v for k, v in per_k.items()},
        }
        if primary["status"] != AVAILABLE:
            return EvaluatorResult(
                score=0.0,
                status=EvaluatorStatus.SKIPPED,
                reason=f"nDCG@{self._ks[0]} not available: {primary['reason']}",
                metadata=metadata,
            )
        return EvaluatorResult(
            score=primary["value"],
            reason=f"nDCG@{self._ks[0]} {primary['value']:.2f} ({judged.kind}).",
            metadata=metadata,
        )

    @staticmethod
    def _ndcg_at(k: int, judged: RankedJudgments, depth: int | None) -> dict[str, Any]:
        ok, note = depth_check(k, len(judged.ids), depth)
        if not ok:
            return {"status": NOT_AVAILABLE, "value": None, "reason": note}
        seen: set[str] = set()
        grades: list[int | None] = []
        for doc_id, grade in zip(judged.ids[:k], judged.grades[:k], strict=True):
            if doc_id not in seen:
                seen.add(doc_id)
                grades.append(grade)
        if judged.policy == QRELS:
            gains = [2.0**g - 1 if g else 0.0 for g in grades]
            ideal = [2.0**g - 1 for g in judged.judged_grades or ()]
        else:  # evidence_match: binary labels from the top k, as document_retrieval
            gains = [1.0 if g else 0.0 for g in grades]
            ideal = gains if any(gains) else [1.0]
        return {
            "status": AVAILABLE,
            "value": round(max(0.0, min(1.0, ndcg_at_k(gains, ideal, k))), 4),
            "unjudged_in_top_k": sum(g is None for g in grades),
            "duplicate_results": min(k, len(judged.ids)) - len(grades),
            "reason": note,
        }
