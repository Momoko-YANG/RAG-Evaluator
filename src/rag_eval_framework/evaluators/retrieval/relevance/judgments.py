"""Relevance policies: one place that decides which retrieved chunks are relevant.

Every ranking metric (Hit, MRR, Recall, nDCG) gets its per-rank relevance from
:meth:`RelevancePolicy.judge`, so two metrics agree whenever they use the same
policy, and differ only when the configuration says they use different ones.

``evidence_match``
    Title + evidence-quote label match (:func:`.matcher.is_relevant_document`).
    Binary.  The relevant set is *open*: any chunk containing the quote counts,
    and how many such chunks exist in the corpus is unknown — so Recall is not
    defined under this policy.

``qrels``
    Judged chunks of a qrels file (:mod:`rag_eval_framework.datasets.qrels`),
    matched by identity.  Graded 0–3; *relevant* means grade ≥ ``relevance_threshold``.
    The relevant set is fixed, so Recall and standard nDCG are defined.  Chunks
    not in the file are *unjudged* (counted, never silently called relevant).

Configuration (per evaluator, under ``evaluator_options.<name>``)::

    relevance_policy: qrels            # or evidence_match
    qrels_path: datasets/my-project/qrels/single-gold-v2.jsonl
    relevance_threshold: 2
    require_reviewed: false
    annotation_version: single-gold-v2
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag_eval_framework.core.schemas import EvaluationRecord
from rag_eval_framework.datasets.qrels import (
    DEFAULT_RELEVANCE_THRESHOLD,
    QrelsEntry,
    load_qrels,
)
from rag_eval_framework.evaluators.retrieval.relevance.matcher import (
    chunk_matches,
    document_identity,
    has_labels,
    is_relevant_document,
    ranked_documents,
)

EVIDENCE_MATCH = "evidence_match"
QRELS = "qrels"
POLICIES = (EVIDENCE_MATCH, QRELS)

JUDGMENT_KIND_LABELS = {
    "evidence_match": "Title + evidence-quote match; open relevant set (no fixed denominator)",
    "single_gold_proxy": "Single-gold qrels (one labelled chunk per question): a proxy, "
    "equal to Hit@k of that exact chunk",
    "pooled_partial": "Known-relevant (pooled) qrels: relevant set may be incomplete",
    "human_reviewed": "Human-reviewed, exhaustive qrels",
}

_QRELS_CACHE: dict[str, dict[str, QrelsEntry]] = {}


def _load(path: str) -> tuple[dict[str, QrelsEntry], str]:
    """Qrels entries and file SHA-256; parsed once per file content (not per path/mtime)."""
    p = Path(path)
    if not p.exists():
        load_qrels(p)  # raises QrelsLoadError with the path
    digest = hashlib.sha256(p.read_bytes()).hexdigest()
    if digest not in _QRELS_CACHE:
        _QRELS_CACHE[digest] = load_qrels(p)
    return _QRELS_CACHE[digest], digest


@dataclass(frozen=True)
class RankedJudgments:
    """Relevance of every stored result of one record, in rank order."""

    policy: str
    kind: str  # judgment kind: evidence_match / single_gold_proxy / pooled_partial / human_reviewed
    relevant: list[bool]
    # qrels: judged grade per rank (None = unjudged); evidence_match: 1 / 0.
    grades: list[int | None]
    # Per-rank id for de-duplication: "qrel:<n>" for a judged chunk, else the chunk's identity.
    ids: list[str]
    matched_by: list[str | None]
    # qrels only: the fixed relevant set and every judged grade (for IDCG).
    relevant_ids: frozenset[str] | None = None
    judged_grades: tuple[int, ...] | None = None
    annotation: dict[str, Any] | None = None

    def first_relevant_rank(self, k: int | None = None) -> int | None:
        for rank, relevant in enumerate(self.relevant[:k] if k else self.relevant, start=1):
            if relevant:
                return rank
        return None


class RelevancePolicy:
    """Parsed relevance options of one evaluator; judges records under that policy."""

    def __init__(
        self,
        evaluator: str,
        options: dict[str, Any],
        *,
        default: str,
        allowed: tuple[str, ...] = POLICIES,
    ) -> None:
        self.evaluator = evaluator
        self.name = options.get("relevance_policy", default)
        if self.name not in allowed:
            raise ValueError(
                f"evaluator_options.{evaluator}.relevance_policy must be one of {list(allowed)}"
                f" (got {self.name!r})."
            )
        threshold = options.get("relevance_threshold", DEFAULT_RELEVANCE_THRESHOLD)
        if isinstance(threshold, bool) or threshold not in (1, 2, 3):
            raise ValueError(
                f"evaluator_options.{evaluator}.relevance_threshold must be 1, 2 or 3."
            )
        self.threshold: int = threshold
        self.require_reviewed = bool(options.get("require_reviewed", False))
        self.annotation_version: str | None = options.get("annotation_version")
        self.qrels_path = ""
        self.qrels_sha256 = ""
        self._qrels: dict[str, QrelsEntry] = {}
        if self.name == QRELS:
            path = options.get("qrels_path")
            if not path:
                raise ValueError(
                    f"evaluator_options.{evaluator}.qrels_path is required "
                    "for relevance_policy qrels."
                )
            self.qrels_path = str(Path(path))
            self._qrels, self.qrels_sha256 = _load(path)

    def describe(self) -> dict[str, Any]:
        """The policy as recorded in reports (so scores can be traced to their labels)."""
        if self.name == EVIDENCE_MATCH:
            return {
                "relevance_policy": EVIDENCE_MATCH,
                "match": (
                    "title in ground_truth_documents and content contains "
                    "a ground_truth_evidence quote"
                ),
            }
        return {
            "relevance_policy": QRELS,
            "qrels_path": self.qrels_path,
            "qrels_sha256": self.qrels_sha256,
            "relevance_threshold": self.threshold,
            "require_reviewed": self.require_reviewed,
            "annotation_version": self.annotation_version,
        }

    def judge(self, record: EvaluationRecord) -> RankedJudgments | str:
        """Judgments for *record*, or the reason it cannot be judged (record is skipped)."""
        if self.name == EVIDENCE_MATCH:
            return self._judge_labels(record)
        return self._judge_qrels(record)

    def _judge_labels(self, record: EvaluationRecord) -> RankedJudgments | str:
        if not has_labels(record):
            return "No ground_truth_documents or ground_truth_evidence label."
        if record.ground_truth_documents and not record.retrieved_documents and record.contexts:
            return (
                "ground_truth_documents needs retrieved_documents with titles; "
                "only contexts are available."
            )
        docs = ranked_documents(record)
        relevant = [is_relevant_document(record, d) for d in docs]
        ids = []
        for position, doc in enumerate(docs):
            identity = document_identity(doc)
            ids.append(
                identity.get("chunk_id") or identity.get("content_sha256") or f"empty:{position}"
            )
        return RankedJudgments(
            policy=EVIDENCE_MATCH,
            kind=EVIDENCE_MATCH,
            relevant=relevant,
            grades=[int(r) for r in relevant],
            ids=ids,
            matched_by=["evidence_match" if r else None for r in relevant],
        )

    def _judge_qrels(self, record: EvaluationRecord) -> RankedJudgments | str:
        entry = self._qrels.get(record.id)
        if entry is None:
            return "No relevance annotation for this question."
        if self.annotation_version and entry.annotation_version != self.annotation_version:
            return (
                f"Annotation version {entry.annotation_version} is not {self.annotation_version}."
            )
        if self.require_reviewed and entry.annotation_status != "reviewed":
            return f"Annotation status is {entry.annotation_status}; reviewed required."
        relevant_ids = frozenset(
            f"qrel:{n}" for n, c in enumerate(entry.relevant_chunks) if c.grade >= self.threshold
        )
        if not relevant_ids:
            return f"No judged chunk with grade >= {self.threshold}; the relevant set is empty."
        relevant: list[bool] = []
        grades: list[int | None] = []
        ids: list[str] = []
        matched_by: list[str | None] = []
        for position, doc in enumerate(ranked_documents(record)):
            identity = document_identity(doc)
            match = next(
                (
                    (n, how)
                    for n, chunk in enumerate(entry.relevant_chunks)
                    if (how := chunk_matches(chunk, identity))
                ),
                None,
            )
            if match is None:
                grades.append(None)
                relevant.append(False)
                ids.append(
                    identity.get("chunk_id")
                    or identity.get("content_sha256")
                    or f"empty:{position}"
                )
                matched_by.append(None)
            else:
                grade = entry.relevant_chunks[match[0]].grade
                grades.append(grade)
                relevant.append(grade >= self.threshold)
                ids.append(f"qrel:{match[0]}")
                matched_by.append(match[1])
        return RankedJudgments(
            policy=QRELS,
            kind=entry.judgment_kind,
            relevant=relevant,
            grades=grades,
            ids=ids,
            matched_by=matched_by,
            relevant_ids=relevant_ids,
            judged_grades=tuple(c.grade for c in entry.relevant_chunks),
            annotation={
                "annotation_version": entry.annotation_version,
                "annotation_status": entry.annotation_status,
                "judgment_scope": entry.judgment_scope,
            },
        )
