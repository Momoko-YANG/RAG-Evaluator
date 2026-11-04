"""How a retrieved document is matched to ground truth.

Two matching rules, used by the relevance policies in :mod:`.judgments`:

* **Label match** (policy ``evidence_match``) — the document's ``title`` is one of the
  record's ``ground_truth_documents`` and its ``content`` contains one of the
  ``ground_truth_evidence`` quotes (both after :func:`normalise_for_match`).  When only
  one of the two labels is given, only that one is checked.  Any chunk that satisfies
  the rule counts, e.g. a neighbouring chunk that overlaps the quoted passage.
* **Identity match** (policy ``qrels``) — the document *is* a judged chunk of a qrels
  file: same Azure AI Search key (``chunk_id``), same key hash, or — for runs collected
  before keys were stored — same hash of the normalised text.
"""

from __future__ import annotations

from typing import Any

from rag_eval_framework.core.schemas import EvaluationRecord
from rag_eval_framework.datasets.qrels import RelevantChunk, content_sha256, sha256_hex
from rag_eval_framework.utils.normalisation import normalise_for_match


def ranked_documents(record: EvaluationRecord) -> list[dict[str, Any]]:
    """Retrieved documents in rank order (contexts when no structured documents)."""
    return record.retrieved_documents or [{"content": c} for c in record.contexts if c.strip()]


def has_labels(record: EvaluationRecord) -> bool:
    return bool(record.ground_truth_documents or record.ground_truth_evidence)


def is_relevant_document(record: EvaluationRecord, doc: dict[str, Any]) -> bool:
    """Label match: ``doc`` matches the record's title and/or evidence labels."""
    titles = {normalise_for_match(t) for t in record.ground_truth_documents}
    if titles and normalise_for_match(str(doc.get("title", ""))) not in titles:
        return False
    evidence = [normalise_for_match(e) for e in record.ground_truth_evidence]
    content = normalise_for_match(str(doc.get("content", "")))
    return not evidence or any(e in content for e in evidence)


def first_relevant_rank(record: EvaluationRecord, k: int | None = None) -> int | None:
    """1-based rank of the first label-matching document in the top k, if any."""
    documents = ranked_documents(record)
    if k is not None:
        documents = documents[:k]
    for rank, doc in enumerate(documents, start=1):
        if is_relevant_document(record, doc):
            return rank
    return None


def document_identity(doc: dict[str, Any]) -> dict[str, str]:
    """Identities of a retrieved document: ``chunk_id`` (when stored) and content hash.

    The rank-based ``id`` written by the collector (``doc-0``, ``doc-1`` …) is
    deliberately not used.
    """
    identity: dict[str, str] = {}
    chunk_id = str(doc.get("chunk_id") or "").strip()
    if chunk_id:
        identity["chunk_id"] = chunk_id
    content = str(doc.get("content") or "")
    if content.strip():
        identity["content_sha256"] = content_sha256(content)
    return identity


def chunk_matches(chunk: RelevantChunk, identity: dict[str, str]) -> str | None:
    """Identity match: the identity used when *chunk* is the retrieved document, else ``None``.

    Keys decide when both sides have one; the content hash is only a fallback,
    since two chunks with different keys can have identical text.
    """
    retrieved_id = identity.get("chunk_id")
    if retrieved_id and (chunk.chunk_id or chunk.chunk_id_sha256):
        if chunk.chunk_id:
            return "chunk_id" if chunk.chunk_id == retrieved_id else None
        return "chunk_id_sha256" if chunk.chunk_id_sha256 == sha256_hex(retrieved_id) else None
    if chunk.content_sha256 and chunk.content_sha256 == identity.get("content_sha256"):
        return "content_sha256"
    return None
