"""Relevance judgments (qrels) for chunk-level retrieval metrics such as Recall@k.

A qrels file is JSONL, one object per question, kept next to (not inside) the
question bank so the bank stays unchanged::

    {"question_id": "q-001",
     "relevant_chunks": [
       {"chunk_id": "<Azure Search key>", "grade": 3},
       {"content_sha256": "<sha256 of the normalised chunk text>", "grade": 2}],
     "annotation_status": "model_draft",
     "judgment_scope": "single_gold",
     "index_identity": "index-my-project",
     "annotation_version": "single-gold-v1"}

Each chunk is identified by at least one of

* ``chunk_id`` — the Azure AI Search document key (stable while the index is
  not rebuilt with different chunking);
* ``chunk_id_sha256`` — SHA-256 of that key, when only the hash was kept;
* ``content_sha256`` — SHA-256 of :func:`normalise_for_match` of the chunk text,
  which also identifies chunks in runs collected before keys were stored.

Grades: 0 irrelevant, 1 related but not enough to answer, 2 partially supports
the answer, 3 fully supports it.  Chunks absent from ``relevant_chunks`` are
*unjudged*, not irrelevant; list grade 0/1 chunks to record them as judged.

``judgment_scope`` says how complete the relevant set is:

* ``single_gold`` — one labelled chunk per question (Recall is a proxy);
* ``pooled`` — chunks judged from a pool of retrieved results (known-relevant
  Recall: other relevant chunks may exist in the corpus);
* ``exhaustive`` — every relevant chunk is believed to be listed.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from rag_eval_framework.utils.normalisation import normalise_for_match

AnnotationStatus = Literal["model_draft", "partially_reviewed", "reviewed"]
JudgmentScope = Literal["single_gold", "pooled", "exhaustive"]
JudgmentKind = Literal["single_gold_proxy", "pooled_partial", "human_reviewed"]

DEFAULT_RELEVANCE_THRESHOLD = 2


class QrelsLoadError(Exception):
    """Raised when a qrels file cannot be read or is invalid."""


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def content_sha256(text: str) -> str:
    """Content identity of a chunk: SHA-256 of its normalised text."""
    return sha256_hex(normalise_for_match(text))


class RelevantChunk(BaseModel):
    """One judged chunk of a question."""

    chunk_id: str | None = None
    chunk_id_sha256: str | None = None
    content_sha256: str | None = None
    grade: int = Field(..., ge=0, le=3)
    # Original labels, kept for audit (not used for matching).
    source_title: str | None = None
    evidence_quote: str | None = None
    label_origin: str | None = None

    @model_validator(mode="after")
    def _needs_identity(self) -> RelevantChunk:
        if not (self.chunk_id or self.chunk_id_sha256 or self.content_sha256):
            raise ValueError("needs chunk_id, chunk_id_sha256 or content_sha256")
        return self


class QrelsEntry(BaseModel):
    """Relevance judgments of one question."""

    question_id: str
    relevant_chunks: list[RelevantChunk] = Field(default_factory=list)
    annotation_status: AnnotationStatus
    judgment_scope: JudgmentScope
    annotation_version: str
    index_identity: str | dict[str, Any] | None = None

    @field_validator("question_id", "annotation_version")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must not be empty")
        return v.strip()

    @property
    def judgment_kind(self) -> JudgmentKind:
        """How metrics computed from these judgments may be described."""
        if self.judgment_scope == "single_gold":
            return "single_gold_proxy"
        if self.judgment_scope == "exhaustive" and self.annotation_status == "reviewed":
            return "human_reviewed"
        return "pooled_partial"

    def relevant(self, threshold: int) -> list[RelevantChunk]:
        """Chunks whose grade reaches *threshold* (binary relevance)."""
        return [c for c in self.relevant_chunks if c.grade >= threshold]


def load_qrels(path: str | Path) -> dict[str, QrelsEntry]:
    """Load a qrels JSONL file keyed by question id."""
    path = Path(path)
    if not path.exists():
        raise QrelsLoadError(f"Qrels file not found: {path}")
    entries: dict[str, QrelsEntry] = {}
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            entry = QrelsEntry(**json.loads(line))
        except (json.JSONDecodeError, TypeError, ValidationError) as exc:
            raise QrelsLoadError(f"{path} line {line_no}: {exc}") from exc
        if entry.question_id in entries:
            raise QrelsLoadError(f"{path} line {line_no}: duplicate {entry.question_id}")
        entries[entry.question_id] = entry
    return entries


def write_qrels(path: str | Path, entries: Iterable[QrelsEntry]) -> None:
    lines = [e.model_dump_json(exclude_none=True) for e in entries]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
