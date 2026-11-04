"""Single-gold qrels from a question bank, without changing the bank.

Each question gets the one chunk it was written from, identified by the hashes
the bank already stores (no new labels are made up):

* ``metadata_source_key_hash`` → ``chunk_id_sha256`` (SHA-256 of the index key);
* ``metadata_content_hash``    → ``content_sha256`` (SHA-256 of the normalised text).

The result is ``judgment_scope: single_gold`` — Recall@k from it is a proxy that
counts only this chunk, not complete recall.  Questions without the hashes are
returned as missing, never given an empty relevant set.

``scripts/prepare_dataset.py`` uses this to write ``qrels/<version>.jsonl``.
"""

from __future__ import annotations

from typing import Any

from rag_eval_framework.datasets.qrels import AnnotationStatus, QrelsEntry, RelevantChunk

DEFAULT_GRADE = 3


def build_single_gold_entries(
    rows: list[dict[str, str]],
    *,
    sets: set[str],
    annotation_version: str,
    annotation_status: AnnotationStatus = "model_draft",
    index_identity: str | dict[str, Any] | None = None,
) -> tuple[list[QrelsEntry], list[str]]:
    """Qrels entries of the rows in *sets*, and the ids that have no chunk hashes."""
    entries: list[QrelsEntry] = []
    missing: list[str] = []
    for row in rows:
        if (row.get("metadata_set") or "").strip() not in sets:
            continue
        key_hash = (row.get("metadata_source_key_hash") or "").strip()
        text_hash = (row.get("metadata_content_hash") or "").strip()
        if not (key_hash or text_hash):
            missing.append(row["id"])
            continue
        chunk = RelevantChunk(
            chunk_id_sha256=key_hash or None,
            content_sha256=text_hash or None,
            grade=DEFAULT_GRADE,
            source_title=(row.get("ground_truth_documents") or "").strip() or None,
            evidence_quote=(row.get("ground_truth_evidence") or "").strip() or None,
            label_origin=(row.get("metadata_label_status") or "").strip() or None,
        )
        entries.append(
            QrelsEntry(
                question_id=row["id"],
                relevant_chunks=[chunk],
                annotation_status=annotation_status,
                judgment_scope="single_gold",
                annotation_version=annotation_version,
                index_identity=index_identity,
            )
        )
    return entries, missing
