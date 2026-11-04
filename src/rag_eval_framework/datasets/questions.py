"""Question banks (CSV) → question dicts for the collectors and layer runners.

Required column ``question``; optional ``id``, ``ground_truth_answer``,
``ground_truth_documents`` / ``ground_truth_evidence`` (``|``-separated),
``expected_tools`` / ``expected_query_terms`` (``|``-separated),
``expected_abstain`` (true/false) and ``metadata_*`` (packed into ``metadata``).
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Retrieval labels: ``|``-separated in the CSV, lists in the JSONL record.
LABEL_COLUMNS = ("ground_truth_documents", "ground_truth_evidence")
# End-to-end expectations: ``|``-separated lists and a true/false column.
EXPECTATION_LIST_COLUMNS = ("expected_tools", "expected_query_terms")
EXPECTATION_BOOL_COLUMNS = ("expected_abstain",)


def read_question_bank(csv_path: Path) -> list[dict[str, Any]]:
    """Read questions CSV and return a list of dicts.

    Required column : ``question``
    Optional columns: ``id``, ``ground_truth_answer``, ``metadata_*``
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Questions file not found: {csv_path}")

    rows: list[dict[str, Any]] = []
    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "question" not in (reader.fieldnames or []):
            raise ValueError(f"CSV must have a 'question' column. Found: {reader.fieldnames}")

        for idx, row in enumerate(reader, start=1):
            question = row.get("question", "").strip()
            if not question:
                logger.warning("Row %d: empty question — skipping.", idx)
                continue

            record: dict[str, Any] = {
                "id": row.get("id", f"q-{idx:04d}").strip(),
                "question": question,
                "ground_truth_answer": row.get("ground_truth_answer", "").strip(),
            }
            for column in EXPECTATION_LIST_COLUMNS:
                values = [v.strip() for v in (row.get(column) or "").split("|") if v.strip()]
                if values:
                    record[column] = values
            for column in EXPECTATION_BOOL_COLUMNS:
                value = (row.get(column) or "").strip().lower()
                if value in {"true", "1", "yes"}:
                    record[column] = True
                elif value in {"false", "0", "no"}:
                    record[column] = False
            for label in LABEL_COLUMNS:
                values = [v.strip() for v in (row.get(label) or "").split("|") if v.strip()]
                if values:
                    record[label] = values

            # Pack metadata_* columns
            metadata: dict[str, str] = {}
            for key, value in row.items():
                if key.startswith("metadata_") and value:
                    meta_key = key[len("metadata_") :]
                    metadata[meta_key] = value.strip()
            record["metadata"] = metadata

            rows.append(record)

    logger.info("Loaded %d questions from %s", len(rows), csv_path)
    return rows


# ---------------------------------------------------------------------------
# Collection loop
# ---------------------------------------------------------------------------
