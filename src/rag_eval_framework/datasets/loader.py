"""JSONL dataset loader."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from rag_eval_framework.core.schemas import EvaluationRecord
from rag_eval_framework.datasets.validation import validate_dataset

logger = logging.getLogger(__name__)


class DatasetLoadError(Exception):
    """Raised when a dataset file cannot be loaded."""


def load_dataset(
    dataset_path: str | Path,
    *,
    strict: bool = True,
) -> list[EvaluationRecord]:
    """Load and validate a JSONL evaluation dataset.

    Parameters
    ----------
    dataset_path:
        Path to the ``.jsonl`` file.
    strict:
        If ``True`` (default), raise on any invalid record.
        If ``False``, skip invalid records and log warnings.

    Returns
    -------
    list[EvaluationRecord]
        Validated evaluation records.

    Raises
    ------
    DatasetLoadError
        If the file cannot be read or (in strict mode) contains invalid records.
    """
    path = Path(dataset_path)

    if not path.exists():
        raise DatasetLoadError(f"Dataset file not found: {path}")

    if path.suffix != ".jsonl":
        raise DatasetLoadError(f"Dataset must be a .jsonl file, got: {path.suffix}")

    lines: list[str] = []
    try:
        lines = path.read_text(encoding="utf-8").strip().splitlines()
    except OSError as exc:
        raise DatasetLoadError(f"Failed to read dataset file: {exc}") from exc

    if not lines:
        raise DatasetLoadError(f"Dataset file is empty: {path}")

    raw_records: list[dict] = []  # type: ignore[type-arg]
    for line_no, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            msg = f"Line {line_no}: invalid JSON — {exc}"
            if strict:
                raise DatasetLoadError(msg) from exc
            logger.warning(msg)
            continue
        if not isinstance(obj, dict):
            msg = f"Line {line_no}: expected JSON object, got {type(obj).__name__}"
            if strict:
                raise DatasetLoadError(msg)
            logger.warning(msg)
            continue
        raw_records.append(obj)

    records, errors = validate_dataset(raw_records)

    if errors and strict:
        error_summary = "\n".join(errors[:10])
        raise DatasetLoadError(
            f"Dataset validation failed with {len(errors)} error(s):\n{error_summary}"
        )

    if errors:
        for err in errors:
            logger.warning("Validation: %s", err)

    logger.info(
        "Loaded %d valid record(s) from '%s' (%d skipped)",
        len(records),
        path,
        len(errors),
    )
    return records
