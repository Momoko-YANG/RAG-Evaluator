"""Dataset record validation logic."""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from rag_eval_framework.core.schemas import EvaluationRecord


def validate_dataset(
    raw_records: list[dict[str, Any]],
) -> tuple[list[EvaluationRecord], list[str]]:
    """Validate a list of raw dictionaries against the EvaluationRecord schema.

    Parameters
    ----------
    raw_records:
        Parsed JSON objects from the dataset file.

    Returns
    -------
    tuple[list[EvaluationRecord], list[str]]
        A tuple of (valid_records, error_messages).
    """
    valid: list[EvaluationRecord] = []
    errors: list[str] = []

    for idx, raw in enumerate(raw_records):
        record_id = raw.get("id", f"<index-{idx}>")
        try:
            record = EvaluationRecord(**raw)
            valid.append(record)
        except ValidationError as exc:
            for err in exc.errors():
                field = " -> ".join(str(loc) for loc in err["loc"])
                errors.append(f"Record '{record_id}', field '{field}': {err['msg']}")

    return valid, errors
