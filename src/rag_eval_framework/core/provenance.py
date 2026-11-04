"""Where the scores of a run came from: framework version, dataset hash, retrieval settings."""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path
from typing import Any

from rag_eval_framework import __version__
from rag_eval_framework.core.schemas import EvaluationRecord

# rag_params keys that identify what was searched and how (written by the collector).
RETRIEVAL_KEYS = (
    "source",
    "searchName",
    "indexName",
    "topK",
    "knn",
    "retrievalOnly",
    "vectorOnly",
    "deployment",
)


def build_provenance(dataset_path: str, records: list[EvaluationRecord]) -> dict[str, Any]:
    """Framework version, dataset SHA-256 and the distinct retrieval settings in the data."""
    path = Path(dataset_path)
    settings: Counter[tuple[tuple[str, Any], ...]] = Counter()
    depths: Counter[int] = Counter()
    for record in records:
        params = (record.metadata or {}).get("rag_params") or {}
        settings[tuple((k, params[k]) for k in RETRIEVAL_KEYS if k in params)] += 1
        depth = (record.metadata or {}).get("retrieval_depth")
        if isinstance(depth, int):
            depths[depth] += 1
    return {
        "framework_version": __version__,
        "dataset": {
            "path": str(dataset_path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None,
            "records": len(records),
        },
        "retrieval_settings": [
            {**dict(key), "records": n} for key, n in settings.most_common() if key
        ],
        "truncated_retrieval_depths": dict(depths),
    }
