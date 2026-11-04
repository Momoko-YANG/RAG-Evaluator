"""Retrieval layer runner: collect retrieval-only results per variant, then score them.

Each variant (e.g. baseline Top-K 3 / KNN 10, top10, knn50) gets
``<output>/<variant>/retrieval.jsonl`` and its reports; ``manifest.json`` lists the
variants, record counts and aggregate scores.  The ranking metrics do not depend on
how many results were collected, so the full file is scored directly (a k beyond the
collected depth is reported as not available).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rag_eval_framework.config.models import ProjectConfig
from rag_eval_framework.core.protocols import Adapter, Scorer
from rag_eval_framework.core.results import EvaluationRunResult
from rag_eval_framework.runners.collection import collect, record_from_response, score


@dataclass(frozen=True)
class RetrievalVariant:
    name: str
    top_k: int
    knn: int | None = None

    @classmethod
    def parse(cls, spec: str) -> RetrievalVariant:
        """``name:topK[:knn]``, e.g. ``knn50:3:50``."""
        parts = spec.split(":")
        if len(parts) not in (2, 3) or not parts[0]:
            raise ValueError(f"Variant must be name:topK[:knn], got {spec!r}")
        return cls(parts[0], int(parts[1]), int(parts[2]) if len(parts) == 3 else None)


class RetrievalRunner:
    def __init__(
        self,
        make_adapter: Callable[[RetrievalVariant], Adapter],
        *,
        scorer: Scorer | None = None,
        retries: int = 3,
    ) -> None:
        self._make_adapter = make_adapter
        self._scorer = scorer
        self._retries = retries

    def run(
        self,
        questions: list[dict[str, Any]],
        variants: list[RetrievalVariant],
        output_dir: Path,
        config: ProjectConfig,
        *,
        questions_path: Path | None = None,
    ) -> dict[str, EvaluationRunResult]:
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest: dict[str, Any] = {
            "layer": "retrieval",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "questions": len(questions),
            "questions_path": str(questions_path) if questions_path else None,
            "questions_sha256": hashlib.sha256(questions_path.read_bytes()).hexdigest()
            if questions_path
            else None,
            "variants": {},
        }
        results: dict[str, EvaluationRunResult] = {}
        for variant in variants:
            adapter = self._make_adapter(variant)
            path = output_dir / variant.name / "retrieval.jsonl"

            def produce(q: dict[str, Any], adapter: Adapter = adapter) -> dict[str, Any]:
                return record_from_response(q, adapter.query(q["question"]))

            collect(questions, produce, path, retries=self._retries)
            result = score(config, path, output_dir / variant.name, self._scorer)
            results[variant.name] = result
            manifest["variants"][variant.name] = {
                **asdict(variant),
                "records": result.total_records,
                "dataset_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "scores": result.aggregate_scores,
            }
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return results
