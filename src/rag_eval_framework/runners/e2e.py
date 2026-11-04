"""End-to-end layer runner: the real assistant flow, with its trace, then e2e metrics.

The adapter runs the system as a user would (for ``rag_eval_api``: ``/api/rag-eval``
with ``e2e: true`` — the extension chat flow where the model chooses whether and how
to search).  Each record keeps the answer, the retrieved chunks and the trace (tool
calls, search text, citations, TTFT, total time, token usage), which the e2e,
generation and retrieval evaluators and the observability summary read.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rag_eval_framework.config.models import ProjectConfig
from rag_eval_framework.core.protocols import Adapter, Scorer
from rag_eval_framework.core.results import EvaluationRunResult
from rag_eval_framework.runners.collection import collect, record_from_response, score


class E2ERunner:
    def __init__(self, adapter: Adapter, *, scorer: Scorer | None = None, retries: int = 3) -> None:
        self._adapter = adapter
        self._scorer = scorer
        self._retries = retries

    def run(
        self,
        questions: list[dict[str, Any]],
        output_path: Path,
        config: ProjectConfig,
    ) -> EvaluationRunResult:
        collect(
            questions,
            lambda q: record_from_response(q, self._adapter.query(q["question"])),
            output_path,
            retries=self._retries,
        )
        return score(config, output_path, output_path.parent, self._scorer)
