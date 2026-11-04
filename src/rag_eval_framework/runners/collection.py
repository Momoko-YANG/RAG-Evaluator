"""Shared by the layer runners: checkpointed collection with retries, then scoring.

Collecting and scoring are separate: a layer runner produces a JSONL dataset
(one record per question, written as it goes so a run can resume), then hands it
to the scoring runner (``LocalRunner`` by default) and the report engine.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from rag_eval_framework.config.models import ProjectConfig
from rag_eval_framework.core.protocols import Scorer
from rag_eval_framework.core.results import EvaluationRunResult
from rag_eval_framework.core.schemas import EMPTY_ANSWER
from rag_eval_framework.reporting.engine import ReportEngine

logger = logging.getLogger(__name__)

TRANSIENT_HTTP = (429, 500, 502, 503, 504)
TRANSIENT_ERRORS = frozenset(
    {
        "ReadTimeout",
        "ConnectTimeout",
        "ConnectionError",
        "Timeout",
        "APITimeoutError",
        "APIConnectionError",
        "RateLimitError",
        "InternalServerError",
    }
)
QUESTION_FIELDS = (
    "ground_truth_answer",
    "ground_truth_documents",
    "ground_truth_evidence",
    "expected_tools",
    "expected_abstain",
    "expected_query_terms",
)


def is_transient(exc: Exception) -> bool:
    """Worth retrying: rate limits, 5xx and network timeouts."""
    return type(exc).__name__ in TRANSIENT_ERRORS or any(
        f"HTTP {code}" in str(exc) for code in TRANSIENT_HTTP
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def collect(
    items: list[dict[str, Any]],
    produce: Callable[[dict[str, Any]], dict[str, Any]],
    output: Path,
    *,
    retries: int = 3,
    backoff_s: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
) -> list[dict[str, Any]]:
    """Produce one record per item, appending to *output*; resume from its existing prefix.

    A transient error is retried (``retries`` attempts in total); any other error
    stops the run with the checkpoint kept.  ``metadata.eval_latency_ms`` is the
    wall time of the successful item including retries; ``metadata.attempts`` counts
    the requests.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    done = read_jsonl(output) if output.exists() else []
    ids = [item["id"] for item in items]
    if [r["id"] for r in done] != ids[: len(done)]:
        raise ValueError(f"{output} does not match the question order; use a new output path.")
    if done:
        logger.info("Resuming %s after %d of %d records.", output, len(done), len(items))
    with output.open("a", encoding="utf-8") as checkpoint:
        for item in items[len(done) :]:
            started = time.perf_counter()
            attempt = 0
            while True:
                attempt += 1
                try:
                    record = produce(item)
                    break
                except Exception as exc:
                    if attempt >= retries or not is_transient(exc):
                        raise
                    logger.warning(
                        "%s: transient %s, retry %d", item["id"], type(exc).__name__, attempt
                    )
                    sleep(backoff_s * attempt)
            metadata = record.setdefault("metadata", {})
            metadata["eval_latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
            metadata["attempts"] = attempt
            if isinstance(record.get("trace"), dict):
                record["trace"]["attempts"] = attempt
            checkpoint.write(json.dumps(record, ensure_ascii=False) + "\n")
            checkpoint.flush()
            done.append(record)
            if len(done) % 10 == 0 or len(done) == len(items):
                logger.info("%s: %d / %d", output.name, len(done), len(items))
    return done


def record_from_response(item: dict[str, Any], response: Any) -> dict[str, Any]:
    """A dataset record from a question dict and an adapter's ``RagResponse``.

    An empty answer is a result, not a collection error: the record gets
    :data:`EMPTY_ANSWER`, ``metadata.empty_answer`` and a ``trace.error``, so it is
    kept (and fails ``task_success``) instead of making the dataset invalid.
    """
    record: dict[str, Any] = {
        "id": item["id"],
        "question": item["question"],
        "response": response.response,
        "contexts": response.contexts,
        **{k: item[k] for k in QUESTION_FIELDS if item.get(k) not in (None, "", [])},
        "metadata": {**item.get("metadata", {}), **response.metadata},
    }
    if response.retrieved_documents:
        record["retrieved_documents"] = response.retrieved_documents
    if getattr(response, "trace", None) is not None:
        record["trace"] = response.trace
    return mark_empty_answer(record)


def mark_empty_answer(record: dict[str, Any]) -> dict[str, Any]:
    """Give a record without an answer :data:`EMPTY_ANSWER` and mark it (in place).

    Sets ``metadata.empty_answer`` (evaluators that need a response skip the record)
    and, if the record has a trace without an error, ``trace.error``.
    """
    if str(record.get("response") or "").strip():
        return record
    record["response"] = EMPTY_ANSWER
    record.setdefault("metadata", {})["empty_answer"] = True
    trace = record.get("trace")
    if isinstance(trace, dict) and not trace.get("error"):
        calls = trace.get("model_calls")
        trace["error"] = "empty answer" + (f" after {calls} model calls" if calls else "")
    return record


def score(
    config: ProjectConfig, dataset: Path, output_dir: Path, scorer: Scorer | None = None
) -> EvaluationRunResult:
    """Score *dataset* with *config*'s evaluators and write the reports to *output_dir*."""
    from rag_eval_framework.runners.local import LocalRunner

    run_config = config.model_copy(
        update={"dataset_path": str(dataset), "output_dir": str(output_dir)}
    )
    result = (scorer or LocalRunner()).run(run_config)
    ReportEngine().generate(run_config, result)
    return result
