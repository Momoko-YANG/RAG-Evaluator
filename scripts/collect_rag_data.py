"""Automated RAG data collection script.

Reads test questions (+ optional ground truth) from a CSV file, calls your
RAG API for each question, captures the response and retrieved contexts,
and writes a JSONL evaluation dataset ready for consumption by the framework.

Usage
-----
    python scripts/collect_rag_data.py \\
        --questions datasets/hr-rag/questions.csv \\
        --output datasets/hr-rag/eval.jsonl \\
        --adapter http \\
        --rag-url http://localhost:8000/api/chat

Adapters
--------
The script ships with several built-in adapters:

* ``http`` — generic REST endpoint (JSON request/response).
* ``azure_chat_stream`` — Azure-style chat completion endpoints that
  return NDJSON streaming (``application/json-lines``).  Covers RAG
  chatbots built with Azure AI that stream responses.
* ``rag_eval_api`` — an app's evaluation endpoint ``/api/rag-eval`` (see README).

The adapters live in ``rag_eval_framework.adapters`` (``rag_eval_api/``, ``generic.py``).
Implement your own by subclassing ``RagAdapter`` and registering it in
``ADAPTER_REGISTRY`` there.

CSV format
----------
The input CSV must have a ``question`` column.  Optional columns:

    id, ground_truth_answer, ground_truth_documents, ground_truth_evidence, metadata_*

``ground_truth_documents`` (file names) and ``ground_truth_evidence`` (passages
the relevant chunk contains) are ``|``-separated lists used by the
``retrieval_hit_rate`` / ``retrieval_mrr`` evaluators.

Columns prefixed with ``metadata_`` are packed into the record's
``metadata`` dict (e.g. ``metadata_scenario`` → ``{"scenario": "..."}``)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_eval_framework.adapters import (  # noqa: E402
    ADAPTER_REGISTRY,
    AzureChatStreamAdapter,
    HttpAdapter,
    RagAdapter,
    RagEvalApiAdapter,
    RagResponse,
)
from rag_eval_framework.datasets.questions import (  # noqa: E402
    EXPECTATION_BOOL_COLUMNS,
    EXPECTATION_LIST_COLUMNS,
    LABEL_COLUMNS,
    read_question_bank,
)
from rag_eval_framework.runners.collection import mark_empty_answer  # noqa: E402

__all__ = [
    "ADAPTER_REGISTRY",
    "AzureChatStreamAdapter",
    "HttpAdapter",
    "RagAdapter",
    "RagEvalApiAdapter",
    "RagResponse",
    "build_parser",
    "collect",
    "load_questions",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CSV reader
# ---------------------------------------------------------------------------

def load_questions(csv_path: Path) -> list[dict[str, Any]]:
    """Read a question bank CSV (see ``rag_eval_framework.datasets.questions``); exit on errors."""
    try:
        return read_question_bank(csv_path)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        sys.exit(1)


def collect(
    adapter: RagAdapter,
    questions: list[dict[str, Any]],
    output_path: Path,
    delay: float = 0.0,
) -> None:
    """Query the RAG system for each question and write the JSONL output."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    success = 0
    errors = 0

    with output_path.open("w", encoding="utf-8") as f:
        for i, q in enumerate(questions, start=1):
            question_text = q["question"]
            labels = {label: q[label] for label in LABEL_COLUMNS if q.get(label)}
            labels |= {
                column: q[column]
                for column in EXPECTATION_LIST_COLUMNS + EXPECTATION_BOOL_COLUMNS
                if q.get(column) is not None and q.get(column) != []
            }
            logger.info("[%d/%d] %s", i, len(questions), question_text[:80])

            try:
                rag_resp = adapter.query(question_text)
            except Exception as exc:
                logger.error("  Error: %s", exc)
                errors += 1
                # Keep the question with a marked empty answer: the dataset stays valid,
                # answer metrics skip it and task_success counts the error.
                record = {
                    "id": q["id"],
                    "question": question_text,
                    "response": "",
                    "contexts": [],
                    "ground_truth_answer": q.get("ground_truth_answer", ""),
                    **labels,
                    "metadata": {**q.get("metadata", {}), "collection_error": str(exc)},
                }
                f.write(json.dumps(mark_empty_answer(record), ensure_ascii=False) + "\n")
                continue

            record = {
                "id": q["id"],
                "question": question_text,
                "response": rag_resp.response,
                "contexts": rag_resp.contexts,
                "ground_truth_answer": q.get("ground_truth_answer", ""),
                **labels,
                "metadata": {**q.get("metadata", {}), **rag_resp.metadata},
            }
            if rag_resp.retrieved_documents:
                record["retrieved_documents"] = rag_resp.retrieved_documents
            if rag_resp.trace is not None:
                record["trace"] = rag_resp.trace
            f.write(json.dumps(mark_empty_answer(record), ensure_ascii=False) + "\n")
            success += 1

            if delay > 0:
                time.sleep(delay)

    logger.info(
        "Done — %d succeeded, %d errors.  Dataset written to %s",
        success,
        errors,
        output_path,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Collect RAG evaluation data by querying your RAG system.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--questions", required=True, type=Path,
        help="Path to CSV file with test questions (must have a 'question' column).",
    )
    p.add_argument(
        "--output", required=True, type=Path,
        help="Output JSONL file path (e.g. datasets/my-project/eval.jsonl).",
    )
    p.add_argument(
        "--adapter", default="http", choices=sorted(ADAPTER_REGISTRY),
        help="RAG adapter to use (default: http).",
    )
    p.add_argument(
        "--rag-url", default="http://localhost:8000/api/chat",
        help="RAG API endpoint URL.",
    )
    p.add_argument(
        "--rag-api-key", default="",
        help="Optional Bearer token / API key for the RAG endpoint.",
    )
    p.add_argument(
        "--response-field", default="response",
        help="JSON field name for the RAG answer (default: response).",
    )
    p.add_argument(
        "--contexts-field", default="contexts",
        help="JSON field name for retrieved contexts (default: contexts).",
    )
    p.add_argument(
        "--chat-thread-id", default="",
        help="(rag_eval_api) Chat thread whose uploaded documents are searched. "
             "Defaults to env RAG_EVAL_CHAT_THREAD_ID, then the server's first allowed thread.",
    )
    p.add_argument(
        "--extension-id", default="",
        help="(rag_eval_api) Search this Azure AI Search extension's index instead of a "
             "thread (must be in RAG_EVAL_ALLOWED_EXTENSION_IDS). Defaults to env "
             "RAG_EVAL_EXTENSION_ID.",
    )
    p.add_argument(
        "--model", default="primary", choices=["primary", "latest"],
        help="(rag_eval_api) Model key sent as selectedModel (default: primary).",
    )
    p.add_argument(
        "--top-k", type=int, default=None,
        help="(rag_eval_api) Chunks to retrieve per question "
             "(default: the app's value).",
    )
    p.add_argument(
        "--knn", type=int, default=None,
        help="(rag_eval_api) Vector candidates in the hybrid search (default: the app's value).",
    )
    p.add_argument(
        "--e2e", action="store_true",
        help="(rag_eval_api, extension) Real chat flow: the model chooses tool calls and search "
        "text; records a trace (tool calls, TTFT, latency, token usage).",
    )
    p.add_argument(
        "--retrieval-only", action="store_true",
        help="(rag_eval_api) Skip answer generation; for retrieval metrics only (much faster).",
    )
    p.add_argument(
        "--system-prompt-file", type=Path, default=None,
        help="(rag_eval_api) UTF-8 file whose text replaces the RAG system prompt.",
    )
    p.add_argument(
        "--request-timeout", type=int, default=60,
        help="HTTP request timeout in seconds (default: 60).",
    )
    p.add_argument(
        "--delay", type=float, default=0.0,
        help="Delay between requests in seconds (for rate limiting).",
    )
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Build adapter
    adapter_cls = ADAPTER_REGISTRY[args.adapter]
    adapter = adapter_cls()
    adapter.setup(args)

    # Load questions
    questions = load_questions(args.questions)
    if not questions:
        logger.error("No questions loaded — exiting.")
        sys.exit(1)

    # Collect data
    collect(adapter, questions, args.output, delay=args.delay)


if __name__ == "__main__":
    main()
