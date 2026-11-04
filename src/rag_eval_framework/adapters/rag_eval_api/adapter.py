"""RAG eval API adapter: calls an evaluation-only endpoint ``POST /api/rag-eval``."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

from rag_eval_framework.adapters.base import RagAdapter, RagResponse
from rag_eval_framework.observability.traces import trace_from_rag_eval

logger = logging.getLogger(__name__)


class RagEvalApiAdapter(RagAdapter):
    """Adapter for an evaluation-only endpoint ``POST /api/rag-eval`` in the RAG app.

    The endpoint (implemented in the app, see README "The /api/rag-eval contract")
    reuses the app's real document retrieval and RAG prompt, and returns plain JSON::

        {
            "response": "answer without citation markup",
            "contexts": ["chunk 1", ...],
            "documents": [{"id": "doc-0", "chunkId": "<index key>", "metadata": "file.pdf",
                           "pageContent": "chunk 1", "score": 0.83}, ...]
        }

    ``id`` is the prompt's citation id (the rank); ``chunkId`` (optional) is
    the search index document key and is stored as ``chunk_id`` for Recall@k.

    The documents searched are a chat thread's uploads (``chatThreadId``) or an
    extension's search index (``extensionId``).  The key is sent in the
    ``x-rag-eval-key`` header (``--rag-api-key`` or env ``RAG_EVAL_API_KEY``).

    ``--top-k``, ``--knn`` and ``--system-prompt-file`` override the app's
    retrieval / prompt settings for tuning experiments.  The settings actually
    used are stored in each record's ``metadata.rag_params``.

    Usage::

        python scripts/collect_rag_data.py \\
            --adapter rag_eval_api \\
            --rag-url http://localhost:3000/api/rag-eval \\
            --chat-thread-id <thread id> \\
            --questions datasets/my-project/sets/golden.csv \\
            --output datasets/my-project/runs/<run name>/golden.jsonl
    """

    KEY_HEADER = "x-rag-eval-key"

    @classmethod
    def create(
        cls,
        *,
        rag_url: str = "http://localhost:3000/api/rag-eval",
        api_key: str = "",
        extension_id: str = "",
        chat_thread_id: str = "",
        model: str = "primary",
        top_k: int | None = None,
        knn: int | None = None,
        retrieval_only: bool = False,
        e2e: bool = False,
        system_prompt_file: str = "",
        request_timeout: int = 180,
    ) -> RagEvalApiAdapter:
        """A configured adapter without going through the command line."""
        adapter = cls()
        adapter.setup(
            argparse.Namespace(
                rag_url=rag_url,
                rag_api_key=api_key,
                extension_id=extension_id,
                chat_thread_id=chat_thread_id,
                model=model,
                top_k=top_k,
                knn=knn,
                retrieval_only=retrieval_only,
                e2e=e2e,
                system_prompt_file=system_prompt_file,
                request_timeout=request_timeout,
            )
        )
        return adapter

    def setup(self, args: argparse.Namespace) -> None:
        try:
            import requests  # noqa: F401
        except ImportError:
            logger.error("Install 'requests' to use this adapter:  pip install requests")
            sys.exit(1)
        api_key = args.rag_api_key or os.environ.get("RAG_EVAL_API_KEY", "")
        if not api_key:
            logger.error("rag_eval_api adapter needs --rag-api-key or env RAG_EVAL_API_KEY.")
            sys.exit(1)
        self._url: str = args.rag_url
        self._headers: dict[str, str] = {
            "Content-Type": "application/json",
            self.KEY_HEADER: api_key,
        }
        # An extension is searched instead of a thread's uploaded documents.
        self._extension_id: str = args.extension_id or os.environ.get("RAG_EVAL_EXTENSION_ID", "")
        self._thread_id: str = (
            ""
            if self._extension_id
            else args.chat_thread_id or os.environ.get("RAG_EVAL_CHAT_THREAD_ID", "")
        )
        if self._extension_id and args.chat_thread_id:
            logger.error("Use either --chat-thread-id or --extension-id, not both.")
            sys.exit(1)
        self._model: str = args.model
        self._timeout: int = args.request_timeout
        self._overrides: dict[str, Any] = {}
        if args.top_k is not None:
            self._overrides["topK"] = args.top_k
        if args.knn is not None:
            self._overrides["knn"] = args.knn
        if args.retrieval_only:
            self._overrides["retrievalOnly"] = True
        if getattr(args, "e2e", False):
            if args.retrieval_only or args.system_prompt_file or not self._extension_id:
                logger.error(
                    "--e2e needs --extension-id, without --retrieval-only / --system-prompt-file."
                )
                sys.exit(1)
            self._overrides["e2e"] = True
        self._prompt_file = ""
        if args.system_prompt_file:
            self._prompt_file = Path(args.system_prompt_file).name
            self._overrides["systemPrompt"] = Path(args.system_prompt_file).read_text(
                encoding="utf-8"
            )
        logger.info("RagEvalApiAdapter configured → %s", self._url)

    def query(self, question: str, **kwargs: Any) -> RagResponse:
        import requests

        payload: dict[str, Any] = {
            "question": question,
            "selectedModel": self._model,
            **self._overrides,
        }
        if self._extension_id:
            payload["extensionId"] = self._extension_id
        elif self._thread_id:
            payload["chatThreadId"] = self._thread_id
        started = time.perf_counter()
        resp = requests.post(
            self._url,
            json=payload,
            headers=self._headers,
            timeout=self._timeout,
        )
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("error", "")
            except ValueError:
                detail = ""
            raise RuntimeError(f"HTTP {resp.status_code} from the RAG eval API: {detail}".strip())
        data = resp.json()
        client_ms = round((time.perf_counter() - started) * 1000, 2)

        # "id" is the citation id of the prompt (doc-0, doc-1 …: the rank); "chunk_id"
        # is the search index key, kept only when the API returns it (chunkId).
        documents = [
            {
                "id": doc.get("id", ""),
                "title": doc.get("metadata", ""),
                "content": doc.get("pageContent", ""),
                "score": doc.get("score"),
                **({"chunk_id": doc["chunkId"]} if doc.get("chunkId") else {}),
            }
            for doc in data.get("documents", [])
        ]
        rag_params: dict[str, Any] = {
            "model": self._model,
            "deployment": data.get("deploymentName", ""),
            **data.get("params", {}),
        }
        if self._prompt_file:
            rag_params["system_prompt_file"] = self._prompt_file
        trace = None
        if isinstance(data.get("trace"), dict):
            trace = trace_from_rag_eval(
                data["trace"], str(data.get("rawResponse", "")), client_ms=client_ms
            ).model_dump(mode="json")
        return RagResponse(
            response=str(data.get("response", "")),
            contexts=[str(c) for c in data.get("contexts", [])],
            retrieved_documents=documents,
            raw=data,
            metadata={"rag_params": rag_params},
            trace=trace,
        )
