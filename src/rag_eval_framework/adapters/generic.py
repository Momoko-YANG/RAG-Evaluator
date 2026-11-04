"""Generic adapters: plain HTTP JSON, Azure-style NDJSON chat streams, Azure agent stub."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any

from rag_eval_framework.adapters.base import RagAdapter, RagResponse

logger = logging.getLogger(__name__)


class HttpAdapter(RagAdapter):
    """Generic HTTP adapter — POST ``{"question": ...}`` to a REST endpoint.

    Expected JSON response format::

        {
            "response": "The answer...",
            "contexts": ["chunk 1", "chunk 2"]
        }

    Customise the field names via ``--response-field`` and ``--contexts-field``.
    """

    def setup(self, args: argparse.Namespace) -> None:
        try:
            import requests  # noqa: F401
        except ImportError:
            logger.error("Install 'requests' to use the HTTP adapter:  pip install requests")
            sys.exit(1)
        self._url: str = args.rag_url
        self._response_field: str = args.response_field
        self._contexts_field: str = args.contexts_field
        self._headers: dict[str, str] = {"Content-Type": "application/json"}
        if args.rag_api_key:
            self._headers["Authorization"] = f"Bearer {args.rag_api_key}"
        self._timeout: int = args.request_timeout
        logger.info("HttpAdapter configured → %s", self._url)

    def query(self, question: str, **kwargs: Any) -> RagResponse:
        import requests

        payload = {"question": question, **kwargs}
        resp = requests.post(
            self._url,
            json=payload,
            headers=self._headers,
            timeout=self._timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        return RagResponse(
            response=str(data.get(self._response_field, "")),
            contexts=data.get(self._contexts_field, []),
            raw=data,
        )


class AzureChatStreamAdapter(RagAdapter):
    """Adapter for Azure-style chat completion endpoints with NDJSON streaming.

    Handles endpoints that:

    * Accept ``{"messages": [{"role": "user", "content": "..."}]}``
    * Return ``application/json-lines`` (NDJSON) where:
        - Line 0: ``role=tool`` → contains a ``citations`` array.
        - Lines 1-N: ``role=assistant`` → token-by-token answer chunks.

    This covers RAG chatbots built with Azure AI that stream
    NDJSON responses.

    Usage::

        python scripts/collect_rag_data.py \\
            --adapter azure_chat_stream \\
            --rag-url http://localhost:8000/conversation \\
            --questions datasets/my-project/questions.csv \\
            --output datasets/my-project/eval.jsonl
    """

    def setup(self, args: argparse.Namespace) -> None:
        try:
            import requests  # noqa: F401
        except ImportError:
            logger.error("Install 'requests' to use this adapter:  pip install requests")
            sys.exit(1)
        self._url: str = args.rag_url
        self._headers: dict[str, str] = {"Content-Type": "application/json"}
        if args.rag_api_key:
            self._headers["Authorization"] = f"Bearer {args.rag_api_key}"
        self._timeout: int = args.request_timeout
        logger.info("AzureChatStreamAdapter configured → %s", self._url)

    def query(self, question: str, **kwargs: Any) -> RagResponse:
        import requests

        payload = {"messages": [{"role": "user", "content": question}]}
        resp = requests.post(
            self._url,
            json=payload,
            headers=self._headers,
            timeout=self._timeout,
            stream=True,
        )
        resp.raise_for_status()

        full_answer = ""
        contexts: list[str] = []
        raw_citations: list[dict[str, Any]] = []

        for raw_line in resp.iter_lines(decode_unicode=True):
            if not raw_line or not raw_line.strip():
                continue

            try:
                obj = json.loads(raw_line)
            except json.JSONDecodeError:
                logger.debug("Skipping non-JSON line: %s", raw_line[:120])
                continue

            for choice in obj.get("choices", []):
                for msg in choice.get("messages", []):
                    role = msg.get("role", "")
                    content = msg.get("content", "")

                    if role == "tool" and content:
                        try:
                            tool_data = json.loads(content)
                            for citation in tool_data.get("citations", []):
                                ctx_text = citation.get("content", "").strip()
                                if ctx_text:
                                    contexts.append(ctx_text)
                                    raw_citations.append(citation)
                        except json.JSONDecodeError:
                            pass

                    elif role == "assistant":
                        full_answer += content

        return RagResponse(
            response=full_answer.strip(),
            contexts=contexts,
            raw={"citations": raw_citations},
        )

