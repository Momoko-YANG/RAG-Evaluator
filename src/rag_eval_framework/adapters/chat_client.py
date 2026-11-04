"""Chat model client (``core.protocols.ChatClient``) over an OpenAI-compatible SDK client.

Streams the completion to measure time to first token, and asks for token usage
in the stream.  Works with ``openai.AzureOpenAI`` / ``openai.OpenAI``; tests pass a
fake with the same ``chat.completions.create`` method.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

from rag_eval_framework.core.protocols import ChatOutput
from rag_eval_framework.observability.traces import TokenUsage


class OpenAIChatClient:
    """``complete()`` = one streamed chat completion with TTFT, total time and usage."""

    def __init__(self, client: Any, *, stage: str = "generation") -> None:
        self._client = client
        self._stage = stage

    @classmethod
    def from_config(cls, config: Any) -> OpenAIChatClient:
        """Azure OpenAI client of the project's ``azure`` / ``judge`` settings."""
        from rag_eval_framework.evaluators.common.llm_judge import _build_openai_client

        return cls(_build_openai_client(config))

    def complete(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        model: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatOutput:
        options: dict[str, Any] = {}
        if temperature is not None:
            options["temperature"] = temperature
        if max_tokens is not None:
            options["max_completion_tokens"] = max_tokens
        started = time.perf_counter()
        stream = self._client.chat.completions.create(
            model=model,
            messages=list(messages),
            stream=True,
            stream_options={"include_usage": True},
            **options,
        )
        parts: list[str] = []
        ttft_ms: float | None = None
        usage: Any = None
        served_model: str | None = None
        for chunk in stream:
            served_model = getattr(chunk, "model", None) or served_model
            if getattr(chunk, "usage", None):
                usage = chunk.usage
            for choice in getattr(chunk, "choices", None) or []:
                delta = getattr(choice.delta, "content", None)
                if delta:
                    if ttft_ms is None:
                        ttft_ms = (time.perf_counter() - started) * 1000
                    parts.append(delta)
        total_ms = (time.perf_counter() - started) * 1000
        return ChatOutput(
            text="".join(parts),
            model=served_model or model,
            usage=TokenUsage(
                stage=self._stage,
                model=served_model or model,
                input_tokens=usage.prompt_tokens,
                output_tokens=usage.completion_tokens,
                calls=1,
            )
            if usage is not None
            else None,
            ttft_ms=round(ttft_ms, 2) if ttft_ms is not None else None,
            total_ms=round(total_ms, 2),
        )
