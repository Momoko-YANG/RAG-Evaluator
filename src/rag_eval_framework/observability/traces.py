"""Trace of one question through a RAG system: tool calls, timings, token usage, citations.

Collected by the generation and end-to-end runners (``record.trace``) and read by
the e2e evaluators and the observability summaries.  Timings are *wall-clock
spans measured where they happen*; nested spans are reported separately and
never added up (``search_ms`` is inside ``total_ms``, which is inside
``client_ms``).
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    """One tool call made by the model."""

    name: str
    arguments: Any = None
    query: str | None = Field(default=None, description="Search text written by the model.")
    latency_ms: float | None = None
    result_count: int | None = None
    status: Literal["executed", "stubbed", "error"] = "executed"
    error: str | None = None


class TokenUsage(BaseModel):
    """Tokens of the model calls of one stage (``generation``, ``judge``, …)."""

    stage: str = "generation"
    model: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int | None = None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class Timings(BaseModel):
    """Measured spans in milliseconds (``None`` = not measured)."""

    client_ms: float | None = Field(
        default=None, description="Collector request start to parsed response (incl. network)."
    )
    total_ms: float | None = Field(
        default=None, description="Server: request start to final answer."
    )
    ttft_ms: float | None = Field(default=None, description="Start to the first answer token.")
    search_ms: float | None = Field(
        default=None, description="Sum of search tool calls (inside total_ms)."
    )


class Citation(BaseModel):
    name: str
    id: str


class Trace(BaseModel):
    """What happened for one question (``mode``: ``e2e`` or ``generation``)."""

    mode: Literal["e2e", "generation", "retrieval"] = "e2e"
    tool_calls: list[ToolCall] = Field(default_factory=list)
    timings: Timings = Field(default_factory=Timings)
    usage: list[TokenUsage] = Field(default_factory=list)
    model_calls: int | None = None
    model: str | None = None
    citations: list[Citation] = Field(default_factory=list)
    attempts: int = Field(default=1, description="Requests sent, including retries.")
    error: str | None = None

    def searches(self) -> list[ToolCall]:
        """Calls that ran a search (they carry the model's search text)."""
        return [c for c in self.tool_calls if c.query is not None]


_CITATION = re.compile(r"\{%\s*citation\s+items=\[(.*?)\]\s*/%\}", re.S)
_ITEM = re.compile(r'\{\s*name\s*:\s*"([^"]*)"\s*,\s*id\s*:\s*"([^"]*)"\s*\}')


def parse_citations(raw_response: str) -> list[Citation]:
    """Citations in the ``{% citation items=[{name:"…",id:"…"}] /%}`` markup."""
    citations: list[Citation] = []
    for block in _CITATION.findall(raw_response or ""):
        citations += [Citation(name=name, id=cid) for name, cid in _ITEM.findall(block)]
    return citations


def trace_from_rag_eval(
    trace: dict[str, Any],
    raw_response: str = "",
    *,
    client_ms: float | None = None,
    attempts: int = 1,
) -> Trace:
    """Convert the ``trace`` object of ``/api/rag-eval`` (camelCase)."""
    calls = [
        ToolCall(
            name=c.get("name", ""),
            arguments=c.get("arguments"),
            query=c.get("query"),
            latency_ms=c.get("latencyMs"),
            result_count=c.get("documentCount"),
            status={"search": "executed"}.get(c.get("executed", ""), c.get("executed", "executed")),
            error=c.get("error"),
        )
        for c in trace.get("toolCalls") or []
    ]
    search_ms = [c.latency_ms for c in calls if c.query is not None and c.latency_ms is not None]
    usage = trace.get("usage")
    return Trace(
        mode="e2e",
        tool_calls=calls,
        timings=Timings(
            client_ms=client_ms,
            total_ms=trace.get("totalMs"),
            ttft_ms=trace.get("ttftMs"),
            search_ms=sum(search_ms) if search_ms else None,
        ),
        usage=[
            TokenUsage(
                stage="generation",
                model=trace.get("model"),
                input_tokens=usage.get("promptTokens", 0),
                output_tokens=usage.get("completionTokens", 0),
                calls=trace.get("modelCalls"),
            )
        ]
        if usage
        else [],
        model_calls=trace.get("modelCalls"),
        model=trace.get("model"),
        citations=parse_citations(raw_response),
        attempts=attempts,
    )
