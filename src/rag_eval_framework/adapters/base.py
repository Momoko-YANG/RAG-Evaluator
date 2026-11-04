"""Adapter interface: one question in, the RAG system's answer and retrieved chunks out."""

from __future__ import annotations

import abc
import argparse
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RagResponse:
    """Structured response from a RAG system."""

    response: str
    contexts: list[str] = field(default_factory=list)
    retrieved_documents: list[dict[str, Any]] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)
    # Merged into the record's metadata (e.g. the RAG settings used for this run).
    metadata: dict[str, Any] = field(default_factory=dict)
    # End-to-end runs: a serialised rag_eval_framework.observability.traces.Trace.
    trace: dict[str, Any] | None = None


class RagAdapter(abc.ABC):
    """Abstract base for RAG system adapters.

    Subclass this and implement ``query()`` to integrate with any backend.
    """

    @abc.abstractmethod
    def query(self, question: str, **kwargs: Any) -> RagResponse:
        """Send a question to the RAG system and return the response."""

    def setup(self, args: argparse.Namespace) -> None:  # noqa: B027
        """Optional one-time setup (called before the first query)."""
