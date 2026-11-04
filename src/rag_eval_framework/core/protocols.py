"""Interfaces between the parts of the framework.

The dependency direction is::

    Adapter ──► EvaluationRecord ──► Runner ──► Evaluators ──► EvaluatorResult[]
                                                              │
                                                Aggregation ◄─┘ ──► Reporting

* An **adapter** turns one system's response into the framework's record fields
  (evaluators never know which system produced a record).
* An **evaluator** scores the fields it is given; it does not collect data.
* A **chat client** is the only thing that calls a generation model, so the layer
  runners and LLM judges can swap SDKs, deployments and test fakes.
* A **scorer** (``LocalRunner`` / ``FoundryRunner``) applies evaluators to a
  dataset; the layer runners collect data and then delegate to it.

These are structural (``typing.Protocol``): existing classes conform without
inheriting from them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from rag_eval_framework.core.results import EvaluationRunResult, EvaluatorResult
from rag_eval_framework.core.schemas import EvaluationRecord
from rag_eval_framework.observability.traces import TokenUsage


@runtime_checkable
class Evaluator(Protocol):
    """A pure scoring component."""

    @property
    def name(self) -> str: ...

    @property
    def version(self) -> str: ...

    @property
    def required_fields(self) -> list[str]: ...

    def setup(self, config: Any) -> None: ...

    def evaluate(self, record: EvaluationRecord) -> EvaluatorResult: ...

    def definition(self) -> dict[str, Any]: ...


@runtime_checkable
class Adapter(Protocol):
    """Sends one question to a RAG system and returns a response object.

    The response has ``response``, ``contexts``, ``retrieved_documents``,
    ``metadata`` and (end-to-end) ``trace`` — see
    :class:`rag_eval_framework.adapters.base.RagResponse`.
    """

    def setup(self, args: Any) -> None: ...

    def query(self, question: str, **kwargs: Any) -> Any: ...


@dataclass
class ChatOutput:
    """What a chat client returns for one completion."""

    text: str
    model: str | None = None
    usage: TokenUsage | None = None
    ttft_ms: float | None = None
    total_ms: float | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class ChatClient(Protocol):
    """Calls a chat model with a list of messages."""

    def complete(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        model: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatOutput: ...


@runtime_checkable
class Scorer(Protocol):
    """Applies the configured evaluators to a dataset (LocalRunner / FoundryRunner)."""

    def run(self, config: Any) -> EvaluationRunResult: ...
