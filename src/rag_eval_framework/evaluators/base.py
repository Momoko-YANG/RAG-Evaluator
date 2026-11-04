"""Base evaluator interface and result model."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from rag_eval_framework.core.results import EvaluatorResult, EvaluatorStatus
from rag_eval_framework.core.schemas import EvaluationRecord

__all__ = ["BaseEvaluator", "EvaluatorResult", "EvaluatorStatus"]

# ---------------------------------------------------------------------------
# Base evaluator ABC
# ---------------------------------------------------------------------------


class BaseEvaluator(ABC):
    """Abstract base class for all evaluators.

    Every evaluator — built-in, Azure-backed, or custom — must implement
    this interface.  The evaluator registry resolves evaluator names to
    concrete subclasses at run-time.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Machine-readable evaluator name (used in configs and reports)."""
        ...

    @property
    def description(self) -> str:
        """Optional human-readable description."""
        return ""

    @property
    def required_fields(self) -> list[str]:
        """Dataset fields this evaluator requires.

        The runner checks these before calling ``evaluate()``.  If a field
        is missing or empty the record is scored ``0.0`` with status
        ``SKIPPED`` and a clear reason.

        Override in subclasses to declare additional dependencies.  The
        default only requires ``question`` and ``response``.
        """
        return ["question", "response"]

    @property
    def version(self) -> str:
        """Version of the metric definition; bump it whenever scores would change."""
        return "1.0"

    def definition(self) -> dict[str, Any]:
        """What a score from this evaluator means, recorded with every run.

        Subclasses add what the score depends on (k, relevance policy and labels,
        judge model and prompt hash, SDK version …) so that the same metric can be
        traced across runs.
        """
        return {"evaluator": self.name, "version": self.version}

    def setup(self, config: Any) -> None:  # noqa: B027
        """Optional lifecycle hook called by the runner after instantiation.

        Override this to extract connection strings, model names, or other
        settings from a ``ProjectConfig`` (or any config object).  The
        default implementation is a no-op.
        """

    @abstractmethod
    def evaluate(self, record: EvaluationRecord) -> EvaluatorResult:
        """Evaluate a single dataset record and return a scored result."""
        ...
