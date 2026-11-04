"""Base runner interface (result models live in :mod:`rag_eval_framework.core.results`)."""

from __future__ import annotations

from abc import ABC, abstractmethod

from rag_eval_framework.config.models import ProjectConfig
from rag_eval_framework.core.results import (
    EvaluationRunResult,
    RecordResult,
    ThresholdBreach,
    ThresholdResult,
)

__all__ = [
    "BaseRunner",
    "EvaluationRunResult",
    "RecordResult",
    "ThresholdBreach",
    "ThresholdResult",
]


class BaseRunner(ABC):
    """Abstract scoring runner: evaluates a dataset and returns aggregated results.

    :class:`~rag_eval_framework.runners.local.LocalRunner` scores in-process,
    :class:`~rag_eval_framework.runners.cloud.FoundryRunner` via Azure AI Foundry.
    The layer runners (``retrieval``, ``generation``, ``e2e``) collect data and
    then delegate scoring to one of these.
    """

    @abstractmethod
    def run(self, config: ProjectConfig) -> EvaluationRunResult:
        """Execute a full evaluation run and return aggregated results."""
        ...
