"""Evaluation runners — local and cloud execution."""

from rag_eval_framework.runners.base import (
    BaseRunner,
    EvaluationRunResult,
    RecordResult,
    ThresholdBreach,
    ThresholdResult,
)
from rag_eval_framework.runners.cloud import FoundryRunner
from rag_eval_framework.runners.local import LocalRunner

__all__ = [
    "BaseRunner",
    "EvaluationRunResult",
    "FoundryRunner",
    "LocalRunner",
    "RecordResult",
    "ThresholdBreach",
    "ThresholdResult",
]
