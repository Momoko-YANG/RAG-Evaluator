"""Result models: what an evaluator returns and what a run produces.

* :class:`EvaluatorResult` — one evaluator on one record (score, status, reason).
* :class:`RecordResult` / :class:`EvaluationRunResult` — a whole run.
* :class:`MetricResult` — one reported metric value with what it depends on
  (see :func:`rag_eval_framework.core.aggregation.metric_results`).
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class EvaluatorStatus(str, Enum):
    """Status of an individual evaluator result."""

    SUCCESS = "success"
    SKIPPED = "skipped"
    ERROR = "error"


class EvaluatorResult(BaseModel):
    """Result produced by a single evaluator for a single record."""

    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Numeric score between 0.0 (worst) and 1.0 (best).",
    )
    reason: str = Field(
        default="",
        description="Human-readable explanation of the score.",
    )
    status: EvaluatorStatus = Field(
        default=EvaluatorStatus.SUCCESS,
        description="Whether the evaluator ran successfully, was skipped, or errored.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Evaluator-specific metadata (e.g. model used, latency).",
    )
    raw_output: dict[str, Any] | None = Field(
        default=None,
        description="Raw structured output from the underlying SDK or LLM call.",
    )


class RecordResult(BaseModel):
    """Evaluation results for a single dataset record."""

    record_id: str
    evaluator_results: dict[str, EvaluatorResult]
    passed: bool = Field(
        default=True,
        description="True if every evaluator score meets its threshold for this record.",
    )


class ThresholdBreach(BaseModel):
    """Details about a threshold that was not met."""

    evaluator: str
    threshold: float
    actual: float
    delta: float = Field(
        description="How far below the threshold the actual score fell.",
    )


class ThresholdResult(BaseModel):
    """Per-evaluator threshold check outcome (used in reports)."""

    evaluator: str
    threshold: float
    actual: float
    passed: bool


class EvaluationRunResult(BaseModel):
    """Aggregated result of a full evaluation run."""

    project_name: str
    timestamp: str
    dataset_path: str
    total_records: int
    evaluators_used: list[str]
    record_results: list[RecordResult]
    aggregate_scores: dict[str, float]
    threshold_breaches: list[ThresholdBreach] = Field(default_factory=list)
    threshold_results: list[ThresholdResult] = Field(default_factory=list)
    passed: bool
    run_metadata: dict[str, Any] = Field(default_factory=dict)
    skipped_evaluators: dict[str, str] = Field(
        default_factory=dict,
        description="Evaluators skipped globally and the reason (e.g. SDK missing).",
    )
    config_reference: str = Field(
        default="",
        description="Path to the project configuration file that produced this run.",
    )
    runner_type: str = Field(
        default="local",
        description="Execution mode: 'local' or 'cloud'.",
    )
    metric_definitions: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description=(
            "Per evaluator: metric version, layer, and what the score depends on "
            "(k, relevance policy and labels, judge model / prompt hash, SDK version)."
        ),
    )
    provenance: dict[str, Any] = Field(
        default_factory=dict,
        description="Framework version, dataset SHA-256 and the retrieval settings in the data.",
    )
    observability: dict[str, Any] = Field(
        default_factory=dict,
        description="Latency per stage, token usage and cost of the evaluated runs.",
    )


class MetricResult(BaseModel):
    """One reported metric value, traceable to its definition and labels."""

    metric: str = Field(..., description="Metric name, with @k when it has a cut-off.")
    evaluator: str
    score: float | None = None
    status: Literal["valid", "not_available"]
    eligible_questions: int = 0
    metric_version: str | None = None
    framework_version: str | None = None
    relevance_policy: str | None = None
    label_source: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)
