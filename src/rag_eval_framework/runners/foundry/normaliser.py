"""Normalise Azure AI Foundry cloud evaluation results.

This module converts the raw result dictionaries returned by the Foundry
service into the framework's :class:`EvaluationRunResult` model so that
reports, threshold checking, and CLI display work identically regardless
of whether the evaluation ran locally or in the cloud.
"""

from __future__ import annotations

import logging
from typing import Any

from rag_eval_framework.evaluators.base import EvaluatorResult, EvaluatorStatus
from rag_eval_framework.runners.base import (
    RecordResult,
    ThresholdBreach,
    ThresholdResult,
)
from rag_eval_framework.utils.normalisation import normalise_1_5

logger = logging.getLogger(__name__)


def normalise_cloud_metrics(
    raw_metrics: dict[str, Any],
    evaluator_names: list[str],
) -> dict[str, float]:
    """Extract per-evaluator aggregate scores from cloud metrics.

    The Foundry SDK returns metric keys like ``groundedness.groundedness``
    or just ``groundedness``.  This function tries several key patterns
    and normalises 1-5 scores to 0.0-1.0.
    """
    scores: dict[str, float] = {}

    for name in evaluator_names:
        # Try several key conventions used by the SDK
        candidates = [
            f"{name}.{name}",  # e.g. "groundedness.groundedness"
            f"{name}.gpt_{name}",  # e.g. "groundedness.gpt_groundedness"
            name,  # direct key
            f"gpt_{name}",  # legacy key
        ]
        raw_val: float | None = None
        for key in candidates:
            if key in raw_metrics:
                raw_val = float(raw_metrics[key])
                break

        if raw_val is not None:
            # Detect 1-5 scale and normalise
            if raw_val > 1.0:
                scores[name] = normalise_1_5(raw_val)
            else:
                scores[name] = round(raw_val, 4)
        else:
            logger.warning(
                "Cloud metrics missing score for evaluator '%s'. Available keys: %s",
                name,
                list(raw_metrics.keys()),
            )
            scores[name] = 0.0

    return scores


def normalise_cloud_rows(
    raw_rows: list[dict[str, Any]],
    evaluator_names: list[str],
) -> list[RecordResult]:
    """Convert per-row cloud results to ``RecordResult`` objects.

    Each row from the Foundry SDK contains the original data fields plus
    evaluator outputs (e.g. ``outputs.groundedness.groundedness``).
    """
    record_results: list[RecordResult] = []

    for idx, row in enumerate(raw_rows):
        record_id = _extract_record_id(row, idx)

        eval_results: dict[str, EvaluatorResult] = {}
        for name in evaluator_names:
            score, reason, status = _extract_evaluator_result(row, name)
            eval_results[name] = EvaluatorResult(
                score=score,
                reason=reason,
                status=status,
            )

        record_results.append(
            RecordResult(
                record_id=record_id,
                evaluator_results=eval_results,
                passed=True,  # will be recalculated by the runner
            )
        )

    return record_results


def build_threshold_results(
    aggregate_scores: dict[str, float],
    thresholds: dict[str, float],
) -> tuple[list[ThresholdResult], list[ThresholdBreach]]:
    """Build threshold results and breaches from aggregate scores."""
    results: list[ThresholdResult] = []
    breaches: list[ThresholdBreach] = []

    for evaluator_name, min_score in thresholds.items():
        actual = aggregate_scores.get(evaluator_name, 0.0)
        passed = actual >= min_score
        results.append(
            ThresholdResult(
                evaluator=evaluator_name,
                threshold=min_score,
                actual=actual,
                passed=passed,
            )
        )
        if not passed:
            breaches.append(
                ThresholdBreach(
                    evaluator=evaluator_name,
                    threshold=min_score,
                    actual=actual,
                    delta=round(min_score - actual, 4),
                )
            )

    return results, breaches


def update_record_pass_fail(
    record_results: list[RecordResult],
    thresholds: dict[str, float],
) -> None:
    """Update ``passed`` on each record based on thresholds (in-place)."""
    for rr in record_results:
        rr.passed = True
        for name, threshold in thresholds.items():
            er = rr.evaluator_results.get(name)
            if er is None:
                continue
            if er.status == EvaluatorStatus.SKIPPED:
                continue
            if er.score < threshold:
                rr.passed = False
                break


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_record_id(row: dict[str, Any], index: int) -> str:
    """Try to extract a record ID from a cloud result row."""
    for key in ("id", "record_id", "case_id", "inputs.id"):
        val = row.get(key)
        if val:
            return str(val)
    return f"cloud-row-{index:04d}"


def _extract_evaluator_result(
    row: dict[str, Any], evaluator_name: str
) -> tuple[float, str, EvaluatorStatus]:
    """Extract a single evaluator's score + reason from a cloud row."""
    # Try nested "outputs.<name>.<name>" pattern
    outputs = row.get("outputs", {})
    evaluator_output = outputs.get(evaluator_name, {})

    # Direct score keys
    score_candidates = [
        evaluator_output.get(evaluator_name),
        evaluator_output.get(f"gpt_{evaluator_name}"),
        row.get(f"outputs.{evaluator_name}.{evaluator_name}"),
        row.get(f"outputs.{evaluator_name}"),
        row.get(evaluator_name),
    ]

    raw_score: float | None = None
    for candidate in score_candidates:
        if candidate is not None:
            try:
                raw_score = float(candidate)
                break
            except (ValueError, TypeError):
                continue

    if raw_score is None:
        return 0.0, f"No score found for '{evaluator_name}' in cloud output.", EvaluatorStatus.ERROR

    # Normalise 1-5 → 0-1
    score = normalise_1_5(raw_score) if raw_score > 1.0 else round(raw_score, 4)

    # Extract reason
    reason_candidates = [
        evaluator_output.get(f"{evaluator_name}_reason"),
        evaluator_output.get(f"gpt_{evaluator_name}_reason"),
        evaluator_output.get("reason"),
    ]
    reason = ""
    for r in reason_candidates:
        if r:
            reason = str(r)
            break

    return score, reason, EvaluatorStatus.SUCCESS
