"""Local evaluation runner — executes evaluators in-process."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from datetime import datetime, timezone

from rag_eval_framework.config.models import EVALUATOR_LAYERS, ProjectConfig
from rag_eval_framework.core.aggregation import mean_scores
from rag_eval_framework.core.provenance import build_provenance
from rag_eval_framework.core.registry import EvaluatorRegistry, default_registry
from rag_eval_framework.core.schemas import EvaluationRecord
from rag_eval_framework.datasets.loader import load_dataset
from rag_eval_framework.evaluators.base import (
    BaseEvaluator,
    EvaluatorResult,
    EvaluatorStatus,
)
from rag_eval_framework.observability.summary import observability_summary
from rag_eval_framework.runners.base import (
    BaseRunner,
    EvaluationRunResult,
    RecordResult,
    ThresholdBreach,
    ThresholdResult,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Field-requirement helpers
# ---------------------------------------------------------------------------

_FIELD_CHECKERS: dict[str, Callable[[EvaluationRecord], bool]] = {
    "question": lambda r: bool(r.question.strip()),
    "response": lambda r: bool(r.response.strip()),
    "contexts": lambda r: any(c.strip() for c in r.contexts),
    "ground_truth_answer": lambda r: bool(r.ground_truth_answer.strip()),
    "retrieved_documents": lambda r: bool(r.retrieved_documents),
    "ground_truth_documents": lambda r: bool(r.ground_truth_documents),
    "ground_truth_evidence": lambda r: bool(r.ground_truth_evidence),
}


def _check_required_fields(evaluator: BaseEvaluator, record: EvaluationRecord) -> str | None:
    """Return a missing-field reason string, or *None* if all fields are OK."""
    if "response" in evaluator.required_fields and record.is_retrieval_only():
        return (
            f"Record is retrieval-only (rag_params.retrievalOnly): its response is a "
            f"placeholder, so evaluator '{evaluator.name}' does not apply."
        )
    if "response" in evaluator.required_fields and record.is_empty_answer():
        return (
            f"The system gave no answer (metadata.empty_answer; see trace.error), so "
            f"evaluator '{evaluator.name}' has nothing to score; task_success counts it as failed."
        )
    for field in evaluator.required_fields:
        checker = _FIELD_CHECKERS.get(field)
        if checker is not None and not checker(record):
            return f"Required field '{field}' is missing or empty for evaluator '{evaluator.name}'."
    return None


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class LocalRunner(BaseRunner):
    """Run evaluators locally in the current Python process."""

    def __init__(self, registry: EvaluatorRegistry | None = None) -> None:
        self._registry = registry or default_registry

    def run(self, config: ProjectConfig) -> EvaluationRunResult:
        """Execute the full local evaluation pipeline."""
        start = time.monotonic()
        timestamp = datetime.now(timezone.utc).isoformat()

        logger.info("Starting LOCAL mode evaluation for project '%s'", config.project_name)

        # 1. Resolve and configure evaluators
        evaluators = self._resolve_evaluators(config.evaluators)
        for ev in evaluators:
            ev.setup(config)
        logger.info("Resolved %d evaluator(s): %s", len(evaluators), config.evaluators)

        # 2. Load dataset
        records = load_dataset(config.dataset_path)
        definitions = {
            ev.name: {**ev.definition(), "layer": EVALUATOR_LAYERS.get(ev.name, "other")}
            for ev in evaluators
        }
        logger.info("Loaded %d record(s) from '%s'", len(records), config.dataset_path)

        # 3. Evaluate every record
        record_results: list[RecordResult] = []
        for record in records:
            eval_results: dict[str, EvaluatorResult] = {}
            for evaluator in evaluators:
                # Check required fields before calling evaluate
                missing = _check_required_fields(evaluator, record)
                if missing:
                    eval_results[evaluator.name] = EvaluatorResult(
                        score=0.0,
                        status=EvaluatorStatus.SKIPPED,
                        reason=missing,
                    )
                    continue
                try:
                    result = evaluator.evaluate(record)
                    eval_results[evaluator.name] = result
                except Exception:
                    logger.exception(
                        "Evaluator '%s' failed on record '%s'",
                        evaluator.name,
                        record.id,
                    )
                    raise

            # Per-record pass/fail
            record_passed = self._record_passes_thresholds(eval_results, config.thresholds)
            record_results.append(
                RecordResult(
                    record_id=record.id,
                    evaluator_results=eval_results,
                    passed=record_passed,
                )
            )

        # 4. Aggregate scores (mean per evaluator, excluding SKIPPED)
        aggregate_scores = mean_scores(record_results, config.evaluators)

        # 5. Check thresholds
        # A threshold on an evaluator that ran but scored nothing is not met (actual 0).
        checked = {
            name: aggregate_scores.get(name, 0.0)
            for name in config.thresholds
            if name in aggregate_scores or name in config.evaluators
        }
        breaches = self._check_thresholds(checked, config.thresholds)
        threshold_results = self._build_threshold_results(checked, config.thresholds)
        passed = len(breaches) == 0

        elapsed = time.monotonic() - start
        logger.info(
            "Evaluation completed in %.2fs — passed=%s, breaches=%d",
            elapsed,
            passed,
            len(breaches),
        )

        return EvaluationRunResult(
            project_name=config.project_name,
            timestamp=timestamp,
            dataset_path=config.dataset_path,
            total_records=len(records),
            evaluators_used=config.evaluators,
            record_results=record_results,
            aggregate_scores=aggregate_scores,
            threshold_breaches=breaches,
            threshold_results=threshold_results,
            passed=passed,
            run_metadata={"elapsed_seconds": round(elapsed, 3), **config.metadata},
            metric_definitions=definitions,
            provenance=build_provenance(config.dataset_path, records),
            observability=observability_summary(records, config.observability.pricing),
            config_reference=config.dataset_path,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_evaluators(self, names: list[str]) -> list[BaseEvaluator]:
        return [self._registry.get(name) for name in names]

    @staticmethod
    def _record_passes_thresholds(
        eval_results: dict[str, EvaluatorResult],
        thresholds: dict[str, float],
    ) -> bool:
        """Return False if any evaluator score is below its threshold."""
        for name, threshold in thresholds.items():
            er = eval_results.get(name)
            if er is None:
                continue
            if er.status == EvaluatorStatus.SKIPPED:
                continue
            if er.score < threshold:
                return False
        return True

    @staticmethod
    def _check_thresholds(
        aggregate_scores: dict[str, float],
        thresholds: dict[str, float],
    ) -> list[ThresholdBreach]:
        breaches: list[ThresholdBreach] = []
        for evaluator_name, min_score in thresholds.items():
            actual = aggregate_scores.get(evaluator_name)
            if actual is None:
                continue
            if actual < min_score:
                breaches.append(
                    ThresholdBreach(
                        evaluator=evaluator_name,
                        threshold=min_score,
                        actual=actual,
                        delta=round(min_score - actual, 4),
                    )
                )
        return breaches

    @staticmethod
    def _build_threshold_results(
        aggregate_scores: dict[str, float],
        thresholds: dict[str, float],
    ) -> list[ThresholdResult]:
        """Build a structured pass/fail result for every configured threshold."""
        results: list[ThresholdResult] = []
        for evaluator_name, min_score in thresholds.items():
            actual = aggregate_scores.get(evaluator_name, 0.0)
            results.append(
                ThresholdResult(
                    evaluator=evaluator_name,
                    threshold=min_score,
                    actual=actual,
                    passed=actual >= min_score,
                )
            )
        return results
