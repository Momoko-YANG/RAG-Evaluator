"""Cloud evaluation runner — submits jobs to Azure AI Foundry.

This runner implements the same :class:`BaseRunner` interface as
:class:`LocalRunner`, ensuring that reports, threshold checking, and CLI
display work identically regardless of execution mode.

The Foundry-specific SDK logic is isolated in the
:mod:`rag_eval_framework.runners.foundry` adapter package.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

from rag_eval_framework.config.models import ProjectConfig
from rag_eval_framework.runners.base import (
    BaseRunner,
    EvaluationRunResult,
)
from rag_eval_framework.runners.foundry.adapter import (
    FoundryAdapter,
    FoundryAdapterError,
)
from rag_eval_framework.runners.foundry.normaliser import (
    build_threshold_results,
    normalise_cloud_metrics,
    normalise_cloud_rows,
    update_record_pass_fail,
)

logger = logging.getLogger(__name__)


class CloudRunnerError(Exception):
    """Raised when the cloud runner encounters a fatal error."""


class FoundryRunner(BaseRunner):
    """Run evaluators via Azure AI Foundry cloud evaluation.

    Usage
    -----
    >>> runner = FoundryRunner()
    >>> result = runner.run(config)

    The runner:

    1. Validates the project config has a ``foundry`` section.
    2. Connects to the Foundry service via the adapter.
    3. Submits the evaluation job.
    4. Polls for completion and retrieves results.
    5. Normalises cloud metrics into the framework's result model.
    6. Checks thresholds and builds the same ``EvaluationRunResult``
       as the local runner.
    """

    def run(self, config: ProjectConfig) -> EvaluationRunResult:
        """Execute a cloud evaluation run via Azure AI Foundry."""
        start = time.monotonic()
        timestamp = datetime.now(timezone.utc).isoformat()

        logger.info("Starting CLOUD mode evaluation for project '%s'", config.project_name)

        # 1. Build and connect the adapter
        try:
            adapter = FoundryAdapter(config)
            adapter.connect()
        except (FoundryAdapterError, ImportError) as exc:
            raise CloudRunnerError(f"Failed to initialise cloud evaluation: {exc}") from exc

        # 2. Submit and wait for results
        try:
            raw_results = adapter.submit_and_wait(
                dataset_path=config.dataset_path,
                evaluators=config.evaluators,
                display_name=config.project_name,
            )
        except FoundryAdapterError as exc:
            raise CloudRunnerError(f"Cloud evaluation run failed: {exc}") from exc

        # 3. Normalise cloud results
        raw_metrics = raw_results.get("metrics", {})
        raw_rows = raw_results.get("rows", [])
        run_id = raw_results.get("run_id", "")

        aggregate_scores = normalise_cloud_metrics(raw_metrics, config.evaluators)

        record_results = normalise_cloud_rows(raw_rows, config.evaluators)
        total_records = len(record_results) if record_results else 0

        # 4. Apply thresholds
        update_record_pass_fail(record_results, config.thresholds)
        threshold_results, threshold_breaches = build_threshold_results(
            aggregate_scores, config.thresholds
        )
        passed = len(threshold_breaches) == 0

        elapsed = time.monotonic() - start
        logger.info(
            "Cloud evaluation completed in %.2fs — passed=%s, breaches=%d",
            elapsed,
            passed,
            len(threshold_breaches),
        )

        return EvaluationRunResult(
            project_name=config.project_name,
            timestamp=timestamp,
            dataset_path=config.dataset_path,
            total_records=total_records,
            evaluators_used=config.evaluators,
            record_results=record_results,
            aggregate_scores=aggregate_scores,
            threshold_breaches=threshold_breaches,
            threshold_results=threshold_results,
            passed=passed,
            runner_type="cloud",
            run_metadata={
                "elapsed_seconds": round(elapsed, 3),
                "cloud_run_id": run_id,
                **config.metadata,
            },
            config_reference=config.dataset_path,
        )
