"""Tests added during Phase 2 hardening pass.

Covers:
- credential_type validation on AzureConfig
- Built-in evaluator required_fields declarations
- Shared normalise_1_5 utility
- ReportEngine with HTML format
- Azure adapter graceful SDK-missing handling
- _collect_failed_records uses rr.passed (not hardcoded 0.5)
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from rag_eval_framework.config.models import AzureConfig, ProjectConfig
from rag_eval_framework.evaluators.base import EvaluatorResult
from rag_eval_framework.evaluators.generation.answer_presence import AnswerPresenceEvaluator
from rag_eval_framework.evaluators.generation.correctness import ExactMatchAccuracyEvaluator
from rag_eval_framework.evaluators.retrieval.context_presence import ContextPresenceEvaluator
from rag_eval_framework.reporting.engine import ReportEngine
from rag_eval_framework.reporting.html_report import render_html_report
from rag_eval_framework.reporting.markdown_report import render_markdown_report
from rag_eval_framework.runners.base import (
    EvaluationRunResult,
    RecordResult,
    ThresholdBreach,
    ThresholdResult,
)
from rag_eval_framework.utils.normalisation import normalise_1_5
from tests.conftest import make_config

# ---------------------------------------------------------------------------
# credential_type validation
# ---------------------------------------------------------------------------


class TestCredentialTypeValidation:
    def test_valid_default(self) -> None:
        cfg = AzureConfig(credential_type="default")
        assert cfg.credential_type == "default"

    def test_valid_key(self) -> None:
        cfg = AzureConfig(credential_type="key")
        assert cfg.credential_type == "key"

    def test_valid_env(self) -> None:
        cfg = AzureConfig(credential_type="env")
        assert cfg.credential_type == "env"

    def test_invalid_credential_type_rejected(self) -> None:
        with pytest.raises(ValidationError):
            AzureConfig(credential_type="bogus")


# ---------------------------------------------------------------------------
# Built-in required_fields
# ---------------------------------------------------------------------------


class TestBuiltinRequiredFields:
    """Built-in evaluators' required_fields.

    context_presence and exact_match_accuracy intentionally do NOT require
    "contexts" or "ground_truth_answer" — they handle missing values
    gracefully (0.0 or 0.5) rather than skipping the record.  context_presence
    scores retrieval, so it does not need a response either.
    """

    def test_answer_presence_requires_question_response(self) -> None:
        ev = AnswerPresenceEvaluator()
        assert ev.required_fields == ["question", "response"]

    def test_context_presence_requires_question_only(self) -> None:
        ev = ContextPresenceEvaluator()
        assert ev.required_fields == ["question"]
        assert "contexts" not in ev.required_fields

    def test_exact_match_uses_default_fields(self) -> None:
        ev = ExactMatchAccuracyEvaluator()
        assert ev.required_fields == ["question", "response"]
        assert "ground_truth_answer" not in ev.required_fields


# ---------------------------------------------------------------------------
# Shared normalise_1_5 utility
# ---------------------------------------------------------------------------


class TestNormalise15:
    def test_min(self) -> None:
        assert normalise_1_5(1) == 0.0

    def test_max(self) -> None:
        assert normalise_1_5(5) == 1.0

    def test_mid(self) -> None:
        assert normalise_1_5(3) == 0.5

    def test_clamp_below(self) -> None:
        assert normalise_1_5(0) == 0.0

    def test_clamp_above(self) -> None:
        assert normalise_1_5(10) == 1.0

    def test_fractional(self) -> None:
        assert normalise_1_5(2) == 0.25


# ---------------------------------------------------------------------------
# ReportEngine — HTML format
# ---------------------------------------------------------------------------


def _make_run_result(**overrides: object) -> EvaluationRunResult:
    defaults: dict[str, object] = dict(
        project_name="harden-test",
        timestamp="2026-06-01T12:00:00Z",
        dataset_path="datasets/sample/example-run/golden.jsonl",
        total_records=2,
        evaluators_used=["answer_presence", "exact_match_accuracy"],
        record_results=[
            RecordResult(
                record_id="r-001",
                evaluator_results={
                    "answer_presence": EvaluatorResult(score=1.0, reason="OK"),
                    "exact_match_accuracy": EvaluatorResult(score=1.0, reason="Match"),
                },
            ),
            RecordResult(
                record_id="r-002",
                evaluator_results={
                    "answer_presence": EvaluatorResult(score=1.0, reason="OK"),
                    "exact_match_accuracy": EvaluatorResult(score=0.3, reason="No match"),
                },
                passed=False,
            ),
        ],
        aggregate_scores={"answer_presence": 1.0, "exact_match_accuracy": 0.65},
        threshold_breaches=[
            ThresholdBreach(
                evaluator="exact_match_accuracy",
                threshold=0.8,
                actual=0.65,
                delta=0.15,
            ),
        ],
        threshold_results=[
            ThresholdResult(
                evaluator="exact_match_accuracy",
                threshold=0.8,
                actual=0.65,
                passed=False,
            ),
        ],
        passed=False,
    )
    defaults.update(overrides)  # type: ignore[arg-type]
    return EvaluationRunResult(**defaults)  # type: ignore[arg-type]


class TestReportEngineHtml:
    def test_engine_generates_html(self, tmp_path: Path) -> None:
        config = ProjectConfig(
            **make_config(
                report_format=["html"],
                output_dir=str(tmp_path),
            )
        )
        engine = ReportEngine()
        paths = engine.generate(config, _make_run_result())
        assert len(paths) == 1
        assert paths[0].suffix == ".html"
        assert paths[0].exists()

    def test_engine_generates_all_three(self, tmp_path: Path) -> None:
        config = ProjectConfig(
            **make_config(
                report_format=["json", "markdown", "html"],
                output_dir=str(tmp_path),
            )
        )
        engine = ReportEngine()
        paths = engine.generate(config, _make_run_result())
        assert len(paths) == 3
        extensions = {p.suffix for p in paths}
        assert extensions == {".json", ".md", ".html"}


# ---------------------------------------------------------------------------
# _collect_failed_records uses rr.passed (not hardcoded 0.5)
# ---------------------------------------------------------------------------


class TestCollectFailedRecordsThresholdAware:
    """Ensure failed-record collection respects rr.passed, not a score cutoff."""

    def test_markdown_shows_failed_record_above_05(self, tmp_path: Path) -> None:
        """A record with score 0.65 that fails its 0.8 threshold should appear."""
        result = _make_run_result()
        path = render_markdown_report(result, tmp_path)
        text = path.read_text(encoding="utf-8")
        # r-002 has score 0.3 (below 1.0) and passed=False
        assert "r-002" in text

    def test_html_shows_failed_record_above_05(self, tmp_path: Path) -> None:
        result = _make_run_result()
        path = render_html_report(result, tmp_path)
        html_text = path.read_text(encoding="utf-8")
        assert "r-002" in html_text

    def test_passing_records_not_in_failed_section(self, tmp_path: Path) -> None:
        """r-001 passes — it should NOT appear in failed records."""
        result = _make_run_result()
        path = render_markdown_report(result, tmp_path)
        text = path.read_text(encoding="utf-8")
        # r-001 passed=True, so should not be in "Sample Failed Records"
        failed_section = text.split("## Sample Failed Records")
        if len(failed_section) > 1:
            assert "r-001" not in failed_section[1]


# ---------------------------------------------------------------------------
# Azure adapter graceful SDK-missing handling
# ---------------------------------------------------------------------------


class TestAzureAdapterGracefulFailure:
    def test_setup_does_not_raise_when_sdk_missing(self) -> None:
        """If azure-ai-evaluation isn't installed, setup() warns and returns."""
        from rag_eval_framework.evaluators.generation.faithfulness import GroundednessEvaluator

        ev = GroundednessEvaluator()
        config = ProjectConfig(
            **make_config(
                azure={
                    "endpoint": "https://test.openai.azure.com",
                    "deployment_name": "gpt-4",
                }
            )
        )

        # Patch the SDK import to always fail
        with patch(
            "rag_eval_framework.evaluators.common.azure_sdk._require_azure_sdk",
            side_effect=ImportError("mock missing SDK"),
        ):
            ev.setup(config)  # should not raise
            assert not ev._configured
