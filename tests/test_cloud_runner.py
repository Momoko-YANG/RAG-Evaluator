"""Tests for Phase 3A — cloud evaluation runner, adapter, normaliser, and CLI.

All Azure SDK interactions are mocked so that tests run without the SDK
installed or any cloud credentials configured.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from rag_eval_framework.config.models import (
    FoundryConfig,
    ProjectConfig,
)
from rag_eval_framework.evaluators.base import EvaluatorResult, EvaluatorStatus
from rag_eval_framework.runners.base import (
    EvaluationRunResult,
    RecordResult,
    ThresholdResult,
)
from rag_eval_framework.runners.foundry.normaliser import (
    build_threshold_results,
    normalise_cloud_metrics,
    normalise_cloud_rows,
    update_record_pass_fail,
)
from tests.conftest import make_config

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cloud_config(**overrides: Any) -> dict[str, Any]:
    """Return a valid cloud-mode project config dict.

    Mode-specific overrides (``evaluators``, ``thresholds``, ``azure``,
    ``foundry``) are routed into the ``cloud_mode`` block automatically.
    """
    _CLOUD_MODE_KEYS = {"evaluators", "thresholds", "azure", "foundry"}

    cloud_updates: dict[str, Any] = {}
    top_updates: dict[str, Any] = {}
    for key, value in overrides.items():
        if key in _CLOUD_MODE_KEYS:
            cloud_updates[key] = value
        else:
            top_updates[key] = value

    base: dict[str, Any] = {
        "project_name": "test-project",
        "dataset_path": "datasets/sample/example-run/golden.jsonl",
        "mode": "cloud",
        "cloud_mode": {
            "evaluators": [
                "groundedness",
                "relevance",
                "retrieval",
                "response_completeness",
            ],
            "azure": {
                "endpoint": "https://test.openai.azure.com",
                "deployment_name": "gpt-4",
                "credential_type": "key",
                "api_key_env": "AZURE_OPENAI_API_KEY",
            },
            "foundry": {
                "subscription_id": "00000000-0000-0000-0000-000000000000",
                "resource_group": "rg-eval-test",
                "project_name": "foundry-project-test",
                "credential_type": "default",
                "poll_interval_seconds": 5,
                "poll_timeout_seconds": 60,
            },
        },
    }
    base["cloud_mode"].update(cloud_updates)
    base.update(top_updates)
    return base


# ---------------------------------------------------------------------------
# FoundryConfig validation
# ---------------------------------------------------------------------------


class TestFoundryConfigValidation:
    """Test Pydantic validation for the FoundryConfig model."""

    def test_default_values(self) -> None:
        cfg = FoundryConfig()
        assert cfg.endpoint == ""
        assert cfg.subscription_id == ""
        assert cfg.credential_type == "default"
        assert cfg.poll_interval_seconds == 10
        assert cfg.poll_timeout_seconds == 1800

    def test_valid_config(self) -> None:
        cfg = FoundryConfig(
            endpoint="https://my-project.eastus.api.azureml.ms",
            subscription_id="abc-123",
            resource_group="rg-test",
            project_name="my-project",
            credential_type="key",
            poll_interval_seconds=15,
            poll_timeout_seconds=300,
        )
        assert cfg.resource_group == "rg-test"
        assert cfg.endpoint == "https://my-project.eastus.api.azureml.ms"

    def test_poll_interval_too_low(self) -> None:
        with pytest.raises(ValidationError):
            FoundryConfig(poll_interval_seconds=0)

    def test_poll_interval_too_high(self) -> None:
        with pytest.raises(ValidationError):
            FoundryConfig(poll_interval_seconds=301)

    def test_poll_timeout_too_low(self) -> None:
        with pytest.raises(ValidationError):
            FoundryConfig(poll_timeout_seconds=10)

    def test_invalid_credential_type(self) -> None:
        with pytest.raises(ValidationError):
            FoundryConfig(credential_type="bogus")

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            FoundryConfig(unknown_field="value")


class TestProjectConfigCloudFields:
    """Test mode and foundry fields on ProjectConfig."""

    def test_default_mode_is_local(self) -> None:
        cfg = ProjectConfig(**make_config())
        assert cfg.mode == "local"
        assert cfg.foundry is None

    def test_cloud_mode_parses(self) -> None:
        cfg = ProjectConfig(**_cloud_config())
        assert cfg.mode == "cloud"
        assert cfg.foundry is not None
        assert cfg.foundry.subscription_id == "00000000-0000-0000-0000-000000000000"

    def test_invalid_mode_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ProjectConfig(
                project_name="test",
                dataset_path="data.jsonl",
                mode="hybrid",  # type: ignore[arg-type]
            )

    def test_foundry_none_in_local_mode(self) -> None:
        cfg = ProjectConfig(**make_config())
        assert cfg.foundry is None


# ---------------------------------------------------------------------------
# normalise_cloud_metrics
# ---------------------------------------------------------------------------


class TestNormaliseCloudMetrics:
    """Tests for normalise_cloud_metrics()."""

    def test_direct_key_0_to_1(self) -> None:
        raw = {"groundedness": 0.85}
        scores = normalise_cloud_metrics(raw, ["groundedness"])
        assert scores["groundedness"] == 0.85

    def test_dotted_key(self) -> None:
        raw = {"groundedness.groundedness": 0.7}
        scores = normalise_cloud_metrics(raw, ["groundedness"])
        assert scores["groundedness"] == 0.7

    def test_gpt_dotted_key(self) -> None:
        raw = {"relevance.gpt_relevance": 0.9}
        scores = normalise_cloud_metrics(raw, ["relevance"])
        assert scores["relevance"] == 0.9

    def test_legacy_gpt_key(self) -> None:
        raw = {"gpt_groundedness": 0.6}
        scores = normalise_cloud_metrics(raw, ["groundedness"])
        assert scores["groundedness"] == 0.6

    def test_normalises_1_5_scale(self) -> None:
        # Score of 3 on 1-5 scale → 0.5 on 0-1 scale
        raw = {"groundedness": 3.0}
        scores = normalise_cloud_metrics(raw, ["groundedness"])
        assert scores["groundedness"] == 0.5

    def test_normalises_5_to_1(self) -> None:
        raw = {"groundedness": 5.0}
        scores = normalise_cloud_metrics(raw, ["groundedness"])
        assert scores["groundedness"] == 1.0

    def test_missing_evaluator_returns_zero(self) -> None:
        raw = {}
        scores = normalise_cloud_metrics(raw, ["groundedness"])
        assert scores["groundedness"] == 0.0

    def test_multiple_evaluators(self) -> None:
        raw = {"groundedness": 0.9, "relevance.relevance": 0.8}
        scores = normalise_cloud_metrics(raw, ["groundedness", "relevance"])
        assert scores["groundedness"] == 0.9
        assert scores["relevance"] == 0.8


# ---------------------------------------------------------------------------
# normalise_cloud_rows
# ---------------------------------------------------------------------------


class TestNormaliseCloudRows:
    """Tests for normalise_cloud_rows()."""

    def test_basic_row_conversion(self) -> None:
        rows = [
            {
                "id": "rec-001",
                "outputs": {
                    "groundedness": {"groundedness": 0.8, "groundedness_reason": "OK"},
                },
            },
        ]
        results = normalise_cloud_rows(rows, ["groundedness"])
        assert len(results) == 1
        assert results[0].record_id == "rec-001"
        assert results[0].evaluator_results["groundedness"].score == 0.8
        assert results[0].evaluator_results["groundedness"].reason == "OK"

    def test_fallback_record_id(self) -> None:
        rows = [{"outputs": {"groundedness": {"groundedness": 0.5}}}]
        results = normalise_cloud_rows(rows, ["groundedness"])
        assert results[0].record_id == "cloud-row-0000"

    def test_record_id_from_case_id(self) -> None:
        rows = [{"case_id": "my-case", "outputs": {}}]
        results = normalise_cloud_rows(rows, [])
        assert results[0].record_id == "my-case"

    def test_missing_evaluator_in_row(self) -> None:
        rows = [{"id": "r-1", "outputs": {}}]
        results = normalise_cloud_rows(rows, ["groundedness"])
        er = results[0].evaluator_results["groundedness"]
        assert er.status == EvaluatorStatus.ERROR
        assert er.score == 0.0

    def test_normalises_1_5_score_in_rows(self) -> None:
        rows = [
            {
                "id": "r-1",
                "outputs": {"groundedness": {"groundedness": 4.0}},
            },
        ]
        results = normalise_cloud_rows(rows, ["groundedness"])
        # 4 on 1-5 → 0.75 on 0-1
        assert results[0].evaluator_results["groundedness"].score == 0.75


# ---------------------------------------------------------------------------
# build_threshold_results
# ---------------------------------------------------------------------------


class TestBuildThresholdResults:
    """Tests for build_threshold_results()."""

    def test_all_pass(self) -> None:
        scores = {"groundedness": 0.9, "relevance": 0.8}
        thresholds = {"groundedness": 0.7, "relevance": 0.7}
        results, breaches = build_threshold_results(scores, thresholds)
        assert len(results) == 2
        assert all(t.passed for t in results)
        assert len(breaches) == 0

    def test_one_breach(self) -> None:
        scores = {"groundedness": 0.5}
        thresholds = {"groundedness": 0.7}
        results, breaches = build_threshold_results(scores, thresholds)
        assert len(breaches) == 1
        assert breaches[0].evaluator == "groundedness"
        assert breaches[0].delta == 0.2

    def test_missing_evaluator_defaults_zero(self) -> None:
        scores = {}
        thresholds = {"groundedness": 0.5}
        results, breaches = build_threshold_results(scores, thresholds)
        assert len(breaches) == 1
        assert results[0].actual == 0.0


# ---------------------------------------------------------------------------
# update_record_pass_fail
# ---------------------------------------------------------------------------


class TestUpdateRecordPassFail:
    """Tests for update_record_pass_fail()."""

    def test_marks_failing_record(self) -> None:
        records = [
            RecordResult(
                record_id="r-1",
                evaluator_results={
                    "answer_presence": EvaluatorResult(score=0.5, reason="low"),
                },
            ),
        ]
        update_record_pass_fail(records, {"answer_presence": 0.8})
        assert records[0].passed is False

    def test_keeps_passing_record(self) -> None:
        records = [
            RecordResult(
                record_id="r-1",
                evaluator_results={
                    "answer_presence": EvaluatorResult(score=0.9, reason="good"),
                },
            ),
        ]
        update_record_pass_fail(records, {"answer_presence": 0.8})
        assert records[0].passed is True

    def test_skipped_evaluator_ignored(self) -> None:
        records = [
            RecordResult(
                record_id="r-1",
                evaluator_results={
                    "answer_presence": EvaluatorResult(
                        score=0.0,
                        reason="skipped",
                        status=EvaluatorStatus.SKIPPED,
                    ),
                },
            ),
        ]
        update_record_pass_fail(records, {"answer_presence": 0.5})
        assert records[0].passed is True


# ---------------------------------------------------------------------------
# FoundryRunner (with mocked adapter)
# ---------------------------------------------------------------------------


class TestFoundryRunner:
    """Test the FoundryRunner orchestration (all SDK calls mocked)."""

    def _make_fake_raw_results(self) -> dict[str, Any]:
        return {
            "metrics": {
                "groundedness.groundedness": 0.85,
                "relevance": 0.6,
            },
            "rows": [
                {
                    "id": "rec-001",
                    "outputs": {
                        "groundedness": {"groundedness": 0.9},
                        "relevance": {"relevance": 0.7},
                    },
                },
                {
                    "id": "rec-002",
                    "outputs": {
                        "groundedness": {"groundedness": 0.8},
                        "relevance": {"relevance": 0.5},
                    },
                },
            ],
            "run_id": "foundry-run-001",
        }

    @patch(
        "rag_eval_framework.runners.cloud.FoundryAdapter",
        autospec=True,
    )
    def test_run_returns_cloud_result(self, mock_adapter_cls: MagicMock) -> None:
        from rag_eval_framework.runners.cloud import FoundryRunner

        mock_adapter = mock_adapter_cls.return_value
        mock_adapter.submit_and_wait.return_value = self._make_fake_raw_results()

        cfg = ProjectConfig(
            **_cloud_config(
                evaluators=["groundedness", "relevance"],
                thresholds={"groundedness": 0.7},
            )
        )
        runner = FoundryRunner()
        result = runner.run(cfg)

        assert result.runner_type == "cloud"
        assert result.total_records == 2
        assert result.passed is True
        assert "groundedness" in result.aggregate_scores
        assert result.run_metadata["cloud_run_id"] == "foundry-run-001"
        mock_adapter.connect.assert_called_once()
        mock_adapter.submit_and_wait.assert_called_once()

    @patch(
        "rag_eval_framework.runners.cloud.FoundryAdapter",
        autospec=True,
    )
    def test_run_detects_threshold_breach(self, mock_adapter_cls: MagicMock) -> None:
        from rag_eval_framework.runners.cloud import FoundryRunner

        mock_adapter = mock_adapter_cls.return_value
        mock_adapter.submit_and_wait.return_value = self._make_fake_raw_results()

        cfg = ProjectConfig(
            **_cloud_config(
                evaluators=["groundedness", "relevance"],
                thresholds={"relevance": 0.9},  # will breach (0.6 < 0.9)
            )
        )
        runner = FoundryRunner()
        result = runner.run(cfg)

        assert result.passed is False
        assert len(result.threshold_breaches) == 1
        assert result.threshold_breaches[0].evaluator == "relevance"

    @patch(
        "rag_eval_framework.runners.cloud.FoundryAdapter",
        autospec=True,
    )
    def test_run_raises_cloud_runner_error_on_adapter_failure(
        self, mock_adapter_cls: MagicMock
    ) -> None:
        from rag_eval_framework.runners.cloud import CloudRunnerError, FoundryRunner
        from rag_eval_framework.runners.foundry.adapter import FoundryAdapterError

        mock_adapter = mock_adapter_cls.return_value
        mock_adapter.connect.side_effect = FoundryAdapterError("no connection")

        cfg = ProjectConfig(**_cloud_config())
        runner = FoundryRunner()

        with pytest.raises(CloudRunnerError, match="Failed to initialise"):
            runner.run(cfg)


# ---------------------------------------------------------------------------
# FoundryAdapter — unit-level tests (SDK mocked)
# ---------------------------------------------------------------------------


class TestFoundryAdapter:
    """Unit tests for FoundryAdapter with mocked Azure SDKs."""

    def test_missing_foundry_config_raises(self) -> None:
        from rag_eval_framework.runners.foundry.adapter import (
            FoundryAdapter,
            FoundryAdapterError,
        )

        cfg = ProjectConfig(**make_config())  # local mode, no foundry
        with pytest.raises(FoundryAdapterError, match="foundry"):
            FoundryAdapter(cfg)

    def test_ensure_connected_before_submit(self) -> None:
        from rag_eval_framework.runners.foundry.adapter import (
            FoundryAdapter,
            FoundryAdapterError,
        )

        cfg = ProjectConfig(**_cloud_config())
        adapter = FoundryAdapter(cfg)
        # Not connected yet
        with pytest.raises(FoundryAdapterError, match="connect"):
            adapter.submit_evaluation(
                dataset_path="data.jsonl",
                evaluators=["groundedness"],
            )

    @patch(
        "rag_eval_framework.runners.foundry.adapter._require_foundry_sdk",
    )
    def test_connect_calls_require_sdk(self, mock_require: MagicMock) -> None:
        from rag_eval_framework.runners.foundry.adapter import FoundryAdapter

        cfg = ProjectConfig(**_cloud_config())
        adapter = FoundryAdapter(cfg)

        # Mock the _build_client to avoid real SDK call
        adapter._build_client = MagicMock(return_value=MagicMock())  # type: ignore[method-assign]
        adapter.connect()

        mock_require.assert_called_once()

    def test_require_foundry_sdk_raises_import_error(self) -> None:
        from rag_eval_framework.runners.foundry.adapter import _require_foundry_sdk

        with (
            patch.dict("sys.modules", {"azure.ai.projects": None}),
            pytest.raises(ImportError, match="Azure AI Projects SDK"),
        ):
            _require_foundry_sdk()


# ---------------------------------------------------------------------------
# CLI runner selection
# ---------------------------------------------------------------------------


class TestCLIRunnerSelection:
    """Test that the CLI _build_runner helper selects the correct runner."""

    def test_local_mode_returns_local_runner(self) -> None:
        from rag_eval_framework.cli.run import _build_runner
        from rag_eval_framework.runners.local import LocalRunner

        runner = _build_runner("local")
        assert isinstance(runner, LocalRunner)

    @patch(
        "rag_eval_framework.runners.cloud.FoundryAdapter",
        autospec=True,
    )
    def test_cloud_mode_returns_foundry_runner(self, _mock: MagicMock) -> None:
        from rag_eval_framework.cli.run import _build_runner
        from rag_eval_framework.runners.cloud import FoundryRunner

        runner = _build_runner("cloud")
        assert isinstance(runner, FoundryRunner)


# ---------------------------------------------------------------------------
# Report runner_type display
# ---------------------------------------------------------------------------


class TestReportRunnerType:
    """Verify that reports include the runner_type field."""

    def _make_cloud_run_result(self) -> EvaluationRunResult:
        return EvaluationRunResult(
            project_name="cloud-test",
            timestamp="2026-06-01T12:00:00Z",
            dataset_path="datasets/sample/example-run/golden.jsonl",
            total_records=1,
            evaluators_used=["groundedness"],
            record_results=[
                RecordResult(
                    record_id="r-001",
                    evaluator_results={
                        "groundedness": EvaluatorResult(score=0.85, reason="OK"),
                    },
                ),
            ],
            aggregate_scores={"groundedness": 0.85},
            threshold_breaches=[],
            threshold_results=[
                ThresholdResult(
                    evaluator="groundedness",
                    threshold=0.7,
                    actual=0.85,
                    passed=True,
                ),
            ],
            passed=True,
            runner_type="cloud",
            run_metadata={"cloud_run_id": "run-abc123"},
        )

    def test_markdown_report_shows_runner_type(self, tmp_path: Path) -> None:
        from rag_eval_framework.reporting.markdown_report import render_markdown_report

        result = self._make_cloud_run_result()
        path = render_markdown_report(result, tmp_path)
        text = path.read_text(encoding="utf-8")
        assert "**Runner**: cloud" in text

    def test_html_report_shows_runner_type(self, tmp_path: Path) -> None:
        from rag_eval_framework.reporting.html_report import render_html_report

        result = self._make_cloud_run_result()
        path = render_html_report(result, tmp_path)
        text = path.read_text(encoding="utf-8")
        assert "cloud" in text

    def test_json_report_includes_runner_type(self, tmp_path: Path) -> None:
        import json

        from rag_eval_framework.reporting.json_report import render_json_report

        result = self._make_cloud_run_result()
        path = render_json_report(result, tmp_path)
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["runner_type"] == "cloud"

    def test_local_runner_type_default(self) -> None:
        """EvaluationRunResult defaults runner_type to 'local'."""
        result = EvaluationRunResult(
            project_name="test",
            timestamp="now",
            dataset_path="data.jsonl",
            total_records=0,
            evaluators_used=[],
            record_results=[],
            aggregate_scores={},
            threshold_breaches=[],
            threshold_results=[],
            passed=True,
        )
        assert result.runner_type == "local"
