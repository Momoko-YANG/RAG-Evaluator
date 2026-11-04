"""Tests for report generation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_eval_framework.config.models import ProjectConfig
from rag_eval_framework.evaluators.base import EvaluatorResult
from rag_eval_framework.reporting.engine import ReportEngine
from rag_eval_framework.reporting.json_report import render_json_report
from rag_eval_framework.reporting.markdown_report import render_markdown_report
from rag_eval_framework.runners.base import (
    EvaluationRunResult,
    RecordResult,
    ThresholdBreach,
)
from tests.conftest import make_config


@pytest.fixture()
def sample_run_result() -> EvaluationRunResult:
    return EvaluationRunResult(
        project_name="test-project",
        timestamp="2026-03-30T12:00:00Z",
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
                    "exact_match_accuracy": EvaluatorResult(score=0.0, reason="No match"),
                },
                passed=False,
            ),
        ],
        aggregate_scores={"answer_presence": 1.0, "exact_match_accuracy": 0.5},
        threshold_breaches=[
            ThresholdBreach(
                evaluator="exact_match_accuracy",
                threshold=0.8,
                actual=0.5,
                delta=0.3,
            )
        ],
        passed=False,
    )


class TestJsonReport:
    def test_renders_valid_json(
        self, tmp_path: Path, sample_run_result: EvaluationRunResult
    ) -> None:
        path = render_json_report(sample_run_result, tmp_path)
        assert path.exists()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["project_name"] == "test-project"
        assert "aggregate_scores" in data

    def test_filename_convention(
        self, tmp_path: Path, sample_run_result: EvaluationRunResult
    ) -> None:
        path = render_json_report(sample_run_result, tmp_path)
        assert path.name == "eval-report-test-project.json"


class TestMarkdownReport:
    def test_renders_markdown(self, tmp_path: Path, sample_run_result: EvaluationRunResult) -> None:
        path = render_markdown_report(sample_run_result, tmp_path)
        assert path.exists()
        text = path.read_text(encoding="utf-8")
        assert "# Evaluation Report" in text
        assert "FAILED" in text
        assert "exact_match_accuracy" in text

    def test_contains_threshold_breaches(
        self, tmp_path: Path, sample_run_result: EvaluationRunResult
    ) -> None:
        path = render_markdown_report(sample_run_result, tmp_path)
        text = path.read_text(encoding="utf-8")
        assert "Threshold Breaches" in text

    def test_contains_failed_records(
        self, tmp_path: Path, sample_run_result: EvaluationRunResult
    ) -> None:
        path = render_markdown_report(sample_run_result, tmp_path)
        text = path.read_text(encoding="utf-8")
        assert "r-002" in text


class TestReportEngine:
    def test_generates_both_formats(
        self, tmp_path: Path, sample_run_result: EvaluationRunResult
    ) -> None:
        config = ProjectConfig(
            **make_config(
                report_format=["json", "markdown"],
                output_dir=str(tmp_path),
            )
        )
        engine = ReportEngine()
        paths = engine.generate(config, sample_run_result)
        assert len(paths) == 2
        extensions = {p.suffix for p in paths}
        assert ".json" in extensions
        assert ".md" in extensions

    def test_creates_output_directory(
        self, tmp_path: Path, sample_run_result: EvaluationRunResult
    ) -> None:
        output_dir = tmp_path / "deep" / "nested"
        config = ProjectConfig(
            **make_config(
                report_format=["json"],
                output_dir=str(output_dir),
            )
        )
        engine = ReportEngine()
        paths = engine.generate(config, sample_run_result)
        assert len(paths) == 1
        assert paths[0].exists()


class TestMarkdownPassedReport:
    """Ensure PASSED reports render correctly (no breaches section)."""

    def test_passed_report_shows_passed(self, tmp_path: Path) -> None:
        result = EvaluationRunResult(
            project_name="ok-project",
            timestamp="2026-03-30T12:00:00Z",
            dataset_path="datasets/ok/eval.jsonl",
            total_records=1,
            evaluators_used=["answer_presence"],
            record_results=[
                RecordResult(
                    record_id="r-001",
                    evaluator_results={
                        "answer_presence": EvaluatorResult(score=1.0, reason="OK"),
                    },
                ),
            ],
            aggregate_scores={"answer_presence": 1.0},
            passed=True,
        )
        path = render_markdown_report(result, tmp_path)
        text = path.read_text(encoding="utf-8")
        assert "PASSED" in text
        assert "Threshold Breaches" not in text
