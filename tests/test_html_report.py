"""Tests for HTML report rendering."""

from __future__ import annotations

from pathlib import Path

from rag_eval_framework.evaluators.base import EvaluatorResult
from rag_eval_framework.reporting.html_report import render_html_report
from rag_eval_framework.runners.base import (
    EvaluationRunResult,
    RecordResult,
    ThresholdBreach,
    ThresholdResult,
)


def _make_run_result(**overrides) -> EvaluationRunResult:  # type: ignore[no-untyped-def]
    defaults = dict(
        project_name="html-test",
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
            ),
        ],
        threshold_results=[
            ThresholdResult(
                evaluator="answer_presence",
                threshold=0.9,
                actual=1.0,
                passed=True,
            ),
            ThresholdResult(
                evaluator="exact_match_accuracy",
                threshold=0.8,
                actual=0.5,
                passed=False,
            ),
        ],
        passed=False,
    )
    defaults.update(overrides)
    return EvaluationRunResult(**defaults)


class TestHtmlReport:
    def test_creates_html_file(self, tmp_path: Path) -> None:
        result = _make_run_result()
        path = render_html_report(result, tmp_path)
        assert path.exists()
        assert path.suffix == ".html"

    def test_filename_convention(self, tmp_path: Path) -> None:
        result = _make_run_result()
        path = render_html_report(result, tmp_path)
        assert path.name == "eval-report-html-test.html"

    def test_contains_project_name(self, tmp_path: Path) -> None:
        result = _make_run_result()
        path = render_html_report(result, tmp_path)
        html_text = path.read_text(encoding="utf-8")
        assert "html-test" in html_text

    def test_contains_status_failed(self, tmp_path: Path) -> None:
        result = _make_run_result(passed=False)
        path = render_html_report(result, tmp_path)
        html_text = path.read_text(encoding="utf-8")
        assert "FAILED" in html_text

    def test_contains_status_passed(self, tmp_path: Path) -> None:
        result = _make_run_result(
            passed=True,
            threshold_breaches=[],
            aggregate_scores={"answer_presence": 1.0},
        )
        path = render_html_report(result, tmp_path)
        html_text = path.read_text(encoding="utf-8")
        assert "PASSED" in html_text

    def test_contains_evaluator_scores(self, tmp_path: Path) -> None:
        result = _make_run_result()
        path = render_html_report(result, tmp_path)
        html_text = path.read_text(encoding="utf-8")
        assert "answer_presence" in html_text
        assert "exact_match_accuracy" in html_text

    def test_contains_threshold_breaches_section(self, tmp_path: Path) -> None:
        result = _make_run_result()
        path = render_html_report(result, tmp_path)
        html_text = path.read_text(encoding="utf-8")
        assert "Threshold Breaches" in html_text

    def test_no_threshold_breaches_when_passing(self, tmp_path: Path) -> None:
        result = _make_run_result(passed=True, threshold_breaches=[])
        path = render_html_report(result, tmp_path)
        html_text = path.read_text(encoding="utf-8")
        assert "Threshold Breaches" not in html_text

    def test_contains_failed_record_id(self, tmp_path: Path) -> None:
        result = _make_run_result()
        path = render_html_report(result, tmp_path)
        html_text = path.read_text(encoding="utf-8")
        assert "r-002" in html_text

    def test_contains_dataset_statistics(self, tmp_path: Path) -> None:
        result = _make_run_result()
        path = render_html_report(result, tmp_path)
        html_text = path.read_text(encoding="utf-8")
        assert "Dataset Statistics" in html_text

    def test_is_valid_html(self, tmp_path: Path) -> None:
        result = _make_run_result()
        path = render_html_report(result, tmp_path)
        html_text = path.read_text(encoding="utf-8")
        assert html_text.startswith("<!DOCTYPE html>")
        assert "</html>" in html_text

    def test_displays_run_metadata(self, tmp_path: Path) -> None:
        result = _make_run_result(run_metadata={"elapsed_seconds": 1.5})
        path = render_html_report(result, tmp_path)
        html_text = path.read_text(encoding="utf-8")
        assert "Run Metadata" in html_text
        assert "elapsed_seconds" in html_text
