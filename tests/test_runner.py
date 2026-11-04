"""Tests for the local evaluation runner."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from rag_eval_framework.config.models import ProjectConfig
from rag_eval_framework.runners.local import LocalRunner
from tests.conftest import make_config, make_record


@pytest.fixture()
def runner_config(tmp_path: Path) -> ProjectConfig:
    """Create a valid config pointing to a temp dataset."""
    dataset_path = tmp_path / "eval.jsonl"
    records = [
        make_record(id="r-001"),
        make_record(id="r-002", contexts=[], ground_truth_answer="wrong answer"),
        make_record(id="r-003", response="ok fine", ground_truth_answer="90 days"),
    ]
    lines = [json.dumps(r) for r in records]
    dataset_path.write_text("\n".join(lines), encoding="utf-8")

    data = make_config(
        dataset_path=str(dataset_path),
        output_dir=str(tmp_path / "output"),
        thresholds={"answer_presence": 0.5},
    )
    return ProjectConfig(**data)


class TestLocalRunner:
    def test_run_produces_result(self, runner_config: ProjectConfig) -> None:
        runner = LocalRunner()
        result = runner.run(runner_config)
        assert result.project_name == "test-project"
        assert result.total_records == 3
        assert len(result.record_results) == 3

    def test_aggregate_scores_present(self, runner_config: ProjectConfig) -> None:
        runner = LocalRunner()
        result = runner.run(runner_config)
        for evaluator_name in runner_config.evaluators:
            assert evaluator_name in result.aggregate_scores

    def test_threshold_pass(self, runner_config: ProjectConfig) -> None:
        runner = LocalRunner()
        result = runner.run(runner_config)
        # All records have multi-word responses → answer_presence should be 1.0,
        # which exceeds the 0.5 threshold.
        assert result.passed is True

    def test_threshold_breach_detected(self, tmp_path: Path) -> None:
        dataset_path = tmp_path / "eval.jsonl"
        # All records have empty contexts → context_presence = 0.0
        records = [
            make_record(id="r-001", contexts=[]),
            make_record(id="r-002", contexts=[]),
        ]
        dataset_path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
        data = make_config(
            dataset_path=str(dataset_path),
            evaluators=["context_presence"],
            thresholds={"context_presence": 0.9},
        )
        config = ProjectConfig(**data)
        runner = LocalRunner()
        result = runner.run(config)
        assert result.passed is False
        assert len(result.threshold_breaches) == 1
        assert result.threshold_breaches[0].evaluator == "context_presence"

    def test_retrieval_only_records_skip_response_evaluators(self, tmp_path: Path) -> None:
        dataset_path = tmp_path / "eval.jsonl"
        retrieval_only = {"rag_params": {"topK": 3, "retrievalOnly": True}}
        records = [
            make_record(id="r-001", response="(retrieval only)", metadata=retrieval_only),
            make_record(id="r-002"),
        ]
        dataset_path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
        data = make_config(
            dataset_path=str(dataset_path),
            evaluators=["answer_presence", "exact_match_accuracy", "context_presence"],
        )
        result = LocalRunner().run(ProjectConfig(**data))

        placeholder = result.record_results[0].evaluator_results
        assert placeholder["answer_presence"].status.value == "skipped"
        assert "retrieval-only" in placeholder["answer_presence"].reason
        assert placeholder["exact_match_accuracy"].status.value == "skipped"
        # context_presence scores retrieval and still applies.
        assert placeholder["context_presence"].status.value == "success"
        # Means come from the answered record only.
        assert result.aggregate_scores["answer_presence"] == 1.0
        answered = result.record_results[1].evaluator_results
        assert answered["answer_presence"].status.value == "success"

    def test_evaluator_with_nothing_to_score_is_not_a_zero(self, tmp_path: Path) -> None:
        dataset_path = tmp_path / "eval.jsonl"
        # No ground truth labels: retrieval_hit_rate skips every record.
        dataset_path.write_text(json.dumps(make_record(id="r-001")), encoding="utf-8")
        data = make_config(
            dataset_path=str(dataset_path),
            output_dir=str(tmp_path / "out"),
            evaluators=["answer_presence", "retrieval_hit_rate"],
        )
        result = LocalRunner().run(ProjectConfig(**data))
        assert "retrieval_hit_rate" not in result.aggregate_scores
        assert result.passed is True
        from rag_eval_framework.reporting.engine import ReportEngine

        config = ProjectConfig(**data)
        md = next(p for p in ReportEngine().generate(config, result) if p.suffix == ".md")
        assert "| retrieval | retrieval_hit_rate | n/a |" in md.read_text(encoding="utf-8")

        # With a threshold, "nothing scored" fails instead of passing silently.
        with_threshold = make_config(
            dataset_path=str(dataset_path),
            evaluators=["answer_presence", "retrieval_hit_rate"],
            thresholds={"retrieval_hit_rate": 0.5},
        )
        failed = LocalRunner().run(ProjectConfig(**with_threshold))
        assert failed.passed is False
        assert failed.threshold_breaches[0].evaluator == "retrieval_hit_rate"

    def test_unknown_evaluator_raises(self) -> None:
        """Unknown evaluator is now rejected at config parse time by the catalog validator."""
        data = make_config(evaluators=["nonexistent"])
        with pytest.raises(ValidationError, match="not in the local catalog"):
            ProjectConfig(**data)
