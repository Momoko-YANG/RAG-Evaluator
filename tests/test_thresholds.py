"""Tests for Phase 2 threshold enforcement improvements."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_eval_framework.config.models import ProjectConfig
from rag_eval_framework.runners.local import LocalRunner
from tests.conftest import make_config, make_record


class TestThresholdResults:
    """Verify that threshold_results are populated with pass/fail detail."""

    @pytest.fixture()
    def runner_config(self, tmp_path: Path) -> ProjectConfig:
        dataset_path = tmp_path / "eval.jsonl"
        records = [
            make_record(id="r-001"),
            make_record(id="r-002", contexts=[]),
        ]
        dataset_path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
        return ProjectConfig(
            **make_config(
                dataset_path=str(dataset_path),
                thresholds={
                    "answer_presence": 0.5,
                    "context_presence": 0.9,
                },
            )
        )

    def test_threshold_results_populated(self, runner_config: ProjectConfig) -> None:
        result = LocalRunner().run(runner_config)
        assert len(result.threshold_results) == 2

    def test_threshold_results_contain_pass_and_fail(self, runner_config: ProjectConfig) -> None:
        result = LocalRunner().run(runner_config)
        by_name = {tr.evaluator: tr for tr in result.threshold_results}
        assert by_name["answer_presence"].passed is True
        assert by_name["context_presence"].passed is False

    def test_threshold_results_have_actual_scores(self, runner_config: ProjectConfig) -> None:
        result = LocalRunner().run(runner_config)
        for tr in result.threshold_results:
            assert 0.0 <= tr.actual <= 1.0
            assert 0.0 <= tr.threshold <= 1.0


class TestPerRecordPassFail:
    """Verify per-record pass/fail is computed."""

    @pytest.fixture()
    def strict_config(self, tmp_path: Path) -> ProjectConfig:
        dataset_path = tmp_path / "eval.jsonl"
        records = [
            make_record(id="r-pass"),
            make_record(id="r-fail", contexts=[]),
        ]
        dataset_path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
        return ProjectConfig(
            **make_config(
                dataset_path=str(dataset_path),
                evaluators=["context_presence"],
                thresholds={"context_presence": 0.9},
            )
        )

    def test_per_record_passed_field(self, strict_config: ProjectConfig) -> None:
        result = LocalRunner().run(strict_config)
        by_id = {rr.record_id: rr for rr in result.record_results}
        assert by_id["r-pass"].passed is True
        assert by_id["r-fail"].passed is False


class TestFieldRequirementSkipping:
    """Verify records are SKIPPED when evaluator's required fields are missing."""

    @pytest.fixture()
    def config_with_contexts_eval(self, tmp_path: Path) -> ProjectConfig:
        dataset_path = tmp_path / "eval.jsonl"
        records = [
            make_record(id="r-has-ctx", contexts=["some context"]),
            make_record(id="r-no-ctx", contexts=[]),
        ]
        dataset_path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
        return ProjectConfig(
            **make_config(
                dataset_path=str(dataset_path),
                evaluators=["context_presence"],
                thresholds={},
            )
        )

    def test_no_crash_on_missing_fields(self, config_with_contexts_eval: ProjectConfig) -> None:
        """Even though contexts are empty, the record should not crash."""
        result = LocalRunner().run(config_with_contexts_eval)
        assert result.total_records == 2
