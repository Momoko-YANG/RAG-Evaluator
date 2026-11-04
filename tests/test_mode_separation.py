"""Tests for the mode separation refactoring.

Validates:
 - New-format config parsing (local_mode / cloud_mode blocks)
 - Catalog-based evaluator validation per mode
 - Legacy backward compatibility (flat format still works)
 - Mode banner in reports
 - Registry mode-aware helpers
 - CLI runner selection
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from rag_eval_framework.config.models import (
    CLOUD_EVALUATOR_CATALOG,
    CLOUD_EVALUATOR_NAMES,
    LOCAL_EVALUATOR_CATALOG,
    LOCAL_EVALUATOR_NAMES,
    CloudModeConfig,
    LocalModeConfig,
    ProjectConfig,
)
from rag_eval_framework.core.registry import default_registry

# ---------------------------------------------------------------------------
# Evaluator catalog definitions
# ---------------------------------------------------------------------------


class TestEvaluatorCatalogDefinitions:
    """Verify catalog constants are correctly defined."""

    def test_local_catalog_has_21_evaluators(self) -> None:
        assert len(LOCAL_EVALUATOR_CATALOG) == 21

    def test_cloud_catalog_has_4_evaluators(self) -> None:
        assert len(CLOUD_EVALUATOR_CATALOG) == 4

    def test_cloud_is_subset_of_local(self) -> None:
        # All cloud evaluator *names* also exist in local
        assert CLOUD_EVALUATOR_NAMES <= LOCAL_EVALUATOR_NAMES

    def test_local_catalog_contains_all_tiers(self) -> None:
        scaffold = {
            "answer_presence",
            "context_presence",
            "exact_match_accuracy",
            "retrieval_hit_rate",
            "retrieval_mrr",
            "retrieval_recall",
            "retrieval_ndcg",
        }
        azure_sdk = {
            "groundedness",
            "relevance",
            "retrieval",
            "response_completeness",
            "document_retrieval",
        }
        judges = {
            "accuracy_judge",
            "hallucination_judge",
            "citation_judge",
            "policy_compliance_judge",
        }
        assert scaffold <= LOCAL_EVALUATOR_NAMES
        assert azure_sdk <= LOCAL_EVALUATOR_NAMES
        assert judges <= LOCAL_EVALUATOR_NAMES

    def test_cloud_catalog_only_sdk_evaluators(self) -> None:
        expected = {"groundedness", "relevance", "retrieval", "response_completeness"}
        assert expected == CLOUD_EVALUATOR_NAMES

    def test_catalogs_have_descriptions(self) -> None:
        for name, desc in LOCAL_EVALUATOR_CATALOG.items():
            assert desc, f"Missing description for local evaluator '{name}'"
        for name, desc in CLOUD_EVALUATOR_CATALOG.items():
            assert desc, f"Missing description for cloud evaluator '{name}'"


# ---------------------------------------------------------------------------
# LocalModeConfig validation
# ---------------------------------------------------------------------------


class TestLocalModeConfig:
    """Validate LocalModeConfig block."""

    def test_valid_scaffold_only(self) -> None:
        cfg = LocalModeConfig(evaluators=["answer_presence", "context_presence"])
        assert len(cfg.evaluators) == 2

    def test_valid_all_evaluators(self) -> None:
        cfg = LocalModeConfig(evaluators=list(LOCAL_EVALUATOR_NAMES))
        assert len(cfg.evaluators) == 21

    def test_invalid_evaluator_rejected(self) -> None:
        with pytest.raises(ValidationError, match="not in the local catalog"):
            LocalModeConfig(evaluators=["nonexistent_evaluator"])

    def test_cloud_only_evaluator_accepted_in_local(self) -> None:
        """Cloud evaluators like 'groundedness' are also in the local catalog."""
        cfg = LocalModeConfig(evaluators=["groundedness"])
        assert cfg.evaluators == ["groundedness"]

    def test_empty_evaluators_rejected(self) -> None:
        with pytest.raises(ValidationError):
            LocalModeConfig(evaluators=[])

    def test_thresholds_validated(self) -> None:
        with pytest.raises(ValidationError, match="between 0.0 and 1.0"):
            LocalModeConfig(
                evaluators=["answer_presence"],
                thresholds={"answer_presence": 1.5},
            )


# ---------------------------------------------------------------------------
# CloudModeConfig validation
# ---------------------------------------------------------------------------


class TestCloudModeConfig:
    """Validate CloudModeConfig block."""

    def test_valid_cloud_evaluators(self) -> None:
        cfg = CloudModeConfig(evaluators=["groundedness", "relevance"])
        assert len(cfg.evaluators) == 2

    def test_scaffold_evaluator_rejected(self) -> None:
        with pytest.raises(ValidationError, match="not in the cloud catalog"):
            CloudModeConfig(evaluators=["answer_presence"])

    def test_judge_evaluator_rejected(self) -> None:
        with pytest.raises(ValidationError, match="not in the cloud catalog"):
            CloudModeConfig(evaluators=["accuracy_judge"])

    def test_all_cloud_evaluators_accepted(self) -> None:
        cfg = CloudModeConfig(evaluators=list(CLOUD_EVALUATOR_NAMES))
        assert len(cfg.evaluators) == 4

    def test_empty_evaluators_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CloudModeConfig(evaluators=[])

    def test_thresholds_validated(self) -> None:
        with pytest.raises(ValidationError, match="between 0.0 and 1.0"):
            CloudModeConfig(
                evaluators=["groundedness"],
                thresholds={"groundedness": -0.1},
            )


# ---------------------------------------------------------------------------
# ProjectConfig — new format
# ---------------------------------------------------------------------------


class TestProjectConfigNewFormat:
    """Test ProjectConfig with new mode-specific blocks."""

    def test_local_mode_new_format(self) -> None:
        cfg = ProjectConfig(
            project_name="test",
            dataset_path="data.jsonl",
            mode="local",
            local_mode=LocalModeConfig(
                evaluators=["answer_presence", "context_presence"],
                thresholds={"answer_presence": 0.8},
            ),
        )
        assert cfg.mode == "local"
        assert cfg.evaluators == ["answer_presence", "context_presence"]
        assert cfg.thresholds == {"answer_presence": 0.8}
        assert cfg.is_local_mode()
        assert not cfg.is_cloud_mode()

    def test_cloud_mode_new_format(self) -> None:
        cfg = ProjectConfig(
            project_name="test",
            dataset_path="data.jsonl",
            mode="cloud",
            cloud_mode=CloudModeConfig(
                evaluators=["groundedness", "relevance"],
                thresholds={"groundedness": 0.7},
            ),
        )
        assert cfg.mode == "cloud"
        assert cfg.evaluators == ["groundedness", "relevance"]
        assert cfg.is_cloud_mode()
        assert not cfg.is_local_mode()

    def test_mode_display_local(self) -> None:
        cfg = ProjectConfig(
            project_name="test",
            dataset_path="data.jsonl",
            mode="local",
            local_mode=LocalModeConfig(evaluators=["answer_presence"]),
        )
        assert cfg.mode_display == "LOCAL"

    def test_mode_display_cloud(self) -> None:
        cfg = ProjectConfig(
            project_name="test",
            dataset_path="data.jsonl",
            mode="cloud",
            cloud_mode=CloudModeConfig(evaluators=["groundedness"]),
        )
        assert cfg.mode_display == "CLOUD (Azure AI Foundry)"


# ---------------------------------------------------------------------------
# Cross-mode validation
# ---------------------------------------------------------------------------


class TestCrossModePrevention:
    """Ensure evaluators cannot cross mode boundaries."""

    def test_scaffold_in_cloud_mode_rejected(self) -> None:
        with pytest.raises(ValidationError, match="not in the cloud catalog"):
            ProjectConfig(
                project_name="test",
                dataset_path="data.jsonl",
                mode="cloud",
                cloud_mode=CloudModeConfig(evaluators=["answer_presence"]),
            )

    def test_judge_in_cloud_mode_rejected(self) -> None:
        with pytest.raises(ValidationError, match="not in the cloud catalog"):
            ProjectConfig(
                project_name="test",
                dataset_path="data.jsonl",
                mode="cloud",
                cloud_mode=CloudModeConfig(evaluators=["accuracy_judge"]),
            )

    def test_empty_evaluators_fails(self) -> None:
        """No evaluators anywhere → should fail."""
        with pytest.raises(ValidationError):
            ProjectConfig(
                project_name="test",
                dataset_path="data.jsonl",
                mode="local",
                local_mode=LocalModeConfig(evaluators=[]),
            )


# ---------------------------------------------------------------------------
# Registry mode-aware methods
# ---------------------------------------------------------------------------


class TestRegistryModeAware:
    """Test the registry's mode-aware helpers."""

    def test_list_for_local_mode(self) -> None:
        names = default_registry.list_for_mode("local")
        assert len(names) == 21
        assert "answer_presence" in names
        assert "accuracy_judge" in names
        assert "groundedness" in names

    def test_list_for_cloud_mode(self) -> None:
        names = default_registry.list_for_mode("cloud")
        assert len(names) == 4
        assert "groundedness" in names
        assert "answer_presence" not in names
        assert "accuracy_judge" not in names

    def test_get_for_mode_valid(self) -> None:
        ev = default_registry.get_for_mode("groundedness", "local")
        assert ev.name == "groundedness"

    def test_get_for_mode_invalid_rejects(self) -> None:
        with pytest.raises(KeyError, match="not available in cloud mode"):
            default_registry.get_for_mode("answer_presence", "cloud")


# ---------------------------------------------------------------------------
# Report mode banners
# ---------------------------------------------------------------------------


class TestReportModeBanners:
    """Verify reports contain mode information."""

    def _make_result(self, runner_type: str = "local") -> Any:
        from rag_eval_framework.evaluators.base import EvaluatorResult
        from rag_eval_framework.runners.base import (
            EvaluationRunResult,
            RecordResult,
        )

        return EvaluationRunResult(
            project_name="mode-test",
            timestamp="2026-06-01T12:00:00Z",
            dataset_path="datasets/test/eval.jsonl",
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
            threshold_breaches=[],
            threshold_results=[],
            passed=True,
            runner_type=runner_type,
        )

    def test_markdown_local_mode_banner(self, tmp_path: Path) -> None:
        from rag_eval_framework.reporting.markdown_report import render_markdown_report

        result = self._make_result("local")
        path = render_markdown_report(result, tmp_path)
        text = path.read_text(encoding="utf-8")
        assert "Evaluation Mode: LOCAL" in text
        assert "in-process" in text

    def test_markdown_cloud_mode_banner(self, tmp_path: Path) -> None:
        from rag_eval_framework.reporting.markdown_report import render_markdown_report

        result = self._make_result("cloud")
        path = render_markdown_report(result, tmp_path)
        text = path.read_text(encoding="utf-8")
        assert "Evaluation Mode: CLOUD" in text
        assert "Azure AI Foundry" in text

    def test_html_local_mode_banner(self, tmp_path: Path) -> None:
        from rag_eval_framework.reporting.html_report import render_html_report

        result = self._make_result("local")
        path = render_html_report(result, tmp_path)
        text = path.read_text(encoding="utf-8")
        assert "Evaluation Mode: LOCAL" in text

    def test_html_cloud_mode_banner(self, tmp_path: Path) -> None:
        from rag_eval_framework.reporting.html_report import render_html_report

        result = self._make_result("cloud")
        path = render_html_report(result, tmp_path)
        text = path.read_text(encoding="utf-8")
        assert "CLOUD" in text
        assert "Azure AI Foundry" in text

    def test_json_includes_evaluation_mode(self, tmp_path: Path) -> None:
        from rag_eval_framework.reporting.json_report import render_json_report

        result = self._make_result("local")
        path = render_json_report(result, tmp_path)
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["evaluation_mode"] == "local"
        assert "evaluator_descriptions" in data

    def test_json_cloud_mode(self, tmp_path: Path) -> None:
        from rag_eval_framework.reporting.json_report import render_json_report

        result = self._make_result("cloud")
        result.evaluators_used = ["groundedness"]
        result.aggregate_scores = {"groundedness": 0.9}
        path = render_json_report(result, tmp_path)
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["evaluation_mode"] == "cloud"
        assert "groundedness" in data["evaluator_descriptions"]


# ---------------------------------------------------------------------------
# Metric descriptions in reports
# ---------------------------------------------------------------------------


class TestMetricDescriptions:
    """Verify evaluator descriptions appear in reports."""

    def _make_result(self) -> Any:
        from rag_eval_framework.evaluators.base import EvaluatorResult
        from rag_eval_framework.runners.base import (
            EvaluationRunResult,
            RecordResult,
        )

        return EvaluationRunResult(
            project_name="desc-test",
            timestamp="2026-06-01T12:00:00Z",
            dataset_path="datasets/test/eval.jsonl",
            total_records=1,
            evaluators_used=["answer_presence", "groundedness"],
            record_results=[
                RecordResult(
                    record_id="r-001",
                    evaluator_results={
                        "answer_presence": EvaluatorResult(score=1.0, reason="OK"),
                        "groundedness": EvaluatorResult(score=0.9, reason="OK"),
                    },
                ),
            ],
            aggregate_scores={"answer_presence": 1.0, "groundedness": 0.9},
            threshold_breaches=[],
            threshold_results=[],
            passed=True,
            runner_type="local",
        )

    def test_markdown_metric_descriptions(self, tmp_path: Path) -> None:
        from rag_eval_framework.reporting.markdown_report import render_markdown_report

        result = self._make_result()
        path = render_markdown_report(result, tmp_path)
        text = path.read_text(encoding="utf-8")
        assert "Metric Descriptions" in text
        assert "answer_presence" in text
        assert "groundedness" in text

    def test_html_metric_descriptions(self, tmp_path: Path) -> None:
        from rag_eval_framework.reporting.html_report import render_html_report

        result = self._make_result()
        path = render_html_report(result, tmp_path)
        text = path.read_text(encoding="utf-8")
        assert "Metric Descriptions" in text
