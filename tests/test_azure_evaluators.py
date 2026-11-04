"""Tests for the Azure evaluator adapter and concrete Azure evaluators.

All tests run without live Azure access — they exercise the adapter
plumbing and normalisation logic using mocks.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from rag_eval_framework.config.models import ProjectConfig
from rag_eval_framework.core.schemas import EvaluationRecord
from rag_eval_framework.evaluators.base import EvaluatorStatus
from rag_eval_framework.evaluators.common.azure_sdk import AzureEvaluatorBase, build_model_config
from rag_eval_framework.evaluators.generation.completeness import ResponseCompletenessEvaluator
from rag_eval_framework.evaluators.generation.faithfulness import GroundednessEvaluator
from rag_eval_framework.evaluators.generation.relevance import RelevanceEvaluator
from rag_eval_framework.evaluators.retrieval.context_relevance import RetrievalEvaluator
from tests.conftest import make_config, make_record


class TestBuildModelConfig:
    def test_returns_empty_when_no_azure(self) -> None:
        config = ProjectConfig(**make_config())
        assert build_model_config(config) == {}

    def test_builds_config_with_azure_section(self) -> None:
        config = ProjectConfig(
            **make_config(
                azure={
                    "endpoint": "https://test.openai.azure.com",
                    "deployment_name": "gpt-4",
                }
            )
        )
        mc = build_model_config(config)
        assert mc["azure_endpoint"] == "https://test.openai.azure.com"
        assert mc["azure_deployment"] == "gpt-4"

    def test_falls_back_to_judge_model_for_deployment(self) -> None:
        config = ProjectConfig(
            **make_config(
                judge_model="gpt-4o",
                azure={"endpoint": "https://test.openai.azure.com"},
            )
        )
        mc = build_model_config(config)
        assert mc["azure_deployment"] == "gpt-4o"

    @patch.dict("os.environ", {"MY_KEY": "secret123"})
    def test_includes_api_key_when_credential_key(self) -> None:
        config = ProjectConfig(
            **make_config(
                azure={
                    "endpoint": "https://test.openai.azure.com",
                    "credential_type": "key",
                    "api_key_env": "MY_KEY",
                }
            )
        )
        mc = build_model_config(config)
        assert mc["api_key"] == "secret123"


class TestAzureEvaluatorNotConfigured:
    """When setup() is not called, evaluators return ERROR status."""

    @pytest.mark.parametrize(
        "evaluator_cls",
        [
            GroundednessEvaluator,
            RelevanceEvaluator,
            RetrievalEvaluator,
            ResponseCompletenessEvaluator,
        ],
    )
    def test_unconfigured_returns_error(self, evaluator_cls: type[AzureEvaluatorBase]) -> None:
        ev = evaluator_cls()
        record = EvaluationRecord(**make_record())
        result = ev.evaluate(record)
        assert result.status == EvaluatorStatus.ERROR
        assert "not been configured" in result.reason


class TestAzureEvaluatorNames:
    def test_groundedness_name(self) -> None:
        assert GroundednessEvaluator().name == "groundedness"

    def test_relevance_name(self) -> None:
        assert RelevanceEvaluator().name == "relevance"

    def test_retrieval_name(self) -> None:
        assert RetrievalEvaluator().name == "retrieval"

    def test_completeness_name(self) -> None:
        assert ResponseCompletenessEvaluator().name == "response_completeness"


class TestAzureEvaluatorRequiredFields:
    def test_groundedness_requires_contexts(self) -> None:
        assert "contexts" in GroundednessEvaluator().required_fields

    def test_relevance_requires_question_response(self) -> None:
        rf = RelevanceEvaluator().required_fields
        assert "question" in rf
        assert "response" in rf

    def test_retrieval_requires_contexts(self) -> None:
        assert "contexts" in RetrievalEvaluator().required_fields

    def test_completeness_requires_ground_truth(self) -> None:
        assert "ground_truth_answer" in ResponseCompletenessEvaluator().required_fields


class TestGroundednessNormalisation:
    """Test SDK output normalisation via a mock SDK evaluator."""

    def test_normalises_1_to_5_score(self) -> None:
        ev = GroundednessEvaluator()
        ev._configured = True
        ev._sdk_evaluator = MagicMock(
            return_value={"groundedness": 5, "groundedness_reason": "Grounded."}
        )
        record = EvaluationRecord(**make_record())
        result = ev.evaluate(record)
        assert result.score == 1.0
        assert result.reason == "Grounded."

    def test_normalises_low_score(self) -> None:
        ev = GroundednessEvaluator()
        ev._configured = True
        ev._sdk_evaluator = MagicMock(
            return_value={"groundedness": 1, "groundedness_reason": "Not grounded."}
        )
        record = EvaluationRecord(**make_record())
        result = ev.evaluate(record)
        assert result.score == 0.0

    def test_handles_sdk_error(self) -> None:
        ev = GroundednessEvaluator()
        ev._configured = True
        ev._sdk_evaluator = MagicMock(side_effect=RuntimeError("SDK boom"))
        record = EvaluationRecord(**make_record())
        result = ev.evaluate(record)
        assert result.status == EvaluatorStatus.ERROR
        assert "SDK error" in result.reason
