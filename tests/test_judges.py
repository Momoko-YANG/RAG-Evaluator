"""Tests for LLM-as-a-judge evaluators.

All tests run without live Azure access — they validate prompt structure,
parsing, normalisation, and error handling using mocks.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from rag_eval_framework.core.schemas import EvaluationRecord
from rag_eval_framework.evaluators.base import EvaluatorStatus
from rag_eval_framework.evaluators.common.llm_judge import BaseLLMJudge
from rag_eval_framework.evaluators.generation.citation import CitationJudge
from rag_eval_framework.evaluators.generation.correctness import AccuracyJudge
from rag_eval_framework.evaluators.generation.faithfulness import HallucinationJudge
from rag_eval_framework.evaluators.generation.policy_compliance import PolicyComplianceJudge
from rag_eval_framework.utils.normalisation import normalise_1_5
from tests.conftest import make_record


class TestJudgeNames:
    def test_accuracy(self) -> None:
        assert AccuracyJudge().name == "accuracy_judge"

    def test_hallucination(self) -> None:
        assert HallucinationJudge().name == "hallucination_judge"

    def test_citation(self) -> None:
        assert CitationJudge().name == "citation_judge"

    def test_policy_compliance(self) -> None:
        assert PolicyComplianceJudge().name == "policy_compliance_judge"


class TestJudgeRequiredFields:
    def test_accuracy_requires_ground_truth(self) -> None:
        assert "ground_truth_answer" in AccuracyJudge().required_fields

    def test_hallucination_requires_contexts(self) -> None:
        assert "contexts" in HallucinationJudge().required_fields

    def test_citation_requires_contexts(self) -> None:
        assert "contexts" in CitationJudge().required_fields

    def test_policy_compliance_minimal(self) -> None:
        rf = PolicyComplianceJudge().required_fields
        assert "question" in rf
        assert "response" in rf


class TestJudgeNotConfigured:
    @pytest.mark.parametrize(
        "judge_cls",
        [AccuracyJudge, HallucinationJudge, CitationJudge, PolicyComplianceJudge],
    )
    def test_unconfigured_returns_error(self, judge_cls: type[BaseLLMJudge]) -> None:
        judge = judge_cls()
        record = EvaluationRecord(**make_record())
        result = judge.evaluate(record)
        assert result.status == EvaluatorStatus.ERROR
        assert "not configured" in result.reason


class TestNormalise1To5:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [(1, 0.0), (3, 0.5), (5, 1.0), (0, 0.0), (6, 1.0)],
    )
    def test_normalisation(self, raw: float, expected: float) -> None:
        assert normalise_1_5(raw) == expected


class TestJudgePromptFormatting:
    def test_accuracy_prompt_includes_fields(self) -> None:
        judge = AccuracyJudge()
        record = EvaluationRecord(**make_record())
        prompt = judge._format_user_prompt(record)
        assert "What is the policy?" in prompt
        assert "90 days" in prompt
        assert "retention period is 90 days" in prompt

    def test_hallucination_prompt_includes_contexts(self) -> None:
        judge = HallucinationJudge()
        record = EvaluationRecord(**make_record(contexts=["ctx1", "ctx2"]))
        prompt = judge._format_user_prompt(record)
        assert "ctx1" in prompt
        assert "ctx2" in prompt


class TestJudgeResponseParsing:
    def test_parse_valid_json(self) -> None:
        raw = json.dumps({"score": 4, "reason": "Good"})
        result = BaseLLMJudge._parse_response(raw)
        assert result["score"] == 4
        assert result["reason"] == "Good"

    def test_parse_invalid_json(self) -> None:
        result = BaseLLMJudge._parse_response("not json at all")
        assert result["score"] == 0
        assert "Failed to parse" in result["reason"]


class TestJudgeWithMockedLLM:
    """Test end-to-end judge evaluation with a mocked LLM client."""

    def test_accuracy_judge_with_mock(self) -> None:
        judge = AccuracyJudge()
        judge._configured = True
        judge._deployment = "gpt-4"

        # Mock the _call_llm method to return a structured response
        judge._call_llm = MagicMock(  # type: ignore[method-assign]
            return_value=json.dumps({"score": 5, "reason": "Perfectly accurate"})
        )

        record = EvaluationRecord(**make_record())
        result = judge.evaluate(record)

        assert result.score == 1.0  # 5 -> 1.0 normalised
        assert result.reason == "Perfectly accurate"
        assert result.metadata["raw_score"] == 5
        assert result.metadata["judge_model"] == "gpt-4"

    def test_judge_handles_llm_error(self) -> None:
        judge = AccuracyJudge()
        judge._configured = True
        judge._deployment = "gpt-4"
        judge._call_llm = MagicMock(  # type: ignore[method-assign]
            side_effect=RuntimeError("Connection refused")
        )

        record = EvaluationRecord(**make_record())
        result = judge.evaluate(record)
        assert result.status == EvaluatorStatus.ERROR
        assert "LLM call failed" in result.reason
