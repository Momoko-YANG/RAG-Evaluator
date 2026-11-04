"""Tests for the evaluator system — base class, registry, and built-ins."""

from __future__ import annotations

import pytest

from rag_eval_framework.core.registry import EvaluatorRegistry, default_registry
from rag_eval_framework.core.schemas import EvaluationRecord
from rag_eval_framework.evaluators.base import BaseEvaluator, EvaluatorResult
from rag_eval_framework.evaluators.generation.answer_presence import AnswerPresenceEvaluator
from rag_eval_framework.evaluators.generation.correctness import ExactMatchAccuracyEvaluator
from rag_eval_framework.evaluators.retrieval.context_presence import ContextPresenceEvaluator
from tests.conftest import make_record

# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


class TestEvaluatorRegistry:
    def test_register_and_get(self) -> None:
        registry = EvaluatorRegistry()
        registry.register(AnswerPresenceEvaluator)
        evaluator = registry.get("answer_presence")
        assert isinstance(evaluator, AnswerPresenceEvaluator)

    def test_get_unknown_raises(self) -> None:
        registry = EvaluatorRegistry()
        with pytest.raises(KeyError, match="no_such_evaluator"):
            registry.get("no_such_evaluator")

    def test_list_evaluators(self) -> None:
        registry = EvaluatorRegistry()
        registry.register(AnswerPresenceEvaluator)
        registry.register(ContextPresenceEvaluator)
        assert registry.list_evaluators() == ["answer_presence", "context_presence"]

    def test_contains(self) -> None:
        registry = EvaluatorRegistry()
        registry.register(AnswerPresenceEvaluator)
        assert "answer_presence" in registry
        assert "nope" not in registry

    def test_len(self) -> None:
        registry = EvaluatorRegistry()
        assert len(registry) == 0
        registry.register(AnswerPresenceEvaluator)
        assert len(registry) == 1

    def test_custom_evaluator_registration(self) -> None:
        class MyEval(BaseEvaluator):
            @property
            def name(self) -> str:
                return "custom_eval"

            def evaluate(self, record: EvaluationRecord) -> EvaluatorResult:
                return EvaluatorResult(score=0.42, reason="custom")

        registry = EvaluatorRegistry()
        registry.register(MyEval)
        result = registry.get("custom_eval").evaluate(EvaluationRecord(**make_record()))
        assert result.score == 0.42


# ---------------------------------------------------------------------------
# Built-in evaluator tests
# ---------------------------------------------------------------------------


class TestAnswerPresenceEvaluator:
    def setup_method(self) -> None:
        self.evaluator = AnswerPresenceEvaluator()

    def test_name(self) -> None:
        assert self.evaluator.name == "answer_presence"

    def test_substantive_response_scores_1(self) -> None:
        record = EvaluationRecord(**make_record(response="The answer is 42."))
        result = self.evaluator.evaluate(record)
        assert result.score == 1.0

    def test_single_word_scores_0(self) -> None:
        record = EvaluationRecord(**make_record(response="Yes"))
        result = self.evaluator.evaluate(record)
        assert result.score == 0.0


class TestContextPresenceEvaluator:
    def setup_method(self) -> None:
        self.evaluator = ContextPresenceEvaluator()

    def test_name(self) -> None:
        assert self.evaluator.name == "context_presence"

    def test_with_contexts_scores_1(self) -> None:
        record = EvaluationRecord(**make_record(contexts=["Some context."]))
        result = self.evaluator.evaluate(record)
        assert result.score == 1.0

    def test_empty_contexts_scores_0(self) -> None:
        record = EvaluationRecord(**make_record(contexts=[]))
        result = self.evaluator.evaluate(record)
        assert result.score == 0.0

    def test_whitespace_only_contexts_scores_0(self) -> None:
        record = EvaluationRecord(**make_record(contexts=["   ", "\t"]))
        result = self.evaluator.evaluate(record)
        assert result.score == 0.0


class TestExactMatchAccuracyEvaluator:
    def setup_method(self) -> None:
        self.evaluator = ExactMatchAccuracyEvaluator()

    def test_name(self) -> None:
        assert self.evaluator.name == "exact_match_accuracy"

    def test_exact_match_scores_1(self) -> None:
        record = EvaluationRecord(
            **make_record(
                response="The retention period is 90 days.",
                ground_truth_answer="90 days",
            )
        )
        result = self.evaluator.evaluate(record)
        assert result.score == 1.0

    def test_no_match_scores_0(self) -> None:
        record = EvaluationRecord(
            **make_record(
                response="I don't know.",
                ground_truth_answer="90 days",
            )
        )
        result = self.evaluator.evaluate(record)
        assert result.score == 0.0

    def test_empty_ground_truth_scores_05(self) -> None:
        record = EvaluationRecord(**make_record(ground_truth_answer=""))
        result = self.evaluator.evaluate(record)
        assert result.score == 0.5

    def test_case_insensitive(self) -> None:
        record = EvaluationRecord(
            **make_record(
                response="The answer is YES definitely.",
                ground_truth_answer="yes",
            )
        )
        result = self.evaluator.evaluate(record)
        assert result.score == 1.0


class TestDefaultRegistry:
    """Verify the module-level default_registry is correctly populated."""

    def test_contains_all_builtins(self) -> None:
        expected = {
            # Phase 1
            "answer_presence",
            "context_presence",
            "exact_match_accuracy",
            # Retrieval ranking
            "retrieval_hit_rate",
            "retrieval_mrr",
            "retrieval_recall",
            "retrieval_ndcg",
            # Phase 2 – Azure SDK
            "groundedness",
            "relevance",
            "retrieval",
            "response_completeness",
            "document_retrieval",
            # Phase 2 – LLM judges
            "accuracy_judge",
            "hallucination_judge",
            "citation_judge",
            "policy_compliance_judge",
            "task_success",
            "tool_selection",
            "query_construction",
            "abstention",
            "access_control",
        }
        assert set(default_registry.list_evaluators()) == expected

    def test_len(self) -> None:
        assert len(default_registry) == 21

    def test_get_returns_instances(self) -> None:
        for name in default_registry.list_evaluators():
            evaluator = default_registry.get(name)
            assert isinstance(evaluator, BaseEvaluator)
