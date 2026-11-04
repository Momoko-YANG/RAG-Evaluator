"""Tests for the retrieval ranking evaluators (Hit@k / MRR)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from rag_eval_framework.core.schemas import EvaluationRecord
from rag_eval_framework.evaluators.base import EvaluatorStatus
from rag_eval_framework.evaluators.retrieval.hit import RetrievalHitRateEvaluator
from rag_eval_framework.evaluators.retrieval.mrr import RetrievalMRREvaluator
from rag_eval_framework.evaluators.retrieval.relevance.matcher import first_relevant_rank
from tests.conftest import make_record


def _docs(*items: tuple[str, str]) -> list[dict[str, Any]]:
    return [{"title": title, "content": content} for title, content in items]


def _record(**overrides: Any) -> EvaluationRecord:
    base = {
        "retrieved_documents": _docs(
            ("旅費規程.pdf", "出張旅費は30日以内に精算する。"),
            ("就業規則.pdf", "第3章 休日について。"),
            ("就業規則.pdf", "年次有給休暇は年２０日付与する。"),
        ),
    }
    return EvaluationRecord(**make_record(**{**base, **overrides}))


def _setup(evaluator: Any, **options: Any) -> Any:
    evaluator.setup(SimpleNamespace(evaluator_options={evaluator.name: options}))
    return evaluator


class TestFirstRelevantRank:
    def test_document_and_evidence_must_both_match(self) -> None:
        record = _record(
            ground_truth_documents=["就業規則.pdf"], ground_truth_evidence=["年20日"]
        )
        # Rank 2 is the right file but not the chunk with the evidence.
        assert first_relevant_rank(record) == 3

    def test_document_only(self) -> None:
        assert first_relevant_rank(_record(ground_truth_documents=["就業規則.pdf"])) == 2

    def test_evidence_only_normalises_width_and_whitespace(self) -> None:
        assert first_relevant_rank(_record(ground_truth_evidence=["年 20 日"])) == 3

    def test_any_of_several_labels(self) -> None:
        record = _record(ground_truth_documents=["存在しない.pdf", "旅費規程.pdf"])
        assert first_relevant_rank(record) == 1

    def test_k_limits_the_search(self) -> None:
        record = _record(ground_truth_evidence=["年20日"])
        assert first_relevant_rank(record, k=2) is None
        assert first_relevant_rank(record, k=3) == 3

    def test_falls_back_to_contexts_for_evidence(self) -> None:
        record = _record(
            retrieved_documents=[],
            contexts=["関係ない", "有給は年20日"],
            ground_truth_evidence=["年20日"],
        )
        assert first_relevant_rank(record) == 2


class TestRetrievalHitRate:
    def test_hit_and_miss(self) -> None:
        evaluator = _setup(RetrievalHitRateEvaluator())
        hit = evaluator.evaluate(_record(ground_truth_evidence=["年20日"]))
        assert hit.score == 1.0
        assert hit.metadata == {
            "rank": 3, "k": None, "retrieved": 3,
            "relevance_policy": "evidence_match", "judgment_kind": "evidence_match",
        }

        miss = evaluator.evaluate(_record(ground_truth_evidence=["記載のない内容"]))
        assert miss.score == 0.0
        assert miss.status == EvaluatorStatus.SUCCESS

    def test_k_option(self) -> None:
        evaluator = _setup(RetrievalHitRateEvaluator(), k=1)
        result = evaluator.evaluate(_record(ground_truth_evidence=["年20日"]))
        assert result.score == 0.0
        assert "top 1" in result.reason

    def test_nothing_retrieved_is_a_miss(self) -> None:
        evaluator = _setup(RetrievalHitRateEvaluator())
        record = _record(retrieved_documents=[], contexts=[], ground_truth_documents=["a.pdf"])
        result = evaluator.evaluate(record)
        assert result.score == 0.0
        assert result.status == EvaluatorStatus.SUCCESS

    def test_skipped_without_labels(self) -> None:
        result = _setup(RetrievalHitRateEvaluator()).evaluate(_record())
        assert result.status == EvaluatorStatus.SKIPPED

    def test_skipped_when_titles_are_unavailable(self) -> None:
        record = _record(
            retrieved_documents=[], contexts=["ctx"], ground_truth_documents=["a.pdf"]
        )
        result = _setup(RetrievalHitRateEvaluator()).evaluate(record)
        assert result.status == EvaluatorStatus.SKIPPED

    @pytest.mark.parametrize("k", [0, -1, 2.5, "3", True])
    def test_invalid_k_rejected(self, k: Any) -> None:
        with pytest.raises(ValueError, match="positive integer"):
            _setup(RetrievalHitRateEvaluator(), k=k)


class TestRetrievalMRR:
    def test_reciprocal_rank(self) -> None:
        evaluator = _setup(RetrievalMRREvaluator())
        assert evaluator.evaluate(_record(ground_truth_documents=["旅費規程.pdf"])).score == 1.0
        assert evaluator.evaluate(_record(ground_truth_documents=["就業規則.pdf"])).score == 0.5
        assert evaluator.evaluate(_record(ground_truth_evidence=["年20日"])).score == 0.3333
        assert evaluator.evaluate(_record(ground_truth_evidence=["なし"])).score == 0.0
