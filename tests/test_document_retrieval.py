"""Tests for the document_retrieval evaluator (Azure SDK DocumentRetrievalEvaluator)."""

from __future__ import annotations

from typing import Any

import pytest

from rag_eval_framework.core.schemas import EvaluationRecord
from rag_eval_framework.evaluators.base import EvaluatorStatus
from rag_eval_framework.evaluators.retrieval.ndcg import (
    NOT_RELEVANT,
    RELEVANT,
    UNRETRIEVED_RELEVANT_ID,
    AzureDocumentRetrievalEvaluator,
    build_sdk_inputs,
)
from tests.conftest import make_record


def _record(contents: list[str], **overrides: Any) -> EvaluationRecord:
    docs = [{"title": "", "content": c} for c in contents]
    return EvaluationRecord(**make_record(retrieved_documents=docs, **overrides))


class TestBuildSdkInputs:
    def test_labels_relevant_chunks_and_keeps_rank_order(self) -> None:
        record = _record(
            ["無関係", "有給休暇は年２０日", "別の規程"], ground_truth_evidence=["年20日"]
        )
        ground_truth, retrieved = build_sdk_inputs(record)

        assert [g["query_relevance_label"] for g in ground_truth] == [
            NOT_RELEVANT,
            RELEVANT,
            NOT_RELEVANT,
        ]
        scores = [d["relevance_score"] for d in retrieved]
        assert scores == sorted(scores, reverse=True)
        assert {d["document_id"] for d in retrieved} == {g["document_id"] for g in ground_truth}

    def test_adds_unretrieved_relevant_document_on_a_miss(self) -> None:
        record = _record(["無関係", "別の規程"], ground_truth_evidence=["年20日"])
        ground_truth, retrieved = build_sdk_inputs(record)
        assert ground_truth[-1] == {
            "document_id": UNRETRIEVED_RELEVANT_ID,
            "query_relevance_label": RELEVANT,
        }
        assert len(retrieved) == 2

    def test_identical_chunks_count_once(self) -> None:
        record = _record(["年20日", "年 20 日", "無関係"], ground_truth_evidence=["年20日"])
        _, retrieved = build_sdk_inputs(record)
        assert len(retrieved) == 2


class TestEvaluator:
    def test_skipped_without_labels(self) -> None:
        result = AzureDocumentRetrievalEvaluator().evaluate(_record(["x"]))
        assert result.status == EvaluatorStatus.SKIPPED

    def test_error_when_sdk_not_configured(self) -> None:
        result = AzureDocumentRetrievalEvaluator().evaluate(
            _record(["年20日"], ground_truth_evidence=["年20日"])
        )
        assert result.status == EvaluatorStatus.ERROR

    def test_ndcg_with_the_sdk(self) -> None:
        pytest.importorskip("azure.ai.evaluation")
        evaluator = AzureDocumentRetrievalEvaluator()
        evaluator.setup(None)

        first = evaluator.evaluate(
            _record(["年20日", "無関係", "別"], ground_truth_evidence=["年20日"])
        )
        third = evaluator.evaluate(
            _record(["無関係", "別", "年20日"], ground_truth_evidence=["年20日"])
        )
        miss = evaluator.evaluate(_record(["無関係", "別"], ground_truth_evidence=["年20日"]))

        assert first.status == EvaluatorStatus.SUCCESS
        assert first.score == 1.0
        assert 0.0 < third.score < first.score
        assert miss.score == 0.0
        assert "ndcg@3" in first.metadata
