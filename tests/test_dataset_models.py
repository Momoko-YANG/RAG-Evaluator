"""Tests for Phase 2 dataset model enhancements."""

from __future__ import annotations

import pytest

from rag_eval_framework.core.schemas import EvaluationRecord
from tests.conftest import make_record


class TestDatasetModelBasics:
    def test_required_fields_only(self) -> None:
        rec = EvaluationRecord(id="d-001", question="Q?", response="A.")
        assert rec.id == "d-001"
        assert rec.contexts == []
        assert rec.ground_truth_answer == ""

    def test_full_record(self) -> None:
        rec = EvaluationRecord(**make_record())
        assert rec.id == "test-001"
        assert len(rec.contexts) == 1


class TestPhase2Fields:
    def test_retrieved_documents_default(self) -> None:
        rec = EvaluationRecord(**make_record())
        assert rec.retrieved_documents == []

    def test_retrieved_documents_populated(self) -> None:
        rec = EvaluationRecord(
            **make_record(
                retrieved_documents=[
                    {"content": "Doc 1 text", "score": 0.95},
                    {"content": "Doc 2 text", "score": 0.80},
                ]
            )
        )
        assert len(rec.retrieved_documents) == 2
        assert rec.retrieved_documents[0]["content"] == "Doc 1 text"

    def test_ground_truth_documents_default(self) -> None:
        rec = EvaluationRecord(**make_record())
        assert rec.ground_truth_documents == []

    def test_ground_truth_documents_populated(self) -> None:
        rec = EvaluationRecord(**make_record(ground_truth_documents=["doc-id-1", "doc-id-2"]))
        assert len(rec.ground_truth_documents) == 2


class TestHelperMethods:
    def test_has_contexts_true(self) -> None:
        rec = EvaluationRecord(**make_record())
        assert rec.has_contexts() is True

    def test_has_contexts_false_empty_list(self) -> None:
        rec = EvaluationRecord(**make_record(contexts=[]))
        assert rec.has_contexts() is False

    def test_has_contexts_false_whitespace(self) -> None:
        rec = EvaluationRecord(**make_record(contexts=["  ", ""]))
        assert rec.has_contexts() is False

    def test_has_ground_truth_true(self) -> None:
        rec = EvaluationRecord(**make_record())
        assert rec.has_ground_truth() is True

    def test_has_ground_truth_false(self) -> None:
        rec = EvaluationRecord(**make_record(ground_truth_answer=""))
        assert rec.has_ground_truth() is False

    def test_has_ground_truth_false_whitespace(self) -> None:
        rec = EvaluationRecord(**make_record(ground_truth_answer="   "))
        assert rec.has_ground_truth() is False


class TestValidators:
    def test_empty_question_rejected(self) -> None:
        with pytest.raises(ValueError, match="Question must not be empty"):
            EvaluationRecord(id="v-001", question="  ", response="A.")

    def test_empty_response_rejected(self) -> None:
        with pytest.raises(ValueError, match="Response must not be empty"):
            EvaluationRecord(id="v-001", question="Q?", response="  ")
