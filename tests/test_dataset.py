"""Tests for dataset loading and validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from rag_eval_framework.core.schemas import EvaluationRecord
from rag_eval_framework.datasets.loader import DatasetLoadError, load_dataset
from rag_eval_framework.datasets.validation import validate_dataset
from tests.conftest import make_record


class TestEvaluationRecordModel:
    """Validation of the EvaluationRecord pydantic model."""

    def test_valid_record(self) -> None:
        r = EvaluationRecord(**make_record())
        assert r.id == "test-001"

    def test_empty_response_fails(self) -> None:
        with pytest.raises(ValidationError, match="empty"):
            EvaluationRecord(**make_record(response="   "))

    def test_empty_question_fails(self) -> None:
        with pytest.raises(ValidationError, match="empty"):
            EvaluationRecord(**make_record(question=""))

    def test_missing_id_fails(self) -> None:
        data = make_record()
        del data["id"]
        with pytest.raises(ValidationError):
            EvaluationRecord(**data)

    def test_optional_fields_default(self) -> None:
        data = make_record()
        del data["contexts"]
        del data["ground_truth_answer"]
        del data["metadata"]
        r = EvaluationRecord(**data)
        assert r.contexts == []
        assert r.ground_truth_answer == ""
        assert r.metadata == {}


class TestValidateDataset:
    """Tests for the validate_dataset helper."""

    def test_valid_records(self) -> None:
        raw = [make_record(id="1"), make_record(id="2")]
        records, errors = validate_dataset(raw)
        assert len(records) == 2
        assert errors == []

    def test_invalid_records_reported(self) -> None:
        raw = [make_record(id="ok"), {"id": "bad", "question": "", "response": "x"}]
        records, errors = validate_dataset(raw)
        assert len(records) == 1
        assert len(errors) >= 1
        assert "bad" in errors[0]


class TestLoadDataset:
    """Tests for the JSONL dataset file loader."""

    def test_load_valid_file(self, tmp_dataset_file: Path) -> None:
        records = load_dataset(tmp_dataset_file)
        assert len(records) == 3

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetLoadError, match="not found"):
            load_dataset(tmp_path / "missing.jsonl")

    def test_wrong_extension_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "data.csv"
        bad.write_text("a,b", encoding="utf-8")
        with pytest.raises(DatasetLoadError, match=".jsonl"):
            load_dataset(bad)

    def test_empty_file_raises(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty.jsonl"
        empty.write_text("", encoding="utf-8")
        with pytest.raises(DatasetLoadError, match="empty"):
            load_dataset(empty)

    def test_malformed_json_strict_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.jsonl"
        bad.write_text("{not json\n", encoding="utf-8")
        with pytest.raises(DatasetLoadError, match="invalid JSON"):
            load_dataset(bad, strict=True)

    def test_malformed_json_non_strict_skips(self, tmp_path: Path) -> None:
        good = json.dumps(make_record())
        content = f"{{bad json\n{good}\n"
        path = tmp_path / "mixed.jsonl"
        path.write_text(content, encoding="utf-8")
        records = load_dataset(path, strict=False)
        assert len(records) == 1

    def test_invalid_record_strict_raises(self, tmp_path: Path) -> None:
        # question is empty → validation should fail
        bad_record = json.dumps({"id": "x", "question": "", "response": "y"})
        path = tmp_path / "invalid.jsonl"
        path.write_text(bad_record, encoding="utf-8")
        with pytest.raises(DatasetLoadError, match="validation failed"):
            load_dataset(path, strict=True)
