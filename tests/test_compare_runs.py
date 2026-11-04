"""Tests for scripts/compare_runs.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "compare_runs.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("compare_runs", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


cmp = _load_module()


def _report(scores: dict[str, Any], evaluator: str = "retrieval_mrr") -> dict[str, Any]:
    records = []
    for rid, value in scores.items():
        if isinstance(value, dict):
            result = value
        elif value is None:
            result = {"score": 0.0, "status": "skipped"}
        else:
            result = {"score": value, "status": "success"}
        records.append({"record_id": rid, "evaluator_results": {evaluator: result}})
    return {"record_results": records}


def test_buckets_and_moves() -> None:
    a = _report({"q1": 1.0, "q2": 0.5, "q3": 0.0, "q4": 0.0, "q5": None, "q6": 1.0})
    b = _report({"q1": 0.5, "q2": 0.0, "q3": 1.0, "q4": 0.0, "q5": 1.0, "q7": 1.0})
    result = cmp.compare(a, b, "retrieval_mrr")

    assert result["buckets"][(True, True)] == ["q1"]
    assert result["buckets"][(True, False)] == ["q2"]
    assert result["buckets"][(False, True)] == ["q3"]
    assert result["buckets"][(False, False)] == ["q4"]
    assert result["unscored"] == ["q5"]  # skipped in A: not a miss
    assert result["only_in_a"] == ["q6"] and result["only_in_b"] == ["q7"]
    assert result["moves"] == {"improved": 1, "worse": 2, "unchanged": 1}

    text = cmp.render(result, "baseline", "knn50", {"q3": "How | long?"})
    assert "baseline (A) vs knn50 (B)" in text
    assert "| q3 | 0.0000 | 1.0000 | How \\| long? |" in text


def test_min_score_and_per_k() -> None:
    def per_k(v3: float, v10: float | None) -> dict[str, Any]:
        ten = (
            {"status": "ok", "value": v10}
            if v10 is not None
            else {"status": "not_available", "value": None, "reason": "depth"}
        )
        return {
            "score": v3,
            "status": "success",
            "metadata": {"per_k": {"3": {"status": "ok", "value": v3}, "10": ten}},
        }

    a = _report({"q1": per_k(0.0, 1.0), "q2": per_k(1.0, None)}, "retrieval_recall")
    b = _report({"q1": per_k(1.0, 1.0), "q2": per_k(1.0, 1.0)}, "retrieval_recall")
    at10 = cmp.compare(a, b, "retrieval_recall", k=10)
    assert at10["metric"] == "retrieval_recall@10"
    assert at10["unscored"] == ["q2"]  # not available at depth 3
    assert at10["buckets"][(True, True)] == ["q1"]

    strict = cmp.compare(_report({"q1": 0.5}), _report({"q1": 1.0}), "retrieval_mrr", min_score=1.0)
    assert strict["buckets"][(False, True)] == ["q1"]

    with pytest.raises(ValueError, match="missing"):
        cmp.compare(a, b, "retrieval_hit_rate")
