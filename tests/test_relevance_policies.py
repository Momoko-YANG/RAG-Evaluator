"""Relevance policies shared by the ranking metrics, depth-invariant nDCG, and traceability."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from rag_eval_framework.config.loader import load_config
from rag_eval_framework.core.aggregation import metric_results
from rag_eval_framework.core.schemas import EvaluationRecord
from rag_eval_framework.datasets.qrels import content_sha256
from rag_eval_framework.evaluators.base import EvaluatorStatus
from rag_eval_framework.evaluators.generation.correctness import AccuracyJudge
from rag_eval_framework.evaluators.retrieval.hit import RetrievalHitRateEvaluator
from rag_eval_framework.evaluators.retrieval.mrr import RetrievalMRREvaluator
from rag_eval_framework.evaluators.retrieval.ndcg import (
    AzureDocumentRetrievalEvaluator,
    RetrievalNDCGEvaluator,
    build_sdk_inputs,
    ndcg_at_k,
)
from rag_eval_framework.evaluators.retrieval.recall import RetrievalRecallEvaluator
from rag_eval_framework.reporting.engine import ReportEngine
from rag_eval_framework.runners.local import LocalRunner
from tests.conftest import make_record

GOLD = "年次有給休暇は年20日付与する。"


def _doc(key: str, text: str | None = None, title: str = "規程.pdf") -> dict[str, Any]:
    return {"chunk_id": key, "title": title, "content": text or f"本文 {key}"}


def _record(
    docs: list[dict[str, Any]], top_k: int | None = None, **overrides: Any
) -> EvaluationRecord:
    metadata: dict[str, Any] = {"rag_params": {"topK": top_k}} if top_k else {}
    base = {"ground_truth_documents": ["規程.pdf"], "ground_truth_evidence": ["年20日"]}
    return EvaluationRecord(
        **make_record(retrieved_documents=docs, metadata=metadata, **{**base, **overrides})
    )


def _qrels(tmp_path: Path, *chunks: dict[str, Any], scope: str = "pooled") -> Path:
    path = tmp_path / "qrels.jsonl"
    path.write_text(
        json.dumps(
            {
                "question_id": "test-001",
                "relevant_chunks": list(chunks),
                "annotation_status": "reviewed",
                "judgment_scope": scope,
                "annotation_version": "v2",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _setup(evaluator: Any, **options: Any) -> Any:
    evaluator.setup(SimpleNamespace(evaluator_options={evaluator.name: options}))
    return evaluator


# ---------------------------------------------------------------------------
# P0 — nDCG@3 does not depend on how many results are stored
# ---------------------------------------------------------------------------


class TestDepthInvariantNDCG:
    def _top10(self) -> list[dict[str, Any]]:
        # Relevant at rank 2 and (beyond the cut) rank 6.
        docs = [_doc(f"d{i}") for i in range(10)]
        docs[1] = _doc("d1", GOLD)
        docs[5] = _doc("d5", "（再掲）" + GOLD)
        return docs

    def test_sdk_inputs_only_label_the_top_3(self) -> None:
        deep = _record(self._top10(), top_k=10)
        shallow = _record(self._top10()[:3], top_k=10)
        assert build_sdk_inputs(deep) == build_sdk_inputs(shallow)

    def test_sdk_ndcg_equal_for_top10_and_top3_input(self) -> None:
        pytest.importorskip("azure.ai.evaluation")
        evaluator = AzureDocumentRetrievalEvaluator()
        evaluator.setup(None)
        deep = evaluator.evaluate(_record(self._top10(), top_k=10))
        shallow = evaluator.evaluate(_record(self._top10()[:3], top_k=10))
        assert deep.score == shallow.score == round(1 / math.log2(3), 4)
        assert evaluator.version == "2.0"

    def test_pure_python_matches_sdk_under_evidence_match(self) -> None:
        pytest.importorskip("azure.ai.evaluation")
        sdk = AzureDocumentRetrievalEvaluator()
        sdk.setup(None)
        ours = _setup(RetrievalNDCGEvaluator(), relevance_policy="evidence_match", k=[3])
        rankings = [
            [_doc("a", GOLD), _doc("b"), _doc("c")],
            [_doc("b"), _doc("c"), _doc("a", GOLD)],
            [_doc("a", GOLD), _doc("b", "別本 " + GOLD), _doc("c")],
            [_doc("b"), _doc("c"), _doc("d")],
            self._top10(),
        ]
        for docs in rankings:
            record = _record(docs, top_k=len(docs))
            assert ours.evaluate(record).score == sdk.evaluate(record).score

    def test_qrels_ndcg_does_not_change_with_depth(self, tmp_path: Path) -> None:
        path = _qrels(tmp_path, {"chunk_id": "d1", "grade": 3}, {"chunk_id": "d5", "grade": 2})
        ev = _setup(RetrievalNDCGEvaluator(), qrels_path=str(path), k=[3])
        deep = ev.evaluate(_record(self._top10(), top_k=10))
        shallow = _record(self._top10()[:3], top_k=10)
        shallow.metadata["retrieval_depth"] = 3
        assert ev.evaluate(shallow).score == deep.score


# ---------------------------------------------------------------------------
# P1 — one relevance layer, explicit policy per metric
# ---------------------------------------------------------------------------


class TestRelevancePolicies:
    def test_graded_ndcg_value(self, tmp_path: Path) -> None:
        path = _qrels(tmp_path, {"chunk_id": "a", "grade": 3}, {"chunk_id": "b", "grade": 2})
        ev = _setup(RetrievalNDCGEvaluator(), qrels_path=str(path), k=[3])
        result = ev.evaluate(_record([_doc("b"), _doc("x"), _doc("a")], top_k=3))
        dcg = 3 / math.log2(2) + 7 / math.log2(4)
        idcg = 7 / math.log2(2) + 3 / math.log2(3)
        assert result.score == round(dcg / idcg, 4)
        assert result.metadata["per_k"]["3"]["unjudged_in_top_k"] == 1
        assert ndcg_at_k([0.0], [0.0], 3) == 0.0

    def test_hit_under_qrels_equals_single_gold_recall(self, tmp_path: Path) -> None:
        path = _qrels(
            tmp_path, {"content_sha256": content_sha256(GOLD), "grade": 3}, scope="single_gold"
        )
        hit = _setup(
            RetrievalHitRateEvaluator(), relevance_policy="qrels", qrels_path=str(path), k=3
        )
        recall = _setup(RetrievalRecallEvaluator(), qrels_path=str(path), k=[3])
        evidence_hit = _setup(RetrievalHitRateEvaluator(), k=3)
        overlap = _doc("n", "前章の続き。" + GOLD + "次章へ。")  # contains the quote, other chunk
        for docs in ([_doc("x"), _doc("g", GOLD), _doc("y")], [overlap, _doc("x"), _doc("y")]):
            record = _record(docs, top_k=3)
            assert hit.evaluate(record).score == recall.evaluate(record).score
        # Under evidence_match the overlapping chunk counts; under qrels it does not.
        record = _record([overlap, _doc("x"), _doc("y")], top_k=3)
        assert evidence_hit.evaluate(record).score == 1.0
        assert hit.evaluate(record).score == 0.0
        assert hit.evaluate(record).metadata["relevance_policy"] == "qrels"

    def test_hit_and_mrr_agree_under_the_same_policy(self, tmp_path: Path) -> None:
        path = _qrels(tmp_path, {"chunk_id": "g", "grade": 2})
        options = {"relevance_policy": "qrels", "qrels_path": str(path), "k": 3}
        hit = _setup(RetrievalHitRateEvaluator(), **options)
        mrr = _setup(RetrievalMRREvaluator(), **options)
        for docs in ([_doc("x"), _doc("g")], [_doc("x"), _doc("y"), _doc("z"), _doc("g")]):
            record = _record(docs, top_k=len(docs))
            assert (hit.evaluate(record).score > 0) == (mrr.evaluate(record).score > 0)

    def test_qrels_policy_skips_unannotated_questions(self, tmp_path: Path) -> None:
        path = _qrels(tmp_path, {"chunk_id": "g", "grade": 2})
        hit = _setup(RetrievalHitRateEvaluator(), relevance_policy="qrels", qrels_path=str(path))
        result = hit.evaluate(
            EvaluationRecord(**make_record(id="other", retrieved_documents=[_doc("g")]))
        )
        assert result.status == EvaluatorStatus.SKIPPED

    def test_recall_rejects_the_open_policy(self) -> None:
        with pytest.raises(ValueError, match="relevance_policy"):
            _setup(RetrievalRecallEvaluator(), relevance_policy="evidence_match")

    def test_unknown_policy_and_missing_qrels_path(self) -> None:
        with pytest.raises(ValueError, match="relevance_policy"):
            _setup(RetrievalHitRateEvaluator(), relevance_policy="semantic")
        with pytest.raises(ValueError, match="qrels_path"):
            _setup(RetrievalNDCGEvaluator())

    def test_ndcg_beyond_depth_is_not_available(self, tmp_path: Path) -> None:
        path = _qrels(tmp_path, {"chunk_id": "g", "grade": 2})
        ev = _setup(RetrievalNDCGEvaluator(), qrels_path=str(path), k=[3, 10])
        result = ev.evaluate(_record([_doc("g"), _doc("x"), _doc("y")], top_k=3))
        assert result.metadata["per_k"]["10"]["status"] == "not_available"
        assert result.score == 1.0


# ---------------------------------------------------------------------------
# P2 — traceable metric results
# ---------------------------------------------------------------------------


class TestTraceability:
    def test_definitions_provenance_and_metric_results(self, tmp_path: Path) -> None:
        dataset = tmp_path / "retrieval.jsonl"
        record = make_record(
            id="test-001",
            retrieved_documents=[_doc("g", GOLD), _doc("x")],
            ground_truth_documents=["規程.pdf"],
            ground_truth_evidence=["年20日"],
            metadata={"rag_params": {"topK": 10, "knn": 50, "indexName": "idx"}},
        )
        dataset.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
        qrels = _qrels(tmp_path, {"chunk_id": "g", "grade": 3})
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            "project_name: trace\n"
            f"dataset_path: {dataset.as_posix()}\n"
            "report_format: [json, markdown, html]\n"
            f"output_dir: {(tmp_path / 'out').as_posix()}\n"
            "local_mode:\n"
            "  evaluators: [retrieval_hit_rate, retrieval_recall, retrieval_ndcg]\n"
            "  evaluator_options:\n"
            "    retrieval_hit_rate: {k: 3}\n"
            f"    retrieval_recall: {{qrels_path: {qrels.as_posix()}, k: [3, 10]}}\n"
            f"    retrieval_ndcg: {{qrels_path: {qrels.as_posix()}, k: [3]}}\n",
            encoding="utf-8",
        )
        config = load_config(config_path)
        result = LocalRunner().run(config)

        recall_def = result.metric_definitions["retrieval_recall"]
        assert recall_def["version"] == "1.0" and recall_def["layer"] == "retrieval"
        assert recall_def["relevance_policy"] == "qrels"
        assert recall_def["qrels_sha256"] == hashlib.sha256(qrels.read_bytes()).hexdigest()
        assert (
            result.metric_definitions["retrieval_hit_rate"]["relevance_policy"] == "evidence_match"
        )
        assert (
            result.provenance["dataset"]["sha256"]
            == hashlib.sha256(dataset.read_bytes()).hexdigest()
        )
        assert result.provenance["retrieval_settings"] == [
            {"indexName": "idx", "topK": 10, "knn": 50, "records": 1}
        ]

        entries = {e["metric"]: e for e in metric_results(result)}
        assert entries["recall@3"]["score"] == 1.0
        assert entries["recall@3"]["relevance_policy"] == "qrels"
        assert entries["recall@10"]["status"] == "valid"  # search returned 2 of 10: complete list
        assert entries["retrieval_hit_rate@3"]["metric_version"] == "1.0"
        assert entries["ndcg@3"]["score"] == 1.0

        paths = {p.suffix: p for p in ReportEngine().generate(config, result)}
        data = json.loads(paths[".json"].read_text(encoding="utf-8"))
        assert {e["metric"] for e in data["metric_results"]} >= {"recall@3", "ndcg@3"}
        assert set(data["per_k_summaries"]) == {"retrieval_recall", "retrieval_ndcg"}
        markdown = paths[".md"].read_text(encoding="utf-8")
        assert "## Metric Definitions" in markdown and "## Provenance" in markdown
        assert "## nDCG@k" in markdown and "## Recall@k" in markdown
        assert "<h2>Metric Definitions</h2>" in paths[".html"].read_text(encoding="utf-8")

    def test_judge_definition_records_prompt_hash(self) -> None:
        judge = AccuracyJudge()
        definition = judge.definition()
        prompt = f"{judge.system_prompt}\n---\n{judge.user_prompt_template}"
        assert definition["prompt_sha256"] == hashlib.sha256(prompt.encode()).hexdigest()[:16]
        assert definition["evaluator"] == "accuracy_judge" and definition["version"] == "1.0"
