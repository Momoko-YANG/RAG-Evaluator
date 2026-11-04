"""Tests for Recall@k (retrieval_recall), qrels loading and the Recall report section."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from rag_eval_framework.config.loader import load_config
from rag_eval_framework.core.aggregation import summarize_per_k
from rag_eval_framework.core.schemas import EvaluationRecord
from rag_eval_framework.datasets.qrels import (
    QrelsEntry,
    QrelsLoadError,
    content_sha256,
    load_qrels,
    sha256_hex,
)
from rag_eval_framework.datasets.single_gold import build_single_gold_entries
from rag_eval_framework.evaluators.base import EvaluatorStatus
from rag_eval_framework.evaluators.retrieval.hit import RetrievalHitRateEvaluator
from rag_eval_framework.evaluators.retrieval.recall import RetrievalRecallEvaluator, recall_at_k
from rag_eval_framework.evaluators.retrieval.relevance.matcher import first_relevant_rank
from rag_eval_framework.reporting.engine import ReportEngine
from rag_eval_framework.runners.local import LocalRunner
from tests.conftest import make_record

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _doc(key: str, text: str | None = None, title: str = "規程.pdf") -> dict[str, Any]:
    return {"chunk_id": key, "title": title, "content": text or f"本文 {key}"}


def _entry(question_id: str = "test-001", **overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "question_id": question_id,
        "relevant_chunks": [{"chunk_id": "a", "grade": 3}, {"chunk_id": "b", "grade": 2}],
        "annotation_status": "reviewed",
        "judgment_scope": "pooled",
        "annotation_version": "v2",
    }
    base.update(overrides)
    return base


def _write_qrels(tmp_path: Path, *entries: dict[str, Any]) -> Path:
    path = tmp_path / "qrels.jsonl"
    path.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in entries), "utf-8")
    return path


def _evaluator(
    tmp_path: Path, *entries: dict[str, Any], **options: Any
) -> RetrievalRecallEvaluator:
    evaluator = RetrievalRecallEvaluator()
    path = _write_qrels(tmp_path, *(entries or (_entry(),)))
    evaluator.setup(
        SimpleNamespace(
            evaluator_options={"retrieval_recall": {"qrels_path": str(path), **options}}
        )
    )
    return evaluator


def _record(
    docs: list[dict[str, Any]], top_k: int | None = None, **overrides: Any
) -> EvaluationRecord:
    metadata: dict[str, Any] = {}
    if top_k is not None:
        metadata["rag_params"] = {"topK": top_k, "knn": 10}
    return EvaluationRecord(**make_record(retrieved_documents=docs, metadata=metadata, **overrides))


def _per_k(result: Any, k: int) -> dict[str, Any]:
    return result.metadata["per_k"][str(k)]


# ---------------------------------------------------------------------------
# recall_at_k (reference definition)
# ---------------------------------------------------------------------------


class TestRecallAtK:
    def test_definition(self) -> None:
        assert recall_at_k(["a", "x", "b"], {"a", "b"}, 3) == 1.0
        assert recall_at_k(["a", "x", "b"], {"a", "b"}, 2) == 0.5

    def test_duplicates_count_once(self) -> None:
        assert recall_at_k(["a", "a", "a"], {"a", "b"}, 3) == 0.5

    def test_empty_relevant_set_is_undefined(self) -> None:
        assert recall_at_k(["a"], set(), 3) is None

    def test_k_must_be_positive(self) -> None:
        with pytest.raises(ValueError):
            recall_at_k(["a"], {"a"}, 0)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class TestRetrievalRecallEvaluator:
    def test_all_relevant_retrieved(self, tmp_path: Path) -> None:
        ev = _evaluator(tmp_path, k=[3])
        result = ev.evaluate(_record([_doc("a"), _doc("x"), _doc("b")], top_k=3))
        assert result.status == EvaluatorStatus.SUCCESS
        assert result.score == 1.0
        assert _per_k(result, 3)["relevant_retrieved"] == 2

    def test_some_relevant_retrieved(self, tmp_path: Path) -> None:
        result = _evaluator(tmp_path, k=[3]).evaluate(
            _record([_doc("a"), _doc("x"), _doc("y")], top_k=3)
        )
        assert result.score == 0.5

    def test_none_retrieved_is_zero_not_skipped(self, tmp_path: Path) -> None:
        result = _evaluator(tmp_path, k=[3]).evaluate(
            _record([_doc("x"), _doc("y"), _doc("z")], top_k=3)
        )
        assert result.status == EvaluatorStatus.SUCCESS
        assert result.score == 0.0
        assert _per_k(result, 3)["unjudged_in_top_k"] == 3

    def test_duplicate_results_count_once(self, tmp_path: Path) -> None:
        result = _evaluator(tmp_path, k=[3]).evaluate(
            _record([_doc("a"), _doc("a"), _doc("a")], top_k=3)
        )
        assert result.score == 0.5
        assert _per_k(result, 3)["duplicate_results"] == 2

    def test_multiple_relevant_chunks_and_several_ks(self, tmp_path: Path) -> None:
        entry = _entry(relevant_chunks=[{"chunk_id": c, "grade": 3} for c in "abcd"])
        docs = [_doc(c) for c in "axbyzcwvut"]  # a@1, b@3, c@6, d missing
        result = _evaluator(tmp_path, entry, k=[3, 10]).evaluate(_record(docs, top_k=10))
        assert _per_k(result, 3)["value"] == 0.5
        assert _per_k(result, 10)["value"] == 0.75
        assert result.score == 0.5  # the first k is the score

    def test_grade_threshold(self, tmp_path: Path) -> None:
        entry = _entry(
            relevant_chunks=[
                {"chunk_id": "a", "grade": 3},
                {"chunk_id": "b", "grade": 2},
                {"chunk_id": "c", "grade": 1},
            ]
        )
        docs = [_doc("a"), _doc("c"), _doc("x")]
        default = _evaluator(tmp_path, entry, k=[3]).evaluate(_record(docs, top_k=3))
        assert default.score == 0.5  # a of {a, b}; c is below 2
        assert _per_k(default, 3)["unjudged_in_top_k"] == 1  # c is judged (grade 1), x is not
        strict = _evaluator(tmp_path, entry, k=[3], relevance_threshold=3).evaluate(
            _record(docs, top_k=3)
        )
        assert strict.score == 1.0
        lenient = _evaluator(tmp_path, entry, k=[3], relevance_threshold=1).evaluate(
            _record(docs, top_k=3)
        )
        assert lenient.score == round(2 / 3, 4)
        assert lenient.metadata["relevance_threshold"] == 1

    def test_no_chunk_at_threshold_is_skipped(self, tmp_path: Path) -> None:
        entry = _entry(relevant_chunks=[{"chunk_id": "a", "grade": 1}])
        result = _evaluator(tmp_path, entry).evaluate(_record([_doc("a")], top_k=3))
        assert result.status == EvaluatorStatus.SKIPPED
        assert "relevant set is empty" in result.reason

    def test_empty_relevant_set_is_skipped(self, tmp_path: Path) -> None:
        result = _evaluator(tmp_path, _entry(relevant_chunks=[])).evaluate(
            _record([_doc("a")], top_k=3)
        )
        assert result.status == EvaluatorStatus.SKIPPED

    def test_missing_annotation_is_skipped_not_zero(self, tmp_path: Path) -> None:
        result = _evaluator(tmp_path, _entry("other-question")).evaluate(
            _record([_doc("a")], top_k=3)
        )
        assert result.status == EvaluatorStatus.SKIPPED
        assert result.metadata["excluded_reason"] == "No relevance annotation for this question."

    def test_require_reviewed_excludes_partial_annotations(self, tmp_path: Path) -> None:
        entry = _entry(annotation_status="partially_reviewed")
        docs = [_doc("a"), _doc("b"), _doc("x")]
        assert _evaluator(tmp_path, entry).evaluate(_record(docs, top_k=3)).score == 1.0
        result = _evaluator(tmp_path, entry, require_reviewed=True).evaluate(_record(docs, top_k=3))
        assert result.status == EvaluatorStatus.SKIPPED
        assert "reviewed required" in result.reason

    def test_annotation_version_mismatch_is_skipped(self, tmp_path: Path) -> None:
        result = _evaluator(tmp_path, annotation_version="v3").evaluate(
            _record([_doc("a")], top_k=3)
        )
        assert result.status == EvaluatorStatus.SKIPPED

    def test_fewer_results_than_requested_is_complete(self, tmp_path: Path) -> None:
        # The search returned 2 of 10 requested: the list is complete, Recall@10 is defined.
        result = _evaluator(tmp_path, k=[10]).evaluate(_record([_doc("a"), _doc("x")], top_k=10))
        assert result.status == EvaluatorStatus.SUCCESS
        assert _per_k(result, 10)["value"] == 0.5
        assert "returned 2 of 10" in _per_k(result, 10)["reason"]

    def test_collection_depth_below_k_is_not_available(self, tmp_path: Path) -> None:
        # topK=3 (e.g. knn=50 run): Recall@10 / @50 cannot be computed, and are not 0.
        docs = [_doc("a"), _doc("x"), _doc("y")]
        result = _evaluator(tmp_path, k=[3, 10, 50]).evaluate(_record(docs, top_k=3))
        assert result.status == EvaluatorStatus.SUCCESS
        assert _per_k(result, 10) == {
            "status": "not_available",
            "value": None,
            "reason": "collection depth topK=3 < k=10",
        }
        assert _per_k(result, 50)["status"] == "not_available"

    def test_truncated_copy_uses_retrieval_depth(self, tmp_path: Path) -> None:
        # A Top-10 run cut to 3 results keeps topK=10 in rag_params.
        record = _record([_doc("a"), _doc("x"), _doc("y")], top_k=10)
        record.metadata["retrieval_depth"] = 3
        result = _evaluator(tmp_path, k=[3, 10]).evaluate(record)
        assert _per_k(result, 10)["status"] == "not_available"

    def test_unknown_depth_with_short_list_is_not_available(self, tmp_path: Path) -> None:
        result = _evaluator(tmp_path, k=[3, 10]).evaluate(_record([_doc("a")] * 1))
        assert _per_k(result, 3)["status"] == "not_available"
        assert result.status == EvaluatorStatus.SKIPPED  # the score k (3) is missing
        assert "collection depth is unknown" in result.reason

    def test_recall_kind(self, tmp_path: Path) -> None:
        docs = [_doc("a")]
        cases = {
            ("model_draft", "single_gold"): "single_gold_proxy",
            ("reviewed", "pooled"): "pooled_partial",
            ("model_draft", "exhaustive"): "pooled_partial",
            ("reviewed", "exhaustive"): "human_reviewed",
        }
        for (status, scope), kind in cases.items():
            entry = _entry(annotation_status=status, judgment_scope=scope)
            result = _evaluator(tmp_path, entry, k=[1]).evaluate(_record(docs, top_k=1))
            assert result.metadata["judgment_kind"] == kind

    def test_requires_qrels_path(self) -> None:
        with pytest.raises(ValueError, match="qrels_path"):
            RetrievalRecallEvaluator().setup(SimpleNamespace(evaluator_options={}))

    @pytest.mark.parametrize("k", [0, [3, 0], True, "3"])
    def test_rejects_bad_k(self, tmp_path: Path, k: Any) -> None:
        with pytest.raises(ValueError, match="k must be"):
            _evaluator(tmp_path, k=k)


class TestChunkIdentity:
    def test_rank_ids_are_ignored(self, tmp_path: Path) -> None:
        # Older runs store only id "doc-0" (the rank); it must not match a chunk id.
        entry = _entry(relevant_chunks=[{"chunk_id": "doc-0", "grade": 3}])
        doc = {"id": "doc-0", "title": "規程.pdf", "content": "本文"}
        result = _evaluator(tmp_path, entry, k=[1]).evaluate(_record([doc], top_k=1))
        assert result.score == 0.0

    def test_historical_jsonl_matches_by_content_hash(self, tmp_path: Path) -> None:
        text = "有給休暇は 年２０日 付与する。"
        entry = _entry(
            relevant_chunks=[
                {"content_sha256": content_sha256("有給休暇は年20日付与する。"), "grade": 3}
            ]
        )
        doc = {"id": "doc-0", "title": "規程.pdf", "content": text}  # no chunk_id
        result = _evaluator(tmp_path, entry, k=[1]).evaluate(_record([doc], top_k=1))
        assert result.score == 1.0
        assert _per_k(result, 1)["matched_by"] == {"content_sha256": 1}

    def test_chunk_id_hash(self, tmp_path: Path) -> None:
        entry = _entry(relevant_chunks=[{"chunk_id_sha256": sha256_hex("key-1"), "grade": 3}])
        result = _evaluator(tmp_path, entry, k=[1]).evaluate(_record([_doc("key-1")], top_k=1))
        assert _per_k(result, 1)["matched_by"] == {"chunk_id_sha256": 1}

    def test_keys_decide_over_identical_text(self, tmp_path: Path) -> None:
        # Two chunks share their text (e.g. a footer) but have different keys.
        entry = _entry(
            relevant_chunks=[
                {"chunk_id": "key-1", "content_sha256": content_sha256("同文"), "grade": 3}
            ]
        )
        result = _evaluator(tmp_path, entry, k=[1]).evaluate(
            _record([_doc("key-2", "同文")], top_k=1)
        )
        assert result.score == 0.0


class TestSingleGoldEquivalence:
    def test_equals_hit_at_k_when_the_gold_chunk_is_the_only_match(self, tmp_path: Path) -> None:
        gold = "年次有給休暇は年20日付与する。"
        entry = _entry(
            relevant_chunks=[{"content_sha256": content_sha256(gold), "grade": 3}],
            annotation_status="model_draft",
            judgment_scope="single_gold",
        )
        hit = RetrievalHitRateEvaluator()
        for docs in (
            [_doc("x", "第3章"), _doc("y", "休日"), _doc("z", gold)],
            [_doc("x", "第3章"), _doc("y", "休日"), _doc("w", "旅費")],
        ):
            for k in (1, 2, 3):
                record = _record(
                    docs,
                    top_k=3,
                    ground_truth_documents=["規程.pdf"],
                    ground_truth_evidence=["年20日"],
                )
                hit.setup(SimpleNamespace(evaluator_options={"retrieval_hit_rate": {"k": k}}))
                recall = _evaluator(tmp_path, entry, k=[k]).evaluate(record)
                assert recall.score == hit.evaluate(record).score
                assert recall.metadata["judgment_kind"] == "single_gold_proxy"


# ---------------------------------------------------------------------------
# Qrels file
# ---------------------------------------------------------------------------


class TestQrels:
    def test_loads_and_rejects_duplicates(self, tmp_path: Path) -> None:
        assert set(load_qrels(_write_qrels(tmp_path, _entry("q1"), _entry("q2")))) == {"q1", "q2"}
        with pytest.raises(QrelsLoadError, match="duplicate"):
            load_qrels(_write_qrels(tmp_path, _entry("q1"), _entry("q1")))

    def test_chunk_needs_an_identity_and_valid_grade(self, tmp_path: Path) -> None:
        with pytest.raises(QrelsLoadError):
            load_qrels(_write_qrels(tmp_path, _entry(relevant_chunks=[{"grade": 3}])))
        with pytest.raises(QrelsLoadError):
            load_qrels(
                _write_qrels(tmp_path, _entry(relevant_chunks=[{"chunk_id": "a", "grade": 4}]))
            )

    def test_rejects_unknown_status(self, tmp_path: Path) -> None:
        with pytest.raises(QrelsLoadError):
            load_qrels(_write_qrels(tmp_path, _entry(annotation_status="human")))

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(QrelsLoadError, match="not found"):
            load_qrels(tmp_path / "none.jsonl")

    def test_single_gold_builder_keeps_bank_hashes(self) -> None:
        rows = [
            {
                "id": "q1",
                "metadata_set": "golden",
                "metadata_source_key_hash": "k1",
                "metadata_content_hash": "c1",
                "ground_truth_documents": "a.pdf",
            },
            {
                "id": "q2",
                "metadata_set": "holdout",
                "metadata_source_key_hash": "k2",
                "metadata_content_hash": "c2",
            },
            {
                "id": "q3",
                "metadata_set": "smoke",
                "metadata_source_key_hash": "",
                "metadata_content_hash": "",
            },
        ]
        entries, missing = build_single_gold_entries(
            rows,
            sets={"smoke", "golden"},
            annotation_version="single-gold-v1",
            annotation_status="model_draft",
            index_identity="idx",
        )
        assert [e.question_id for e in entries] == ["q1"]  # holdout left out
        assert missing == ["q3"]
        chunk = entries[0].relevant_chunks[0]
        assert (chunk.chunk_id_sha256, chunk.content_sha256, chunk.source_title) == (
            "k1",
            "c1",
            "a.pdf",
        )
        assert entries[0].judgment_kind == "single_gold_proxy"
        assert isinstance(entries[0], QrelsEntry)


# ---------------------------------------------------------------------------
# Runner, aggregation and reports
# ---------------------------------------------------------------------------


def _run(tmp_path: Path, records: list[dict[str, Any]], *entries: dict[str, Any]) -> Any:
    dataset = tmp_path / "retrieval.jsonl"
    dataset.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), "utf-8")
    qrels = _write_qrels(tmp_path, *entries)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "project_name: recall-test\n"
        f"dataset_path: {dataset.as_posix()}\n"
        "report_format: [json, markdown, html]\n"
        f"output_dir: {(tmp_path / 'out').as_posix()}\n"
        "local_mode:\n"
        "  evaluators: [retrieval_hit_rate, retrieval_mrr, retrieval_recall]\n"
        "  evaluator_options:\n"
        "    retrieval_hit_rate: {k: 3}\n"
        "    retrieval_mrr: {k: 3}\n"
        "    retrieval_recall:\n"
        f"      qrels_path: {qrels.as_posix()}\n"
        "      k: [3, 10]\n",
        encoding="utf-8",
    )
    config = load_config(config_path)
    return config, LocalRunner().run(config)


class TestAggregationAndReports:
    def _records(self) -> list[dict[str, Any]]:
        def rec(qid: str, keys: str, top_k: int) -> dict[str, Any]:
            return make_record(
                id=qid,
                retrieved_documents=[_doc(k, title="規程.pdf") for k in keys],
                ground_truth_documents=["規程.pdf"],
                ground_truth_evidence=["本文 a"],
                metadata={"rag_params": {"topK": top_k}},
            )

        return [rec("q1", "abxyzuvwst", 10), rec("q2", "xya", 3), rec("q3", "xyz", 3)]

    def test_macro_recall_and_reports(self, tmp_path: Path) -> None:
        config, result = _run(
            tmp_path,
            self._records(),
            _entry("q1"),
            _entry("q2"),
            _entry("q4"),  # no record: harmless
        )
        summary = summarize_per_k(result, "retrieval_recall")
        assert summary is not None
        # q1: 1.0, q2: 0.5 at k=3; q3 has no qrels.
        assert summary["per_k"]["3"]["macro"] == 0.75
        assert summary["per_k"]["3"]["eligible_questions"] == 2
        assert summary["per_k"]["10"]["eligible_questions"] == 1
        assert summary["per_k"]["10"]["not_available_reasons"] == {
            "collection depth topK=3 < k=10": 1
        }
        judgments = summary["relevance_judgments"]
        assert judgments["relevance_threshold"] == 2
        assert judgments["excluded_questions"] == 1
        assert judgments["judgment_scopes"] == {"pooled": 2}
        assert summary["values"] == {"recall_at_3": 0.75, "recall_at_10": 1.0}
        assert result.aggregate_scores["retrieval_recall"] == 0.75

        paths = {p.suffix: p for p in ReportEngine().generate(config, result)}
        data = json.loads(paths[".json"].read_text(encoding="utf-8"))
        assert data["per_k_summaries"]["retrieval_recall"]["values"]["recall_at_3"] == 0.75
        markdown = paths[".md"].read_text(encoding="utf-8")
        assert "| Recall@3 | 0.7500 | 2 | 0 | pooled_partial 0.7500 (n=2) |" in markdown
        assert "Recall@10 not available for 1 question(s)" in markdown
        html = paths[".html"].read_text(encoding="utf-8")
        assert "<h2>Recall@k</h2>" in html and "pooled_partial" in html

    def test_existing_evaluators_unchanged(self, tmp_path: Path) -> None:
        records = self._records()
        _, result = _run(tmp_path, records, _entry("q1"))
        for record, rr in zip(records, result.record_results, strict=True):
            rank = first_relevant_rank(EvaluationRecord(**record), 3)
            assert rr.evaluator_results["retrieval_hit_rate"].score == (1.0 if rank else 0.0)
            assert rr.evaluator_results["retrieval_mrr"].score == (
                round(1 / rank, 4) if rank else 0.0
            )

    def test_reports_without_recall_have_no_section(self, tmp_path: Path) -> None:
        config, result = _run(tmp_path, self._records(), _entry("q1"))
        result = result.model_copy(update={"evaluators_used": ["retrieval_hit_rate"]})
        paths = {p.suffix: p for p in ReportEngine().generate(config, result)}
        assert "per_k_summaries" not in json.loads(paths[".json"].read_text(encoding="utf-8"))
        assert "## Recall@k" not in paths[".md"].read_text(encoding="utf-8")
