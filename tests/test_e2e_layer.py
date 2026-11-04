"""E2E evaluators, observability (traces, latency, tokens, cost) and the layer runners."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from rag_eval_framework.adapters.base import RagResponse
from rag_eval_framework.adapters.chat_client import OpenAIChatClient
from rag_eval_framework.adapters.rag_eval_api.prompts import RAG_SYSTEM_PROMPT, build_rag_messages
from rag_eval_framework.config.loader import load_config
from rag_eval_framework.core.protocols import ChatClient, ChatOutput
from rag_eval_framework.core.schemas import EvaluationRecord
from rag_eval_framework.evaluators.base import EvaluatorStatus
from rag_eval_framework.evaluators.e2e.abstention import AbstentionEvaluator
from rag_eval_framework.evaluators.e2e.access_control import AccessControlEvaluator
from rag_eval_framework.evaluators.e2e.query_construction import QueryConstructionEvaluator
from rag_eval_framework.evaluators.e2e.task_success import TaskSuccessEvaluator
from rag_eval_framework.evaluators.e2e.tool_selection import ToolSelectionEvaluator
from rag_eval_framework.observability.cost import PricingConfig, cost_summary
from rag_eval_framework.observability.latency import latency_summary, nearest_rank
from rag_eval_framework.observability.token_usage import usage_summary
from rag_eval_framework.observability.traces import TokenUsage, parse_citations, trace_from_rag_eval
from rag_eval_framework.runners.collection import collect
from rag_eval_framework.runners.e2e import E2ERunner
from rag_eval_framework.runners.generation import GenerationRunner
from rag_eval_framework.runners.retrieval import RetrievalRunner, RetrievalVariant

GOLD = "年次有給休暇は年20日付与する。"
RAW = '20日です。 {% citation items=[{name:"就業規則.pdf",id:"doc-0"}] /%}'
API_TRACE = {
    "mode": "e2e",
    "toolCalls": [
        {
            "name": "aisearch",
            "arguments": {"body": {"search": "有給 日数"}},
            "query": "有給 日数",
            "latencyMs": 120,
            "documentCount": 3,
            "executed": "search",
        },
        {
            "name": "propose_remember",
            "arguments": {"content": "x"},
            "latencyMs": 0,
            "executed": "stubbed",
        },
    ],
    "ttftMs": 900,
    "totalMs": 2400,
    "modelCalls": 2,
    "model": "gpt-5-mini-2025-08-07",
    "usage": {"promptTokens": 1000, "completionTokens": 200, "totalTokens": 1200},
}


def _docs() -> list[dict[str, Any]]:
    return [
        {"id": "doc-0", "title": "就業規則.pdf", "content": GOLD},
        {"id": "doc-1", "title": "旅費規程.pdf", "content": "出張旅費は30日以内に精算する。"},
    ]


def _record(**overrides: Any) -> EvaluationRecord:
    base: dict[str, Any] = {
        "id": "q1",
        "question": "有給は何日？",
        "response": "20日です。",
        "retrieved_documents": _docs(),
        "ground_truth_documents": ["就業規則.pdf"],
        "ground_truth_evidence": ["年20日"],
        "trace": trace_from_rag_eval(API_TRACE, RAW, client_ms=2600).model_dump(),
        "metadata": {"rag_params": {"indexName": "index-a", "searchName": "svc-a", "topK": 3}},
    }
    return EvaluationRecord(**{**base, **overrides})


def _setup(evaluator: Any, **options: Any) -> Any:
    evaluator.setup(SimpleNamespace(evaluator_options={evaluator.name: options}))
    return evaluator


# ---------------------------------------------------------------------------
# Traces
# ---------------------------------------------------------------------------


class TestTraces:
    def test_from_rag_eval(self) -> None:
        trace = trace_from_rag_eval(API_TRACE, RAW, client_ms=2600)
        assert [c.status for c in trace.tool_calls] == ["executed", "stubbed"]
        assert trace.searches()[0].query == "有給 日数"
        assert trace.timings.search_ms == 120 and trace.timings.total_ms == 2400
        assert (
            trace.usage[0].total_tokens == 1200 and trace.usage[0].model == "gpt-5-mini-2025-08-07"
        )
        assert [(c.name, c.id) for c in trace.citations] == [("就業規則.pdf", "doc-0")]

    def test_parse_citations_handles_several_blocks(self) -> None:
        text = (
            RAW + ' 又 {% citation items=[{name:"a.pdf",id:"doc-1"}, {name:"b.pdf",id:"doc-2"}] /%}'
        )
        assert [c.id for c in parse_citations(text)] == ["doc-0", "doc-1", "doc-2"]
        assert parse_citations("no markup") == []


# ---------------------------------------------------------------------------
# E2E evaluators
# ---------------------------------------------------------------------------


class TestToolSelection:
    def test_expected_called_extra_allowed(self) -> None:
        result = _setup(ToolSelectionEvaluator(), expected_tools=["aisearch"]).evaluate(_record())
        assert result.score == 1.0 and result.metadata["extra"] == ["propose_remember"]

    def test_extra_not_allowed_and_forbidden(self) -> None:
        strict = _setup(ToolSelectionEvaluator(), expected_tools=["aisearch"], allow_extra=False)
        assert strict.evaluate(_record()).score == 0.0
        forbid = _setup(
            ToolSelectionEvaluator(),
            expected_tools=["aisearch"],
            forbidden_tools=["propose_remember"],
        )
        assert forbid.evaluate(_record()).metadata["forbidden_called"] == ["propose_remember"]

    def test_record_expectation_wins_and_missing_tool(self) -> None:
        result = _setup(ToolSelectionEvaluator(), expected_tools=["aisearch"]).evaluate(
            _record(expected_tools=["servicenow"])
        )
        assert result.score == 0.0 and result.metadata["missing"] == ["servicenow"]

    def test_skips_without_trace_or_expectation(self) -> None:
        ev = _setup(ToolSelectionEvaluator())
        assert ev.evaluate(_record()).status == EvaluatorStatus.SKIPPED
        assert (
            ev.evaluate(_record(trace=None, expected_tools=["a"])).status == EvaluatorStatus.SKIPPED
        )


class TestQueryConstruction:
    def test_term_coverage(self) -> None:
        result = QueryConstructionEvaluator().evaluate(
            _record(expected_query_terms=["有給", "付与"])
        )
        assert result.score == 0.5 and result.metadata["terms_missing"] == ["付与"]

    def test_no_search_scores_zero_and_no_terms_skip(self) -> None:
        no_search = trace_from_rag_eval({**API_TRACE, "toolCalls": []}).model_dump()
        assert (
            QueryConstructionEvaluator()
            .evaluate(_record(trace=no_search, expected_query_terms=["有給"]))
            .score
            == 0.0
        )
        assert QueryConstructionEvaluator().evaluate(_record()).status == EvaluatorStatus.SKIPPED


class TestAbstention:
    def test_answered_labelled_question(self) -> None:
        result = AbstentionEvaluator().evaluate(_record())
        assert result.score == 1.0 and result.metadata["outcome"] == "answered"

    def test_false_and_correct_abstention(self) -> None:
        ev = AbstentionEvaluator()
        refusal = "提示された資料には情報が見当たりません。"
        assert ev.evaluate(_record(response=refusal)).metadata["outcome"] == "false abstention"
        unanswerable = _record(
            response=refusal, ground_truth_documents=[], ground_truth_evidence=[]
        )
        assert ev.evaluate(unanswerable).score == 1.0
        assert (
            ev.evaluate(_record(expected_abstain=True)).metadata["outcome"] == "missed abstention"
        )

    def test_retrieval_only_is_skipped(self) -> None:
        record = _record(metadata={"rag_params": {"retrievalOnly": True}})
        assert AbstentionEvaluator().evaluate(record).status == EvaluatorStatus.SKIPPED


class TestAccessControl:
    def test_rules(self) -> None:
        ok = _setup(AccessControlEvaluator(), allowed_indexes=["index-a"]).evaluate(_record())
        assert ok.score == 1.0
        wrong = _setup(AccessControlEvaluator(), allowed_indexes=["index-b"]).evaluate(_record())
        assert wrong.score == 0.0 and "index" in wrong.metadata["violations"][0]
        pattern = _setup(AccessControlEvaluator(), forbidden_title_patterns=["旅費"]).evaluate(
            _record()
        )
        assert pattern.score == 0.0
        per_record = _setup(AccessControlEvaluator()).evaluate(
            _record(metadata={"forbidden_documents": ["就業規則.pdf"]})
        )
        assert per_record.score == 0.0

    def test_no_rule_skips(self) -> None:
        assert (
            _setup(AccessControlEvaluator()).evaluate(_record()).status == EvaluatorStatus.SKIPPED
        )


class TestTaskSuccess:
    def test_rule_passes_when_citing_the_relevant_chunk(self) -> None:
        result = _setup(TaskSuccessEvaluator()).evaluate(_record())
        assert result.score == 1.0 and result.metadata["source"] == "rule"

    def test_rule_fails_on_wrong_citation_or_wrong_abstention(self) -> None:
        wrong = trace_from_rag_eval(API_TRACE, RAW.replace("doc-0", "doc-1")).model_dump()
        result = _setup(TaskSuccessEvaluator()).evaluate(_record(trace=wrong))
        assert result.score == 0.0 and result.metadata["checks"]["cites_relevant_chunk"] is False
        refusal = _setup(TaskSuccessEvaluator()).evaluate(
            _record(response="情報が見当たりません。")
        )
        assert refusal.metadata["checks"]["abstention_correct"] is False

    def test_human_label_and_error(self) -> None:
        human = _setup(TaskSuccessEvaluator()).evaluate(
            _record(metadata={"human_task_success": "false"})
        )
        assert human.score == 0.0 and human.metadata["source"] == "human"
        errored = _record()
        errored.trace.error = "timeout"  # type: ignore[union-attr]
        assert (
            _setup(TaskSuccessEvaluator()).evaluate(errored).metadata["checks"]["no_error"] is False
        )

    def test_judge_source(self) -> None:
        ev = _setup(TaskSuccessEvaluator(), judge_threshold=0.75)
        ev._judge = SimpleNamespace(  # type: ignore[assignment]
            evaluate=lambda record: SimpleNamespace(
                status=EvaluatorStatus.SUCCESS, score=0.5, reason="ok"
            )
        )
        result = ev.evaluate(_record())
        assert result.metadata["source"] == "llm_judge" and result.score == 0.0


# ---------------------------------------------------------------------------
# Observability
# ---------------------------------------------------------------------------


class TestObservability:
    def test_latency_stages_are_not_added(self) -> None:
        summary = latency_summary([_record(), _record()])
        stages = summary["stages"]
        assert stages["client_ms"]["mean_ms"] == 2600 and stages["total_ms"]["mean_ms"] == 2400
        assert stages["search_ms"]["mean_ms"] == 120 and stages["ttft_ms"]["p95_ms"] == 900
        assert nearest_rank([float(i) for i in range(1, 21)], 0.95) == 19.0

    def test_usage_and_cost(self) -> None:
        records = [_record(), _record(trace=None, metadata={})]
        usage = usage_summary(records)
        assert usage["input_tokens"] == 1000 and usage["records_without_usage"] == 1
        pricing = PricingConfig(
            source="test sheet",
            models={
                "gpt-5-mini": {"input_per_1m": 0.25, "output_per_1m": 2.0},
                "gpt-5": {"input_per_1m": 9.0, "output_per_1m": 9.0},
            },
            fixed_monthly={"search": 250.0},
        )
        cost = cost_summary(records, pricing)
        assert cost["by_model"] == {"gpt-5-mini": round((1000 * 0.25 + 200 * 2.0) / 1e6, 6)}
        assert cost["status"] == "partial" and cost["records_without_usage"] == 1
        assert cost["fixed_monthly_not_allocated"] == {"search": 250.0}
        assert cost_summary(records, None)["status"] == "not_configured"
        unpriced = cost_summary([_record()], PricingConfig(source="x", models={}))
        assert unpriced["unpriced_tokens_by_model"] == {"gpt-5-mini-2025-08-07": 1200}


# ---------------------------------------------------------------------------
# Collection, chat client and layer runners
# ---------------------------------------------------------------------------


class TestCollect:
    def test_retries_transient_and_resumes(self, tmp_path: Path) -> None:
        calls = {"n": 0}

        def produce(item: dict[str, Any]) -> dict[str, Any]:
            calls["n"] += 1
            if item["id"] == "a" and calls["n"] == 1:
                raise RuntimeError("HTTP 503 from the RAG eval API")
            return {"id": item["id"], "question": "q", "response": "r"}

        out = tmp_path / "x.jsonl"
        items = [{"id": "a"}, {"id": "b"}]
        records = collect(items[:1], produce, out, sleep=lambda s: None)
        assert records[0]["metadata"]["attempts"] == 2
        records = collect(items, produce, out, sleep=lambda s: None)  # resumes after "a"
        assert [r["id"] for r in records] == ["a", "b"] and calls["n"] == 3

    def test_non_transient_error_stops_and_order_is_checked(self, tmp_path: Path) -> None:
        out = tmp_path / "x.jsonl"
        with pytest.raises(ValueError, match="bad"):
            collect([{"id": "a"}], lambda i: (_ for _ in ()).throw(ValueError("bad")), out)
        collect([{"id": "a"}], lambda i: {"id": "a", "question": "q", "response": "r"}, out)
        with pytest.raises(ValueError, match="question order"):
            collect([{"id": "z"}], lambda i: {}, out)


class _FakeStream:
    def __init__(self) -> None:
        def chunk(content: str | None, usage: Any = None) -> Any:
            delta = SimpleNamespace(content=content)
            choices = [SimpleNamespace(delta=delta)] if content is not None else []
            return SimpleNamespace(model="gpt-4.1-2025", choices=choices, usage=usage)

        self.chunks = [
            chunk("20日"),
            chunk("です。"),
            chunk(None, SimpleNamespace(prompt_tokens=500, completion_tokens=10)),
        ]
        self.kwargs: dict[str, Any] = {}
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        return iter(self.chunks)


def test_chat_client_streams_usage_and_ttft() -> None:
    fake = _FakeStream()
    out = OpenAIChatClient(fake).complete([{"role": "user", "content": "q"}], model="gpt-4.1")
    assert out.text == "20日です。" and out.model == "gpt-4.1-2025"
    assert out.usage is not None and out.usage.input_tokens == 500
    assert out.ttft_ms is not None and fake.kwargs["stream_options"] == {"include_usage": True}
    assert isinstance(OpenAIChatClient(fake), ChatClient)


def test_prompt_mirrors_rag_eval_api() -> None:
    messages = build_rag_messages(_docs(), "有給は何日？")
    assert messages[0]["content"] == RAG_SYSTEM_PROMPT
    assert "[0]. file name: 就業規則.pdf \n file id: doc-0" in messages[2]["content"]
    assert messages[2]["content"].rstrip().endswith("有給は何日？")


def _config(tmp_path: Path, evaluators: str, options: str = "", pricing: bool = False) -> Any:
    path = tmp_path / "config.yaml"
    path.write_text(
        "project_name: layer\n"
        f"dataset_path: {(tmp_path / 'none.jsonl').as_posix()}\n"
        "report_format: [json, markdown]\n"
        f"output_dir: {(tmp_path / 'out').as_posix()}\n"
        + (
            "observability:\n  pricing:\n    source: test\n    models:\n"
            "      gpt-4.1: {input_per_1m: 2.0, output_per_1m: 8.0}\n"
            if pricing
            else ""
        )
        + f"local_mode:\n  evaluators: [{evaluators}]\n"
        + (f"  evaluator_options:\n{options}" if options else ""),
        encoding="utf-8",
    )
    return load_config(path)


class _FakeAdapter:
    def __init__(self, top_k: int = 3, e2e: bool = False) -> None:
        self.top_k = top_k
        self.e2e = e2e

    def setup(self, args: Any) -> None: ...

    def query(self, question: str, **kwargs: Any) -> RagResponse:
        docs = _docs()[: self.top_k]
        trace = trace_from_rag_eval(API_TRACE, RAW, client_ms=2600).model_dump(mode="json")
        return RagResponse(
            response="20日です。" if self.e2e else "(retrieval only)",
            contexts=[d["content"] for d in docs],
            retrieved_documents=docs,
            metadata={
                "rag_params": {
                    "topK": self.top_k,
                    "retrievalOnly": not self.e2e,
                    "indexName": "index-a",
                }
            },
            trace=trace if self.e2e else None,
        )


QUESTIONS = [
    {
        "id": "q1",
        "question": "有給は何日？",
        "ground_truth_answer": "20日",
        "ground_truth_documents": ["就業規則.pdf"],
        "ground_truth_evidence": ["年20日"],
        "metadata": {"set": "golden"},
    }
]


class TestLayerRunners:
    def test_retrieval_runner(self, tmp_path: Path) -> None:
        config = _config(tmp_path, "retrieval_hit_rate", "    retrieval_hit_rate: {k: 3}\n")
        results = RetrievalRunner(lambda v: _FakeAdapter(v.top_k)).run(
            QUESTIONS,
            [RetrievalVariant.parse("baseline:3:10"), RetrievalVariant.parse("one:1")],
            tmp_path / "run",
            config,
        )
        assert results["baseline"].aggregate_scores["retrieval_hit_rate"] == 1.0
        manifest = json.loads((tmp_path / "run" / "manifest.json").read_text(encoding="utf-8"))
        assert (
            manifest["variants"]["one"]["top_k"] == 1 and manifest["variants"]["one"]["knn"] is None
        )
        assert results["baseline"].observability["latency"]["stages"]["client_ms"]["n"] == 1

    def test_generation_runner_uses_fixed_contexts(self, tmp_path: Path) -> None:
        retrieval = tmp_path / "retrieval.jsonl"
        retrieval.write_text(
            json.dumps(
                {
                    **QUESTIONS[0],
                    "response": "(retrieval only)",
                    "retrieved_documents": _docs(),
                    "metadata": {
                        "rag_params": {"topK": 10, "retrievalOnly": True},
                        "eval_latency_ms": 300,
                    },
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

        class Client:
            def __init__(self) -> None:
                self.messages: list[Any] = []

            def complete(
                self, messages: Any, *, model: str, temperature: Any = None, max_tokens: Any = None
            ) -> ChatOutput:
                self.messages = list(messages)
                return ChatOutput(
                    text=RAW,
                    model="gpt-4.1-2025",
                    ttft_ms=300.0,
                    total_ms=900.0,
                    usage=TokenUsage(model="gpt-4.1-2025", input_tokens=800, output_tokens=20),
                )

        client = Client()
        config = _config(tmp_path, "answer_presence, abstention, task_success", pricing=True)
        out = tmp_path / "gen" / "gpt.jsonl"
        result = GenerationRunner(client, model="gpt-4.1", context_k=1).run(retrieval, out, config)
        record = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
        assert record["response"] == "20日です。" and len(record["retrieved_documents"]) == 1
        assert record["metadata"]["rag_params"]["retrievalOnly"] is False
        assert record["metadata"]["retrieval_latency_ms"] == 300
        assert record["metadata"]["generation"]["prompt_sha256"]
        assert "file id: doc-0" in client.messages[2]["content"]
        assert result.aggregate_scores == {
            "answer_presence": 1.0,
            "abstention": 1.0,
            "task_success": 1.0,
        }
        assert result.observability["cost"]["by_model"] == {
            "gpt-4.1": round((800 * 2 + 20 * 8) / 1e6, 6)
        }

    def test_e2e_runner_scores_trace_and_reports_observability(self, tmp_path: Path) -> None:
        options = (
            "    tool_selection: {expected_tools: [aisearch]}\n"
            "    access_control: {allowed_indexes: [index-a]}\n"
        )
        config = _config(
            tmp_path, "tool_selection, abstention, access_control, task_success", options
        )
        out = tmp_path / "e2e" / "e2e.jsonl"
        result = E2ERunner(_FakeAdapter(e2e=True)).run(QUESTIONS, out, config)
        assert set(result.aggregate_scores.values()) == {1.0}
        assert result.metric_definitions["tool_selection"]["layer"] == "e2e"
        assert result.observability["token_usage"]["input_tokens"] == 1000
        assert result.observability["cost"]["status"] == "not_configured"
        markdown = next((tmp_path / "e2e").rglob("*.md")).read_text(encoding="utf-8")
        assert "## Observability" in markdown and "| ttft_ms | 1 |" in markdown
        assert "| e2e | tool_selection |" in markdown

    def test_e2e_empty_answer_is_a_failed_task_not_an_invalid_dataset(self, tmp_path: Path) -> None:
        class _NoAnswer(_FakeAdapter):
            def query(self, question: str, **kwargs: Any) -> RagResponse:
                return dataclasses.replace(super().query(question), response="")

        config = _config(tmp_path, "answer_presence, abstention, task_success, retrieval_hit_rate")
        out = tmp_path / "e2e" / "e2e.jsonl"
        result = E2ERunner(_NoAnswer(e2e=True)).run(QUESTIONS, out, config)

        record = json.loads(out.read_text(encoding="utf-8"))
        assert record["response"] == "(empty answer)"
        assert record["metadata"]["empty_answer"] is True
        assert record["trace"]["error"] == "empty answer after 2 model calls"
        results = result.record_results[0].evaluator_results
        assert results["answer_presence"].status == EvaluatorStatus.SKIPPED
        assert results["abstention"].status == EvaluatorStatus.SKIPPED
        assert results["task_success"].score == 0.0  # counted as a failure
        assert results["retrieval_hit_rate"].score == 1.0  # retrieval still scored


def test_answer_presence_counts_cjk_characters() -> None:
    from rag_eval_framework.evaluators.generation.answer_presence import AnswerPresenceEvaluator

    ev = AnswerPresenceEvaluator()
    assert ev.version == "2.0"
    assert ev.evaluate(_record(response="20日です。")).score == 1.0
    assert ev.evaluate(_record(response="はい。")).score == 0.0
    assert ev.evaluate(_record(response="Yes")).score == 0.0
    assert ev.evaluate(_record(response="Twenty days")).score == 1.0


def test_tool_selection_expect_search() -> None:
    ev = _setup(ToolSelectionEvaluator(), expect_search=True)
    assert ev.evaluate(_record()).score == 1.0
    no_search = trace_from_rag_eval(
        {**API_TRACE, "toolCalls": API_TRACE["toolCalls"][1:]}
    ).model_dump()
    result = ev.evaluate(_record(trace=no_search))
    assert result.score == 0.0 and result.metadata["missing"] == ["<search>"]
