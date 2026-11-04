"""Tests for the rag_eval_api adapter in scripts/collect_rag_data.py."""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from rag_eval_framework.core.schemas import EvaluationRecord

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "collect_rag_data.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("collect_rag_data", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict[str, Any]) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> dict[str, Any]:
        return self._payload


@pytest.fixture()
def fake_requests(monkeypatch: pytest.MonkeyPatch):
    calls: list[dict[str, Any]] = []
    state: dict[str, Any] = {"response": _FakeResponse(200, {})}

    def post(url: str, **kwargs: Any) -> _FakeResponse:
        calls.append({"url": url, **kwargs})
        return state["response"]

    module = types.ModuleType("requests")
    module.post = post  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "requests", module)
    return calls, state


def _args(mod, **overrides: Any):
    argv = ["--questions", "q.csv", "--output", "out.jsonl", "--adapter", "rag_eval_api"]
    args = mod.build_parser().parse_args(argv)
    args.rag_url = "http://localhost:3000/api/rag-eval"
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_rag_eval_api_adapter_registered() -> None:
    mod = _load_module()
    assert mod.ADAPTER_REGISTRY["rag_eval_api"] is mod.RagEvalApiAdapter


def test_rag_eval_api_query_maps_response(fake_requests, monkeypatch) -> None:
    monkeypatch.delenv("RAG_EVAL_API_KEY", raising=False)
    calls, state = fake_requests
    state["response"] = _FakeResponse(
        200,
        {
            "response": "有給休暇は年20日です。",
            "contexts": ["有給休暇は年20日付与される。"],
            "documents": [
                {
                    "id": "doc-0",
                    "metadata": "就業規則.pdf",
                    "pageContent": "有給休暇は年20日付与される。",
                    "score": 0.91,
                }
            ],
        },
    )
    mod = _load_module()
    adapter = mod.RagEvalApiAdapter()
    adapter.setup(_args(mod, rag_api_key="k" * 32, chat_thread_id="thread-1", model="latest"))

    result = adapter.query("有給休暇は何日？")

    assert calls[0]["url"] == "http://localhost:3000/api/rag-eval"
    assert calls[0]["headers"]["x-rag-eval-key"] == "k" * 32
    assert calls[0]["json"] == {
        "question": "有給休暇は何日？",
        "selectedModel": "latest",
        "chatThreadId": "thread-1",
    }
    assert result.response == "有給休暇は年20日です。"
    assert result.contexts == ["有給休暇は年20日付与される。"]
    assert result.retrieved_documents == [
        {
            "id": "doc-0",
            "title": "就業規則.pdf",
            "content": "有給休暇は年20日付与される。",
            "score": 0.91,
        }
    ]


def test_rag_eval_api_keeps_index_key_as_chunk_id(fake_requests, monkeypatch) -> None:
    calls, state = fake_requests
    state["response"] = _FakeResponse(
        200,
        {
            "response": "ok",
            "contexts": ["c1", "c2"],
            "documents": [
                {"id": "doc-0", "chunkId": "key-9", "metadata": "a.pdf", "pageContent": "c1"},
                {"id": "doc-1", "metadata": "b.pdf", "pageContent": "c2"},  # older server
            ],
        },
    )
    mod = _load_module()
    adapter = mod.RagEvalApiAdapter()
    adapter.setup(_args(mod, rag_api_key="k" * 32, extension_id="ext-1"))

    docs = adapter.query("q").retrieved_documents

    assert docs[0]["chunk_id"] == "key-9"
    assert docs[0]["id"] == "doc-0"  # the prompt's citation id is kept as before
    assert "chunk_id" not in docs[1]


def test_rag_eval_api_key_from_env_and_thread_optional(fake_requests, monkeypatch) -> None:
    monkeypatch.setenv("RAG_EVAL_API_KEY", "e" * 32)
    monkeypatch.delenv("RAG_EVAL_CHAT_THREAD_ID", raising=False)
    calls, state = fake_requests
    state["response"] = _FakeResponse(200, {"response": "ok", "contexts": []})
    mod = _load_module()
    adapter = mod.RagEvalApiAdapter()
    adapter.setup(_args(mod))

    adapter.query("q")

    assert calls[0]["headers"]["x-rag-eval-key"] == "e" * 32
    assert "chatThreadId" not in calls[0]["json"]
    assert calls[0]["json"]["selectedModel"] == "primary"


def test_rag_eval_api_requires_key(fake_requests, monkeypatch) -> None:
    monkeypatch.delenv("RAG_EVAL_API_KEY", raising=False)
    mod = _load_module()
    with pytest.raises(SystemExit):
        mod.RagEvalApiAdapter().setup(_args(mod))


def test_rag_eval_api_http_error_raises(fake_requests, monkeypatch) -> None:
    monkeypatch.setenv("RAG_EVAL_API_KEY", "e" * 32)
    _, state = fake_requests
    state["response"] = _FakeResponse(403, {"error": "chatThreadId is not allowed."})
    mod = _load_module()
    adapter = mod.RagEvalApiAdapter()
    adapter.setup(_args(mod))

    with pytest.raises(RuntimeError, match="403"):
        adapter.query("q")


def test_collect_writes_retrieved_documents(tmp_path: Path) -> None:
    mod = _load_module()

    class _Stub(mod.RagAdapter):
        def query(self, question: str, **kwargs: Any):
            return mod.RagResponse(
                response="answer",
                contexts=["ctx"],
                retrieved_documents=[{"id": "doc-0", "content": "ctx", "score": 0.5}],
            )

    out = tmp_path / "eval.jsonl"
    questions = [{"id": "q1", "question": "Q?", "ground_truth_answer": "A", "metadata": {}}]
    mod.collect(_Stub(), questions, out)

    record = json.loads(out.read_text(encoding="utf-8").strip())
    assert record["retrieved_documents"][0]["content"] == "ctx"
    EvaluationRecord.model_validate(record)


def test_rag_eval_api_sends_tuning_overrides(fake_requests, monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("RAG_EVAL_API_KEY", "e" * 32)
    calls, state = fake_requests
    state["response"] = _FakeResponse(
        200,
        {
            "response": "ok",
            "contexts": [],
            "deploymentName": "gpt-primary",
            "params": {"topK": 3, "knn": 50, "systemPrompt": "custom"},
        },
    )
    prompt = tmp_path / "strict.txt"
    prompt.write_text("資料に書かれた内容だけで答えてください。", encoding="utf-8")
    mod = _load_module()
    adapter = mod.RagEvalApiAdapter()
    adapter.setup(_args(mod, top_k=3, knn=50, system_prompt_file=prompt))

    result = adapter.query("q")

    assert calls[0]["json"]["topK"] == 3
    assert calls[0]["json"]["knn"] == 50
    assert calls[0]["json"]["systemPrompt"] == "資料に書かれた内容だけで答えてください。"
    assert result.metadata == {
        "rag_params": {
            "model": "primary",
            "deployment": "gpt-primary",
            "topK": 3,
            "knn": 50,
            "systemPrompt": "custom",
            "system_prompt_file": "strict.txt",
        }
    }


def test_rag_eval_api_omits_overrides_by_default(fake_requests, monkeypatch) -> None:
    monkeypatch.setenv("RAG_EVAL_API_KEY", "e" * 32)
    calls, state = fake_requests
    state["response"] = _FakeResponse(200, {"response": "ok", "contexts": []})
    mod = _load_module()
    adapter = mod.RagEvalApiAdapter()
    adapter.setup(_args(mod))

    adapter.query("q")

    assert not {"topK", "knn", "systemPrompt"} & set(calls[0]["json"])


def test_load_questions_reads_retrieval_labels(tmp_path: Path) -> None:
    csv_path = tmp_path / "questions.csv"
    csv_path.write_text(
        "id,question,ground_truth_answer,ground_truth_documents,ground_truth_evidence\n"
        "q1,有給は？,20日,就業規則.pdf | 補足.pdf,年20日\n"
        "q2,範囲外,情報が見当たりません,,\n",
        encoding="utf-8",
    )
    mod = _load_module()

    rows = mod.load_questions(csv_path)

    assert rows[0]["ground_truth_documents"] == ["就業規則.pdf", "補足.pdf"]
    assert rows[0]["ground_truth_evidence"] == ["年20日"]
    assert "ground_truth_documents" not in rows[1]


def test_collect_writes_labels_and_rag_params(tmp_path: Path) -> None:
    mod = _load_module()

    class _Stub(mod.RagAdapter):
        def query(self, question: str, **kwargs: Any):
            return mod.RagResponse(
                response="answer",
                contexts=["ctx"],
                metadata={"rag_params": {"topK": 3}},
            )

    out = tmp_path / "eval.jsonl"
    questions = [
        {
            "id": "q1",
            "question": "Q?",
            "ground_truth_answer": "A",
            "ground_truth_documents": ["a.pdf"],
            "ground_truth_evidence": ["ctx"],
            "metadata": {"scenario": "s"},
        }
    ]
    mod.collect(_Stub(), questions, out)

    record = EvaluationRecord.model_validate(json.loads(out.read_text(encoding="utf-8")))
    assert record.ground_truth_documents == ["a.pdf"]
    assert record.ground_truth_evidence == ["ctx"]
    assert record.metadata == {"scenario": "s", "rag_params": {"topK": 3}}


def test_rag_eval_api_searches_extension(fake_requests, monkeypatch) -> None:
    monkeypatch.setenv("RAG_EVAL_API_KEY", "e" * 32)
    monkeypatch.delenv("RAG_EVAL_CHAT_THREAD_ID", raising=False)
    calls, state = fake_requests
    state["response"] = _FakeResponse(
        200,
        {
            "response": "ok",
            "contexts": [],
            "params": {"source": "extension", "topK": 3, "indexName": "idx"},
        },
    )
    mod = _load_module()
    adapter = mod.RagEvalApiAdapter()
    adapter.setup(_args(mod, extension_id="ext-1"))

    result = adapter.query("q")

    assert calls[0]["json"]["extensionId"] == "ext-1"
    assert "chatThreadId" not in calls[0]["json"]
    assert result.metadata["rag_params"]["source"] == "extension"
    assert result.metadata["rag_params"]["indexName"] == "idx"


def test_rag_eval_api_rejects_thread_and_extension_together(fake_requests, monkeypatch) -> None:
    monkeypatch.setenv("RAG_EVAL_API_KEY", "e" * 32)
    mod = _load_module()
    with pytest.raises(SystemExit):
        mod.RagEvalApiAdapter().setup(_args(mod, extension_id="ext-1", chat_thread_id="t"))


def test_rag_eval_api_e2e_records_a_trace(fake_requests, tmp_path: Path) -> None:
    calls, state = fake_requests
    state["response"] = _FakeResponse(
        200,
        {
            "response": "20日です。",
            "rawResponse": '20日です。 {% citation items=[{name:"a.pdf",id:"doc-0"}] /%}',
            "contexts": ["c1"],
            "documents": [
                {"id": "doc-0", "chunkId": "k1", "metadata": "a.pdf", "pageContent": "c1"}
            ],
            "params": {"source": "extension", "topK": 3, "knn": 10, "e2e": True},
            "trace": {
                "mode": "e2e",
                "toolCalls": [{"name": "aisearch", "query": "有給", "latencyMs": 50,
                               "documentCount": 1, "executed": "search"}],
                "ttftMs": 700, "totalMs": 1500, "modelCalls": 2, "model": "m",
                "usage": {"promptTokens": 10, "completionTokens": 2, "totalTokens": 12},
            },
        },
    )
    mod = _load_module()
    adapter = mod.RagEvalApiAdapter()
    adapter.setup(_args(mod, rag_api_key="k" * 32, extension_id="ext-1", e2e=True))

    result = adapter.query("有給は何日？")

    assert calls[0]["json"]["e2e"] is True
    trace = result.trace
    assert trace["tool_calls"][0]["query"] == "有給"
    assert trace["timings"]["total_ms"] == 1500 and trace["timings"]["client_ms"] is not None
    assert trace["citations"] == [{"name": "a.pdf", "id": "doc-0"}]
    record = EvaluationRecord(id="q1", question="q", response=result.response, trace=trace)
    assert record.trace is not None and record.trace.usage[0].input_tokens == 10

    out = tmp_path / "e2e.jsonl"
    mod.collect(adapter, [{"id": "q1", "question": "q", "expected_tools": ["aisearch"],
                           "metadata": {}}], out)
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["trace"]["model_calls"] == 2 and written["expected_tools"] == ["aisearch"]


def test_rag_eval_api_e2e_needs_an_extension(fake_requests) -> None:
    mod = _load_module()
    with pytest.raises(SystemExit):
        mod.RagEvalApiAdapter().setup(
            _args(mod, rag_api_key="k" * 32, chat_thread_id="t", e2e=True)
        )
    with pytest.raises(SystemExit):
        mod.RagEvalApiAdapter().setup(
            _args(mod, rag_api_key="k" * 32, extension_id="ext-1", e2e=True, retrieval_only=True)
        )


def test_load_questions_reads_e2e_expectations(tmp_path: Path) -> None:
    csv_path = tmp_path / "q.csv"
    csv_path.write_text(
        "id,question,expected_tools,expected_abstain,expected_query_terms\n"
        "q1,Q?,aisearch|other,false,有給|日数\n"
        "q2,R?,,TRUE,\n",
        encoding="utf-8",
    )
    rows = _load_module().load_questions(csv_path)
    assert rows[0]["expected_tools"] == ["aisearch", "other"]
    assert rows[0]["expected_abstain"] is False and rows[1]["expected_abstain"] is True
    assert rows[0]["expected_query_terms"] == ["有給", "日数"] and "expected_tools" not in rows[1]
