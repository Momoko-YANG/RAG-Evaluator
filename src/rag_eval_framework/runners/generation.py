"""Generation layer runner: answers from *fixed* contexts, then generation metrics.

Every model sees the same question and the same top-k chunks of a saved retrieval
run, so generation models can be compared without retrieval noise.  The prompt is
the RAG prompt of :mod:`rag_eval_framework.adapters.rag_eval_api.prompts`;
its hash, the model and k are recorded per record and the trace keeps TTFT, time and
token usage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rag_eval_framework.adapters.rag_eval_api.prompts import (
    build_rag_messages,
    prompt_sha256,
    strip_citation_markup,
)
from rag_eval_framework.config.models import ProjectConfig
from rag_eval_framework.core.protocols import ChatClient, Scorer
from rag_eval_framework.core.results import EvaluationRunResult
from rag_eval_framework.core.schemas import EMPTY_ANSWER
from rag_eval_framework.observability.traces import Timings, Trace, parse_citations
from rag_eval_framework.runners.collection import QUESTION_FIELDS, collect, read_jsonl, score


class GenerationRunner:
    def __init__(
        self,
        client: ChatClient,
        *,
        model: str,
        context_k: int = 3,
        system_prompt: str | None = None,
        persona: str = "",
        temperature: float | None = None,
        max_tokens: int | None = None,
        scorer: Scorer | None = None,
        retries: int = 3,
    ) -> None:
        self._client = client
        self._model = model
        self._k = context_k
        self._system_prompt = system_prompt
        self._persona = persona
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._scorer = scorer
        self._retries = retries

    @property
    def prompt_sha256(self) -> str:
        return prompt_sha256(self._system_prompt, self._persona)

    def generate(self, source: dict[str, Any]) -> dict[str, Any]:
        """One answer from the first ``context_k`` retrieved chunks of a saved record."""
        documents = (source.get("retrieved_documents") or [])[: self._k]
        out = self._client.complete(
            build_rag_messages(
                documents,
                source["question"],
                persona=self._persona,
                system_prompt=self._system_prompt,
            ),
            model=self._model,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        answer = strip_citation_markup(out.text)
        metadata = dict(source.get("metadata") or {})
        if "eval_latency_ms" in metadata:
            metadata["retrieval_latency_ms"] = metadata.pop("eval_latency_ms")
        metadata["rag_params"] = {**(metadata.get("rag_params") or {}), "retrievalOnly": False}
        if not answer:
            metadata["empty_answer"] = True
        metadata["generation"] = {
            "model": self._model,
            "served_model": out.model,
            "context_k": self._k,
            "prompt_sha256": self.prompt_sha256,
            "temperature": self._temperature,
        }
        return {
            "id": source["id"],
            "question": source["question"],
            "response": answer or EMPTY_ANSWER,
            "contexts": [str(d.get("content", "")) for d in documents],
            "retrieved_documents": documents,
            **{k: source[k] for k in QUESTION_FIELDS if source.get(k) not in (None, "", [])},
            "metadata": metadata,
            "trace": Trace(
                mode="generation",
                timings=Timings(total_ms=out.total_ms, ttft_ms=out.ttft_ms),
                usage=[out.usage] if out.usage else [],
                model_calls=1,
                model=out.model,
                citations=parse_citations(out.text),
                error=None if answer else "empty answer",
            ).model_dump(mode="json"),
        }

    def run(
        self,
        retrieval_path: Path,
        output_path: Path,
        config: ProjectConfig,
        *,
        limit: int | None = None,
    ) -> EvaluationRunResult:
        sources = read_jsonl(retrieval_path)[:limit]
        collect(sources, self.generate, output_path, retries=self._retries)
        return score(config, output_path, output_path.parent, self._scorer)
