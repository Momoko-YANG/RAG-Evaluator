"""Pydantic models for evaluation dataset records."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

from rag_eval_framework.observability.traces import Trace

# ``response`` of a record whose system produced no answer (marked ``metadata.empty_answer``).
EMPTY_ANSWER = "(empty answer)"


class EvaluationRecord(BaseModel):
    """A single evaluation dataset record.

    Core fields (``id``, ``question``, ``response``) are always required.
    Other fields are optional and consumed only by evaluators that declare
    them in their ``required_fields`` property.
    """

    id: str = Field(..., description="Unique identifier for this test case.")
    question: str = Field(..., description="The user question / query.")
    response: str = Field(..., description="The RAG system's generated response.")
    contexts: list[str] = Field(
        default_factory=list,
        description="Retrieved context passages used to generate the response.",
    )
    ground_truth_answer: str = Field(
        default="",
        description="Expected correct answer for comparison evaluators.",
    )

    # --- Phase 2 additions ------------------------------------------------

    retrieved_documents: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Structured retrieved document objects, each with at least a "
            "'content' key.  Used by document-level evaluators."
        ),
    )
    ground_truth_documents: list[str] = Field(
        default_factory=list,
        description=(
            "Titles (file names) of the documents that answer the question; matched "
            "against retrieved_documents[].title by the retrieval ranking evaluators."
        ),
    )
    ground_truth_evidence: list[str] = Field(
        default_factory=list,
        description=(
            "Short passages that a relevant retrieved chunk must contain; matched "
            "against retrieved_documents[].content by the retrieval ranking evaluators."
        ),
    )

    # --- End-to-end layer (all optional) -----------------------------------

    trace: Trace | None = Field(
        default=None,
        description="Tool calls, timings and token usage of the run (generation / e2e runners).",
    )
    expected_tools: list[str] = Field(
        default_factory=list,
        description="Tools the assistant should call for this question (tool_selection).",
    )
    expected_abstain: bool | None = Field(
        default=None,
        description="True when the assistant should say it cannot answer (abstention).",
    )
    expected_query_terms: list[str] = Field(
        default_factory=list,
        description="Terms the model's search text should contain (query_construction).",
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary metadata for filtering or grouping.",
    )

    # --- Validators -------------------------------------------------------

    @field_validator("response")
    @classmethod
    def response_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Response must not be empty.")
        return v

    @field_validator("question")
    @classmethod
    def question_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Question must not be empty.")
        return v

    # --- Convenience helpers ----------------------------------------------

    def has_contexts(self) -> bool:
        """Return *True* if at least one non-empty context is present."""
        return any(c.strip() for c in self.contexts)

    def has_ground_truth(self) -> bool:
        """Return *True* if a ground-truth answer is present."""
        return bool(self.ground_truth_answer.strip())

    def is_retrieval_only(self) -> bool:
        """Return *True* for a retrieval-only collection: ``response`` is a placeholder."""
        return bool(((self.metadata or {}).get("rag_params") or {}).get("retrievalOnly"))

    def is_empty_answer(self) -> bool:
        """Return *True* when the system gave no answer: ``response`` is :data:`EMPTY_ANSWER`."""
        return bool((self.metadata or {}).get("empty_answer"))
