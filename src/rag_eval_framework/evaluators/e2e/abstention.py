"""Abstention: does the assistant decline exactly when it should?

Rule-based.  *Expected*: the record's ``expected_abstain``, or — with
``infer_from_labels: true`` (default) — True for questions without ground-truth
labels (unanswerable) and False for labelled ones.  *Detected*: the answer contains
an abstention phrase (``patterns``; default: the default prompt's "No information
found" and common English / Japanese variants).  Score 1.0 when detected == expected.

Phrase matching can miss paraphrased refusals and can flag an answer that says
"could not find" about one detail; review the ``matched`` phrase in the metadata.
"""

from __future__ import annotations

from typing import Any

from rag_eval_framework.core.results import EvaluatorResult
from rag_eval_framework.core.schemas import EvaluationRecord
from rag_eval_framework.evaluators.e2e.base import (
    DEFAULT_ABSTAIN_PATTERNS,
    E2EEvaluatorBase,
    detect_abstention,
    is_retrieval_only,
    skip,
)
from rag_eval_framework.evaluators.retrieval.relevance.matcher import has_labels


def expected_abstention(record: EvaluationRecord, infer_from_labels: bool) -> bool | None:
    if record.expected_abstain is not None:
        return record.expected_abstain
    return (not has_labels(record)) if infer_from_labels else None


class AbstentionEvaluator(E2EEvaluatorBase):
    def __init__(self) -> None:
        self._patterns: tuple[str, ...] = DEFAULT_ABSTAIN_PATTERNS
        self._infer = True

    @property
    def name(self) -> str:
        return "abstention"

    @property
    def description(self) -> str:
        return "Rule: the answer declines exactly when the question is unanswerable."

    def setup(self, config: Any) -> None:
        options = self._options(config)
        self._patterns = tuple(options.get("patterns", DEFAULT_ABSTAIN_PATTERNS))
        self._infer = bool(options.get("infer_from_labels", True))

    def definition(self) -> dict[str, Any]:
        return {
            **super().definition(),
            "patterns": list(self._patterns),
            "infer_from_labels": self._infer,
        }

    def evaluate(self, record: EvaluationRecord) -> EvaluatorResult:
        if is_retrieval_only(record):
            return skip("Retrieval-only record: no answer was generated.")
        if record.is_empty_answer():
            return skip("The system gave no answer (see trace.error); task_success counts it.")
        expected = expected_abstention(record, self._infer)
        if expected is None:
            return skip("No expected_abstain and inference from labels is off.")
        matched = detect_abstention(record.response, self._patterns)
        detected = matched is not None
        outcome = {
            (True, True): "correct abstention",
            (False, False): "answered",
            (True, False): "false abstention",
            (False, True): "missed abstention",
        }
        return EvaluatorResult(
            score=1.0 if detected == expected else 0.0,
            reason=outcome[(detected, expected)].capitalize() + ".",
            metadata={
                "expected_abstain": expected,
                "detected_abstain": detected,
                "matched": matched,
                "outcome": outcome[(detected, expected)],
            },
        )
