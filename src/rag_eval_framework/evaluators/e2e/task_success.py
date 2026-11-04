"""Task success: did the user get what they asked for, end to end?

The score comes from the first available source, recorded as ``source``:

1. ``human`` — a reviewer's label in ``metadata.human_task_success`` (true/false/0–1).
2. ``llm_judge`` — with ``use_llm_judge: true``: the run has no error, the answer
   neither abstains wrongly nor answers an unanswerable question, and
   ``accuracy_judge`` scores it ≥ ``judge_threshold`` (default 0.75 = 4/5).
3. ``rule`` — the run has no error, the abstention is correct, and (answerable
   questions) the answer cites a retrieved chunk that matches the ground-truth labels
   (title + evidence quote).  ``require_citation: false`` drops the citation check.

``task_success`` is a strict, explainable check: each condition is kept in
``metadata.checks``.  The rule source cannot tell a correct paraphrase from a wrong
answer that cites the right chunk; use human labels or the judge for that.
"""

from __future__ import annotations

from typing import Any

from rag_eval_framework.core.results import EvaluatorResult, EvaluatorStatus
from rag_eval_framework.core.schemas import EvaluationRecord
from rag_eval_framework.evaluators.e2e.abstention import expected_abstention
from rag_eval_framework.evaluators.e2e.base import (
    DEFAULT_ABSTAIN_PATTERNS,
    E2EEvaluatorBase,
    detect_abstention,
    is_retrieval_only,
    skip,
)
from rag_eval_framework.evaluators.generation.correctness import AccuracyJudge
from rag_eval_framework.evaluators.retrieval.relevance.matcher import is_relevant_document


def _human_label(value: Any) -> float | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)) and 0 <= value <= 1:
        return float(value)
    if isinstance(value, str) and value.strip().lower() in {"true", "false", "1", "0"}:
        return 1.0 if value.strip().lower() in {"true", "1"} else 0.0
    return None


class TaskSuccessEvaluator(E2EEvaluatorBase):
    def __init__(self) -> None:
        self._use_judge = False
        self._threshold = 0.75
        self._require_citation = True
        self._patterns: tuple[str, ...] = DEFAULT_ABSTAIN_PATTERNS
        self._judge: AccuracyJudge | None = None

    @property
    def name(self) -> str:
        return "task_success"

    @property
    def description(self) -> str:
        return "Human label, else LLM judge, else rule: correct abstention + cited relevant chunk."

    def setup(self, config: Any) -> None:
        options = self._options(config)
        self._use_judge = bool(options.get("use_llm_judge", False))
        self._threshold = float(options.get("judge_threshold", 0.75))
        self._require_citation = bool(options.get("require_citation", True))
        self._patterns = tuple(options.get("abstain_patterns", DEFAULT_ABSTAIN_PATTERNS))
        if self._use_judge:
            self._judge = AccuracyJudge()
            self._judge.setup(config)

    def definition(self) -> dict[str, Any]:
        definition = {
            **super().definition(),
            "use_llm_judge": self._use_judge,
            "require_citation": self._require_citation,
        }
        if self._judge is not None:
            definition |= {"judge_threshold": self._threshold, "judge": self._judge.definition()}
        return definition

    def evaluate(self, record: EvaluationRecord) -> EvaluatorResult:
        if is_retrieval_only(record):
            return skip("Retrieval-only record: no answer was generated.")
        human = _human_label((record.metadata or {}).get("human_task_success"))
        if human is not None:
            return EvaluatorResult(score=human, reason="Human label.", metadata={"source": "human"})

        checks: dict[str, bool] = {}
        error = record.trace.error if record.trace is not None else None
        checks["no_error"] = not error and not (record.metadata or {}).get("collection_error")
        expected = expected_abstention(record, infer_from_labels=True)
        abstained = detect_abstention(record.response, self._patterns) is not None
        checks["abstention_correct"] = abstained == bool(expected)

        source = "rule"
        if not expected:
            if self._judge is not None:
                judged = self._judge.evaluate(record)
                if judged.status != EvaluatorStatus.SUCCESS:
                    return EvaluatorResult(
                        score=0.0,
                        status=judged.status,
                        reason=f"Judge: {judged.reason}",
                        metadata={"source": "llm_judge", "checks": checks},
                    )
                checks["judge_score_at_threshold"] = judged.score >= self._threshold
                source = "llm_judge"
            elif self._require_citation:
                checks["cites_relevant_chunk"] = self._cites_relevant_chunk(record)

        failed = [name for name, ok in checks.items() if not ok]
        return EvaluatorResult(
            score=0.0 if failed else 1.0,
            reason=f"Failed: {', '.join(failed)}." if failed else "All checks passed.",
            metadata={"source": source, "checks": checks},
        )

    @staticmethod
    def _cites_relevant_chunk(record: EvaluationRecord) -> bool:
        if record.trace is None or not record.trace.citations:
            return False
        by_id = {str(d.get("id")): d for d in record.retrieved_documents}
        return any(
            c.id in by_id and is_relevant_document(record, by_id[c.id])
            for c in record.trace.citations
        )
