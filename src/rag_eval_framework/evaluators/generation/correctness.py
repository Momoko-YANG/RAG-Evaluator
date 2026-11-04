"""Correctness — does the answer agree with the ground-truth answer?

* ``accuracy_judge`` — LLM judge comparing the response with ``ground_truth_answer``.
* ``exact_match_accuracy`` — deterministic: the normalised ground truth appears in the
  normalised response (no paraphrases).

**Required dataset fields**: ``question``, ``response``, ``ground_truth_answer``
"""

from __future__ import annotations

import re

from rag_eval_framework.core.schemas import EvaluationRecord
from rag_eval_framework.evaluators.base import BaseEvaluator, EvaluatorResult
from rag_eval_framework.evaluators.common.judge_prompts import ACCURACY_SYSTEM, ACCURACY_USER
from rag_eval_framework.evaluators.common.llm_judge import BaseLLMJudge


def _normalize(text: str) -> str:
    """Lower-case, strip whitespace, collapse runs of whitespace, remove punctuation."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


class ExactMatchAccuracyEvaluator(BaseEvaluator):
    """Case-insensitive normalised comparison of response to ground truth.

    Scoring
    -------
    * **1.0** — normalised response *contains* the normalised ground truth answer.
    * **0.5** — ground truth answer is empty (cannot determine accuracy).
    * **0.0** — normalised ground truth answer is not found in the response.

    This evaluator deliberately uses substring containment (rather than
    strict equality) so that longer responses that include the expected
    answer still score well.  More sophisticated semantic matching should
    be added via LLM-as-a-judge evaluators in Phase 2.
    """

    @property
    def name(self) -> str:
        return "exact_match_accuracy"

    @property
    def description(self) -> str:
        return (
            "Checks whether the normalised ground-truth answer appears in the normalised response."
        )

    def evaluate(self, record: EvaluationRecord) -> EvaluatorResult:
        if not record.ground_truth_answer.strip():
            return EvaluatorResult(
                score=0.5,
                reason="No ground truth answer provided; accuracy indeterminate.",
            )

        norm_response = _normalize(record.response)
        norm_truth = _normalize(record.ground_truth_answer)

        if norm_truth in norm_response:
            return EvaluatorResult(
                score=1.0,
                reason="Ground truth answer found in response.",
            )

        return EvaluatorResult(
            score=0.0,
            reason="Ground truth answer NOT found in response.",
        )


class AccuracyJudge(BaseLLMJudge):
    """LLM judge that evaluates answer accuracy against ground truth."""

    @property
    def name(self) -> str:
        return "accuracy_judge"

    @property
    def description(self) -> str:
        return "LLM-based assessment of answer accuracy against ground truth."

    @property
    def required_fields(self) -> list[str]:
        return ["question", "response", "ground_truth_answer"]

    @property
    def system_prompt(self) -> str:
        return ACCURACY_SYSTEM

    @property
    def user_prompt_template(self) -> str:
        return ACCURACY_USER
