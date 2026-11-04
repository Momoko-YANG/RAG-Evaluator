"""Answer presence evaluator — checks whether the response is non-trivial.

A deterministic check that *some* answer was given (not whether it is correct,
grounded or an abstention — see correctness / faithfulness / abstention).

Version 2.0: text containing CJK characters (Japanese, Chinese) is measured in
characters, because it has no spaces between words — version 1.0 counted
whitespace-separated words, so 「20日です。」 was one word and scored 0.
"""

from __future__ import annotations

import re

from rag_eval_framework.core.schemas import EvaluationRecord
from rag_eval_framework.evaluators.base import BaseEvaluator, EvaluatorResult

# Minimum size to consider a response substantive
_MIN_WORD_COUNT = 2
_MIN_CJK_CHARS = 4
_CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿豈-﫿ｦ-ﾟ]")
_PUNCTUATION_OR_SPACE = re.compile(r"[\s\W_]", re.UNICODE)


class AnswerPresenceEvaluator(BaseEvaluator):
    """Deterministic check: does the response contain a substantive answer?

    Scoring
    -------
    * **1.0** — at least ``_MIN_WORD_COUNT`` words, or (text with CJK characters) at
      least ``_MIN_CJK_CHARS`` characters other than spaces and punctuation.
    * **0.0** — response is missing or too short.
    """

    @property
    def name(self) -> str:
        return "answer_presence"

    @property
    def description(self) -> str:
        return "Checks whether the response contains a substantive answer."

    @property
    def version(self) -> str:
        return "2.0"

    def evaluate(self, record: EvaluationRecord) -> EvaluatorResult:
        text = record.response.strip()
        if _CJK.search(text):
            chars = len(_PUNCTUATION_OR_SPACE.sub("", text))
            ok = chars >= _MIN_CJK_CHARS
            return EvaluatorResult(
                score=1.0 if ok else 0.0,
                reason=f"Response contains {chars} character(s)"
                + (", meets minimum." if ok else f", below minimum of {_MIN_CJK_CHARS}."),
            )
        words = text.split()
        if len(words) >= _MIN_WORD_COUNT:
            return EvaluatorResult(
                score=1.0,
                reason=f"Response contains {len(words)} word(s), meets minimum.",
            )
        return EvaluatorResult(
            score=0.0,
            reason=(
                f"Response contains only {len(words)} word(s), below minimum of {_MIN_WORD_COUNT}."
            ),
        )
