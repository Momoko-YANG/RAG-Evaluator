"""Tool selection: did the model call the tools it should (and none it must not)?

Rule-based, from the trace.  Expected tools come from the record
(``expected_tools``) or, for every record, the ``expected_tools`` option::

    tool_selection:
      expected_tools: [aisearch]       # default when a record has none
      expect_search: true              # or: any search call (a call with search text) counts
      forbidden_tools: []
      allow_extra: true                # other calls (e.g. propose_remember) do not fail

Score 1.0 when every expected tool was called, no forbidden tool was called and
(with ``allow_extra: false``) nothing else was called; else 0.0.
"""

from __future__ import annotations

from typing import Any

from rag_eval_framework.core.results import EvaluatorResult
from rag_eval_framework.core.schemas import EvaluationRecord
from rag_eval_framework.evaluators.e2e.base import E2EEvaluatorBase, skip


class ToolSelectionEvaluator(E2EEvaluatorBase):
    def __init__(self) -> None:
        self._expected: list[str] = []
        self._forbidden: list[str] = []
        self._allow_extra = True
        self._expect_search = False

    @property
    def name(self) -> str:
        return "tool_selection"

    @property
    def description(self) -> str:
        return "Rule: the model called the expected tools and no forbidden one (from the trace)."

    def setup(self, config: Any) -> None:
        options = self._options(config)
        self._expected = list(options.get("expected_tools", []))
        self._forbidden = list(options.get("forbidden_tools", []))
        self._allow_extra = bool(options.get("allow_extra", True))
        self._expect_search = bool(options.get("expect_search", False))

    def definition(self) -> dict[str, Any]:
        return {
            **super().definition(),
            "default_expected_tools": self._expected,
            "forbidden_tools": self._forbidden,
            "allow_extra": self._allow_extra,
            "expect_search": self._expect_search,
        }

    def evaluate(self, record: EvaluationRecord) -> EvaluatorResult:
        if record.trace is None:
            return skip("No trace: collect with the e2e runner.")
        expected = record.expected_tools or self._expected
        if not expected and not self._expect_search:
            return skip("No expected tools for this question.")
        called = [c.name for c in record.trace.tool_calls]
        missing = [t for t in expected if t not in called]
        searched = [c.name for c in record.trace.searches()]
        if self._expect_search and not expected:
            expected = sorted(set(searched)) or ["<search>"]
            missing = [] if searched else ["<search>"]
        forbidden = [t for t in called if t in self._forbidden]
        extra = sorted({t for t in called if t not in expected})
        ok = not missing and not forbidden and (self._allow_extra or not extra)
        if ok:
            reason = f"Called {', '.join(called)}."
        else:
            problems = [
                f"missing {missing}" if missing else "",
                f"forbidden {forbidden}" if forbidden else "",
                f"unexpected {extra}" if extra and not self._allow_extra else "",
            ]
            reason = "; ".join(p for p in problems if p) + "."
        return EvaluatorResult(
            score=1.0 if ok else 0.0,
            reason=reason,
            metadata={
                "expected": expected,
                "called": called,
                "missing": missing,
                "forbidden_called": forbidden,
                "extra": extra,
            },
        )
