"""Access control: did the run only touch the sources it is allowed to?

Rule-based assertions on what was searched, retrieved and cited::

    access_control:
      allowed_indexes: [index-my-project]   # rag_params.indexName
      allowed_search_services: [my-search-service]
      allowed_title_patterns: []                    # regex; if set, every title must match
      forbidden_title_patterns: ["(?i)confidential-hr"]

A record may also list ``metadata.forbidden_documents`` (titles it must not see).
Score 1.0 when nothing violates a rule, else 0.0 (violations in the metadata).
Without any rule the record is skipped.  This checks the evaluation run's
behaviour; it does not test the application's per-user permission model.
"""

from __future__ import annotations

import re
from typing import Any

from rag_eval_framework.core.results import EvaluatorResult
from rag_eval_framework.core.schemas import EvaluationRecord
from rag_eval_framework.evaluators.e2e.base import E2EEvaluatorBase, skip
from rag_eval_framework.utils.normalisation import normalise_for_match


class AccessControlEvaluator(E2EEvaluatorBase):
    def __init__(self) -> None:
        self._indexes: list[str] = []
        self._services: list[str] = []
        self._allowed: list[re.Pattern[str]] = []
        self._forbidden: list[re.Pattern[str]] = []

    @property
    def name(self) -> str:
        return "access_control"

    @property
    def description(self) -> str:
        return "Rule: only allowed indexes / documents were searched, retrieved and cited."

    def setup(self, config: Any) -> None:
        options = self._options(config)
        self._indexes = list(options.get("allowed_indexes", []))
        self._services = list(options.get("allowed_search_services", []))
        self._allowed = [re.compile(p) for p in options.get("allowed_title_patterns", [])]
        self._forbidden = [re.compile(p) for p in options.get("forbidden_title_patterns", [])]

    def definition(self) -> dict[str, Any]:
        return {
            **super().definition(),
            "allowed_indexes": self._indexes,
            "allowed_search_services": self._services,
            "allowed_title_patterns": [p.pattern for p in self._allowed],
            "forbidden_title_patterns": [p.pattern for p in self._forbidden],
        }

    def evaluate(self, record: EvaluationRecord) -> EvaluatorResult:
        forbidden_docs = {
            normalise_for_match(t) for t in (record.metadata or {}).get("forbidden_documents", [])
        }
        if not (
            self._indexes or self._services or self._allowed or self._forbidden or forbidden_docs
        ):
            return skip("No access rule configured.")
        params = (record.metadata or {}).get("rag_params") or {}
        violations: list[str] = []
        if self._indexes and params.get("indexName") not in self._indexes:
            violations.append(f"index {params.get('indexName')!r} not allowed")
        if self._services and params.get("searchName") not in self._services:
            violations.append(f"search service {params.get('searchName')!r} not allowed")
        titles = [str(d.get("title", "")) for d in record.retrieved_documents]
        if record.trace is not None:
            titles += [c.name for c in record.trace.citations]
        for title in dict.fromkeys(titles):
            if self._allowed and not any(p.search(title) for p in self._allowed):
                violations.append(f"title {title!r} matches no allowed pattern")
            if any(p.search(title) for p in self._forbidden):
                violations.append(f"title {title!r} is forbidden")
            if normalise_for_match(title) in forbidden_docs:
                violations.append(f"title {title!r} is forbidden for this question")
        return EvaluatorResult(
            score=0.0 if violations else 1.0,
            reason="; ".join(violations) + "." if violations else "No access rule violated.",
            metadata={"violations": violations, "checked_titles": len(set(titles))},
        )
