"""Query construction: did the model's search text contain the terms it needs?

Rule-based, from the trace.  Score = share of the record's ``expected_query_terms``
found (normalised) in any search text the model wrote; 0.0 when it did not search.
Records without expected terms are skipped.  The queries are kept in the metadata
so that they can be reviewed even where no terms are annotated.
"""

from __future__ import annotations

from rag_eval_framework.core.results import EvaluatorResult
from rag_eval_framework.core.schemas import EvaluationRecord
from rag_eval_framework.evaluators.e2e.base import E2EEvaluatorBase, skip
from rag_eval_framework.utils.normalisation import normalise_for_match


class QueryConstructionEvaluator(E2EEvaluatorBase):
    @property
    def name(self) -> str:
        return "query_construction"

    @property
    def description(self) -> str:
        return "Rule: share of the expected terms present in the model's search text."

    def evaluate(self, record: EvaluationRecord) -> EvaluatorResult:
        if record.trace is None:
            return skip("No trace: collect with the e2e runner.")
        queries = [c.query for c in record.trace.searches() if c.query]
        terms = record.expected_query_terms
        if not terms:
            return skip("No expected_query_terms for this question.", queries=queries)
        if not queries:
            return EvaluatorResult(
                score=0.0,
                reason="The model did not search.",
                metadata={"queries": [], "terms_missing": terms},
            )
        text = " ".join(normalise_for_match(q) for q in queries)
        found = [t for t in terms if normalise_for_match(t) in text]
        missing = [t for t in terms if t not in found]
        return EvaluatorResult(
            score=round(len(found) / len(terms), 4),
            reason=f"{len(found)} of {len(terms)} expected terms in the search text.",
            metadata={
                "queries": queries,
                "terms_found": found,
                "terms_missing": missing,
                "query_is_question": any(
                    normalise_for_match(q) == normalise_for_match(record.question) for q in queries
                ),
            },
        )
