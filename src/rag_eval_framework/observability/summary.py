"""Run-level observability summary: latency per stage, token usage, cost."""

from __future__ import annotations

from typing import Any

from rag_eval_framework.core.schemas import EvaluationRecord
from rag_eval_framework.observability.cost import PricingConfig, cost_summary
from rag_eval_framework.observability.latency import latency_summary
from rag_eval_framework.observability.token_usage import usage_summary


def observability_summary(
    records: list[EvaluationRecord], pricing: PricingConfig | None = None
) -> dict[str, Any]:
    """Empty when the records carry no timing or trace at all (e.g. a toy dataset)."""
    latency = latency_summary(records)
    traced = sum(r.trace is not None for r in records)
    if not latency["stages"] and not traced:
        return {}
    retries = sum(max(0, r.trace.attempts - 1) for r in records if r.trace is not None)
    errors = sum(1 for r in records if r.trace is not None and r.trace.error)
    return {
        "latency": latency,
        "token_usage": usage_summary(records),
        "cost": cost_summary(records, pricing),
        "traced_records": traced,
        "retries": retries,
        "errored_records": errors,
    }
