"""Model cost from token usage and a user-supplied price table.

No prices are built in (they change and differ by contract): configure them with
their source and date.  Cost that cannot be priced is reported as *unknown*, never
as 0.  Fixed monthly costs (e.g. the Azure AI Search service) are listed as such and
not spread over questions, so they are not counted once per run or per question.

    observability:
      pricing:
        currency: USD
        source: "Azure retail price sheet, 2026-10"
        models:
          gpt-4.1: {input_per_1m: 2.0, output_per_1m: 8.0}
          gpt-5-mini: {input_per_1m: 0.25, output_per_1m: 2.0}
        fixed_monthly:
          azure-ai-search-standard: 250.0
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from rag_eval_framework.core.schemas import EvaluationRecord


class ModelPrice(BaseModel):
    input_per_1m: float = Field(..., ge=0)
    output_per_1m: float = Field(..., ge=0)


class PricingConfig(BaseModel):
    currency: str = "USD"
    source: str = Field(..., description="Where the prices come from (sheet / contract and date).")
    models: dict[str, ModelPrice] = Field(default_factory=dict)
    fixed_monthly: dict[str, float] = Field(
        default_factory=dict, description="Reported, never allocated to questions."
    )

    def price_for(self, model: str | None) -> tuple[str, ModelPrice] | None:
        """Exact name, else the longest configured prefix.

        E.g. a ``gpt-5-mini`` price applies to the served model ``gpt-5-mini-2025-08-07``.
        """
        if not model:
            return None
        if model in self.models:
            return model, self.models[model]
        prefixes = [name for name in self.models if model.startswith(name)]
        if not prefixes:
            return None
        name = max(prefixes, key=len)
        return name, self.models[name]


def cost_summary(records: list[EvaluationRecord], pricing: PricingConfig | None) -> dict[str, Any]:
    """Priced cost of the model calls in the records' traces."""
    if pricing is None:
        return {
            "status": "not_configured",
            "note": "No observability.pricing in the config; cost is unknown, not zero.",
        }
    total = 0.0
    priced_records = 0
    by_model: dict[str, float] = {}
    unpriced: dict[str, int] = {}
    without_usage = 0
    for record in records:
        usage = record.trace.usage if record.trace is not None else []
        if not usage:
            without_usage += 1
            continue
        record_priced = True
        for u in usage:
            match = pricing.price_for(u.model)
            if match is None:
                unpriced[u.model or "unknown"] = (
                    unpriced.get(u.model or "unknown", 0) + u.total_tokens
                )
                record_priced = False
                continue
            name, price = match
            cost = (
                u.input_tokens * price.input_per_1m + u.output_tokens * price.output_per_1m
            ) / 1e6
            by_model[name] = by_model.get(name, 0.0) + cost
            total += cost
        priced_records += record_priced
    return {
        "status": "complete" if not unpriced and not without_usage else "partial",
        "currency": pricing.currency,
        "source": pricing.source,
        "priced_total": round(total, 6),
        "by_model": {k: round(v, 6) for k, v in sorted(by_model.items())},
        "mean_per_priced_record": round(total / priced_records, 6) if priced_records else None,
        "fully_priced_records": priced_records,
        "records_without_usage": without_usage,
        "unpriced_tokens_by_model": unpriced,
        "fixed_monthly_not_allocated": pricing.fixed_monthly,
        "not_included": "Embedding of the question and Search queries (not metered per request), "
        "LLM-judge calls of the evaluation itself.",
    }
