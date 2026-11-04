"""Token usage per stage and model, summed over the records of a run."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from rag_eval_framework.core.schemas import EvaluationRecord


def usage_summary(records: list[EvaluationRecord]) -> dict[str, Any]:
    """Input / output tokens by (stage, model); records without usage are counted, not zeroed."""
    totals: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {"input_tokens": 0, "output_tokens": 0, "records": 0}
    )
    with_usage = 0
    for record in records:
        usage = record.trace.usage if record.trace is not None else []
        if usage:
            with_usage += 1
        for u in usage:
            entry = totals[(u.stage, u.model or "unknown")]
            entry["input_tokens"] += u.input_tokens
            entry["output_tokens"] += u.output_tokens
            entry["records"] += 1
    rows = [
        {
            "stage": stage,
            "model": model,
            **v,
            "total_tokens": v["input_tokens"] + v["output_tokens"],
        }
        for (stage, model), v in sorted(totals.items())
    ]
    total_in = sum(v["input_tokens"] for v in totals.values())
    total_out = sum(v["output_tokens"] for v in totals.values())
    return {
        "by_stage_model": rows,
        "input_tokens": total_in,
        "output_tokens": total_out,
        "records_with_usage": with_usage,
        "records_without_usage": len(records) - with_usage,
        "mean_total_tokens_per_record": round((total_in + total_out) / with_usage, 1)
        if with_usage
        else None,
    }
