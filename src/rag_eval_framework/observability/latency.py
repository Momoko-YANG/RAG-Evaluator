"""Latency summaries per measured stage.

Stages are nested, not consecutive: ``search_ms`` ⊂ ``total_ms`` (server) ⊂
``client_ms`` (collector, incl. network and retries).  Each stage is summarised on
its own; they are never added together.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable
from typing import Any

from rag_eval_framework.core.schemas import EvaluationRecord

STAGES: dict[str, str] = {
    "client_ms": "Collector: request start to parsed response (network, app, retries).",
    "total_ms": "Server: request start to final answer (inside client_ms).",
    "ttft_ms": "Server: request start to first answer token.",
    "search_ms": "Search tool calls, summed per question (inside total_ms).",
}


def nearest_rank(values: list[float], q: float) -> float:
    """Percentile by the nearest-rank method (``ceil(q * n)``-th smallest value)."""
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


def summarize(values: Iterable[float]) -> dict[str, Any] | None:
    """n / mean / P50 / P95 / max of a set of millisecond values (``None`` if empty)."""
    data = [float(v) for v in values]
    if not data:
        return None
    return {
        "n": len(data),
        "mean_ms": round(statistics.mean(data), 2),
        "p50_ms": round(statistics.median(data), 2),
        "p95_ms": round(nearest_rank(data, 0.95), 2),
        "max_ms": round(max(data), 2),
    }


def stage_values(record: EvaluationRecord) -> dict[str, float]:
    """Measured stages of one record (trace timings; ``eval_latency_ms`` as client time)."""
    values: dict[str, float] = {}
    if record.trace is not None:
        for stage in STAGES:
            value = getattr(record.trace.timings, stage)
            if value is not None:
                values[stage] = float(value)
    client = (record.metadata or {}).get("eval_latency_ms")
    if "client_ms" not in values and isinstance(client, (int, float)):
        values["client_ms"] = float(client)
    return values


def latency_summary(records: list[EvaluationRecord]) -> dict[str, Any]:
    """Per-stage latency summary over the records that measured that stage."""
    by_stage: dict[str, list[float]] = {stage: [] for stage in STAGES}
    for record in records:
        for stage, value in stage_values(record).items():
            by_stage[stage].append(value)
    stages = {stage: s for stage, values in by_stage.items() if (s := summarize(values))}
    return {
        "stages": stages,
        "definitions": {stage: STAGES[stage] for stage in stages},
        "percentile_method": "P50 median; P95 nearest rank (ceil(0.95 n)).",
        "records": len(records),
    }
