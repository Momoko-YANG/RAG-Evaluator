"""Aggregate per-record evaluator results into run-level scores.

Aggregation only summarises what the evaluators returned; it never re-derives a
metric.  Per-k metrics (Recall@k, nDCG@k) keep one value per k in each result's
``metadata.per_k``; :func:`summarize_per_k` averages them over the questions where
the value is available (macro average).
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from rag_eval_framework.evaluators.base import EvaluatorStatus
from rag_eval_framework.evaluators.retrieval.base import AVAILABLE
from rag_eval_framework.evaluators.retrieval.relevance.judgments import JUDGMENT_KIND_LABELS

# Evaluators that report one value per k, and the metric name used in reports.
PER_K_METRICS = {"retrieval_recall": "recall", "retrieval_ndcg": "ndcg"}


def mean_scores(record_results: list[Any], evaluator_names: list[str]) -> dict[str, float]:
    """Mean score per evaluator over its SUCCESS results (skipped / errors excluded).

    An evaluator without any SUCCESS result has no mean and is left out (reports show
    it as n/a), so "nothing to score" is never reported as a score of 0.
    """
    scores: dict[str, list[float]] = {name: [] for name in evaluator_names}
    for rr in record_results:
        for name in evaluator_names:
            er = rr.evaluator_results.get(name)
            if er is not None and er.status == EvaluatorStatus.SUCCESS:
                scores[name].append(er.score)
    return {name: round(sum(vals) / len(vals), 4) for name, vals in scores.items() if vals}


def _rows(result: Any, name: str) -> list[Any]:
    return [
        rr.evaluator_results[name] for rr in result.record_results if name in rr.evaluator_results
    ]


def summarize_per_k(result: Any, name: str) -> dict[str, Any] | None:
    """Macro average per k and per judgment kind of a per-k evaluator, or ``None``."""
    rows = _rows(result, name)
    if not rows or name not in PER_K_METRICS:
        return None
    metric = PER_K_METRICS[name]
    definition: dict[str, Any] = (getattr(result, "metric_definitions", None) or {}).get(
        name
    ) or next((r.metadata for r in rows if r.metadata.get("ks")), {})
    ks: list[int] = definition.get("ks", [])
    per_k: dict[str, dict[str, Any]] = {}
    for k in ks:
        values: dict[str, list[float]] = {}
        unavailable: Counter[str] = Counter()
        for r in rows:
            item = (r.metadata.get("per_k") or {}).get(str(k))
            if item is None:
                continue
            if item["status"] == AVAILABLE:
                values.setdefault(r.metadata["judgment_kind"], []).append(item["value"])
            else:
                unavailable[item["reason"]] += 1
        eligible = sum(len(v) for v in values.values())
        per_k[str(k)] = {
            "macro": round(sum(sum(v) for v in values.values()) / eligible, 4)
            if eligible
            else None,
            "eligible_questions": eligible,
            "not_available": sum(unavailable.values()),
            "not_available_reasons": dict(unavailable),
            "by_kind": {
                kind: {"macro": round(sum(v) / len(v), 4), "questions": len(v)}
                for kind, v in sorted(values.items())
            },
        }
    excluded = Counter(
        r.metadata.get("excluded_reason") for r in rows if r.metadata.get("excluded_reason")
    )
    judged = [r.metadata for r in rows if "per_k" in r.metadata]
    annotated = [m for m in judged if m.get("annotation_status")]
    kinds = Counter(m["judgment_kind"] for m in judged)
    return {
        "metric": metric,
        "evaluator": name,
        "values": {f"{metric}_at_{k}": per_k[str(k)]["macro"] for k in ks},
        "per_k": per_k,
        "judgment_kinds": dict(kinds),
        "judgment_kind_labels": {kind: JUDGMENT_KIND_LABELS[kind] for kind in kinds},
        "relevance_judgments": {
            "relevance_policy": definition.get("relevance_policy"),
            "qrels_path": definition.get("qrels_path"),
            "qrels_sha256": definition.get("qrels_sha256"),
            "annotation_versions": sorted({m["annotation_version"] for m in annotated}),
            "relevance_threshold": definition.get("relevance_threshold"),
            "require_reviewed": definition.get("require_reviewed"),
            "judgment_scopes": dict(Counter(m["judgment_scope"] for m in annotated)),
            "annotation_status": dict(Counter(m["annotation_status"] for m in annotated)),
            "total_questions": len(rows),
            "eligible_questions": len(judged),
            "excluded_questions": sum(excluded.values()),
            "excluded_reasons": dict(excluded),
        },
    }


def metric_results(result: Any) -> list[dict[str, Any]]:
    """One traceable entry per reported metric value (per k for per-k metrics).

    Each entry names the metric, its value and status, how many questions it covers,
    and the definition behind it (evaluator version, relevance policy, label source).
    """
    definitions = getattr(result, "metric_definitions", None) or {}
    framework = (getattr(result, "provenance", None) or {}).get("framework_version")
    entries: list[dict[str, Any]] = []
    for name in result.evaluators_used:
        definition = definitions.get(name, {})
        trace = {
            "evaluator": name,
            "metric_version": definition.get("version"),
            "framework_version": framework,
            "relevance_policy": definition.get("relevance_policy"),
            "label_source": definition.get("qrels_path")
            and f"{definition.get('qrels_path')} "
            f"(sha256 {str(definition.get('qrels_sha256'))[:12]})",
        }
        summary = summarize_per_k(result, name)
        if summary is not None:
            for k, item in summary["per_k"].items():
                entries.append(
                    {
                        "metric": f"{summary['metric']}@{k}",
                        "score": item["macro"],
                        "status": "valid" if item["eligible_questions"] else "not_available",
                        "eligible_questions": item["eligible_questions"],
                        "not_available": item["not_available"],
                        "judgment_kinds": {
                            kind: v["questions"] for kind, v in item["by_kind"].items()
                        },
                        **trace,
                    }
                )
            continue
        statuses = Counter(r.status.value for r in _rows(result, name))
        k = definition.get("k")
        entries.append(
            {
                "metric": f"{name}@{k}" if k else name,
                "score": result.aggregate_scores.get(name) if statuses.get("success") else None,
                "status": "valid" if statuses.get("success") else "not_available",
                "eligible_questions": statuses.get("success", 0),
                "skipped": statuses.get("skipped", 0),
                "errors": statuses.get("error", 0),
                **trace,
            }
        )
    return entries
