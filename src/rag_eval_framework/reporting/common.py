"""Helpers shared by all report renderers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rag_eval_framework.config.models import (
    CLOUD_EVALUATOR_CATALOG,
    EVALUATOR_LAYERS,
    LAYERS,
    LOCAL_EVALUATOR_CATALOG,
)
from rag_eval_framework.core.aggregation import PER_K_METRICS, summarize_per_k
from rag_eval_framework.evaluators.base import EvaluatorStatus
from rag_eval_framework.runners.base import EvaluationRunResult, RecordResult

MODE_DESCRIPTIONS: dict[str, str] = {
    "local": (
        "Evaluators ran in-process on this machine "
        "(scaffold + Azure AI Evaluation SDK + LLM judges)."
    ),
    "cloud": "Evaluators ran remotely via Azure AI Foundry (Azure AI Evaluation SDK).",
}


@dataclass(frozen=True)
class DatasetStats:
    total_records: int
    passed_records: int
    failed_records: int
    skipped_pairs: int


def report_path(output_dir: Path, project_name: str, extension: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"eval-report-{project_name}.{extension}"


def mode_label(result: EvaluationRunResult) -> str:
    return result.runner_type.upper()


def mode_description(result: EvaluationRunResult) -> str:
    return MODE_DESCRIPTIONS.get(result.runner_type, "")


def evaluator_descriptions(result: EvaluationRunResult) -> dict[str, str]:
    """Return a description for every evaluator used, preferring the active mode's catalog."""
    primary, secondary = (
        (CLOUD_EVALUATOR_CATALOG, LOCAL_EVALUATOR_CATALOG)
        if result.runner_type == "cloud"
        else (LOCAL_EVALUATOR_CATALOG, CLOUD_EVALUATOR_CATALOG)
    )
    return {
        name: primary.get(name) or secondary.get(name) or "Custom evaluator."
        for name in result.evaluators_used
    }


def dataset_stats(result: EvaluationRunResult) -> DatasetStats:
    passed = sum(1 for rr in result.record_results if rr.passed)
    skipped = sum(
        1
        for rr in result.record_results
        for er in rr.evaluator_results.values()
        if er.status == EvaluatorStatus.SKIPPED
    )
    return DatasetStats(
        total_records=result.total_records,
        passed_records=passed,
        failed_records=len(result.record_results) - passed,
        skipped_pairs=skipped,
    )


def failed_records(result: EvaluationRunResult, limit: int | None = None) -> list[RecordResult]:
    """Records that did not meet their thresholds (uses ``RecordResult.passed``)."""
    failed = [rr for rr in result.record_results if not rr.passed]
    return failed if limit is None else failed[:limit]


def evaluator_layer(name: str) -> str:
    """Layer of an evaluator (``retrieval`` / ``generation``; ``other`` for custom ones)."""
    return EVALUATOR_LAYERS.get(name, "other")


def _layer_order(name: str) -> tuple[int, str]:
    layer = evaluator_layer(name)
    return (LAYERS.index(layer) if layer in LAYERS else len(LAYERS), name)


def score_text(score: float | None) -> str:
    """A mean score for display; ``n/a`` when the evaluator scored no record."""
    return f"{score:.4f}" if score is not None else "n/a"


def threshold_rows(
    result: EvaluationRunResult,
) -> list[tuple[str, float | None, float | None, str]]:
    """Rows of (evaluator, score or None, threshold, status), ordered by layer then name."""
    thresholds = {tr.evaluator: tr for tr in result.threshold_results}
    breaches = {b.evaluator: b for b in result.threshold_breaches}
    rows: list[tuple[str, float | None, float | None, str]] = []
    names = list(dict.fromkeys([*result.evaluators_used, *result.aggregate_scores]))
    for name in sorted(names, key=_layer_order):
        score = result.aggregate_scores.get(name)
        if name in thresholds:
            tr = thresholds[name]
            rows.append((name, score, tr.threshold, "PASS" if tr.passed else "FAIL"))
        elif name in breaches:
            rows.append((name, score, breaches[name].threshold, "FAIL"))
        else:
            rows.append((name, score, None, "—"))
    return rows


LABELS = {"recall": "Recall", "ndcg": "nDCG"}
_DEFINITION_KEYS_SHOWN = ("evaluator", "version", "layer", "relevance_policy")


def per_k_summaries(result: EvaluationRunResult) -> list[dict]:  # type: ignore[type-arg]
    """Summaries of the per-k evaluators used in the run (Recall@k, nDCG@k)."""
    summaries = [
        summarize_per_k(result, name) for name in result.evaluators_used if name in PER_K_METRICS
    ]
    return [s for s in summaries if s is not None]


def per_k_title(summary: dict) -> str:  # type: ignore[type-arg]
    return f"{LABELS.get(summary['metric'], summary['metric'])}@k"


def per_k_rows(summary: dict) -> list[tuple[str, str, int, int, str]]:  # type: ignore[type-arg]
    """Rows of (metric, macro value, eligible, not available, by judgment kind)."""
    label = LABELS.get(summary["metric"], summary["metric"])
    rows = []
    for k, item in summary["per_k"].items():
        value = "N/A" if item["macro"] is None else f"{item['macro']:.4f}"
        kinds = "; ".join(
            f"{kind} {v['macro']:.4f} (n={v['questions']})" for kind, v in item["by_kind"].items()
        )
        rows.append(
            (f"{label}@{k}", value, item["eligible_questions"], item["not_available"], kinds)
        )
    return rows


def per_k_notes(summary: dict) -> list[str]:  # type: ignore[type-arg]
    """Relevance policy, judgments and reasons for missing values, one sentence each."""
    label = LABELS.get(summary["metric"], summary["metric"])
    j = summary["relevance_judgments"]
    if j["relevance_policy"] == "qrels":
        source = (
            f"Relevance policy qrels: {j['qrels_path']} "
            f"(versions: {', '.join(j['annotation_versions']) or '—'}); "
            f"relevant = grade >= {j['relevance_threshold']}; "
            f"require_reviewed = {j['require_reviewed']}."
        )
    else:
        source = f"Relevance policy {j['relevance_policy']} (title + evidence-quote labels)."
    notes = [
        source,
        f"Judgment scopes: {j['judgment_scopes']}; annotation status: {j['annotation_status']}.",
        f"Questions: {j['total_questions']} total, {j['eligible_questions']} with judgments, "
        f"{j['excluded_questions']} excluded {j['excluded_reasons'] or ''}".rstrip()
        + ".",
    ]
    for kind, text in summary["judgment_kind_labels"].items():
        notes.append(f"{kind}: {text}.")
    for k, item in summary["per_k"].items():
        for reason, count in item["not_available_reasons"].items():
            notes.append(f"{label}@{k} not available for {count} question(s): {reason}.")
    return notes


def definition_rows(result: EvaluationRunResult) -> list[tuple[str, str, str, str, str]]:
    """Rows of (layer, evaluator, version, relevance policy, other settings)."""
    rows = []
    for name in sorted(result.metric_definitions, key=_layer_order):
        d = result.metric_definitions[name]
        details = "; ".join(
            f"{k}={v}"
            for k, v in d.items()
            if k not in _DEFINITION_KEYS_SHOWN and v not in (None, "", [], {})
        )
        rows.append(
            (
                str(d.get("layer", "")),
                name,
                str(d.get("version", "")),
                str(d.get("relevance_policy") or "—"),
                details,
            )
        )
    return rows


def provenance_lines(result: EvaluationRunResult) -> list[str]:
    p = result.provenance
    if not p:
        return []
    dataset = p.get("dataset") or {}
    lines = [
        f"Framework version: {p.get('framework_version')}",
        f"Dataset: {dataset.get('path')} (sha256 {dataset.get('sha256')}, "
        f"{dataset.get('records')} records)",
    ]
    for setting in p.get("retrieval_settings") or []:
        lines.append("Retrieval: " + ", ".join(f"{k}={v}" for k, v in setting.items()))
    if p.get("truncated_retrieval_depths"):
        lines.append(f"Truncated copies (retrieval_depth): {p['truncated_retrieval_depths']}")
    return lines


def latency_rows(result: EvaluationRunResult) -> list[tuple[str, int, str, str, str, str, str]]:
    """Rows of (stage, n, mean, p50, p95, max, definition) in milliseconds."""
    latency = (result.observability or {}).get("latency") or {}
    rows = []
    for stage, s in (latency.get("stages") or {}).items():
        rows.append(
            (
                stage,
                s["n"],
                f"{s['mean_ms']:.1f}",
                f"{s['p50_ms']:.1f}",
                f"{s['p95_ms']:.1f}",
                f"{s['max_ms']:.1f}",
                (latency.get("definitions") or {}).get(stage, ""),
            )
        )
    return rows


def token_rows(result: EvaluationRunResult) -> list[tuple[str, str, int, int, int, int]]:
    """Rows of (stage, model, records, input, output, total) tokens."""
    usage = (result.observability or {}).get("token_usage") or {}
    return [
        (
            r["stage"],
            r["model"],
            r["records"],
            r["input_tokens"],
            r["output_tokens"],
            r["total_tokens"],
        )
        for r in usage.get("by_stage_model") or []
    ]


def observability_notes(result: EvaluationRunResult) -> list[str]:
    """Cost and coverage, one sentence each (cost that cannot be priced is not shown as 0)."""
    obs = result.observability or {}
    if not obs:
        return []
    notes = []
    usage = obs.get("token_usage") or {}
    notes.append(
        f"Token usage recorded for {usage.get('records_with_usage', 0)} record(s); "
        f"{usage.get('records_without_usage', 0)} without."
    )
    cost = obs.get("cost") or {}
    if cost.get("status") == "not_configured":
        notes.append(f"Cost: {cost.get('note')}")
    elif cost:
        notes.append(
            f"Cost ({cost['status']}): {cost['priced_total']} {cost['currency']} priced "
            f"(by model {cost['by_model']}; source: {cost['source']})."
        )
        if cost.get("unpriced_tokens_by_model"):
            notes.append(f"Unpriced tokens (cost unknown): {cost['unpriced_tokens_by_model']}.")
        if cost.get("fixed_monthly_not_allocated"):
            notes.append(f"Fixed monthly, not allocated: {cost['fixed_monthly_not_allocated']}.")
        notes.append(f"Not included: {cost['not_included']}")
    notes.append(
        f"Retries: {obs.get('retries', 0)}; errored records: {obs.get('errored_records', 0)}."
    )
    return notes
