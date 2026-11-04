"""Markdown report renderer."""

from __future__ import annotations

from pathlib import Path

from rag_eval_framework.reporting.common import (
    dataset_stats,
    definition_rows,
    evaluator_descriptions,
    evaluator_layer,
    failed_records,
    latency_rows,
    mode_description,
    mode_label,
    observability_notes,
    per_k_notes,
    per_k_rows,
    per_k_summaries,
    per_k_title,
    provenance_lines,
    report_path,
    score_text,
    threshold_rows,
    token_rows,
)
from rag_eval_framework.runners.base import EvaluationRunResult

MAX_FAILED_RECORDS = 20


def _cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_markdown_report(result: EvaluationRunResult, output_dir: Path) -> Path:
    """Write a human-readable Markdown report and return the file path."""
    path = report_path(Path(output_dir), result.project_name, "md")
    stats = dataset_stats(result)
    status = "PASSED" if result.passed else "FAILED"

    lines: list[str] = [
        f"# Evaluation Report — {result.project_name}",
        "",
        f"> **Evaluation Mode: {mode_label(result)}** — {mode_description(result)}",
        "",
        "## Summary",
        "",
        f"- **Status**: {status}",
        f"- **Runner**: {result.runner_type}",
        f"- **Timestamp**: {result.timestamp}",
        f"- **Dataset**: `{result.dataset_path}`",
        f"- **Records**: {stats.total_records} "
        f"(passed {stats.passed_records}, failed {stats.failed_records})",
        f"- **Evaluators**: {', '.join(result.evaluators_used)}",
        "",
        "## Dataset Statistics",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Total records | {stats.total_records} |",
        f"| Passed records | {stats.passed_records} |",
        f"| Failed records | {stats.failed_records} |",
        f"| Skipped evaluator/record pairs | {stats.skipped_pairs} |",
        "",
        "## Evaluator Summary",
        "",
        "| Layer | Evaluator | Mean score | Threshold | Status |",
        "|---|---|---:|---:|:---:|",
    ]
    for name, score, threshold, row_status in threshold_rows(result):
        thr = f"{threshold:.2f}" if threshold is not None else "—"
        layer, mean = evaluator_layer(name), score_text(score)
        lines.append(f"| {layer} | {name} | {mean} | {thr} | {row_status} |")

    for summary in per_k_summaries(result):
        lines += [
            "",
            f"## {per_k_title(summary)}",
            "",
            "| Metric | Macro | Eligible | Not available | By judgment kind |",
            "|---|---:|---:|---:|---|",
        ]
        for metric, value, eligible, missing, kinds in per_k_rows(summary):
            lines.append(f"| {metric} | {value} | {eligible} | {missing} | {_cell(kinds)} |")
        lines += [""] + [f"- {_cell(note)}" for note in per_k_notes(summary)]

    if result.metric_definitions:
        lines += [
            "",
            "## Metric Definitions",
            "",
            "| Layer | Evaluator | Version | Relevance policy | Settings |",
            "|---|---|---|---|---|",
        ]
        for layer, name, version, policy, details in definition_rows(result):
            lines.append(f"| {layer} | {name} | {version} | {policy} | {_cell(details)} |")
    provenance = provenance_lines(result)
    if provenance:
        lines += ["", "## Provenance", ""] + [f"- {_cell(line)}" for line in provenance]

    if result.observability:
        lines += ["", "## Observability", ""]
        latency = latency_rows(result)
        if latency:
            lines += [
                "| Stage | n | Mean (ms) | P50 (ms) | P95 (ms) | Max (ms) | Measures |",
                "|---|---:|---:|---:|---:|---:|---|",
            ]
            for stage, n, mean, p50, p95, mx, what in latency:
                lines.append(f"| {stage} | {n} | {mean} | {p50} | {p95} | {mx} | {_cell(what)} |")
            lines.append("")
        tokens = token_rows(result)
        if tokens:
            lines += [
                "| Stage | Model | Records | Input tokens | Output tokens | Total |",
                "|---|---|---:|---:|---:|---:|",
            ]
            for stage, model, n, tin, tout, total in tokens:
                lines.append(f"| {stage} | {model} | {n} | {tin} | {tout} | {total} |")
            lines.append("")
        lines += [f"- {_cell(note)}" for note in observability_notes(result)]

    lines += ["", "## Metric Descriptions", ""]
    for name, desc in evaluator_descriptions(result).items():
        lines.append(f"- **{name}**: {desc}")

    if result.threshold_breaches:
        lines += [
            "",
            "## Threshold Breaches",
            "",
            "| Evaluator | Threshold | Actual | Delta |",
            "|---|---:|---:|---:|",
        ]
        for b in result.threshold_breaches:
            lines.append(f"| {b.evaluator} | {b.threshold:.4f} | {b.actual:.4f} | {b.delta:.4f} |")

    failed = failed_records(result, MAX_FAILED_RECORDS)
    if failed:
        lines += ["", "## Sample Failed Records", ""]
        for rr in failed:
            lines.append(f"- **{rr.record_id}**")
            for ev_name, er in rr.evaluator_results.items():
                lines.append(
                    f"  - {ev_name}: {er.score:.4f} ({er.status.value}) — {_cell(er.reason)}"
                )

    if result.skipped_evaluators:
        lines += ["", "## Skipped Evaluators", ""]
        for name, reason in result.skipped_evaluators.items():
            lines.append(f"- **{name}**: {reason}")

    if result.run_metadata or result.config_reference:
        lines += ["", "## Run Metadata", ""]
        for key, value in result.run_metadata.items():
            lines.append(f"- **{key}**: {_cell(value)}")
        if result.config_reference:
            lines.append(f"- **config_reference**: `{result.config_reference}`")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
