"""Self-contained HTML report renderer (inline CSS, no external assets)."""

from __future__ import annotations

from html import escape
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

_CSS = """
body{font-family:Segoe UI,Meiryo,sans-serif;margin:2rem;color:#222}
h1{margin-bottom:.25rem}h2{margin-top:2rem;border-bottom:1px solid #ddd}
table{border-collapse:collapse;margin:.5rem 0}
th,td{border:1px solid #ccc;padding:.35rem .6rem;text-align:left;vertical-align:top}
th{background:#f4f4f4}.num{text-align:right}
.badge{display:inline-block;padding:.2rem .6rem;border-radius:4px;color:#fff;font-weight:bold}
.pass{background:#2e7d32}.fail{background:#c62828}.na{background:#757575}
.banner{background:#e3f2fd;border-left:4px solid #1565c0;padding:.6rem 1rem;margin:1rem 0}
"""


def _e(value: object) -> str:
    return escape(str(value))


def _badge(status: str) -> str:
    css = {"PASS": "pass", "PASSED": "pass", "FAIL": "fail", "FAILED": "fail"}.get(status, "na")
    return f'<span class="badge {css}">{_e(status)}</span>'


def render_html_report(result: EvaluationRunResult, output_dir: Path) -> Path:
    """Write a self-contained HTML report and return the file path."""
    path = report_path(Path(output_dir), result.project_name, "html")
    stats = dataset_stats(result)
    status = "PASSED" if result.passed else "FAILED"

    parts: list[str] = [
        "<!DOCTYPE html>",
        '<html lang="en"><head><meta charset="utf-8">',
        f"<title>Evaluation Report — {_e(result.project_name)}</title>",
        f"<style>{_CSS}</style></head><body>",
        f"<h1>Evaluation Report — {_e(result.project_name)}</h1>",
        f'<div class="banner"><strong>Evaluation Mode: {_e(mode_label(result))}</strong>'
        f" — {_e(mode_description(result))}</div>",
        "<h2>Summary</h2><table>",
        f"<tr><th>Status</th><td>{_badge(status)}</td></tr>",
        f"<tr><th>Runner</th><td>{_e(result.runner_type)}</td></tr>",
        f"<tr><th>Timestamp</th><td>{_e(result.timestamp)}</td></tr>",
        f"<tr><th>Dataset</th><td><code>{_e(result.dataset_path)}</code></td></tr>",
        f"<tr><th>Evaluators</th><td>{_e(', '.join(result.evaluators_used))}</td></tr>",
        "</table>",
        "<h2>Dataset Statistics</h2><table>",
        f'<tr><th>Total records</th><td class="num">{stats.total_records}</td></tr>',
        f'<tr><th>Passed records</th><td class="num">{stats.passed_records}</td></tr>',
        f'<tr><th>Failed records</th><td class="num">{stats.failed_records}</td></tr>',
        "<tr><th>Skipped evaluator/record pairs</th>"
        f'<td class="num">{stats.skipped_pairs}</td></tr>',
        "</table>",
        "<h2>Evaluator Summary</h2><table>",
        "<tr><th>Layer</th><th>Evaluator</th><th>Mean score</th><th>Threshold</th>"
        "<th>Status</th></tr>",
    ]
    for name, score, threshold, row_status in threshold_rows(result):
        thr = f"{threshold:.2f}" if threshold is not None else "—"
        parts.append(
            f"<tr><td>{_e(evaluator_layer(name))}</td>"
            f'<td>{_e(name)}</td><td class="num">{score_text(score)}</td>'
            f'<td class="num">{thr}</td><td>{_badge(row_status)}</td></tr>'
        )
    parts.append("</table>")

    for summary in per_k_summaries(result):
        parts += [
            f"<h2>{_e(per_k_title(summary))}</h2><table>",
            "<tr><th>Metric</th><th>Macro</th><th>Eligible</th>"
            "<th>Not available</th><th>By judgment kind</th></tr>",
        ]
        for metric, value, eligible, missing, kinds in per_k_rows(summary):
            parts.append(
                f'<tr><td>{_e(metric)}</td><td class="num">{_e(value)}</td>'
                f'<td class="num">{eligible}</td><td class="num">{missing}</td>'
                f"<td>{_e(kinds)}</td></tr>"
            )
        parts.append("</table><ul>")
        parts += [f"<li>{_e(note)}</li>" for note in per_k_notes(summary)]
        parts.append("</ul>")

    if result.metric_definitions:
        parts += [
            "<h2>Metric Definitions</h2><table>",
            "<tr><th>Layer</th><th>Evaluator</th><th>Version</th>"
            "<th>Relevance policy</th><th>Settings</th></tr>",
        ]
        for layer, name, version, policy, details in definition_rows(result):
            parts.append(
                f"<tr><td>{_e(layer)}</td><td>{_e(name)}</td><td>{_e(version)}</td>"
                f"<td>{_e(policy)}</td><td>{_e(details)}</td></tr>"
            )
        parts.append("</table>")
    provenance = provenance_lines(result)
    if provenance:
        parts.append("<h2>Provenance</h2><ul>")
        parts += [f"<li>{_e(line)}</li>" for line in provenance]
        parts.append("</ul>")

    if result.observability:
        parts.append("<h2>Observability</h2>")
        latency = latency_rows(result)
        if latency:
            parts.append(
                "<table><tr><th>Stage</th><th>n</th><th>Mean (ms)</th><th>P50 (ms)</th>"
                "<th>P95 (ms)</th><th>Max (ms)</th><th>Measures</th></tr>"
            )
            for stage, n, mean, p50, p95, mx, what in latency:
                parts.append(
                    f'<tr><td>{_e(stage)}</td><td class="num">{n}</td><td class="num">{mean}</td>'
                    f'<td class="num">{p50}</td><td class="num">{p95}</td><td class="num">{mx}</td>'
                    f"<td>{_e(what)}</td></tr>"
                )
            parts.append("</table>")
        tokens = token_rows(result)
        if tokens:
            parts.append(
                "<table><tr><th>Stage</th><th>Model</th><th>Records</th><th>Input tokens</th>"
                "<th>Output tokens</th><th>Total</th></tr>"
            )
            for stage, model, n, tin, tout, total in tokens:
                parts.append(
                    f"<tr><td>{_e(stage)}</td><td>{_e(model)}</td><td class=\"num\">{n}</td>"
                    f"<td class=\"num\">{tin}</td><td class=\"num\">{tout}</td>"
                    f"<td class=\"num\">{total}</td></tr>"
                )
            parts.append("</table>")
        parts.append("<ul>")
        parts += [f"<li>{_e(note)}</li>" for note in observability_notes(result)]
        parts.append("</ul>")

    parts.append("<h2>Metric Descriptions</h2><ul>")
    for name, desc in evaluator_descriptions(result).items():
        parts.append(f"<li><strong>{_e(name)}</strong>: {_e(desc)}</li>")
    parts.append("</ul>")

    if result.threshold_breaches:
        parts += [
            "<h2>Threshold Breaches</h2><table>",
            "<tr><th>Evaluator</th><th>Threshold</th><th>Actual</th><th>Delta</th></tr>",
        ]
        for b in result.threshold_breaches:
            parts.append(
                f'<tr><td>{_e(b.evaluator)}</td><td class="num">{b.threshold:.4f}</td>'
                f'<td class="num">{b.actual:.4f}</td><td class="num">{b.delta:.4f}</td></tr>'
            )
        parts.append("</table>")

    failed = failed_records(result, MAX_FAILED_RECORDS)
    if failed:
        parts += [
            "<h2>Sample Failed Records</h2><table>",
            "<tr><th>Record</th><th>Evaluator</th><th>Score</th>"
            "<th>Status</th><th>Reason</th></tr>",
        ]
        for rr in failed:
            for ev_name, er in rr.evaluator_results.items():
                parts.append(
                    f"<tr><td>{_e(rr.record_id)}</td><td>{_e(ev_name)}</td>"
                    f'<td class="num">{er.score:.4f}</td><td>{_e(er.status.value)}</td>'
                    f"<td>{_e(er.reason)}</td></tr>"
                )
        parts.append("</table>")

    if result.skipped_evaluators:
        parts.append("<h2>Skipped Evaluators</h2><ul>")
        for name, reason in result.skipped_evaluators.items():
            parts.append(f"<li><strong>{_e(name)}</strong>: {_e(reason)}</li>")
        parts.append("</ul>")

    if result.run_metadata or result.config_reference:
        parts.append("<h2>Run Metadata</h2><table>")
        for key, value in result.run_metadata.items():
            parts.append(f"<tr><th>{_e(key)}</th><td>{_e(value)}</td></tr>")
        if result.config_reference:
            parts.append(
                "<tr><th>config_reference</th>"
                f"<td><code>{_e(result.config_reference)}</code></td></tr>"
            )
        parts.append("</table>")

    parts.append("</body></html>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")
    return path
