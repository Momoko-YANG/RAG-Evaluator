"""Report generation (JSON, Markdown, HTML)."""

from rag_eval_framework.reporting.engine import ReportEngine
from rag_eval_framework.reporting.html_report import render_html_report
from rag_eval_framework.reporting.json_report import render_json_report
from rag_eval_framework.reporting.markdown_report import render_markdown_report

__all__ = [
    "ReportEngine",
    "render_html_report",
    "render_json_report",
    "render_markdown_report",
]
