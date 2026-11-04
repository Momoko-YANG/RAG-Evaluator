"""Report engine — dispatches to the configured report renderers."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from rag_eval_framework.config.models import ProjectConfig
from rag_eval_framework.reporting.html_report import render_html_report
from rag_eval_framework.reporting.json_report import render_json_report
from rag_eval_framework.reporting.markdown_report import render_markdown_report
from rag_eval_framework.runners.base import EvaluationRunResult

logger = logging.getLogger(__name__)

Renderer = Callable[[EvaluationRunResult, Path], Path]

RENDERERS: dict[str, Renderer] = {
    "json": render_json_report,
    "markdown": render_markdown_report,
    "html": render_html_report,
}


class ReportEngine:
    """Generate every report format listed in ``config.report_format``."""

    def generate(self, config: ProjectConfig, result: EvaluationRunResult) -> list[Path]:
        output_dir = Path(config.output_dir) / config.project_name
        output_dir.mkdir(parents=True, exist_ok=True)

        paths: list[Path] = []
        for fmt in config.report_format:
            renderer = RENDERERS.get(fmt)
            if renderer is None:
                logger.warning("Unsupported report format '%s' — skipped.", fmt)
                continue
            path = renderer(result, output_dir)
            logger.info("Wrote %s report → %s", fmt, path)
            paths.append(path)
        return paths
