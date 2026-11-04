"""JSON report renderer."""

from __future__ import annotations

import json
from pathlib import Path

from rag_eval_framework.core.aggregation import metric_results
from rag_eval_framework.reporting.common import (
    evaluator_descriptions,
    evaluator_layer,
    per_k_summaries,
    report_path,
)
from rag_eval_framework.runners.base import EvaluationRunResult


def render_json_report(result: EvaluationRunResult, output_dir: Path) -> Path:
    """Write the full run result (all records) as JSON and return the file path."""
    path = report_path(Path(output_dir), result.project_name, "json")
    data = {
        "evaluation_mode": result.runner_type,
        "evaluator_descriptions": evaluator_descriptions(result),
        "evaluator_layers": {name: evaluator_layer(name) for name in result.evaluators_used},
        **result.model_dump(mode="json"),
    }
    data["metric_results"] = metric_results(result)
    summaries = per_k_summaries(result)
    if summaries:
        data["per_k_summaries"] = {s["evaluator"]: s for s in summaries}
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
