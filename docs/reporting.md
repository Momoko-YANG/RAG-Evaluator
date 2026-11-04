# Reporting

`ReportEngine` (`reporting/engine.py`) writes every format in `report_format` to
`{output_dir}/{project_name}/eval-report-{project_name}.{json|md|html}`. File names
are fixed, so give each run its own `--output` directory.

## Sections

The Markdown and HTML reports (`markdown_report.py`, `html_report.py`; the HTML file
is self-contained) show:

| Section | Content |
|---|---|
| Mode banner | LOCAL or CLOUD |
| Summary / Dataset Statistics | Status, runner, timestamp, dataset, records passed / failed, skipped evaluator–record pairs |
| Evaluator Summary | Layer, evaluator, mean score (SUCCESS results only), threshold, status |
| Recall@k / nDCG@k | Macro value per k, eligible questions, not-available count and reasons, by judgment kind |
| Metric Definitions | Version, relevance policy and settings of each evaluator |
| Provenance | Framework version, dataset SHA-256, retrieval settings found in the data |
| Observability | Latency per stage, tokens per stage / model, cost notes (see [evaluation-layers.md](evaluation-layers.md#observability)) |
| Metric Descriptions | From the evaluator catalogs |
| Threshold Breaches | Threshold, actual, delta |
| Sample Failed Records | Up to 20 records that failed a threshold, with every evaluator's score, status and reason |
| Run Metadata | `elapsed_seconds`, config `metadata`, `cloud_run_id` (cloud) |

Sections without data (e.g. per-k tables, observability in cloud mode) are omitted.

## JSON

The JSON report is the full `EvaluationRunResult` (all records) plus:

- `evaluation_mode`, `evaluator_descriptions`, `evaluator_layers`
- `metric_results`: one entry per reported value (metric, score, `valid` /
  `not_available`, eligible questions, metric version, relevance policy, label source)
- `per_k_summaries`: per-k evaluators only

```json
{
  "evaluation_mode": "local",
  "project_name": "my-project-retrieval",
  "total_records": 200,
  "aggregate_scores": {"retrieval_hit_rate": 0.8},
  "threshold_results": [],
  "passed": true,
  "record_results": [
    {"record_id": "q-001", "passed": true,
     "evaluator_results": {"retrieval_hit_rate": {"score": 1.0, "status": "success", "reason": "...", "metadata": {"rank": 1}}}}
  ],
  "metric_definitions": {"retrieval_hit_rate": {"version": "...", "layer": "retrieval", "relevance_policy": "evidence_match", "k": 3}},
  "provenance": {"framework_version": "...", "dataset": {"path": "...", "sha256": "...", "records": 200}, "retrieval_settings": []},
  "observability": {"latency": {}, "token_usage": {}, "cost": {"status": "not_configured"}, "traced_records": 0, "retries": 0, "errored_records": 0},
  "metric_results": [{"metric": "retrieval_hit_rate@3", "score": 0.8, "status": "valid", "eligible_questions": 200, "skipped": 0, "errors": 0}]
}
```

Scripts that read JSON reports:

- `scripts/summarize_layers.py --report <json> [--report ...] --dataset <jsonl> --output layers.md`
  combines reports into scores per layer and the failing layer per question.
- `scripts/compare_runs.py --a <json> --b <json> --evaluator <name> [--k N] [--dataset <jsonl>] --output compare.md`
  compares two runs question by question: both hit / only A / only B / neither, and
  improved / worse / unchanged values. Skipped and `not_available` values are counted
  separately, never as misses. Compare runs scored with the same config and labels.

## Adding a format

Write a `render_<format>_report(result, output_dir) -> Path` function in
`reporting/`, add it to `RENDERERS` in `engine.py`, and add the name to
`report_format_must_be_supported` in `config/models.py`.
