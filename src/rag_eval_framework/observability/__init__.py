"""Cross-layer measurements: traces, latency per stage, token usage and cost.

These are not quality metrics, so they are not evaluators: runners record them in
``record.trace`` / ``record.metadata`` and :func:`.summary.observability_summary`
summarises them for the reports.
"""
