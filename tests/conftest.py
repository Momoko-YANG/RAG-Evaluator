"""Shared test fixtures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Sample data factories
# ---------------------------------------------------------------------------


def make_record(**overrides: Any) -> dict[str, Any]:
    """Return a minimal valid record dict, with optional overrides."""
    base: dict[str, Any] = {
        "id": "test-001",
        "question": "What is the policy?",
        "response": "The retention period is 90 days.",
        "contexts": ["Policy states retention is 90 days."],
        "ground_truth_answer": "90 days",
        "metadata": {"scenario": "test"},
    }
    base.update(overrides)
    return base


def make_config(**overrides: Any) -> dict[str, Any]:
    """Return a minimal valid config dict, with optional overrides.

    Mode-specific keys (``evaluators``, ``thresholds``, ``azure``, ``judge``,
    ``judge_model``, ``evaluator_options``) are automatically routed into the
    ``local_mode`` block so callers don't have to nest them manually.
    """
    _LOCAL_MODE_KEYS = {
        "evaluators",
        "thresholds",
        "azure",
        "judge",
        "judge_model",
        "evaluator_options",
    }

    local_mode_updates: dict[str, Any] = {}
    top_updates: dict[str, Any] = {}
    for key, value in overrides.items():
        if key in _LOCAL_MODE_KEYS:
            local_mode_updates[key] = value
        else:
            top_updates[key] = value

    base: dict[str, Any] = {
        "project_name": "test-project",
        "dataset_path": "datasets/sample/example-run/golden.jsonl",
        "mode": "local",
        "local_mode": {
            "evaluators": [
                "answer_presence",
                "context_presence",
                "exact_match_accuracy",
            ],
            "thresholds": {"answer_presence": 0.9},
        },
    }
    base["local_mode"].update(local_mode_updates)
    base.update(top_updates)
    return base


def make_phase2_config(**overrides: Any) -> dict[str, Any]:
    """Return a config dict that includes Phase 2 fields (azure + judge)."""
    base: dict[str, Any] = {
        "project_name": "test-project",
        "dataset_path": "datasets/sample/example-run/golden.jsonl",
        "mode": "local",
        "local_mode": {
            "evaluators": [
                "answer_presence",
                "context_presence",
                "exact_match_accuracy",
            ],
            "thresholds": {"answer_presence": 0.9},
            "azure": {
                "endpoint": "https://test.openai.azure.com",
                "deployment_name": "gpt-4",
                "api_version": "2024-12-01-preview",
                "credential_type": "key",
                "api_key_env": "AZURE_OPENAI_API_KEY",
            },
            "judge": {
                "model": "gpt-4.1",
                "temperature": 0.0,
                "max_tokens": 512,
            },
            "evaluator_options": {"groundedness": {"model": "gpt-4"}},
        },
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_config_dict() -> dict[str, Any]:
    return make_config()


@pytest.fixture()
def tmp_config_file(tmp_path: Path, sample_config_dict: dict[str, Any]) -> Path:
    """Write a valid YAML config to a temp file and return its path."""
    import yaml

    path = tmp_path / "test.yaml"
    path.write_text(yaml.dump(sample_config_dict), encoding="utf-8")
    return path


@pytest.fixture()
def tmp_dataset_file(tmp_path: Path) -> Path:
    """Write a valid JSONL dataset to a temp file and return its path."""
    records = [
        make_record(id="rec-001"),
        make_record(id="rec-002", response="Short."),
        make_record(id="rec-003", contexts=[], ground_truth_answer="wrong answer here"),
    ]
    path = tmp_path / "eval.jsonl"
    lines = [json.dumps(r) for r in records]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
