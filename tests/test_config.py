"""Tests for configuration loading and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from rag_eval_framework.config.loader import ConfigLoadError, load_config
from rag_eval_framework.config.models import ProjectConfig
from tests.conftest import make_config


class TestProjectConfigModel:
    """Unit tests for the ProjectConfig pydantic model."""

    def test_valid_minimal_config(self) -> None:
        data = make_config()
        config = ProjectConfig(**data)
        assert config.project_name == "test-project"
        assert config.evaluators == ["answer_presence", "context_presence", "exact_match_accuracy"]

    def test_defaults_applied(self) -> None:
        data = make_config()
        config = ProjectConfig(**data)
        assert config.report_format == ["json", "markdown"]
        assert config.output_dir == "output"

    def test_missing_project_name_fails(self) -> None:
        data = make_config()
        del data["project_name"]
        with pytest.raises(ValidationError):
            ProjectConfig(**data)

    def test_missing_mode_block_fails(self) -> None:
        with pytest.raises(ValidationError):
            ProjectConfig(
                project_name="test",
                dataset_path="data.jsonl",
                mode="local",
            )

    def test_invalid_threshold_range(self) -> None:
        data = make_config()
        data["local_mode"]["thresholds"] = {"foo": 1.5}
        with pytest.raises(ValidationError):
            ProjectConfig(**data)

    def test_unsupported_report_format(self) -> None:
        data = make_config(report_format=["pdf"])
        with pytest.raises(ValidationError):
            ProjectConfig(**data)

    def test_extra_fields_rejected(self) -> None:
        data = make_config(unknown_field="surprise")
        with pytest.raises(ValidationError, match="extra"):
            ProjectConfig(**data)

    def test_threshold_zero_and_one_valid(self) -> None:
        data = make_config()
        data["local_mode"]["thresholds"] = {"answer_presence": 0.0, "context_presence": 1.0}
        config = ProjectConfig(**data)
        assert config.thresholds == {"answer_presence": 0.0, "context_presence": 1.0}


class TestLoadConfig:
    """Tests for the YAML config file loader."""

    def test_load_valid_yaml(self, tmp_config_file: Path) -> None:
        config = load_config(tmp_config_file)
        assert config.project_name == "test-project"

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigLoadError, match="not found"):
            load_config(tmp_path / "nope.yaml")

    def test_non_yaml_extension_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "config.txt"
        bad.write_text("hello", encoding="utf-8")
        with pytest.raises(ConfigLoadError, match=".yaml or .yml"):
            load_config(bad)

    def test_invalid_yaml_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text("}{not yaml", encoding="utf-8")
        with pytest.raises(ConfigLoadError, match="Invalid YAML"):
            load_config(bad)

    def test_non_mapping_yaml_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "list.yaml"
        bad.write_text("- a\n- b\n", encoding="utf-8")
        with pytest.raises(ConfigLoadError, match="YAML mapping"):
            load_config(bad)

    def test_validation_error_raised(self, tmp_path: Path) -> None:
        bad_data: dict[str, Any] = {"project_name": "x"}
        path = tmp_path / "incomplete.yaml"
        path.write_text(yaml.dump(bad_data), encoding="utf-8")
        with pytest.raises(ConfigLoadError, match="validation failed"):
            load_config(path)
