"""Tests for Phase 2 configuration model enhancements."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from rag_eval_framework.config.models import AzureConfig, JudgeConfig, ProjectConfig
from tests.conftest import make_config, make_phase2_config


class TestAzureConfig:
    def test_defaults(self) -> None:
        cfg = AzureConfig()
        assert cfg.endpoint == ""
        assert cfg.credential_type == "default"
        assert cfg.api_version == "2024-12-01-preview"

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError, match="extra"):
            AzureConfig(endpoint="x", bogus="y")  # type: ignore[call-arg]


class TestJudgeConfig:
    def test_defaults(self) -> None:
        cfg = JudgeConfig()
        assert cfg.model == ""
        assert cfg.temperature == 0.0
        assert cfg.max_tokens == 1024

    def test_temperature_range(self) -> None:
        JudgeConfig(temperature=2.0)
        with pytest.raises(ValidationError):
            JudgeConfig(temperature=2.1)

    def test_max_tokens_positive(self) -> None:
        with pytest.raises(ValidationError):
            JudgeConfig(max_tokens=0)


class TestProjectConfigPhase2Fields:
    def test_azure_section_accepted(self) -> None:
        data = make_phase2_config()
        config = ProjectConfig(**data)
        assert config.azure is not None
        assert config.azure.endpoint == "https://test.openai.azure.com"

    def test_judge_section_accepted(self) -> None:
        data = make_phase2_config()
        config = ProjectConfig(**data)
        assert config.judge is not None
        assert config.judge.model == "gpt-4.1"

    def test_evaluator_options_accepted(self) -> None:
        data = make_phase2_config()
        config = ProjectConfig(**data)
        assert "groundedness" in config.evaluator_options

    def test_backward_compatible_no_azure(self) -> None:
        """Phase 1 config (no azure/judge) still works."""
        data = make_config()
        config = ProjectConfig(**data)
        assert config.azure is None
        assert config.judge is None
        assert config.evaluator_options == {}

    def test_effective_judge_model_from_judge_section(self) -> None:
        data = make_phase2_config()
        config = ProjectConfig(**data)
        assert config.effective_judge_model() == "gpt-4.1"

    def test_effective_judge_model_falls_back_to_local_mode_judge_model(self) -> None:
        """When judge.model is empty, falls back to local_mode.judge_model."""
        data = make_phase2_config()
        data["local_mode"]["judge"]["model"] = ""
        data["local_mode"]["judge_model"] = "gpt-4o"
        config = ProjectConfig(**data)
        assert config.effective_judge_model() == "gpt-4o"

    def test_effective_judge_endpoint(self) -> None:
        data = make_phase2_config()
        config = ProjectConfig(**data)
        assert config.effective_judge_endpoint() == "https://test.openai.azure.com"
