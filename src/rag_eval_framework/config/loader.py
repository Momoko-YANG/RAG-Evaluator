"""YAML configuration file loader with validation."""

from __future__ import annotations

import logging
from pathlib import Path

import yaml
from pydantic import ValidationError

from rag_eval_framework.config.models import ProjectConfig

logger = logging.getLogger(__name__)


class ConfigLoadError(Exception):
    """Raised when a configuration file cannot be loaded or validated."""


def load_config(config_path: str | Path) -> ProjectConfig:
    """Load and validate a project YAML configuration file.

    Parameters
    ----------
    config_path:
        Path to the YAML configuration file.

    Returns
    -------
    ProjectConfig
        Validated project configuration.

    Raises
    ------
    ConfigLoadError
        If the file cannot be read, parsed, or fails validation.
    """
    path = Path(config_path)

    if not path.exists():
        raise ConfigLoadError(f"Configuration file not found: {path}")

    if path.suffix not in (".yaml", ".yml"):
        raise ConfigLoadError(f"Configuration file must be .yaml or .yml, got: {path.suffix}")

    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigLoadError(f"Failed to read configuration file: {exc}") from exc

    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise ConfigLoadError(f"Invalid YAML in configuration file: {exc}") from exc

    if not isinstance(data, dict):
        raise ConfigLoadError("Configuration file must contain a YAML mapping at the top level.")

    try:
        config = ProjectConfig(**data)
    except ValidationError as exc:
        raise ConfigLoadError(f"Configuration validation failed:\n{exc}") from exc

    logger.info("Loaded configuration for project '%s'", config.project_name)
    return config  # noqa: RET504
