"""Base adapter for Azure AI Evaluation SDK evaluators.

This module provides :class:`AzureEvaluatorBase`, a thin adapter that
handles:

* Lazy-importing the ``azure-ai-evaluation`` and ``azure-identity``
  packages (with a clear error when they are not installed).
* Building a model configuration dict understood by the SDK evaluators.
* Normalising the SDK result dictionary into the framework's
  :class:`EvaluatorResult`.
* Field-requirement checking before calling the SDK.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from rag_eval_framework.config.models import ProjectConfig
from rag_eval_framework.core.schemas import EvaluationRecord
from rag_eval_framework.evaluators.base import (
    BaseEvaluator,
    EvaluatorResult,
    EvaluatorStatus,
)

logger = logging.getLogger(__name__)

_SDK_INSTALL_HINT = (
    "Azure AI Evaluation SDK is required for this evaluator. "
    "Install with:  pip install 'rag-eval-framework[azure]'"
)


def _require_azure_sdk() -> None:
    """Raise ``ImportError`` with an actionable message when the SDK is absent."""
    try:
        import azure.ai.evaluation  # noqa: F401
    except ImportError as exc:
        raise ImportError(_SDK_INSTALL_HINT) from exc


def build_model_config(config: ProjectConfig) -> dict[str, Any]:
    """Build the ``model_config`` dict expected by Azure SDK evaluators.

    The dict follows the pattern::

        {
            "azure_endpoint": "https://...",
            "azure_deployment": "gpt-4",
            "api_version": "2024-12-01-preview",
            "api_key": "<key>",          # only when credential_type == "key"
        }
    """
    azure = config.azure
    if azure is None:
        return {}

    mc: dict[str, Any] = {
        "azure_endpoint": azure.endpoint,
        "azure_deployment": azure.deployment_name or config.judge_model,
        "api_version": azure.api_version,
    }

    if azure.credential_type == "key":
        api_key = os.environ.get(azure.api_key_env, "")
        if api_key:
            mc["api_key"] = api_key
        else:
            logger.warning(
                "credential_type='key' but environment variable '%s' is not set.",
                azure.api_key_env,
            )

    return mc


class AzureEvaluatorBase(BaseEvaluator):
    """Abstract adapter that all Azure SDK-backed evaluators inherit from.

    Subclasses must override:

    * :pyattr:`name` — evaluator registry name.
    * :pyattr:`_sdk_class_name` — fully-qualified SDK evaluator class name
      (e.g. ``"azure.ai.evaluation.GroundednessEvaluator"``).
    * :meth:`_build_sdk_input` — map an ``EvaluationRecord`` to the dict
      the SDK evaluator expects.
    * :meth:`_normalise_sdk_output` — map the SDK result dict to an
      ``EvaluatorResult``.
    """

    _sdk_evaluator: Any = None
    _configured: bool = False
    _deployment: str = ""

    # -- subclass API ------------------------------------------------------

    @property
    def _sdk_class_name(self) -> str:  # pragma: no cover
        raise NotImplementedError

    def _build_sdk_input(self, record: EvaluationRecord) -> dict[str, Any]:
        """Map a dataset record to the dict accepted by the SDK evaluator."""
        raise NotImplementedError  # pragma: no cover

    def _normalise_sdk_output(self, sdk_result: dict[str, Any]) -> EvaluatorResult:
        """Convert the SDK evaluator output to a framework ``EvaluatorResult``."""
        raise NotImplementedError  # pragma: no cover

    # -- lifecycle ---------------------------------------------------------

    def setup(self, config: Any) -> None:
        """Import the SDK, build model config, and instantiate the evaluator."""
        try:
            _require_azure_sdk()
        except ImportError:
            logger.warning("Azure evaluator '%s' disabled: %s", self.name, _SDK_INSTALL_HINT)
            return

        if not isinstance(config, ProjectConfig):
            return

        model_cfg = build_model_config(config)
        sdk_cls = self._import_sdk_class()

        # Forward the is_reasoning_model flag so the SDK uses
        # max_completion_tokens instead of max_tokens when required.
        extra_kwargs: dict[str, Any] = {}
        if config.azure and config.azure.is_reasoning_model:
            extra_kwargs["is_reasoning_model"] = True

        self._sdk_evaluator = sdk_cls(model_config=model_cfg, **extra_kwargs)
        self._configured = True
        self._deployment = str(model_cfg.get("azure_deployment") or "")
        logger.info(
            "Azure evaluator '%s' initialised (deployment=%s).",
            self.name,
            model_cfg.get("azure_deployment", "?"),
        )

    def definition(self) -> dict[str, Any]:
        try:
            from importlib.metadata import version

            sdk: str | None = version("azure-ai-evaluation")
        except Exception:  # not installed
            sdk = None
        return {
            **super().definition(),
            "backend": self._sdk_class_name,
            "azure_ai_evaluation_version": sdk,
            "judge_model": self._deployment or None,
            "scale": "1-5 normalised to 0-1",
        }

    def _import_sdk_class(self) -> type:
        """Dynamically import the SDK evaluator class."""
        parts = self._sdk_class_name.rsplit(".", 1)
        if len(parts) != 2:
            raise ImportError(f"Invalid SDK class name: {self._sdk_class_name}")
        module_path, class_name = parts
        import importlib

        module = importlib.import_module(module_path)
        return getattr(module, class_name)  # type: ignore[no-any-return]

    # -- evaluation --------------------------------------------------------

    def evaluate(self, record: EvaluationRecord) -> EvaluatorResult:
        if not self._configured:
            return EvaluatorResult(
                score=0.0,
                status=EvaluatorStatus.ERROR,
                reason=(
                    f"Azure evaluator '{self.name}' has not been configured. "
                    "Ensure the project config includes an 'azure' section and "
                    "the SDK is installed."
                ),
            )

        sdk_input = self._build_sdk_input(record)

        try:
            sdk_result = self._sdk_evaluator(**sdk_input)
        except Exception as exc:
            logger.exception("SDK evaluator '%s' raised an error.", self.name)
            return EvaluatorResult(
                score=0.0,
                status=EvaluatorStatus.ERROR,
                reason=f"SDK error: {exc}",
            )

        return self._normalise_sdk_output(sdk_result)
