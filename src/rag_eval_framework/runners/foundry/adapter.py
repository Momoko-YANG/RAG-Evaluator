"""Azure AI Foundry SDK adapter — isolates all cloud SDK interactions.

This module provides :class:`FoundryAdapter`, a thin layer between the
framework's :class:`FoundryRunner` and the ``azure-ai-projects`` /
``azure-ai-evaluation`` SDKs.

Responsibilities
----------------
* Lazy-import the Azure SDKs (with clear install hints).
* Build ``AIProjectClient`` from config / credentials.
* Submit an evaluation run to Azure AI Foundry.
* Poll for run completion.
* Return raw results as plain dicts for normalisation.

All Azure-specific imports happen **inside** functions so that the rest of
the framework (and its tests) can operate without the SDK installed.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from rag_eval_framework.config.models import FoundryConfig, ProjectConfig

logger = logging.getLogger(__name__)

_SDK_INSTALL_HINT = (
    "Azure AI Projects SDK is required for cloud evaluation. "
    "Install with:  pip install 'rag-eval-framework[cloud]'"
)


class FoundryAdapterError(Exception):
    """Raised when the Foundry adapter encounters a configuration or SDK error."""


class FoundryAdapter:
    """Encapsulates all Azure AI Foundry SDK interactions.

    Parameters
    ----------
    config : ProjectConfig
        The full project configuration.  Must include a populated ``foundry``
        section and an ``azure`` section.
    """

    def __init__(self, config: ProjectConfig) -> None:
        self._config = config
        self._foundry_cfg: FoundryConfig = self._require_foundry_config(config)
        self._client: Any = None
        self._cached_results: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Build the ``AIProjectClient`` and verify connectivity.

        Raises
        ------
        ImportError
            If the ``azure-ai-projects`` package is not installed.
        FoundryAdapterError
            If credentials or project scope are misconfigured.
        """
        _require_foundry_sdk()
        self._client = self._build_client()
        logger.info(
            "Connected to Azure AI Foundry project '%s' (subscription=%s, resource_group=%s).",
            self._foundry_cfg.project_name,
            self._foundry_cfg.subscription_id,
            self._foundry_cfg.resource_group,
        )

    def submit_evaluation(
        self,
        *,
        dataset_path: str,
        evaluators: list[str],
        display_name: str = "",
    ) -> str:
        """Run a cloud evaluation and cache the result.

        With ``azure-ai-evaluation`` ≥1.x the ``evaluate()`` function is
        **synchronous** — it executes all evaluators and returns an
        ``EvaluationResult`` (a dict-like object with ``metrics`` and
        ``rows``) directly.  There is no job-ID / polling flow.

        We cache the result on the adapter so that :meth:`get_results` can
        retrieve it.  The returned "run ID" is a synthetic identifier.

        Parameters
        ----------
        dataset_path
            Path to the evaluation JSONL dataset.
        evaluators
            List of evaluator names to run in the cloud.
        display_name
            Human-readable run name (defaults to project_name).

        Returns
        -------
        str
            A synthetic run ID (memory address of the result object).
        """
        self._ensure_connected()

        from azure.ai.evaluation import (  # type: ignore[import-untyped]
            evaluate,
        )

        model_config = self._build_model_config()
        evaluator_map = self._build_evaluator_map(evaluators, model_config)

        run_name = display_name or self._config.project_name
        logger.info(
            "Running cloud evaluation '%s' with %d evaluator(s)…",
            run_name,
            len(evaluator_map),
        )

        result = evaluate(
            data=dataset_path,
            evaluators=evaluator_map,
            evaluator_config=self._build_evaluator_config(evaluators),
            azure_ai_project=self._build_project_scope(),
            output_path=None,
            evaluation_name=run_name,
        )

        # azure-ai-evaluation ≥1.x returns results synchronously.
        # Extract metrics and rows directly from the result dict.
        run_id = str(id(result))
        self._cached_results[run_id] = {
            "metrics": dict(result.get("metrics", {})),
            "rows": list(result.get("rows", [])),
            "run_id": run_id,
            "status": "Completed",
            "display_name": run_name,
            "studio_url": result.get("studio_url", ""),
        }

        studio_url = result.get("studio_url", "")
        if studio_url:
            logger.info("Cloud evaluation complete — studio URL: %s", studio_url)
        logger.info("Cloud evaluation complete — run_id=%s", run_id)
        return run_id

    def get_results(self, run_id: str) -> dict[str, Any]:
        """Retrieve the results for a completed cloud evaluation run.

        With ``azure-ai-evaluation`` ≥1.x the results are already available
        from the synchronous ``evaluate()`` call — no polling required.

        Returns
        -------
        dict
            Results dict containing ``metrics`` and per-row ``rows``.
        """
        if run_id in self._cached_results:
            return self._cached_results[run_id]

        raise FoundryAdapterError(
            f"No cached results for run '{run_id}'.  "
            "With azure-ai-evaluation ≥1.x, evaluate() is synchronous — "
            "results should already be available from submit_evaluation()."
        )

    def submit_and_wait(
        self,
        *,
        dataset_path: str,
        evaluators: list[str],
        display_name: str = "",
    ) -> dict[str, Any]:
        """Run evaluators via ``evaluate()`` and return results."""
        run_id = self.submit_evaluation(
            dataset_path=dataset_path,
            evaluators=evaluators,
            display_name=display_name,
        )
        return self.get_results(run_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _require_foundry_config(config: ProjectConfig) -> FoundryConfig:
        if config.foundry is None:
            raise FoundryAdapterError(
                "Cloud evaluation requires a 'foundry' section in the cloud_mode "
                "config block.  See docs/configuration.md for details."
            )
        return config.foundry

    def _build_client(self) -> Any:
        """Build an ``AIProjectClient`` from config credentials.

        The current ``azure-ai-projects`` SDK (1.x) uses::

            AIProjectClient(endpoint, credential)

        The ``endpoint`` comes from the ``foundry.endpoint`` config field.
        """
        from azure.ai.projects import AIProjectClient  # type: ignore[import-untyped]

        credential = self._build_credential()
        endpoint = self._foundry_cfg.endpoint
        if not endpoint:
            raise FoundryAdapterError(
                "Cloud evaluation requires 'foundry.endpoint' to be set. "
                "Find this in the Azure AI Foundry portal under project "
                "Settings → Overview (e.g. https://<project>.region.api.azureml.ms)."
            )
        return AIProjectClient(
            endpoint=endpoint,
            credential=credential,
        )

    def _build_credential(self) -> Any:
        """Return the appropriate Azure credential object."""
        cred_type = self._foundry_cfg.credential_type
        if cred_type == "key":
            azure_cfg = self._config.azure
            if azure_cfg:
                api_key = os.environ.get(azure_cfg.api_key_env, "")
                if api_key:
                    from azure.core.credentials import (  # type: ignore[import-untyped]
                        AzureKeyCredential,
                    )

                    return AzureKeyCredential(api_key)
            raise FoundryAdapterError("credential_type='key' requires azure.api_key_env to be set.")
        if cred_type == "env":
            conn_str = self._get_connection_string()
            if not conn_str:
                raise FoundryAdapterError(
                    "credential_type='env' requires the connection string "
                    f"env var '{self._foundry_cfg.connection_string_env}' to be set."
                )
            # connection_string auth is handled by from_connection_string
            from azure.identity import DefaultAzureCredential  # type: ignore[import-untyped]

            return DefaultAzureCredential()

        # default → DefaultAzureCredential
        from azure.identity import DefaultAzureCredential  # type: ignore[import-untyped]

        return DefaultAzureCredential()

    def _get_connection_string(self) -> str:
        return os.environ.get(self._foundry_cfg.connection_string_env, "")

    def _build_project_scope(self) -> str | dict[str, str]:
        """Return the ``azure_ai_project`` value for ``evaluate()``.

        The ``azure.ai.evaluation.evaluate`` function accepts either:
        * A plain endpoint URL string (preferred for SDK ≥1.x).
        * A legacy dict with ``subscription_id``, ``resource_group_name``,
          ``project_name``.

        We prefer the endpoint string when available and fall back to the
        dict for backward-compatibility.
        """
        if self._foundry_cfg.endpoint:
            return self._foundry_cfg.endpoint
        return {
            "subscription_id": self._foundry_cfg.subscription_id,
            "resource_group_name": self._foundry_cfg.resource_group,
            "project_name": self._foundry_cfg.project_name,
        }

    def _build_model_config(self) -> dict[str, Any]:
        """Build model_config dict for SDK evaluators."""
        from rag_eval_framework.evaluators.common.azure_sdk import build_model_config

        return build_model_config(self._config)

    def _build_evaluator_map(
        self,
        evaluator_names: list[str],
        model_config: dict[str, Any],
    ) -> dict[str, Any]:
        """Map evaluator names to SDK evaluator instances for cloud submission.

        Only Azure SDK evaluators and LLM judges are submittable to the cloud.
        Built-in scaffold evaluators are skipped (they run locally).
        """
        evaluator_map: dict[str, Any] = {}

        # Azure SDK evaluator name → SDK class mapping
        _SDK_EVALUATORS: dict[str, str] = {
            "groundedness": "azure.ai.evaluation.GroundednessEvaluator",
            "relevance": "azure.ai.evaluation.RelevanceEvaluator",
            "retrieval": "azure.ai.evaluation.RetrievalEvaluator",
            "response_completeness": "azure.ai.evaluation.ResponseCompletenessEvaluator",
        }

        # Forward is_reasoning_model so the SDK uses max_completion_tokens
        # instead of max_tokens for reasoning / o-series models.
        extra_kwargs: dict[str, Any] = {}
        if self._config.azure and self._config.azure.is_reasoning_model:
            extra_kwargs["is_reasoning_model"] = True

        import importlib

        for name in evaluator_names:
            if name in _SDK_EVALUATORS:
                class_path = _SDK_EVALUATORS[name]
                module_path, class_name = class_path.rsplit(".", 1)
                mod = importlib.import_module(module_path)
                cls = getattr(mod, class_name)
                evaluator_map[name] = cls(model_config=model_config, **extra_kwargs)

        return evaluator_map

    def _build_evaluator_config(self, evaluator_names: list[str]) -> dict[str, dict[str, Any]]:
        """Build column-mapping config for cloud evaluators.

        The ``azure.ai.evaluation.evaluate`` function expects each entry in
        ``evaluator_config`` to be an ``EvaluatorConfig`` typed-dict with a
        ``column_mapping`` key::

            {"groundedness": {"column_mapping": {"query": "${data.question}", ...}}}
        """
        _COLUMN_MAPS: dict[str, dict[str, str]] = {
            "groundedness": {
                "query": "${data.question}",
                "response": "${data.response}",
                "context": "${data.contexts}",
            },
            "relevance": {
                "query": "${data.question}",
                "response": "${data.response}",
            },
            "retrieval": {
                "query": "${data.question}",
                "context": "${data.contexts}",
            },
            "response_completeness": {
                "query": "${data.question}",
                "response": "${data.response}",
                "ground_truth": "${data.ground_truth_answer}",
            },
        }
        return {
            name: {"column_mapping": _COLUMN_MAPS[name]}
            for name in evaluator_names
            if name in _COLUMN_MAPS
        }

    def _ensure_connected(self) -> None:
        if self._client is None:
            raise FoundryAdapterError(
                "FoundryAdapter.connect() must be called before submitting evaluation runs."
            )


def _require_foundry_sdk() -> None:
    """Raise ``ImportError`` with an actionable message when the SDK is absent."""
    try:
        import azure.ai.projects  # type: ignore[import-untyped]  # noqa: F401
    except ImportError as exc:
        raise ImportError(_SDK_INSTALL_HINT) from exc
    try:
        import azure.ai.evaluation  # type: ignore[import-untyped]  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Azure AI Evaluation SDK is required for cloud evaluation. "
            "Install with:  pip install 'rag-eval-framework[cloud]'"
        ) from exc
