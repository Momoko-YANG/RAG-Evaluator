"""Pydantic models for project configuration.

This module defines the configuration schema for evaluation projects.
The framework supports two execution modes — **local** and **cloud** —
each with its own evaluator catalog and settings.  Configuration uses
``mode`` with dedicated ``local_mode`` / ``cloud_mode`` blocks.
"""

from __future__ import annotations

import logging
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from rag_eval_framework.observability.cost import PricingConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Evaluator catalogs — authoritative list per mode
# ---------------------------------------------------------------------------

LOCAL_EVALUATOR_CATALOG: dict[str, str] = {
    # Built-in / scaffold (no external dependency)
    "answer_presence": "Checks response contains a substantive answer (≥ 2 words).",
    "context_presence": "Checks at least one non-empty context passage was retrieved.",
    "exact_match_accuracy": (
        "Checks ground-truth answer appears in the response"
        " (normalised substring match)."
    ),
    # Retrieval ranking (needs ground_truth_documents / ground_truth_evidence labels)
    "retrieval_hit_rate": (
        "Checks a labelled relevant document is among the top-k"
        " retrieved documents (Hit@k)."
    ),
    "retrieval_mrr": (
        "Reciprocal rank of the first labelled relevant retrieved"
        " document (MRR@k)."
    ),
    "retrieval_ndcg": (
        "nDCG@k (pure Python) from qrels grades or top-k evidence labels; policy recorded."
    ),
    "retrieval_recall": (
        "Share of the judged relevant chunks (qrels file) retrieved in the"
        " top k (Recall@k; proxy unless judgments are exhaustive and reviewed)."
    ),
    # End-to-end (needs a trace from the e2e runner, except abstention / access_control)
    "task_success": (
        "End-to-end task success: human label, else LLM judge, else rule"
        " (correct abstention + cited relevant chunk)."
    ),
    "tool_selection": "The model called the expected tools and no forbidden one (rule).",
    "query_construction": "Share of expected terms in the model's search text (rule).",
    "abstention": "The answer declines exactly when the question is unanswerable (rule).",
    "access_control": "Only allowed indexes / documents were searched, retrieved, cited (rule).",
    # Azure AI Evaluation SDK (run locally via SDK)
    "groundedness": (
        "Azure SDK — measures whether the response is grounded"
        " in the provided context."
    ),
    "relevance": (
        "Azure SDK — measures whether the response is relevant"
        " to the question."
    ),
    "retrieval": (
        "Azure SDK — measures the quality of the retrieved"
        " context passages."
    ),
    "response_completeness": (
        "Azure SDK — measures how completely the response addresses"
        " the question relative to the ground truth."
    ),
    "document_retrieval": (
        "Azure SDK — nDCG@3 and other ranking metrics from labelled"
        " relevant chunks (no LLM)."
    ),
    # LLM-as-a-judge (run locally via Azure OpenAI)
    "accuracy_judge": (
        "LLM judge — assesses factual accuracy of the response"
        " vs. ground truth (1-5 scale → 0-1)."
    ),
    "hallucination_judge": (
        "LLM judge — detects claims not supported by the"
        " provided context (1-5 scale → 0-1)."
    ),
    "citation_judge": (
        "LLM judge — assesses how well the response uses"
        " evidence from context (1-5 scale → 0-1)."
    ),
    "policy_compliance_judge": (
        "LLM judge — checks the response complies with a"
        " supplied policy document (1-5 scale → 0-1)."
    ),
}

CLOUD_EVALUATOR_CATALOG: dict[str, str] = {
    # Azure AI Evaluation SDK (run via Azure AI Foundry cloud)
    "groundedness": (
        "Azure SDK (cloud) — measures whether the response"
        " is grounded in the provided context."
    ),
    "relevance": (
        "Azure SDK (cloud) — measures whether the response"
        " is relevant to the question."
    ),
    "retrieval": (
        "Azure SDK (cloud) — measures the quality of the"
        " retrieved context passages."
    ),
    "response_completeness": (
        "Azure SDK (cloud) — measures how completely the response"
        " addresses the question relative to the ground truth."
    ),
}

# Evaluation layer of each evaluator (matches evaluators/retrieval/ and evaluators/generation/).
LAYERS: tuple[str, ...] = ("retrieval", "generation", "e2e")
EVALUATOR_LAYERS: dict[str, str] = {
    "context_presence": "retrieval",
    "retrieval_hit_rate": "retrieval",
    "retrieval_mrr": "retrieval",
    "retrieval_recall": "retrieval",
    "retrieval_ndcg": "retrieval",
    "document_retrieval": "retrieval",
    "retrieval": "retrieval",
    "answer_presence": "generation",
    "groundedness": "generation",
    "hallucination_judge": "generation",
    "relevance": "generation",
    "exact_match_accuracy": "generation",
    "accuracy_judge": "generation",
    "response_completeness": "generation",
    "citation_judge": "generation",
    "policy_compliance_judge": "generation",
    "task_success": "e2e",
    "tool_selection": "e2e",
    "query_construction": "e2e",
    "abstention": "e2e",
    "access_control": "e2e",
}

# Flat sets for quick membership tests
LOCAL_EVALUATOR_NAMES: frozenset[str] = frozenset(LOCAL_EVALUATOR_CATALOG)
CLOUD_EVALUATOR_NAMES: frozenset[str] = frozenset(CLOUD_EVALUATOR_CATALOG)

# ---------------------------------------------------------------------------
# Sub-configuration models
# ---------------------------------------------------------------------------


class AzureConfig(BaseModel):
    """Azure AI services connection settings for SDK-backed evaluators."""

    model_config = ConfigDict(extra="forbid")

    endpoint: str = Field(
        default="",
        description="Azure AI project or OpenAI endpoint URL.",
    )
    deployment_name: str = Field(
        default="",
        description="Azure OpenAI deployment name (e.g. 'gpt-4').",
    )
    api_version: str = Field(
        default="2024-12-01-preview",
        description="Azure API version string.",
    )
    credential_type: Literal["default", "key", "env"] = Field(
        default="default",
        description=(
            "Credential strategy: 'default' (DefaultAzureCredential), "
            "'key' (API key from environment), or 'env' (connection string)."
        ),
    )
    api_key_env: str = Field(
        default="AZURE_OPENAI_API_KEY",
        description="Environment variable holding the API key (credential_type='key').",
    )
    project_connection_string_env: str = Field(
        default="AZURE_AI_PROJECT_CONNECTION_STRING",
        description="Environment variable for the Azure AI project connection string.",
    )
    is_reasoning_model: bool = Field(
        default=False,
        description=(
            "Set to true when the deployment uses a reasoning / o-series model "
            "(or any model that requires 'max_completion_tokens' instead of "
            "'max_tokens').  This flag is forwarded to the Azure AI Evaluation "
            "SDK evaluators."
        ),
    )


class FoundryConfig(BaseModel):
    """Azure AI Foundry project settings for cloud evaluation.

    These settings are required when ``mode`` is ``"cloud"``.
    The cloud runner uses them to submit evaluation jobs to Azure AI Foundry
    and retrieve results.
    """

    model_config = ConfigDict(extra="forbid")

    endpoint: str = Field(
        default="",
        description=(
            "Azure AI Foundry project endpoint URL. "
            "Find this in the AI Foundry portal under project Settings → Overview. "
            "Example: https://<project>.region.api.azureml.ms"
        ),
    )
    subscription_id: str = Field(
        default="",
        description="Azure subscription ID.",
    )
    resource_group: str = Field(
        default="",
        description="Azure resource group containing the AI Foundry project.",
    )
    project_name: str = Field(
        default="",
        description="Azure AI Foundry project name (not the eval project_name).",
    )
    credential_type: Literal["default", "key", "env"] = Field(
        default="default",
        description=(
            "Credential strategy: 'default' (DefaultAzureCredential), "
            "'key' (API key from environment), or 'env' (connection string)."
        ),
    )
    connection_string_env: str = Field(
        default="AZURE_AI_PROJECT_CONNECTION_STRING",
        description=(
            "Environment variable holding the Foundry project connection string. "
            "Used when credential_type='env'."
        ),
    )
    poll_interval_seconds: int = Field(
        default=10,
        ge=1,
        le=300,
        description="How often (in seconds) to poll for cloud run completion.",
    )
    poll_timeout_seconds: int = Field(
        default=1800,
        ge=30,
        description="Maximum wait time (in seconds) before declaring a timeout.",
    )


class JudgeConfig(BaseModel):
    """Settings for LLM-as-a-judge evaluators backed by Azure OpenAI."""

    model_config = ConfigDict(extra="forbid")

    model: str = Field(
        default="",
        description="Azure OpenAI deployment name for the judge model.",
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for judge calls.",
    )
    max_tokens: int = Field(
        default=1024,
        gt=0,
        description="Maximum tokens in the judge response.",
    )
    azure_endpoint: str = Field(
        default="",
        description="Azure OpenAI endpoint URL. Falls back to azure.endpoint if empty.",
    )
    api_version: str = Field(
        default="2024-12-01-preview",
        description="Azure OpenAI API version.",
    )


# ---------------------------------------------------------------------------
# Mode-specific configuration blocks
# ---------------------------------------------------------------------------


class LocalModeConfig(BaseModel):
    """Settings specific to **local** evaluation mode.

    Local mode runs all evaluators in the current Python process.  It
    supports three evaluator tiers: built-in/scaffold, Azure AI Evaluation
    SDK, and LLM-as-a-judge.
    """

    model_config = ConfigDict(extra="forbid")

    evaluators: list[str] = Field(
        ...,
        min_length=1,
        description="Evaluators to run (must be from the local evaluator catalog).",
    )
    thresholds: dict[str, float] = Field(
        default_factory=dict,
        description="Minimum acceptable score per evaluator (0.0–1.0).",
    )
    azure: AzureConfig | None = Field(
        default=None,
        description="Azure OpenAI connection settings for SDK evaluators.",
    )
    judge: JudgeConfig | None = Field(
        default=None,
        description="LLM-as-a-judge configuration (model, temperature, etc.).",
    )
    judge_model: str = Field(
        default="gpt-4.1",
        description="Shorthand for the judge deployment name.",
    )
    evaluator_options: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Per-evaluator option overrides keyed by evaluator name.",
    )

    @field_validator("evaluators")
    @classmethod
    def _validate_evaluator_names(cls, v: list[str]) -> list[str]:
        cleaned: list[str] = []
        for name in v:
            name = name.strip()
            if not name:
                raise ValueError("Evaluator names must be non-empty strings.")
            if name not in LOCAL_EVALUATOR_NAMES:
                raise ValueError(
                    f"Evaluator '{name}' is not in the local catalog.  "
                    f"Available: {sorted(LOCAL_EVALUATOR_NAMES)}"
                )
            cleaned.append(name)
        return cleaned

    @field_validator("thresholds")
    @classmethod
    def _validate_thresholds(cls, v: dict[str, float]) -> dict[str, float]:
        for name, score in v.items():
            if not 0.0 <= score <= 1.0:
                raise ValueError(
                    f"Threshold for '{name}' must be between 0.0 and 1.0, got {score}."
                )
        return v


class CloudModeConfig(BaseModel):
    """Settings specific to **cloud** evaluation mode.

    Cloud mode submits evaluation jobs to Azure AI Foundry.  Only
    evaluators supported by the Azure AI Evaluation SDK can be used.
    """

    model_config = ConfigDict(extra="forbid")

    evaluators: list[str] = Field(
        ...,
        min_length=1,
        description="Evaluators to run (must be from the cloud evaluator catalog).",
    )
    thresholds: dict[str, float] = Field(
        default_factory=dict,
        description="Minimum acceptable score per evaluator (0.0–1.0).",
    )
    azure: AzureConfig | None = Field(
        default=None,
        description="Azure OpenAI connection settings for SDK evaluators.",
    )
    foundry: FoundryConfig | None = Field(
        default=None,
        description="Azure AI Foundry project settings.",
    )

    @field_validator("evaluators")
    @classmethod
    def _validate_evaluator_names(cls, v: list[str]) -> list[str]:
        cleaned: list[str] = []
        for name in v:
            name = name.strip()
            if not name:
                raise ValueError("Evaluator names must be non-empty strings.")
            if name not in CLOUD_EVALUATOR_NAMES:
                raise ValueError(
                    f"Evaluator '{name}' is not in the cloud catalog.  "
                    f"Available: {sorted(CLOUD_EVALUATOR_NAMES)}"
                )
            cleaned.append(name)
        return cleaned

    @field_validator("thresholds")
    @classmethod
    def _validate_thresholds(cls, v: dict[str, float]) -> dict[str, float]:
        for name, score in v.items():
            if not 0.0 <= score <= 1.0:
                raise ValueError(
                    f"Threshold for '{name}' must be between 0.0 and 1.0, got {score}."
                )
        return v


# ---------------------------------------------------------------------------
# Top-level project configuration
# ---------------------------------------------------------------------------


class ObservabilityConfig(BaseModel):
    """Latency / token / cost reporting settings (see ``observability/cost.py``)."""

    model_config = ConfigDict(extra="forbid")

    pricing: PricingConfig | None = Field(
        default=None,
        description="Model prices with their source; without it cost is reported as unknown.",
    )


class ProjectConfig(BaseModel):
    """Top-level configuration for a RAG evaluation project.

    Uses ``mode`` with dedicated ``local_mode`` / ``cloud_mode`` blocks::

        mode: local          # or "cloud"
        local_mode:
          evaluators: [...]
          azure: { ... }

    See ``docs/configuration.md`` and ``docs/modes.md`` for details.
    """

    model_config = ConfigDict(extra="forbid")

    # --- Common fields (both modes) ----------------------------------------

    project_name: str = Field(
        ...,
        description="Unique human-readable name for the project (e.g. 'hr-rag').",
    )
    dataset_path: str = Field(
        ...,
        description="Path to the JSONL evaluation dataset, relative to repo root.",
    )
    report_format: list[str] = Field(
        default=["json", "markdown"],
        description="Output report formats. Supported: 'json', 'markdown', 'html'.",
    )
    output_dir: str = Field(
        default="output",
        description="Directory where reports and artifacts are written.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary metadata attached to every run (e.g. team, environment).",
    )

    observability: ObservabilityConfig = Field(
        default_factory=ObservabilityConfig,
        description="Latency, token usage and cost reporting.",
    )

    # --- Mode selection ----------------------------------------------------

    mode: Literal["local", "cloud"] = Field(
        default="local",
        description=(
            "Execution mode: 'local' runs evaluators in-process; "
            "'cloud' submits to Azure AI Foundry."
        ),
    )
    local_mode: LocalModeConfig | None = Field(
        default=None,
        description="Settings specific to local evaluation mode.",
    )
    cloud_mode: CloudModeConfig | None = Field(
        default=None,
        description="Settings specific to cloud evaluation mode.",
    )

    # ------------------------------------------------------------------
    # Field validators
    # ------------------------------------------------------------------

    @field_validator("report_format")
    @classmethod
    def report_format_must_be_supported(cls, v: list[str]) -> list[str]:
        supported = {"json", "markdown", "html"}
        for fmt in v:
            if fmt not in supported:
                raise ValueError(
                    f"Unsupported report format '{fmt}'. Supported: {sorted(supported)}"
                )
        return v

    # ------------------------------------------------------------------
    # Post-validator — ensure consistency
    # ------------------------------------------------------------------

    @model_validator(mode="after")
    def _ensure_consistency(self) -> Self:
        """Ensure the active mode block is present."""
        if self.mode == "local" and self.local_mode is None:
            raise ValueError("mode is 'local' but no 'local_mode' block was provided.")
        if self.mode == "cloud" and self.cloud_mode is None:
            raise ValueError("mode is 'cloud' but no 'cloud_mode' block was provided.")
        return self

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def evaluators(self) -> list[str]:
        """Return the evaluator list from the active mode block."""
        if self.mode == "local" and self.local_mode:
            return self.local_mode.evaluators
        if self.mode == "cloud" and self.cloud_mode:
            return self.cloud_mode.evaluators
        return []

    @property
    def thresholds(self) -> dict[str, float]:
        """Return the thresholds from the active mode block."""
        if self.mode == "local" and self.local_mode:
            return self.local_mode.thresholds
        if self.mode == "cloud" and self.cloud_mode:
            return self.cloud_mode.thresholds
        return {}

    @property
    def azure(self) -> AzureConfig | None:
        """Return the Azure config from the active mode block."""
        if self.mode == "local" and self.local_mode:
            return self.local_mode.azure
        if self.mode == "cloud" and self.cloud_mode:
            return self.cloud_mode.azure
        return None

    @property
    def foundry(self) -> FoundryConfig | None:
        """Return the Foundry config from the active mode block."""
        if self.mode == "cloud" and self.cloud_mode:
            return self.cloud_mode.foundry
        return None

    @property
    def judge(self) -> JudgeConfig | None:
        """Return the judge config (local mode only)."""
        if self.mode == "local" and self.local_mode:
            return self.local_mode.judge
        return None

    @property
    def evaluator_options(self) -> dict[str, dict[str, Any]]:
        """Return per-evaluator options (local mode only)."""
        if self.mode == "local" and self.local_mode:
            return self.local_mode.evaluator_options
        return {}

    @property
    def judge_model(self) -> str:
        """Return the judge model deployment name (local mode only)."""
        if self.mode == "local" and self.local_mode:
            return self.local_mode.judge_model
        return "gpt-4.1"

    def effective_judge_model(self) -> str:
        """Return the judge deployment name, preferring ``judge.model``."""
        if self.judge and self.judge.model:
            return self.judge.model
        return self.judge_model

    def effective_judge_endpoint(self) -> str:
        """Return the judge endpoint, falling back to ``azure.endpoint``."""
        if self.judge and self.judge.azure_endpoint:
            return self.judge.azure_endpoint
        if self.azure and self.azure.endpoint:
            return self.azure.endpoint
        return ""

    def is_cloud_mode(self) -> bool:
        """Return True if the project is configured for cloud execution."""
        return self.mode == "cloud"

    def is_local_mode(self) -> bool:
        """Return True if the project is configured for local execution."""
        return self.mode == "local"

    @property
    def mode_display(self) -> str:
        """Human-readable mode label for reports and CLI output."""
        return "LOCAL" if self.mode == "local" else "CLOUD (Azure AI Foundry)"
