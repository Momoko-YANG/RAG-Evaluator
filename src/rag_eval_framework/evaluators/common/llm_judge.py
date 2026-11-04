"""Base class for LLM-as-a-judge evaluators.

:class:`BaseLLMJudge` provides the full lifecycle:

1. ``setup()`` — lazily import the Azure OpenAI client SDK and store
   connection settings from the project config.
2. ``evaluate()`` — format the prompt pair, call the LLM, parse the
   structured JSON response, normalise to ``EvaluatorResult``.

Concrete judges only need to supply:

* ``name`` / ``description``
* ``system_prompt`` / ``user_prompt_template``
* ``required_fields``
"""

from __future__ import annotations

import hashlib
import json
import logging
from abc import abstractmethod
from typing import Any

from rag_eval_framework.core.schemas import EvaluationRecord
from rag_eval_framework.evaluators.base import (
    BaseEvaluator,
    EvaluatorResult,
    EvaluatorStatus,
)
from rag_eval_framework.utils.normalisation import normalise_1_5

logger = logging.getLogger(__name__)

_OPENAI_INSTALL_HINT = (
    "The 'openai' package is required for LLM judge evaluators.  "
    "Install with:  pip install openai azure-identity"
)


class BaseLLMJudge(BaseEvaluator):
    """Abstract base for LLM-as-a-judge evaluators.

    The judge sends a system prompt + user prompt to an Azure OpenAI
    deployment and expects a JSON response with ``score`` (1-5) and
    ``reason`` keys.
    """

    _client: Any = None
    _deployment: str = ""
    _temperature: float = 0.0
    _max_tokens: int = 1024
    _configured: bool = False

    # -- subclass API ------------------------------------------------------

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """System-level instruction for the judge LLM."""
        ...

    @property
    @abstractmethod
    def user_prompt_template(self) -> str:
        """User prompt template with ``{placeholders}`` for record fields."""
        ...

    def definition(self) -> dict[str, Any]:
        prompt = f"{self.system_prompt}\n---\n{self.user_prompt_template}"
        return {
            **super().definition(),
            "backend": "llm_judge",
            "judge_model": self._deployment or None,
            "temperature": self._temperature,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16],
            "scale": "1-5 normalised to 0-1",
        }

    # -- lifecycle ---------------------------------------------------------

    def setup(self, config: Any) -> None:
        """Initialise the Azure OpenAI client from project config."""
        from rag_eval_framework.config.models import ProjectConfig

        if not isinstance(config, ProjectConfig):
            return

        endpoint = config.effective_judge_endpoint()
        deployment = config.effective_judge_model()

        if not endpoint or not deployment:
            logger.warning(
                "Judge '%s': endpoint or deployment not configured. "
                "Set azure.endpoint + judge_model (or judge section).",
                self.name,
            )
            return

        temperature = self._temperature
        max_tokens = self._max_tokens

        if config.judge:
            temperature = config.judge.temperature
            max_tokens = config.judge.max_tokens

        # Per-evaluator overrides
        opts = config.evaluator_options.get(self.name, {})
        temperature = opts.get("temperature", temperature)
        max_tokens = opts.get("max_tokens", max_tokens)

        try:
            self._client = _build_openai_client(config)
        except ImportError:
            logger.warning(_OPENAI_INSTALL_HINT)
            return

        self._deployment = deployment
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._configured = True
        logger.info(
            "Judge '%s' configured (deployment=%s, endpoint=%s).",
            self.name,
            deployment,
            endpoint,
        )

    # -- evaluation --------------------------------------------------------

    def evaluate(self, record: EvaluationRecord) -> EvaluatorResult:
        if not self._configured:
            return EvaluatorResult(
                score=0.0,
                status=EvaluatorStatus.ERROR,
                reason=(
                    f"Judge '{self.name}' is not configured. "
                    "Provide azure + judge settings in the project config."
                ),
            )

        user_prompt = self._format_user_prompt(record)

        try:
            raw = self._call_llm(user_prompt)
            parsed = self._parse_response(raw)
        except Exception as exc:
            logger.exception("Judge '%s' failed on record '%s'.", self.name, record.id)
            return EvaluatorResult(
                score=0.0,
                status=EvaluatorStatus.ERROR,
                reason=f"LLM call failed: {exc}",
            )

        raw_score = parsed.get("score", 0)
        reason = parsed.get("reason", "")
        score = normalise_1_5(float(raw_score))

        return EvaluatorResult(
            score=score,
            reason=str(reason),
            metadata={"raw_score": raw_score, "judge_model": self._deployment},
            raw_output=parsed,
        )

    # -- internals ---------------------------------------------------------

    def _format_user_prompt(self, record: EvaluationRecord) -> str:
        """Fill the user prompt template from the record."""
        context_str = "\n---\n".join(record.contexts) if record.contexts else "(none)"
        policy = record.metadata.get("policy", "(no policy provided)")
        return self.user_prompt_template.format(
            question=record.question,
            response=record.response,
            contexts=context_str,
            ground_truth_answer=record.ground_truth_answer or "(not provided)",
            policy=policy,
        )

    def _call_llm(self, user_prompt: str) -> str:
        """Send the prompt pair to the Azure OpenAI deployment."""
        try:
            response = self._client.chat.completions.create(
                model=self._deployment,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                response_format={"type": "json_object"},
            )
        except Exception as first_err:
            # Some newer models require max_completion_tokens instead of max_tokens.
            if "max_tokens" in str(first_err):
                logger.debug(
                    "Retrying with max_completion_tokens for model %s",
                    self._deployment,
                )
                response = self._client.chat.completions.create(
                    model=self._deployment,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self._temperature,
                    max_completion_tokens=self._max_tokens,
                    response_format={"type": "json_object"},
                )
            else:
                raise
        return response.choices[0].message.content or ""

    @staticmethod
    def _parse_response(raw: str) -> dict[str, Any]:
        """Parse the JSON response from the judge LLM."""
        try:
            return json.loads(raw)  # type: ignore[no-any-return]
        except json.JSONDecodeError:
            return {"score": 0, "reason": f"Failed to parse judge response: {raw[:200]}"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_openai_client(config: Any) -> Any:
    """Build an ``AzureOpenAI`` client from project config."""
    try:
        from openai import AzureOpenAI
    except ImportError as exc:
        raise ImportError(_OPENAI_INSTALL_HINT) from exc

    from rag_eval_framework.config.models import ProjectConfig

    if not isinstance(config, ProjectConfig):
        raise TypeError("Expected ProjectConfig")

    endpoint = config.effective_judge_endpoint()
    api_version = config.judge.api_version if config.judge else "2024-12-01-preview"

    # Determine credential
    azure_cfg = config.azure
    if azure_cfg and azure_cfg.credential_type == "key":
        import os

        api_key = os.environ.get(azure_cfg.api_key_env, "")
        return AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
        )

    # Default: use DefaultAzureCredential via token provider
    try:
        from azure.identity import DefaultAzureCredential, get_bearer_token_provider

        credential = DefaultAzureCredential()
        token_provider = get_bearer_token_provider(
            credential,
            "https://cognitiveservices.azure.com/.default",
        )
        return AzureOpenAI(
            azure_endpoint=endpoint,
            azure_ad_token_provider=token_provider,
            api_version=api_version,
        )
    except ImportError:
        # Fall back to key-less client (user may provide API key via env)
        return AzureOpenAI(
            azure_endpoint=endpoint,
            api_version=api_version,
        )
