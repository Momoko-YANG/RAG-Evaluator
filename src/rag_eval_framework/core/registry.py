"""Evaluator registry — maps string names to evaluator instances.

The framework maintains two **evaluator catalogs** that define which
evaluators are available in each execution mode:

* **Local catalog** — all 21 evaluators (retrieval and generation layers).
* **Cloud catalog** — the 4 Azure SDK evaluators supported by Foundry.

The catalogs are defined in ``config.models`` and exposed here for
convenience.  The :class:`EvaluatorRegistry` resolves names to instances;
the catalogs control which names are *valid* for a given mode.
"""

from __future__ import annotations

import logging
from typing import Literal

from rag_eval_framework.config.models import (
    CLOUD_EVALUATOR_NAMES,
    LOCAL_EVALUATOR_NAMES,
)
from rag_eval_framework.evaluators.base import BaseEvaluator

logger = logging.getLogger(__name__)


class EvaluatorRegistry:
    """A registry that maps evaluator names to their concrete classes.

    Evaluators are registered once and instantiated on demand.  This keeps
    configuration files human-readable (just evaluator names) while the
    registry handles resolution.

    Usage
    -----
    >>> registry = EvaluatorRegistry()
    >>> registry.register(MyEvaluator)
    >>> evaluator = registry.get("my_evaluator")
    """

    def __init__(self) -> None:
        self._evaluators: dict[str, type[BaseEvaluator]] = {}

    def register(self, evaluator_cls: type[BaseEvaluator]) -> None:
        """Register an evaluator class.

        The evaluator's ``name`` property is used as the registry key.
        """
        # Instantiate briefly to read the name property
        instance = evaluator_cls()
        name = instance.name
        if name in self._evaluators:
            logger.warning("Overwriting existing evaluator '%s'", name)
        self._evaluators[name] = evaluator_cls
        logger.debug("Registered evaluator '%s' -> %s", name, evaluator_cls.__name__)

    def get(self, name: str) -> BaseEvaluator:
        """Return a fresh instance of the evaluator registered under *name*.

        Raises
        ------
        KeyError
            If no evaluator is registered with that name.
        """
        if name not in self._evaluators:
            available = sorted(self._evaluators.keys())
            raise KeyError(f"No evaluator registered with name '{name}'. Available: {available}")
        return self._evaluators[name]()

    def list_evaluators(self) -> list[str]:
        """Return sorted list of registered evaluator names."""
        return sorted(self._evaluators.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._evaluators

    def __len__(self) -> int:
        return len(self._evaluators)

    # ------------------------------------------------------------------
    # Mode-aware helpers
    # ------------------------------------------------------------------

    def list_for_mode(self, mode: Literal["local", "cloud"]) -> list[str]:
        """Return sorted evaluator names available for the given mode."""
        catalog = LOCAL_EVALUATOR_NAMES if mode == "local" else CLOUD_EVALUATOR_NAMES
        return sorted(name for name in self._evaluators if name in catalog)

    def get_for_mode(self, name: str, mode: Literal["local", "cloud"]) -> BaseEvaluator:
        """Return a fresh instance, validating *name* against the mode catalog.

        Raises
        ------
        KeyError
            If the evaluator is not registered or not in the mode's catalog.
        """
        catalog = LOCAL_EVALUATOR_NAMES if mode == "local" else CLOUD_EVALUATOR_NAMES
        if name not in catalog:
            raise KeyError(
                f"Evaluator '{name}' is not available in {mode} mode. Available: {sorted(catalog)}"
            )
        return self.get(name)


def _create_default_registry() -> EvaluatorRegistry:
    """Create a registry pre-loaded with every built-in evaluator, by layer.

    Azure SDK evaluators and LLM judges are always registered; they produce a
    clear error at evaluation time when their dependencies are missing.
    """
    from rag_eval_framework.evaluators.e2e.abstention import AbstentionEvaluator
    from rag_eval_framework.evaluators.e2e.access_control import AccessControlEvaluator
    from rag_eval_framework.evaluators.e2e.query_construction import QueryConstructionEvaluator
    from rag_eval_framework.evaluators.e2e.task_success import TaskSuccessEvaluator
    from rag_eval_framework.evaluators.e2e.tool_selection import ToolSelectionEvaluator
    from rag_eval_framework.evaluators.generation.answer_presence import AnswerPresenceEvaluator
    from rag_eval_framework.evaluators.generation.citation import CitationJudge
    from rag_eval_framework.evaluators.generation.completeness import (
        ResponseCompletenessEvaluator,
    )
    from rag_eval_framework.evaluators.generation.correctness import (
        AccuracyJudge,
        ExactMatchAccuracyEvaluator,
    )
    from rag_eval_framework.evaluators.generation.faithfulness import (
        GroundednessEvaluator,
        HallucinationJudge,
    )
    from rag_eval_framework.evaluators.generation.policy_compliance import (
        PolicyComplianceJudge,
    )
    from rag_eval_framework.evaluators.generation.relevance import RelevanceEvaluator
    from rag_eval_framework.evaluators.retrieval.context_presence import (
        ContextPresenceEvaluator,
    )
    from rag_eval_framework.evaluators.retrieval.context_relevance import RetrievalEvaluator
    from rag_eval_framework.evaluators.retrieval.hit import RetrievalHitRateEvaluator
    from rag_eval_framework.evaluators.retrieval.mrr import RetrievalMRREvaluator
    from rag_eval_framework.evaluators.retrieval.ndcg import (
        AzureDocumentRetrievalEvaluator,
        RetrievalNDCGEvaluator,
    )
    from rag_eval_framework.evaluators.retrieval.recall import RetrievalRecallEvaluator

    registry = EvaluatorRegistry()

    # Retrieval layer
    registry.register(ContextPresenceEvaluator)
    registry.register(RetrievalHitRateEvaluator)
    registry.register(RetrievalMRREvaluator)
    registry.register(RetrievalRecallEvaluator)
    registry.register(RetrievalNDCGEvaluator)
    registry.register(AzureDocumentRetrievalEvaluator)
    registry.register(RetrievalEvaluator)

    # Generation layer
    registry.register(AnswerPresenceEvaluator)
    registry.register(GroundednessEvaluator)
    registry.register(HallucinationJudge)
    registry.register(RelevanceEvaluator)
    registry.register(ExactMatchAccuracyEvaluator)
    registry.register(AccuracyJudge)
    registry.register(ResponseCompletenessEvaluator)
    registry.register(CitationJudge)
    registry.register(PolicyComplianceJudge)

    # End-to-end layer
    registry.register(TaskSuccessEvaluator)
    registry.register(ToolSelectionEvaluator)
    registry.register(QueryConstructionEvaluator)
    registry.register(AbstentionEvaluator)
    registry.register(AccessControlEvaluator)

    return registry


# Module-level singleton used by runners and CLI
default_registry: EvaluatorRegistry = _create_default_registry()
