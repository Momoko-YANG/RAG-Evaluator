"""Evaluators, grouped by layer.

* ``retrieval/``  — did the retriever return the right chunks? (Hit, MRR, Recall, nDCG, ...)
* ``generation/`` — is the answer faithful, relevant and correct? (groundedness, judges, ...)
* ``common/``     — shared bases: Azure AI Evaluation SDK wrapper, LLM judge, judge prompts.

The registry of evaluator names is :mod:`rag_eval_framework.core.registry`.
"""

from rag_eval_framework.evaluators.base import (
    BaseEvaluator,
    EvaluatorResult,
    EvaluatorStatus,
)

__all__ = ["BaseEvaluator", "EvaluatorResult", "EvaluatorStatus"]
