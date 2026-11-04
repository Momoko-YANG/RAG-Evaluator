"""Azure AI Foundry integration layer for cloud evaluation.

This package isolates all ``azure-ai-projects`` SDK interactions so that
the rest of the framework never imports Azure-specific code directly.
"""

from rag_eval_framework.runners.foundry.adapter import FoundryAdapter

__all__ = ["FoundryAdapter"]
