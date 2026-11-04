"""RAG system adapters used by ``scripts/collect_rag_data.py`` (``--adapter <name>``).

Add an adapter by subclassing :class:`RagAdapter` and registering it in
:data:`ADAPTER_REGISTRY`.
"""

from rag_eval_framework.adapters.base import RagAdapter, RagResponse
from rag_eval_framework.adapters.generic import (
    AzureChatStreamAdapter,
    HttpAdapter,
)
from rag_eval_framework.adapters.rag_eval_api import RagEvalApiAdapter

ADAPTER_REGISTRY: dict[str, type[RagAdapter]] = {
    "http": HttpAdapter,
    "azure_chat_stream": AzureChatStreamAdapter,
    "rag_eval_api": RagEvalApiAdapter,
}

__all__ = [
    "ADAPTER_REGISTRY",
    "AzureChatStreamAdapter",
    "HttpAdapter",
    "RagAdapter",
    "RagEvalApiAdapter",
    "RagResponse",
]
