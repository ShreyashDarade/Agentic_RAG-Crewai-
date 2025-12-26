"""
Retriever Module - Production Grade (Milvus Cloud)

Provides:
- Advanced Retriever with RRF fusion, re-ranking, and MMR diversity
- Web Retriever for online search
- Hybrid Retriever combining local and web search
"""

from .advanced_retriever import (
    AdvancedRetriever,
    AdvancedRetrieverConfig,
    AdvancedRetrieverError,
    RetrievalResult,
    create_advanced_retriever,
)
from .web_retriever import (
    WebRetriever,
    WebRetrieverConfig,
    WebSearchResult,
)

# Hybrid retriever (uses AdvancedRetriever + Web)
try:
    from .hybrid_retriever import (
        HybridRetriever,
        HybridRetrieverConfig,
        HybridRetrieverError,
        HybridResult,
        create_hybrid_retriever,
    )
except ImportError:
    HybridRetriever = None
    HybridRetrieverConfig = None
    HybridRetrieverError = None
    HybridResult = None
    create_hybrid_retriever = None

__all__ = [
    # Advanced Retriever (Primary - Milvus Cloud)
    "AdvancedRetriever",
    "AdvancedRetrieverConfig",
    "AdvancedRetrieverError",
    "RetrievalResult",
    "create_advanced_retriever",
    # Web Retriever
    "WebRetriever",
    "WebRetrieverConfig",
    "WebSearchResult",
    # Hybrid Retriever
    "HybridRetriever",
    "HybridRetrieverConfig",
    "HybridRetrieverError",
    "HybridResult",
    "create_hybrid_retriever",
]
