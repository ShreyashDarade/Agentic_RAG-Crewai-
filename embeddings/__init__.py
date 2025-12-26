"""
Embeddings Module - Production Grade (OpenAI + Milvus Cloud)

Provides:
- OpenAI embeddings (text-embedding-3-small/large)
- Milvus Cloud vector store with HNSW indexing
"""

from .openai_embedder import (
    OpenAIEmbedder,
    OpenAIEmbeddingConfig,
    OpenAIEmbeddingError,
    create_openai_embedder,
)
from .milvus_store import (
    MilvusCloudStore,
    MilvusCloudConfig,
    MilvusCloudError,
    MilvusSearchResult,
    create_milvus_cloud_store,
)

# Optional chunk tagging
try:
    from .chunk_tags import ChunkTagger, ChunkTagConfig
except ImportError:
    ChunkTagger = None
    ChunkTagConfig = None

__all__ = [
    # OpenAI Embeddings
    "OpenAIEmbedder",
    "OpenAIEmbeddingConfig",
    "OpenAIEmbeddingError",
    "create_openai_embedder",
    # Milvus Cloud Vector Store
    "MilvusCloudStore",
    "MilvusCloudConfig",
    "MilvusCloudError",
    "MilvusSearchResult",
    "create_milvus_cloud_store",
    # Chunk Tags
    "ChunkTagger",
    "ChunkTagConfig",
]
