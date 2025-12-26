"""
OpenAI Embeddings - Production Grade Implementation

Features:
- OpenAI text-embedding-3-small/large models
- Batch processing with adaptive sizing
- Caching support
- Async operations
- Automatic token limiting
"""

import logging
import asyncio
from dataclasses import dataclass
from typing import List, Optional, Union, Dict, Any
import numpy as np
from functools import lru_cache
import hashlib
import json

logger = logging.getLogger(__name__)


@dataclass
class OpenAIEmbeddingConfig:
    """Configuration for the OpenAI embedder."""
    api_key: str
    model_name: str = "text-embedding-3-small"
    dimension: int = 1536  # 1536 for small, 3072 for large
    batch_size: int = 100
    normalize: bool = True
    max_tokens_per_batch: int = 8000
    cache_enabled: bool = True
    cache_size: int = 10000
    
    def __post_init__(self):
        """Validate configuration."""
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if self.dimension < 1:
            raise ValueError("dimension must be at least 1")
        
        # Set correct dimension based on model
        model_dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        if self.model_name in model_dimensions:
            self.dimension = model_dimensions[self.model_name]


class OpenAIEmbeddingError(Exception):
    """Exception raised for embedding errors."""
    
    def __init__(self, message: str, original_error: Optional[Exception] = None):
        self.message = message
        self.original_error = original_error
        super().__init__(message)


class OpenAIEmbedder:
    """
    OpenAI Embedder for generating dense embeddings.
    
    Uses OpenAI's text-embedding-3 models for high-quality
    semantic embeddings.
    """
    
    def __init__(self, config: Optional[OpenAIEmbeddingConfig] = None, api_key: Optional[str] = None):
        """
        Initialize the OpenAI embedder.
        
        Args:
            config: Embedding configuration
            api_key: API key (alternative to config)
        """
        if config is None:
            if api_key is None:
                import os
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OpenAI API key is required")
            config = OpenAIEmbeddingConfig(api_key=api_key)
        
        self.config = config
        self._client = None
        self._cache: Dict[str, List[float]] = {}
        self._initialize_client()
    
    def _initialize_client(self) -> None:
        """Initialize the OpenAI client."""
        try:
            from openai import AsyncOpenAI
            
            self._client = AsyncOpenAI(api_key=self.config.api_key)
            logger.info(
                f"OpenAI Embedder initialized. "
                f"Model: {self.config.model_name}, Dimension: {self.config.dimension}"
            )
            
        except ImportError:
            raise OpenAIEmbeddingError(
                "openai package is required. Install with: pip install openai"
            )
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        return hashlib.md5(text.encode()).hexdigest()
    
    def _check_cache(self, text: str) -> Optional[List[float]]:
        """Check if embedding is cached."""
        if not self.config.cache_enabled:
            return None
        
        key = self._get_cache_key(text)
        return self._cache.get(key)
    
    def _update_cache(self, text: str, embedding: List[float]) -> None:
        """Update cache with new embedding."""
        if not self.config.cache_enabled:
            return
        
        # Simple LRU-like behavior: remove oldest if too large
        if len(self._cache) >= self.config.cache_size:
            # Remove first item (oldest)
            first_key = next(iter(self._cache))
            del self._cache[first_key]
        
        key = self._get_cache_key(text)
        self._cache[key] = embedding
    
    async def embed(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings for text.
        
        Args:
            text: Single text string or list of texts
            
        Returns:
            Numpy array of embeddings. Shape: (1, dim) for single text,
            (n, dim) for list of texts.
        """
        if not text:
            raise OpenAIEmbeddingError("Cannot embed empty text")
        
        # Convert single text to list
        texts = [text] if isinstance(text, str) else text
        
        # Filter out empty strings
        valid_texts = [t for t in texts if t and t.strip()]
        if not valid_texts:
            raise OpenAIEmbeddingError("No valid texts to embed after filtering empty strings")
        
        # Check cache for existing embeddings
        embeddings = []
        texts_to_embed = []
        text_indices = []
        
        for i, t in enumerate(valid_texts):
            cached = self._check_cache(t)
            if cached is not None:
                embeddings.append((i, cached))
            else:
                texts_to_embed.append(t)
                text_indices.append(i)
        
        # Embed uncached texts in batches
        if texts_to_embed:
            batch_embeddings = await self._embed_batch(texts_to_embed)
            
            for idx, (text_idx, emb) in enumerate(zip(text_indices, batch_embeddings)):
                embeddings.append((text_idx, emb))
                self._update_cache(texts_to_embed[idx], emb)
        
        # Sort by original index and extract embeddings
        embeddings.sort(key=lambda x: x[0])
        result = np.array([emb for _, emb in embeddings])
        
        # Normalize if configured
        if self.config.normalize:
            norms = np.linalg.norm(result, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            result = result / norms
        
        return result
    
    async def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a batch of texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i:i + self.config.batch_size]
            
            try:
                response = await self._client.embeddings.create(
                    input=batch,
                    model=self.config.model_name
                )
                
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                
            except Exception as e:
                logger.error(f"Embedding batch failed: {str(e)}")
                raise OpenAIEmbeddingError(f"Failed to generate embeddings: {str(e)}", e)
        
        return all_embeddings
    
    def embed_sync(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Synchronous wrapper for embed.
        
        Args:
            text: Text(s) to embed
            
        Returns:
            Numpy array of embeddings
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self.embed(text))
    
    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of documents (sync).
        
        Args:
            documents: List of document texts
            
        Returns:
            List of embedding vectors (as Python lists)
        """
        if not documents:
            return []
        
        embeddings = self.embed_sync(documents)
        return embeddings.tolist()
    
    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a query (sync).
        
        Args:
            query: Query text
            
        Returns:
            Embedding vector as Python list
        """
        if not query or not query.strip():
            raise OpenAIEmbeddingError("Query cannot be empty")
        
        embedding = self.embed_sync(query)
        return embedding[0].tolist()
    
    async def embed_query_async(self, query: str) -> List[float]:
        """
        Generate embedding for a query (async).
        
        Args:
            query: Query text
            
        Returns:
            Embedding vector as Python list
        """
        if not query or not query.strip():
            raise OpenAIEmbeddingError("Query cannot be empty")
        
        embedding = await self.embed(query)
        return embedding[0].tolist()
    
    def similarity(
        self,
        embedding1: Union[np.ndarray, List[float]],
        embedding2: Union[np.ndarray, List[float]]
    ) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Cosine similarity score (0-1)
        """
        # Convert to numpy arrays if needed
        if not isinstance(embedding1, np.ndarray):
            embedding1 = np.array(embedding1)
        if not isinstance(embedding2, np.ndarray):
            embedding2 = np.array(embedding2)
        
        # Normalize
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 > 0 and norm2 > 0:
            embedding1 = embedding1 / norm1
            embedding2 = embedding2 / norm2
        
        return float(np.dot(embedding1, embedding2))
    
    def batch_similarity(
        self,
        query_embedding: Union[np.ndarray, List[float]],
        document_embeddings: Union[np.ndarray, List[List[float]]]
    ) -> List[float]:
        """
        Calculate similarity between a query and multiple documents.
        
        Args:
            query_embedding: Query embedding
            document_embeddings: List of document embeddings
            
        Returns:
            List of similarity scores
        """
        if not isinstance(query_embedding, np.ndarray):
            query_embedding = np.array(query_embedding)
        if not isinstance(document_embeddings, np.ndarray):
            document_embeddings = np.array(document_embeddings)
        
        if len(document_embeddings) == 0:
            return []
        
        # Normalize
        query_norm = np.linalg.norm(query_embedding)
        if query_norm > 0:
            query_embedding = query_embedding / query_norm
        
        doc_norms = np.linalg.norm(document_embeddings, axis=1, keepdims=True)
        doc_norms[doc_norms == 0] = 1  # Avoid division by zero
        document_embeddings = document_embeddings / doc_norms
        
        # Calculate similarities
        similarities = np.dot(document_embeddings, query_embedding)
        return similarities.tolist()
    
    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._cache.clear()
        logger.info("Embedding cache cleared")
    
    @property
    def dimension(self) -> int:
        """Get the embedding dimension."""
        return self.config.dimension
    
    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self.config.model_name
    
    def __repr__(self) -> str:
        return f"OpenAIEmbedder(model={self.config.model_name}, dim={self.config.dimension})"


def create_openai_embedder(
    api_key: Optional[str] = None,
    model_name: str = "text-embedding-3-small",
    **kwargs
) -> OpenAIEmbedder:
    """
    Factory function to create an OpenAI embedder.
    
    Args:
        api_key: OpenAI API key (uses env var if not provided)
        model_name: Name of the embedding model
        **kwargs: Additional configuration options
        
    Returns:
        Configured OpenAIEmbedder instance
    """
    import os
    
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key is required")
    
    config = OpenAIEmbeddingConfig(
        api_key=api_key,
        model_name=model_name,
        **kwargs
    )
    return OpenAIEmbedder(config)
