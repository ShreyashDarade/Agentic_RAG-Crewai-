import logging
from dataclasses import dataclass
from typing import List, Optional, Union
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingConfig:
    """Configuration for the embedder."""
    model_name: str = "all-MiniLM-L6-v2"
    dimension: int = 384
    batch_size: int = 32
    normalize: bool = True
    device: str = "cpu"  # "cpu", "cuda", "mps"
    show_progress: bool = False
    
    def __post_init__(self):
        """Validate configuration."""
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if self.dimension < 1:
            raise ValueError("dimension must be at least 1")


class EmbeddingError(Exception):
    """Exception raised for embedding errors."""
    
    def __init__(self, message: str, original_error: Optional[Exception] = None):
        self.message = message
        self.original_error = original_error
        super().__init__(message)


class Embedder:
    """
    Embedder class for generating dense embeddings.
    
    Uses sentence-transformers models (default: MiniLM) for
    generating semantic embeddings of text.
    """
    
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        """
        Initialize the embedder.
        
        Args:
            config: Embedding configuration. Uses defaults if not provided.
        """
        self.config = config or EmbeddingConfig()
        self._model = None
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        """Initialize the sentence transformer model."""
        try:
            from sentence_transformers import SentenceTransformer
            
            logger.info(f"Loading embedding model: {self.config.model_name}")
            
            self._model = SentenceTransformer(
                self.config.model_name,
                device=self.config.device
            )
            
            # Update dimension from model if different
            actual_dim = self._model.get_sentence_embedding_dimension()
            if actual_dim != self.config.dimension:
                logger.warning(
                    f"Model dimension ({actual_dim}) differs from config ({self.config.dimension}). "
                    f"Using model dimension."
                )
                self.config.dimension = actual_dim
            
            logger.info(
                f"Embedding model loaded successfully. "
                f"Dimension: {self.config.dimension}, Device: {self.config.device}"
            )
            
        except ImportError:
            raise EmbeddingError(
                "sentence-transformers is required. Install with: pip install sentence-transformers"
            )
        except Exception as e:
            raise EmbeddingError(f"Failed to load embedding model: {str(e)}", e)
    
    def embed(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings for text.
        
        Args:
            text: Single text string or list of texts
            
        Returns:
            Numpy array of embeddings. Shape: (1, dim) for single text,
            (n, dim) for list of texts.
        """
        if not text:
            raise EmbeddingError("Cannot embed empty text")
        
        # Convert single text to list
        texts = [text] if isinstance(text, str) else text
        
        # Filter out empty strings
        valid_texts = [t for t in texts if t and t.strip()]
        if not valid_texts:
            raise EmbeddingError("No valid texts to embed after filtering empty strings")
        
        try:
            embeddings = self._model.encode(
                valid_texts,
                batch_size=self.config.batch_size,
                normalize_embeddings=self.config.normalize,
                show_progress_bar=self.config.show_progress,
                convert_to_numpy=True
            )
            
            return embeddings
            
        except Exception as e:
            raise EmbeddingError(f"Failed to generate embeddings: {str(e)}", e)
    
    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of documents.
        
        Args:
            documents: List of document texts
            
        Returns:
            List of embedding vectors (as Python lists)
        """
        if not documents:
            return []
        
        embeddings = self.embed(documents)
        return embeddings.tolist()
    
    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a query.
        
        Args:
            query: Query text
            
        Returns:
            Embedding vector as Python list
        """
        if not query or not query.strip():
            raise EmbeddingError("Query cannot be empty")
        
        embedding = self.embed(query)
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
        
        # Normalize if needed
        if self.config.normalize:
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
    
    @property
    def dimension(self) -> int:
        """Get the embedding dimension."""
        return self.config.dimension
    
    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self.config.model_name
    
    def __repr__(self) -> str:
        return f"Embedder(model={self.config.model_name}, dim={self.config.dimension})"


def create_embedder(
    model_name: str = "all-MiniLM-L6-v2",
    **kwargs
) -> Embedder:
    """
    Factory function to create an embedder.
    
    Args:
        model_name: Name of the sentence-transformers model
        **kwargs: Additional configuration options
        
    Returns:
        Configured Embedder instance
    """
    config = EmbeddingConfig(model_name=model_name, **kwargs)
    return Embedder(config)

