import json
import logging
import os
import shutil
import shutil
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class VectorStoreConfig:
    """Configuration for the vector store."""
    collection_name: str = "documents"
    persist_directory: str = "./data/chromadb"
    distance_metric: str = "cosine"  # cosine, l2, ip
    embedding_dimension: int = 384
    
    def __post_init__(self):
        """Validate configuration."""
        valid_metrics = ["cosine", "l2", "ip"]
        if self.distance_metric not in valid_metrics:
            raise ValueError(f"distance_metric must be one of {valid_metrics}")


@dataclass
class SearchResult:
    """Represents a single search result."""
    id: str
    content: str
    metadata: Dict[str, Any]
    score: float
    embedding: Optional[List[float]] = None


class VectorStoreError(Exception):
    """Exception raised for vector store errors."""
    
    def __init__(self, message: str, original_error: Optional[Exception] = None):
        self.message = message
        self.original_error = original_error
        super().__init__(message)


class VectorStore:
    """
    Vector store interface using ChromaDB.
    
    Provides methods for storing, retrieving, and searching
    document embeddings.
    """
    
    def __init__(self, config: Optional[VectorStoreConfig] = None):
        """
        Initialize the vector store.
        
        Args:
            config: Vector store configuration
        """
        self.config = config or VectorStoreConfig()
        self._client = None
        self._collection = None
        self._initialize_store()

    def _to_list(self, value):
        """Convert Chroma return values to plain Python lists."""
        if value is None:
            return []
        if isinstance(value, list):
            return value
        if hasattr(value, "tolist"):
            try:
                return value.tolist()
            except Exception:
                pass
        try:
            return list(value)
        except TypeError:
            return [value]
    
    def _create_client(self, persist_path: Path):
        import chromadb
        try:
            from chromadb.config import Settings as ChromaSettings
            settings = ChromaSettings(anonymized_telemetry=False)
            return chromadb.PersistentClient(path=str(persist_path), settings=settings)
        except Exception:
            return chromadb.PersistentClient(path=str(persist_path))
    
    def _initialize_store(self) -> None:
        """Initialize ChromaDB client and collection."""
        try:
            import chromadb
            persist_path = Path(self.config.persist_directory)
            persist_path.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Initializing ChromaDB at: {persist_path}")
            
            self._client = self._create_client(persist_path)
            self._collection = self._client.get_or_create_collection(
                name=self.config.collection_name,
                metadata={"hnsw:space": self.config.distance_metric}
            )
            
            logger.info(
                f"ChromaDB initialized. Collection: {self.config.collection_name}, "
                f"Documents: {self._collection.count()}"
            )
            
        except ImportError:
            raise VectorStoreError(
                "chromadb is required. Install with: pip install chromadb"
            )
        except Exception as e:
            if "_type" in str(e):
                persist_path = Path(self.config.persist_directory)
                logger.warning(
                    "Detected legacy Chroma persistence at %s. Resetting directory.",
                    persist_path
                )
                shutil.rmtree(persist_path, ignore_errors=True)
                persist_path.mkdir(parents=True, exist_ok=True)
                self._client = self._create_client(persist_path)
                self._collection = self._client.get_or_create_collection(
                    name=self.config.collection_name,
                    metadata={"hnsw:space": self.config.distance_metric}
                )
                logger.info("Chroma persistence directory reset.")
            else:
                raise VectorStoreError(f"Failed to initialize vector store: {str(e)}", e)
    
    def add(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """
        Add documents to the vector store.
        
        Args:
            ids: Unique identifiers for each document
            embeddings: Embedding vectors
            documents: Document texts (optional)
            metadatas: Metadata dictionaries (optional)
        """
        if not ids:
            logger.warning("No documents to add")
            return
        
        if len(ids) != len(embeddings):
            raise VectorStoreError(
                f"Mismatch: {len(ids)} ids but {len(embeddings)} embeddings"
            )
        
        # Validate and clean metadatas
        if metadatas:
            metadatas = [self._clean_metadata(m) for m in metadatas]
        
        try:
            # Check for existing IDs and filter them out
            existing_ids = set()
            try:
                existing = self._collection.get(ids=ids)
                existing_ids = set(existing["ids"])
            except Exception:
                pass
            
            # Filter out existing documents
            new_indices = [i for i, id_ in enumerate(ids) if id_ not in existing_ids]
            
            if not new_indices:
                logger.info("All documents already exist in the collection")
                return
            
            if len(new_indices) < len(ids):
                logger.info(f"Skipping {len(ids) - len(new_indices)} existing documents")
            
            # Prepare data for insertion
            new_ids = [ids[i] for i in new_indices]
            new_embeddings = [embeddings[i] for i in new_indices]
            new_documents = [documents[i] for i in new_indices] if documents else None
            new_metadatas = [metadatas[i] for i in new_indices] if metadatas else None
            
            # Add to collection
            self._collection.add(
                ids=new_ids,
                embeddings=new_embeddings,
                documents=new_documents,
                metadatas=new_metadatas
            )
            
            logger.info(f"Added {len(new_ids)} documents to collection")
            
        except Exception as e:
            raise VectorStoreError(f"Failed to add documents: {str(e)}", e)
    
    def _clean_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean metadata to ensure ChromaDB compatibility.
        
        ChromaDB only supports str, int, float, and bool values.
        """
        if not metadata:
            return {}
        
        cleaned = {}
        for key, value in metadata.items():
            if value is None:
                continue
            if isinstance(value, (str, int, float, bool)):
                cleaned[key] = value
            elif isinstance(value, (list, dict)):
                # Convert complex types to string
                import json
                cleaned[key] = json.dumps(value)
            else:
                cleaned[key] = str(value)
        
        return cleaned
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None,
        include_embeddings: bool = False
    ) -> List[SearchResult]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            where: Metadata filter conditions
            where_document: Document content filter conditions
            include_embeddings: Whether to include embeddings in results
            
        Returns:
            List of SearchResult objects
        """
        try:
            include = ["documents", "metadatas", "distances"]
            if include_embeddings:
                include.append("embeddings")
            
            raw_results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, self._collection.count() or 1),
                where=where,
                where_document=where_document,
                include=include
            )
            
            search_results = []
            
            ids_outer = self._to_list(raw_results.get("ids"))
            docs_outer = self._to_list(raw_results.get("documents"))
            metas_outer = self._to_list(raw_results.get("metadatas"))
            dists_outer = self._to_list(raw_results.get("distances"))
            embeds_outer = self._to_list(raw_results.get("embeddings")) if include_embeddings else []

            if ids_outer:
                first_ids = self._to_list(ids_outer[0])
                first_docs = self._to_list(docs_outer[0]) if docs_outer else []
                first_metas = self._to_list(metas_outer[0]) if metas_outer else []
                first_dists = self._to_list(dists_outer[0]) if dists_outer else []
                first_embeds = self._to_list(embeds_outer[0]) if embeds_outer else []

                for i, id_ in enumerate(first_ids):
                    # Convert distance to similarity score
                    distance = first_dists[i] if i < len(first_dists) else 0
                    
                    # For cosine distance, similarity = 1 - distance
                    if self.config.distance_metric == "cosine":
                        score = 1 - distance
                    else:
                        score = 1 / (1 + distance)  # Convert other distances
                    
                    result = SearchResult(
                        id=id_,
                        content=first_docs[i] if i < len(first_docs) else "",
                        metadata=first_metas[i] if i < len(first_metas) else {},
                        score=score,
                        embedding=first_embeds[i] if include_embeddings and i < len(first_embeds) else None
                    )
                    search_results.append(result)
            
            return search_results
            
        except Exception as e:
            raise VectorStoreError(f"Search failed: {str(e)}", e)
    
    def get(
        self,
        ids: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None
    ) -> List[SearchResult]:
        """
        Get documents by ID or filter.
        
        Args:
            ids: Document IDs to retrieve
            where: Metadata filter conditions
            limit: Maximum number of results
            
        Returns:
            List of SearchResult objects
        """
        try:
            raw_results = self._collection.get(
                ids=ids,
                where=where,
                limit=limit,
                include=["documents", "metadatas", "embeddings"]
            )
            
            search_results = []
            
            ids_data = self._to_list(raw_results.get("ids"))
            docs_data = self._to_list(raw_results.get("documents"))
            metas_data = self._to_list(raw_results.get("metadatas"))
            embeds_data = self._to_list(raw_results.get("embeddings"))

            if ids_data:
                for i, id_ in enumerate(ids_data):
                    result = SearchResult(
                        id=id_,
                        content=docs_data[i] if i < len(docs_data) else "",
                        metadata=metas_data[i] if i < len(metas_data) else {},
                        score=1.0,  # Perfect match for direct retrieval
                        embedding=embeds_data[i] if embeds_data and i < len(embeds_data) else None
                    )
                    search_results.append(result)
            
            return search_results
            
        except Exception as e:
            raise VectorStoreError(f"Get failed: {str(e)}", e)
    
    def update(
        self,
        ids: List[str],
        embeddings: Optional[List[List[float]]] = None,
        documents: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """
        Update existing documents.
        
        Args:
            ids: Document IDs to update
            embeddings: New embeddings (optional)
            documents: New document texts (optional)
            metadatas: New metadata (optional)
        """
        if not ids:
            return
        
        try:
            if metadatas:
                metadatas = [self._clean_metadata(m) for m in metadatas]
            
            self._collection.update(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            
            logger.info(f"Updated {len(ids)} documents")
            
        except Exception as e:
            raise VectorStoreError(f"Update failed: {str(e)}", e)
    
    def delete(
        self,
        ids: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Delete documents from the store.
        
        Args:
            ids: Document IDs to delete
            where: Metadata filter for deletion
        """
        try:
            if ids:
                self._collection.delete(ids=ids)
                logger.info(f"Deleted {len(ids)} documents by ID")
            elif where:
                self._collection.delete(where=where)
                logger.info(f"Deleted documents matching filter: {where}")
            else:
                logger.warning("No deletion criteria provided")
                
        except Exception as e:
            raise VectorStoreError(f"Delete failed: {str(e)}", e)
    
    def count(self) -> int:
        """Get the total number of documents."""
        return self._collection.count()
    
    def reset(self) -> None:
        """Reset the collection (delete all documents)."""
        try:
            self._client.delete_collection(self.config.collection_name)
            self._collection = self._client.get_or_create_collection(
                name=self.config.collection_name,
                metadata={"hnsw:space": self.config.distance_metric}
            )
            logger.info(f"Collection {self.config.collection_name} reset")
            
        except Exception as e:
            raise VectorStoreError(f"Reset failed: {str(e)}", e)
    
    def list_collections(self) -> List[str]:
        """List all collections in the database."""
        collections = self._client.list_collections()
        return [c.name for c in collections]
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the current collection."""
        return {
            "name": self.config.collection_name,
            "count": self._collection.count(),
            "distance_metric": self.config.distance_metric,
            "persist_directory": self.config.persist_directory
        }
    
    def __repr__(self) -> str:
        return f"VectorStore(collection={self.config.collection_name}, count={self.count()})"


def create_vector_store(
    collection_name: str = "documents",
    persist_directory: str = "./data/chromadb",
    **kwargs
) -> VectorStore:
    """
    Factory function to create a vector store.
    
    Args:
        collection_name: Name of the collection
        persist_directory: Directory for persistence
        **kwargs: Additional configuration options
        
    Returns:
        Configured VectorStore instance
    """
    config = VectorStoreConfig(
        collection_name=collection_name,
        persist_directory=persist_directory,
        **kwargs
    )
    return VectorStore(config)

