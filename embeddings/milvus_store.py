"""
Milvus Vector Store - Production Grade with Cloud Support

Features:
- Milvus Cloud (Zilliz) support
- HNSW indexing for optimal performance
- Multi-modal data support (text, tables, images)
- Advanced filtering and search
- Batch operations with retry logic
"""

import logging
import os
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class MilvusCloudConfig:
    """Configuration for Milvus Cloud (Zilliz) vector store."""
    # Collection settings
    collection_name: str = "documents"
    
    # Cloud connection (Zilliz Cloud)
    uri: str = ""  # Zilliz Cloud URI
    token: str = ""  # API token
    
    # Schema settings
    embedding_dimension: int = 1536  # OpenAI text-embedding-3-small
    id_max_length: int = 256
    content_max_length: int = 65535
    
    # HNSW Index settings (BEST for production)
    index_type: str = "HNSW"
    metric_type: str = "COSINE"  # COSINE for normalized embeddings
    hnsw_m: int = 32  # Higher M = better recall, more memory
    hnsw_ef_construction: int = 360  # Higher = better index quality
    hnsw_ef_search: int = 128  # Higher = better search quality
    
    # Search settings
    consistency_level: str = "Strong"  # Strong, Bounded, Eventually
    
    # Retry settings
    max_retries: int = 3
    retry_delay: float = 1.0
    
    def __post_init__(self):
        """Validate and load from environment."""
        # Load from environment if not set
        if not self.uri:
            self.uri = os.getenv("MILVUS_URI", os.getenv("ZILLIZ_URI", ""))
        if not self.token:
            self.token = os.getenv("MILVUS_TOKEN", os.getenv("ZILLIZ_TOKEN", ""))
        
        if not self.uri:
            raise ValueError(
                "Milvus Cloud URI is required. Set MILVUS_URI or ZILLIZ_URI environment variable"
            )


@dataclass
class MilvusSearchResult:
    """Represents a search result from Milvus."""
    id: str
    content: str
    metadata: Dict[str, Any]
    score: float
    content_type: str = "text"
    embedding: Optional[List[float]] = None
    
    # Cross-reference fields
    linked_elements: List[str] = field(default_factory=list)
    parent_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
            "score": self.score,
            "content_type": self.content_type,
            "linked_elements": self.linked_elements,
            "parent_id": self.parent_id
        }


class MilvusCloudError(Exception):
    """Exception for Milvus Cloud operations."""
    
    def __init__(self, message: str, original_error: Optional[Exception] = None):
        self.message = message
        self.original_error = original_error
        super().__init__(message)


class MilvusCloudStore:
    """
    Milvus Cloud Vector Store with HNSW indexing.
    
    Optimized for:
    - High-performance semantic search
    - Multi-modal data (text, tables, images)
    - Cross-reference links between chunks
    - Production-grade reliability
    """
    
    # Schema field definitions
    SCHEMA_FIELDS = {
        "id": {"type": "VARCHAR", "max_length": 256, "is_primary": True},
        "content": {"type": "VARCHAR", "max_length": 65535},
        "content_type": {"type": "VARCHAR", "max_length": 50},  # text, table, image
        "source": {"type": "VARCHAR", "max_length": 1024},
        "file_name": {"type": "VARCHAR", "max_length": 256},
        "chunk_index": {"type": "INT64"},
        "parent_id": {"type": "VARCHAR", "max_length": 256},
        "linked_elements": {"type": "VARCHAR", "max_length": 2048},  # JSON array
        "metadata_json": {"type": "VARCHAR", "max_length": 65535},
        "embedding": {"type": "FLOAT_VECTOR"},  # dimension set dynamically
    }
    
    def __init__(self, config: Optional[MilvusCloudConfig] = None):
        """
        Initialize Milvus Cloud store.
        
        Args:
            config: Cloud configuration
        """
        self.config = config or MilvusCloudConfig()
        self._client = None
        self._initialized = False
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize Milvus Cloud connection."""
        try:
            from pymilvus import MilvusClient
            
            logger.info(f"Connecting to Milvus Cloud: {self.config.uri[:50]}...")
            
            self._client = MilvusClient(
                uri=self.config.uri,
                token=self.config.token
            )
            
            # Ensure collection exists
            self._ensure_collection()
            
            self._initialized = True
            logger.info(f"Milvus Cloud connected. Collection: {self.config.collection_name}")
            
        except ImportError:
            raise MilvusCloudError(
                "pymilvus is required. Install with: pip install pymilvus"
            )
        except Exception as e:
            raise MilvusCloudError(f"Failed to connect to Milvus Cloud: {str(e)}", e)
    
    def _ensure_collection(self) -> None:
        """Ensure collection exists with HNSW index."""
        try:
            if self._client.has_collection(self.config.collection_name):
                logger.info(f"Collection '{self.config.collection_name}' exists")
                return
            
            logger.info(f"Creating collection '{self.config.collection_name}' with HNSW index")
            
            # Create schema
            schema = self._client.create_schema(
                auto_id=False,
                enable_dynamic_field=True
            )
            
            # Add fields
            schema.add_field(
                field_name="id",
                datatype="VARCHAR",
                is_primary=True,
                max_length=self.config.id_max_length
            )
            schema.add_field(
                field_name="content",
                datatype="VARCHAR",
                max_length=self.config.content_max_length
            )
            schema.add_field(
                field_name="content_type",
                datatype="VARCHAR",
                max_length=50
            )
            schema.add_field(
                field_name="embedding",
                datatype="FLOAT_VECTOR",
                dim=self.config.embedding_dimension
            )
            
            # Create HNSW index parameters (BEST for production)
            index_params = self._client.prepare_index_params()
            index_params.add_index(
                field_name="embedding",
                index_type="HNSW",
                metric_type=self.config.metric_type,
                params={
                    "M": self.config.hnsw_m,
                    "efConstruction": self.config.hnsw_ef_construction
                }
            )
            
            # Create collection
            self._client.create_collection(
                collection_name=self.config.collection_name,
                schema=schema,
                index_params=index_params,
                consistency_level=self.config.consistency_level
            )
            
            logger.info(f"Collection created with HNSW index (M={self.config.hnsw_m})")
            
        except Exception as e:
            raise MilvusCloudError(f"Failed to create collection: {str(e)}", e)
    
    def _with_retry(self, operation, *args, **kwargs):
        """Execute operation with retry logic."""
        last_error = None
        
        for attempt in range(self.config.max_retries):
            try:
                return operation(*args, **kwargs)
            except Exception as e:
                last_error = e
                if attempt < self.config.max_retries - 1:
                    wait_time = self.config.retry_delay * (2 ** attempt)
                    logger.warning(f"Operation failed, retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)
        
        raise MilvusCloudError(f"Operation failed after {self.config.max_retries} retries", last_error)
    
    def _clean_metadata(self, metadata: Dict[str, Any]) -> str:
        """Convert metadata to JSON string."""
        if not metadata:
            return "{}"
        
        cleaned = {}
        for key, value in metadata.items():
            if value is None:
                continue
            try:
                json.dumps(value)
                cleaned[key] = value
            except (TypeError, ValueError):
                cleaned[key] = str(value)
        
        return json.dumps(cleaned)
    
    def add(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        content_types: Optional[List[str]] = None
    ) -> None:
        """
        Add documents to the vector store.
        
        Args:
            ids: Document IDs
            embeddings: Embedding vectors
            documents: Document texts
            metadatas: Metadata dictionaries
            content_types: Content types (text, table, image)
        """
        if not ids:
            return
        
        if len(ids) != len(embeddings):
            raise MilvusCloudError(f"Mismatch: {len(ids)} ids, {len(embeddings)} embeddings")
        
        def _insert():
            data = []
            for i, (doc_id, embedding) in enumerate(zip(ids, embeddings)):
                record = {
                    "id": doc_id,
                    "embedding": embedding,
                    "content": documents[i] if documents else "",
                    "content_type": content_types[i] if content_types else "text",
                }
                
                if metadatas and i < len(metadatas):
                    meta = metadatas[i] or {}
                    record["source"] = meta.get("source", "")
                    record["file_name"] = meta.get("file_name", "")
                    record["chunk_index"] = meta.get("chunk_index", 0)
                    record["parent_id"] = meta.get("parent_id", "")
                    record["linked_elements"] = json.dumps(meta.get("linked_elements", []))
                    record["metadata_json"] = self._clean_metadata(meta)
                
                data.append(record)
            
            self._client.insert(
                collection_name=self.config.collection_name,
                data=data
            )
        
        self._with_retry(_insert)
        logger.info(f"Added {len(ids)} documents to Milvus Cloud")
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filter_expr: Optional[str] = None,
        content_type: Optional[str] = None,
        include_embeddings: bool = False
    ) -> List[MilvusSearchResult]:
        """
        Search for similar documents using HNSW.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results
            filter_expr: Milvus filter expression
            content_type: Filter by content type
            include_embeddings: Include vectors in results
            
        Returns:
            List of search results
        """
        def _search():
            # Build filter
            filters = []
            if filter_expr:
                filters.append(filter_expr)
            if content_type:
                filters.append(f'content_type == "{content_type}"')
            filter_string = " and ".join(filters) if filters else None
            
            # HNSW search parameters
            search_params = {
                "ef": self.config.hnsw_ef_search
            }
            
            # Output fields
            output_fields = [
                "id", "content", "content_type", "source", 
                "file_name", "chunk_index", "parent_id",
                "linked_elements", "metadata_json"
            ]
            if include_embeddings:
                output_fields.append("embedding")
            
            results = self._client.search(
                collection_name=self.config.collection_name,
                data=[query_embedding],
                limit=top_k,
                filter=filter_string,
                output_fields=output_fields,
                search_params=search_params
            )
            
            return results
        
        try:
            results = self._with_retry(_search)
            search_results = []
            
            if results and len(results) > 0:
                for hit in results[0]:
                    entity = hit.get("entity", {})
                    
                    # Parse metadata
                    metadata = {}
                    if "metadata_json" in entity:
                        try:
                            metadata = json.loads(entity.get("metadata_json", "{}"))
                        except json.JSONDecodeError:
                            pass
                    
                    # Parse linked elements
                    linked = []
                    if "linked_elements" in entity:
                        try:
                            linked = json.loads(entity.get("linked_elements", "[]"))
                        except json.JSONDecodeError:
                            pass
                    
                    # Score is already similarity for COSINE
                    score = hit.get("distance", 0)
                    
                    result = MilvusSearchResult(
                        id=entity.get("id", ""),
                        content=entity.get("content", ""),
                        metadata=metadata,
                        score=score,
                        content_type=entity.get("content_type", "text"),
                        linked_elements=linked,
                        parent_id=entity.get("parent_id"),
                        embedding=entity.get("embedding") if include_embeddings else None
                    )
                    search_results.append(result)
            
            return search_results
            
        except Exception as e:
            raise MilvusCloudError(f"Search failed: {str(e)}", e)
    
    def search_by_type(
        self,
        query_embedding: List[float],
        content_type: str,
        top_k: int = 5
    ) -> List[MilvusSearchResult]:
        """Search for specific content type."""
        return self.search(
            query_embedding=query_embedding,
            top_k=top_k,
            content_type=content_type
        )
    
    def get(
        self,
        ids: Optional[List[str]] = None,
        filter_expr: Optional[str] = None,
        limit: int = 100
    ) -> List[MilvusSearchResult]:
        """Get documents by ID or filter."""
        try:
            if ids:
                filter_expr = f'id in {ids}'
            
            results = self._client.query(
                collection_name=self.config.collection_name,
                filter=filter_expr or "",
                limit=limit,
                output_fields=["id", "content", "content_type", "metadata_json", "parent_id", "linked_elements"]
            )
            
            search_results = []
            for entity in results:
                metadata = {}
                if "metadata_json" in entity:
                    try:
                        metadata = json.loads(entity.get("metadata_json", "{}"))
                    except:
                        pass
                
                linked = []
                if "linked_elements" in entity:
                    try:
                        linked = json.loads(entity.get("linked_elements", "[]"))
                    except:
                        pass
                
                result = MilvusSearchResult(
                    id=entity.get("id", ""),
                    content=entity.get("content", ""),
                    metadata=metadata,
                    score=1.0,
                    content_type=entity.get("content_type", "text"),
                    linked_elements=linked,
                    parent_id=entity.get("parent_id")
                )
                search_results.append(result)
            
            return search_results
            
        except Exception as e:
            raise MilvusCloudError(f"Get failed: {str(e)}", e)
    
    def get_with_children(self, parent_id: str) -> List[MilvusSearchResult]:
        """Get parent chunk with all its children."""
        filter_expr = f'parent_id == "{parent_id}" or id == "{parent_id}"'
        return self.get(filter_expr=filter_expr)
    
    def get_linked_elements(self, chunk_id: str) -> List[MilvusSearchResult]:
        """Get all elements linked to a chunk."""
        try:
            # First get the chunk
            results = self.get(ids=[chunk_id])
            if not results:
                return []
            
            chunk = results[0]
            if not chunk.linked_elements:
                return []
            
            # Get linked elements
            return self.get(ids=chunk.linked_elements)
            
        except Exception as e:
            logger.warning(f"Failed to get linked elements: {e}")
            return []
    
    def delete(
        self,
        ids: Optional[List[str]] = None,
        filter_expr: Optional[str] = None
    ) -> None:
        """Delete documents."""
        try:
            if ids:
                self._client.delete(
                    collection_name=self.config.collection_name,
                    ids=ids
                )
                logger.info(f"Deleted {len(ids)} documents")
            elif filter_expr:
                self._client.delete(
                    collection_name=self.config.collection_name,
                    filter=filter_expr
                )
                logger.info(f"Deleted documents matching filter")
        except Exception as e:
            raise MilvusCloudError(f"Delete failed: {str(e)}", e)
    
    def count(self) -> int:
        """Get total document count."""
        try:
            stats = self._client.get_collection_stats(self.config.collection_name)
            return stats.get("row_count", 0)
        except:
            return 0
    
    def reset(self) -> None:
        """Reset collection (delete and recreate)."""
        try:
            self._client.drop_collection(self.config.collection_name)
            self._ensure_collection()
            logger.info("Collection reset complete")
        except Exception as e:
            raise MilvusCloudError(f"Reset failed: {str(e)}", e)
    
    def get_info(self) -> Dict[str, Any]:
        """Get collection info."""
        try:
            stats = self._client.get_collection_stats(self.config.collection_name)
            return {
                "name": self.config.collection_name,
                "count": stats.get("row_count", 0),
                "index_type": "HNSW",
                "metric_type": self.config.metric_type,
                "dimension": self.config.embedding_dimension,
                "hnsw_m": self.config.hnsw_m,
                "hnsw_ef_construction": self.config.hnsw_ef_construction
            }
        except Exception as e:
            return {"name": self.config.collection_name, "error": str(e)}


def create_milvus_cloud_store(
    uri: Optional[str] = None,
    token: Optional[str] = None,
    collection_name: str = "documents",
    **kwargs
) -> MilvusCloudStore:
    """
    Factory function for Milvus Cloud store.
    
    Args:
        uri: Zilliz Cloud URI
        token: API token
        collection_name: Collection name
        **kwargs: Additional config
        
    Returns:
        Configured MilvusCloudStore
    """
    config = MilvusCloudConfig(
        uri=uri or "",
        token=token or "",
        collection_name=collection_name,
        **kwargs
    )
    return MilvusCloudStore(config)
