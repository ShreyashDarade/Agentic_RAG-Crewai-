"""
Hybrid Retriever - Production Grade (Milvus + Web Search)

Combines local Milvus search with web search for comprehensive retrieval.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .advanced_retriever import AdvancedRetriever, AdvancedRetrieverConfig, RetrievalResult
from .web_retriever import WebRetriever, WebRetrieverConfig, WebSearchResult

logger = logging.getLogger(__name__)


@dataclass
class HybridRetrieverConfig:
    """Configuration for hybrid retrieval."""
    # Milvus settings
    collection_name: str = "documents"
    milvus_top_k: int = 10
    
    # Web search settings
    web_provider: str = "duckduckgo"
    web_api_key: Optional[str] = None
    web_max_results: int = 5
    
    # Hybrid settings
    enable_web_search: bool = True
    web_search_fallback_only: bool = False
    local_results_threshold: int = 3
    similarity_threshold: float = 0.3
    
    # Scoring weights
    local_weight: float = 0.7
    web_weight: float = 0.3
    
    # Deduplication
    deduplicate: bool = True
    dedup_threshold: float = 0.9


@dataclass
class HybridResult:
    """Represents a hybrid retrieval result."""
    id: str
    content: str
    source_type: str  # "local" or "web"
    source: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    url: Optional[str] = None
    title: Optional[str] = None
    content_type: str = "text"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "source_type": self.source_type,
            "source": self.source,
            "score": self.score,
            "metadata": self.metadata,
            "url": self.url,
            "title": self.title,
            "content_type": self.content_type
        }


class HybridRetrieverError(Exception):
    """Exception for hybrid retrieval errors."""
    pass


class HybridRetriever:
    """
    Hybrid retriever combining Milvus Cloud and web search.
    
    Features:
    - Milvus Cloud with HNSW indexing
    - Web search integration
    - Result fusion and deduplication
    - Configurable fallback strategies
    """
    
    def __init__(self, config: Optional[HybridRetrieverConfig] = None):
        """
        Initialize the hybrid retriever.
        
        Args:
            config: Hybrid retriever configuration
        """
        self.config = config or HybridRetrieverConfig()
        self._milvus_retriever: Optional[AdvancedRetriever] = None
        self._web_retriever: Optional[WebRetriever] = None
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize retriever components."""
        try:
            # Initialize Milvus-based advanced retriever
            milvus_config = AdvancedRetrieverConfig(
                collection_name=self.config.collection_name,
                top_k=self.config.milvus_top_k,
                similarity_threshold=self.config.similarity_threshold,
                enable_rerank=True,
                enable_diversity=True,
                enable_bm25=True
            )
            self._milvus_retriever = AdvancedRetriever(milvus_config)
            logger.info("Milvus retriever initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Milvus retriever: {e}")
            self._milvus_retriever = None
        
        # Initialize web retriever
        if self.config.enable_web_search:
            try:
                web_config = WebRetrieverConfig(
                    provider=self.config.web_provider,
                    api_key=self.config.web_api_key,
                    max_results=self.config.web_max_results
                )
                self._web_retriever = WebRetriever(web_config)
                logger.info("Web retriever initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize web retriever: {e}")
                self._web_retriever = None
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        use_web: Optional[bool] = None
    ) -> List[HybridResult]:
        """
        Perform hybrid search.
        
        Args:
            query: Search query
            top_k: Number of results
            use_web: Override web search setting
            
        Returns:
            List of HybridResult
        """
        all_results = []
        
        # Local Milvus search
        local_results = self._search_local(query, top_k)
        all_results.extend(local_results)
        
        # Determine if web search is needed
        should_search_web = use_web if use_web is not None else self.config.enable_web_search
        
        if self.config.web_search_fallback_only:
            should_search_web = len(local_results) < self.config.local_results_threshold
        
        # Web search
        if should_search_web and self._web_retriever:
            web_results = self._search_web(query, self.config.web_max_results)
            all_results.extend(web_results)
        
        # Deduplicate
        if self.config.deduplicate:
            all_results = self._deduplicate(all_results)
        
        # Sort by score
        all_results.sort(key=lambda x: x.score, reverse=True)
        
        return all_results[:top_k]
    
    def _search_local(self, query: str, top_k: int) -> List[HybridResult]:
        """Search local Milvus store."""
        if not self._milvus_retriever:
            return []
        
        try:
            results = self._milvus_retriever.search(
                query=query,
                top_k=top_k,
                methods=["dense", "bm25"]
            )
            
            hybrid_results = []
            for r in results:
                weighted_score = r.score * self.config.local_weight
                
                hybrid_results.append(HybridResult(
                    id=r.id,
                    content=r.content,
                    source_type="local",
                    source=r.source or r.metadata.get("source", "documents"),
                    score=weighted_score,
                    metadata=r.metadata,
                    content_type=r.content_type
                ))
            
            return hybrid_results
            
        except Exception as e:
            logger.error(f"Local search failed: {e}")
            return []
    
    def _search_web(self, query: str, max_results: int) -> List[HybridResult]:
        """Search web sources."""
        if not self._web_retriever:
            return []
        
        try:
            results = self._web_retriever.search(query, max_results)
            
            hybrid_results = []
            for i, r in enumerate(results):
                weighted_score = r.relevance_score * self.config.web_weight
                
                hybrid_results.append(HybridResult(
                    id=f"web_{i}_{hash(r.url) % 10000}",
                    content=r.snippet or r.content,
                    source_type="web",
                    source=r.source or r.url,
                    score=weighted_score,
                    metadata={"query": query},
                    url=r.url,
                    title=r.title
                ))
            
            return hybrid_results
            
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return []
    
    def _deduplicate(self, results: List[HybridResult]) -> List[HybridResult]:
        """Remove duplicate results."""
        seen_content = {}
        unique_results = []
        
        for result in results:
            content_key = result.content[:200].lower().strip()
            
            if content_key not in seen_content:
                seen_content[content_key] = result
                unique_results.append(result)
            else:
                existing = seen_content[content_key]
                if result.score > existing.score:
                    unique_results.remove(existing)
                    seen_content[content_key] = result
                    unique_results.append(result)
        
        return unique_results
    
    def search_local_only(self, query: str, top_k: int = 10) -> List[HybridResult]:
        """Search only local Milvus store."""
        return self.search(query, top_k, use_web=False)
    
    def search_web_only(self, query: str, max_results: int = 5) -> List[HybridResult]:
        """Search only web sources."""
        return self._search_web(query, max_results)
    
    @property
    def document_count(self) -> int:
        """Get local document count."""
        if self._milvus_retriever:
            return self._milvus_retriever.document_count
        return 0


def create_hybrid_retriever(**kwargs) -> HybridRetriever:
    """Factory function to create hybrid retriever."""
    config = HybridRetrieverConfig(**kwargs)
    return HybridRetriever(config)
