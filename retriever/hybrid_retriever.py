import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from .chroma_retriever import ChromaRetriever, ChromaRetrieverConfig, RetrievalResult
from .web_retriever import WebRetriever, WebRetrieverConfig, WebSearchResult
from common.async_utils import run_async_task

logger = logging.getLogger(__name__)


@dataclass
class HybridRetrieverConfig:
    """Configuration for hybrid retrieval."""
    # Chroma settings
    collection_name: str = "documents"
    persist_directory: str = "./data/chromadb"
    chroma_top_k: int = 10
    
    # Web search settings
    web_provider: str = "duckduckgo"
    web_api_key: Optional[str] = None
    web_max_results: int = 5
    
    # Hybrid settings
    enable_web_search: bool = True
    web_search_fallback_only: bool = False  # Only use web if local results are poor
    local_results_threshold: int = 3  # Min local results before triggering web search
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
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "source_type": self.source_type,
            "source": self.source,
            "score": self.score,
            "metadata": self.metadata,
            "url": self.url,
            "title": self.title
        }


class HybridRetrieverError(Exception):
    """Exception raised for hybrid retrieval errors."""
    pass


class HybridRetriever:
    """
    Hybrid retriever combining local (ChromaDB) and web search.
    
    Features:
    - Seamless local and web search combination
    - Configurable search strategies
    - Result deduplication
    - Flexible scoring and ranking
    - Fallback mechanisms
    """
    
    def __init__(self, config: Optional[HybridRetrieverConfig] = None):
        """
        Initialize the hybrid retriever.
        
        Args:
            config: Hybrid retriever configuration
        """
        self.config = config or HybridRetrieverConfig()
        self._chroma_retriever: Optional[ChromaRetriever] = None
        self._web_retriever: Optional[WebRetriever] = None
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize retriever components."""
        # Initialize ChromaDB retriever
        chroma_config = ChromaRetrieverConfig(
            collection_name=self.config.collection_name,
            persist_directory=self.config.persist_directory,
            top_k=self.config.chroma_top_k,
            similarity_threshold=self.config.similarity_threshold
        )
        self._chroma_retriever = ChromaRetriever(chroma_config)
        
        # Initialize web retriever if enabled
        if self.config.enable_web_search:
            web_config = WebRetrieverConfig(
                provider=self.config.web_provider,
                api_key=self.config.web_api_key,
                max_results=self.config.web_max_results
            )
            self._web_retriever = WebRetriever(web_config)
        
        logger.info("Hybrid retriever initialized")
    
    def _convert_local_result(self, result: RetrievalResult) -> HybridResult:
        """Convert local retrieval result to hybrid result."""
        return HybridResult(
            id=result.id,
            content=result.content,
            source_type="local",
            source=result.source or "chromadb",
            score=result.score * self.config.local_weight,
            metadata=result.metadata,
            title=result.metadata.get("file_name")
        )
    
    def _convert_web_result(self, result: WebSearchResult, index: int) -> HybridResult:
        """Convert web search result to hybrid result."""
        return HybridResult(
            id=f"web_{index}_{hash(result.url)}",
            content=result.snippet,
            source_type="web",
            source=result.source,
            score=result.score * self.config.web_weight,
            metadata=result.metadata,
            url=result.url,
            title=result.title
        )
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity for deduplication."""
        if not text1 or not text2:
            return 0.0
        
        # Simple word overlap similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _deduplicate_results(
        self,
        results: List[HybridResult]
    ) -> List[HybridResult]:
        """Remove duplicate or highly similar results."""
        if not self.config.deduplicate or len(results) <= 1:
            return results
        
        unique_results = []
        
        for result in results:
            is_duplicate = False
            
            for existing in unique_results:
                similarity = self._calculate_text_similarity(
                    result.content,
                    existing.content
                )
                
                if similarity >= self.config.dedup_threshold:
                    # Keep the one with higher score
                    if result.score > existing.score:
                        unique_results.remove(existing)
                        unique_results.append(result)
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_results.append(result)
        
        return unique_results
    
    async def search(
        self,
        query: str,
        top_k: int = 10,
        include_web: Optional[bool] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[HybridResult]:
        """
        Perform hybrid search combining local and web results.
        
        Args:
            query: Search query
            top_k: Total number of results to return
            include_web: Override web search setting
            filters: Metadata filters for local search
            
        Returns:
            List of HybridResult objects
        """
        if not query or not query.strip():
            return []
        
        results: List[HybridResult] = []
        
        # Get local results
        local_results = self._chroma_retriever.search(
            query,
            top_k=self.config.chroma_top_k,
            filters=filters
        )
        
        for r in local_results:
            results.append(self._convert_local_result(r))
        
        logger.info(f"Local search returned {len(local_results)} results")
        
        # Determine if web search should be performed
        should_search_web = (
            include_web if include_web is not None
            else self.config.enable_web_search
        )
        
        if should_search_web and self._web_retriever:
            # Check if fallback-only mode
            if self.config.web_search_fallback_only:
                # Only search web if local results are insufficient
                high_quality_local = [
                    r for r in results
                    if r.score >= self.config.similarity_threshold
                ]
                should_search_web = len(high_quality_local) < self.config.local_results_threshold
            
            if should_search_web:
                try:
                    web_results = await self._web_retriever.search(
                        query,
                        max_results=self.config.web_max_results
                    )
                    
                    for i, r in enumerate(web_results):
                        results.append(self._convert_web_result(r, i))
                    
                    logger.info(f"Web search returned {len(web_results)} results")
                    
                except Exception as e:
                    logger.warning(f"Web search failed: {str(e)}")
        
        # Deduplicate
        results = self._deduplicate_results(results)
        
        # Sort by score
        results = sorted(results, key=lambda x: x.score, reverse=True)
        
        return results[:top_k]
    
    def search_sync(
        self,
        query: str,
        top_k: int = 10,
        include_web: Optional[bool] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[HybridResult]:
        """
        Synchronous wrapper for search.
        
        Args:
            query: Search query
            top_k: Number of results
            include_web: Override web search setting
            filters: Metadata filters
            
        Returns:
            List of HybridResult objects
        """
        return run_async_task(
            self.search(
                query=query,
                top_k=top_k,
                include_web=include_web,
                filters=filters
            )
        )
    
    async def search_local_only(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[HybridResult]:
        """
        Search only local documents.
        
        Args:
            query: Search query
            top_k: Number of results
            filters: Metadata filters
            
        Returns:
            List of HybridResult objects
        """
        return await self.search(query, top_k, include_web=False, filters=filters)
    
    async def search_web_only(
        self,
        query: str,
        max_results: int = 5
    ) -> List[HybridResult]:
        """
        Search only web sources.
        
        Args:
            query: Search query
            max_results: Number of results
            
        Returns:
            List of HybridResult objects
        """
        if not self._web_retriever:
            return []
        
        web_results = await self._web_retriever.search(query, max_results)
        
        results = []
        for i, r in enumerate(web_results):
            results.append(self._convert_web_result(r, i))
        
        return results
    
    def format_results_as_context(
        self,
        results: List[HybridResult],
        include_sources: bool = True,
        max_length: Optional[int] = None
    ) -> str:
        """
        Format results as context for LLM.
        
        Args:
            results: Search results
            include_sources: Include source information
            max_length: Maximum context length
            
        Returns:
            Formatted context string
        """
        if not results:
            return "No relevant information found."
        
        context_parts = []
        
        for i, result in enumerate(results, 1):
            parts = [f"[{i}] "]
            
            if result.title:
                parts.append(f"{result.title}\n")
            
            parts.append(result.content)
            
            if include_sources:
                source_info = f"\n(Source: {result.source_type}"
                if result.url:
                    source_info += f" - {result.url}"
                elif result.source:
                    source_info += f" - {result.source}"
                source_info += f", Score: {result.score:.2f})"
                parts.append(source_info)
            
            context_parts.append("".join(parts))
        
        context = "\n\n".join(context_parts)
        
        # Truncate if needed
        if max_length and len(context) > max_length:
            context = context[:max_length] + "\n\n[Context truncated...]"
        
        return context
    
    def get_source_summary(
        self,
        results: List[HybridResult]
    ) -> Dict[str, Any]:
        """
        Get summary of result sources.
        
        Args:
            results: Search results
            
        Returns:
            Summary dictionary
        """
        local_count = sum(1 for r in results if r.source_type == "local")
        web_count = sum(1 for r in results if r.source_type == "web")
        
        sources = {}
        for r in results:
            source = r.source or "unknown"
            sources[source] = sources.get(source, 0) + 1
        
        return {
            "total_results": len(results),
            "local_results": local_count,
            "web_results": web_count,
            "sources": sources
        }
    
    @property
    def local_document_count(self) -> int:
        """Get count of local documents."""
        return self._chroma_retriever.document_count
    
    def refresh_local_index(self) -> None:
        """Refresh the local search index."""
        self._chroma_retriever.refresh_index()


def create_hybrid_retriever(
    collection_name: str = "documents",
    enable_web_search: bool = True,
    **kwargs
) -> HybridRetriever:
    """
    Factory function to create a hybrid retriever.
    
    Args:
        collection_name: ChromaDB collection name
        enable_web_search: Whether to enable web search
        **kwargs: Additional configuration
        
    Returns:
        Configured HybridRetriever instance
    """
    config = HybridRetrieverConfig(
        collection_name=collection_name,
        enable_web_search=enable_web_search,
        **kwargs
    )
    return HybridRetriever(config)

