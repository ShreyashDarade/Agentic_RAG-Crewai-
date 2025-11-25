import logging
from typing import Any, List, Optional, Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class OnlineSearchInput(BaseModel):
    """Input schema for online search tool."""
    query: str = Field(
        ...,
        description="The search query to look up online"
    )
    max_results: int = Field(
        default=5,
        description="Maximum number of results to return",
        ge=1,
        le=10
    )


class OnlineSearchTool(BaseTool):
    """
    Tool for searching the web for real-time information.
    
    Uses DuckDuckGo, Serper, or Tavily for web searches.
    """
    
    name: str = "online_search"
    description: str = """
    Search the web for current information, news, or facts.
    Use this tool when:
    - The query requires up-to-date information
    - Local documents don't contain the answer
    - You need to verify information from external sources
    Returns web search results with titles, snippets, and URLs.
    """
    args_schema: Type[BaseModel] = OnlineSearchInput
    
    _web_retriever: Any = None
    _provider: str = "duckduckgo"
    _api_key: Optional[str] = None
    
    def __init__(
        self,
        web_retriever: Any = None,
        provider: str = "duckduckgo",
        api_key: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the online search tool.
        
        Args:
            web_retriever: WebRetriever instance
            provider: Search provider name
            api_key: API key for the provider
            **kwargs: Additional arguments
        """
        super().__init__(**kwargs)
        self._web_retriever = web_retriever
        self._provider = provider
        self._api_key = api_key
    
    def _initialize_retriever(self) -> None:
        """Initialize web retriever if not provided."""
        if self._web_retriever is None:
            try:
                from ...retriever import WebRetriever, WebRetrieverConfig
                
                config = WebRetrieverConfig(
                    provider=self._provider,
                    api_key=self._api_key
                )
                self._web_retriever = WebRetriever(config)
                logger.info(f"Initialized WebRetriever with provider: {self._provider}")
                
            except Exception as e:
                logger.error(f"Failed to initialize web retriever: {e}")
                raise
    
    def _run(
        self,
        query: str,
        max_results: int = 5
    ) -> str:
        """
        Execute the web search.
        
        Args:
            query: Search query
            max_results: Number of results
            
        Returns:
            Formatted search results
        """
        self._initialize_retriever()
        
        try:
            logger.info(f"Online search: {query[:50]}... (max_results={max_results})")
            
            # Run async search in sync context
            results = self._web_retriever.search_sync(query, max_results)
            
            if not results:
                return "No web results found for this query."
            
            # Format results
            output_parts = [f"Web Search Results ({len(results)} found):\n"]
            
            for i, result in enumerate(results, 1):
                output_parts.append(f"\n--- Result {i} ---")
                output_parts.append(f"Title: {result.title}")
                output_parts.append(f"URL: {result.url}")
                output_parts.append(f"Snippet: {result.snippet}")
                
                if result.published_date:
                    output_parts.append(f"Published: {result.published_date}")
                
                output_parts.append("")
            
            return "\n".join(output_parts)
            
        except Exception as e:
            error_msg = f"Online search failed: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    async def _arun(
        self,
        query: str,
        max_results: int = 5
    ) -> str:
        """Async version of the search."""
        self._initialize_retriever()
        
        try:
            results = await self._web_retriever.search(query, max_results)
            
            if not results:
                return "No web results found for this query."
            
            output_parts = [f"Web Search Results ({len(results)} found):\n"]
            
            for i, result in enumerate(results, 1):
                output_parts.append(f"\n[{i}] {result.title}")
                output_parts.append(f"    URL: {result.url}")
                output_parts.append(f"    {result.snippet}")
            
            return "\n".join(output_parts)
            
        except Exception as e:
            return f"Online search failed: {str(e)}"


class HybridSearchInput(BaseModel):
    """Input schema for hybrid search."""
    query: str = Field(
        ...,
        description="The search query"
    )
    include_local: bool = Field(
        default=True,
        description="Include local document search"
    )
    include_web: bool = Field(
        default=True,
        description="Include web search"
    )
    top_k: int = Field(
        default=10,
        description="Total number of results"
    )


class HybridSearchTool(BaseTool):
    """
    Tool for combined local and web search.
    
    Provides comprehensive search across all available sources.
    """
    
    name: str = "hybrid_search"
    description: str = """
    Search both local documents and the web simultaneously.
    Use this for comprehensive information gathering from all sources.
    Results are combined and ranked by relevance.
    """
    args_schema: Type[BaseModel] = HybridSearchInput
    
    _hybrid_retriever: Any = None
    
    def __init__(self, hybrid_retriever: Any = None, **kwargs):
        """
        Initialize the hybrid search tool.
        
        Args:
            hybrid_retriever: HybridRetriever instance
            **kwargs: Additional arguments
        """
        super().__init__(**kwargs)
        self._hybrid_retriever = hybrid_retriever
    
    def _initialize_retriever(self) -> None:
        """Initialize hybrid retriever if not provided."""
        if self._hybrid_retriever is None:
            try:
                from ...retriever import HybridRetriever
                self._hybrid_retriever = HybridRetriever()
                logger.info("Initialized HybridRetriever")
            except Exception as e:
                logger.error(f"Failed to initialize hybrid retriever: {e}")
                raise
    
    def _run(
        self,
        query: str,
        include_local: bool = True,
        include_web: bool = True,
        top_k: int = 10
    ) -> str:
        """Execute hybrid search."""
        self._initialize_retriever()
        
        try:
            logger.info(f"Hybrid search: {query[:50]}...")
            
            results = self._hybrid_retriever.search_sync(
                query=query,
                top_k=top_k,
                include_web=include_web
            )
            
            if not results:
                return "No results found from any source."
            
            # Group results by source type
            local_results = [r for r in results if r.source_type == "local"]
            web_results = [r for r in results if r.source_type == "web"]
            
            output_parts = [f"Combined Search Results:\n"]
            output_parts.append(f"Local: {len(local_results)} | Web: {len(web_results)}\n")
            
            for i, result in enumerate(results, 1):
                source_label = "📄" if result.source_type == "local" else "🌐"
                output_parts.append(f"\n{source_label} [{i}] Score: {result.score:.3f}")
                
                if result.title:
                    output_parts.append(f"    Title: {result.title}")
                
                if result.url:
                    output_parts.append(f"    URL: {result.url}")
                elif result.source:
                    output_parts.append(f"    Source: {result.source}")
                
                output_parts.append(f"    {result.content[:200]}...")
            
            return "\n".join(output_parts)
            
        except Exception as e:
            error_msg = f"Hybrid search failed: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    async def _arun(self, **kwargs) -> str:
        """Async version."""
        self._initialize_retriever()
        
        try:
            results = await self._hybrid_retriever.search(**kwargs)
            
            if not results:
                return "No results found."
            
            return self._hybrid_retriever.format_results_as_context(results)
            
        except Exception as e:
            return f"Hybrid search failed: {str(e)}"


def create_online_search_tool(
    provider: str = "duckduckgo",
    api_key: Optional[str] = None
) -> OnlineSearchTool:
    """
    Factory function to create an online search tool.
    
    Args:
        provider: Search provider
        api_key: API key
        
    Returns:
        Configured OnlineSearchTool
    """
    return OnlineSearchTool(provider=provider, api_key=api_key)


def create_hybrid_search_tool(hybrid_retriever: Any = None) -> HybridSearchTool:
    """
    Factory function to create a hybrid search tool.
    
    Args:
        hybrid_retriever: HybridRetriever instance
        
    Returns:
        Configured HybridSearchTool
    """
    return HybridSearchTool(hybrid_retriever=hybrid_retriever)

