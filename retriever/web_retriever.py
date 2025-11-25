import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.parse import quote_plus

from common.async_utils import run_async_task

logger = logging.getLogger(__name__)


@dataclass
class WebRetrieverConfig:
    """Configuration for web retrieval."""
    provider: str = "duckduckgo"  # duckduckgo, serper, tavily
    max_results: int = 5
    timeout: int = 30
    api_key: Optional[str] = None
    safe_search: bool = True
    region: str = "wt-wt"  # World-wide


@dataclass
class WebSearchResult:
    """Represents a web search result."""
    title: str
    url: str
    snippet: str
    source: str
    score: float = 0.0
    published_date: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "source": self.source,
            "score": self.score,
            "published_date": self.published_date,
            "metadata": self.metadata
        }


class WebRetrieverError(Exception):
    """Exception raised for web retrieval errors."""
    pass


class DuckDuckGoSearcher:
    """DuckDuckGo search implementation."""
    
    def __init__(self, timeout: int = 30):
        self.timeout = timeout
    
    async def search(
        self,
        query: str,
        max_results: int = 5,
        region: str = "wt-wt"
    ) -> List[WebSearchResult]:
        """Perform DuckDuckGo search."""
        try:
            from duckduckgo_search import DDGS
            
            results = []
            
            # Run sync search in executor
            loop = asyncio.get_event_loop()
            
            def do_search():
                with DDGS() as ddgs:
                    return list(ddgs.text(
                        query,
                        region=region,
                        max_results=max_results
                    ))
            
            raw_results = await loop.run_in_executor(None, do_search)
            
            for i, r in enumerate(raw_results):
                results.append(WebSearchResult(
                    title=r.get("title", ""),
                    url=r.get("href", r.get("link", "")),
                    snippet=r.get("body", r.get("snippet", "")),
                    source="duckduckgo",
                    score=1.0 - (i * 0.1),  # Rank-based score
                    metadata={"raw": r}
                ))
            
            return results
            
        except ImportError:
            raise WebRetrieverError(
                "duckduckgo-search is required. Install with: pip install duckduckgo-search"
            )
        except Exception as e:
            logger.error(f"DuckDuckGo search failed: {str(e)}")
            raise WebRetrieverError(f"Search failed: {str(e)}")


class SerperSearcher:
    """Serper.dev search implementation."""
    
    def __init__(self, api_key: str, timeout: int = 30):
        self.api_key = api_key
        self.timeout = timeout
        self.base_url = "https://google.serper.dev/search"
    
    async def search(
        self,
        query: str,
        max_results: int = 5
    ) -> List[WebSearchResult]:
        """Perform Serper search."""
        try:
            import aiohttp
            
            headers = {
                "X-API-KEY": self.api_key,
                "Content-Type": "application/json"
            }
            
            payload = {
                "q": query,
                "num": max_results
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.base_url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout
                ) as response:
                    if response.status != 200:
                        raise WebRetrieverError(f"Serper API error: {response.status}")
                    
                    data = await response.json()
            
            results = []
            
            # Process organic results
            organic = data.get("organic", [])
            for i, r in enumerate(organic[:max_results]):
                results.append(WebSearchResult(
                    title=r.get("title", ""),
                    url=r.get("link", ""),
                    snippet=r.get("snippet", ""),
                    source="serper",
                    score=1.0 - (i * 0.1),
                    published_date=r.get("date"),
                    metadata={"position": r.get("position")}
                ))
            
            return results
            
        except ImportError:
            raise WebRetrieverError("aiohttp is required. Install with: pip install aiohttp")
        except Exception as e:
            logger.error(f"Serper search failed: {str(e)}")
            raise WebRetrieverError(f"Serper search failed: {str(e)}")


class TavilySearcher:
    """Tavily search implementation."""
    
    def __init__(self, api_key: str, timeout: int = 30):
        self.api_key = api_key
        self.timeout = timeout
    
    async def search(
        self,
        query: str,
        max_results: int = 5
    ) -> List[WebSearchResult]:
        """Perform Tavily search."""
        try:
            from tavily import TavilyClient
            
            client = TavilyClient(api_key=self.api_key)
            
            # Run sync call in executor
            loop = asyncio.get_event_loop()
            
            def do_search():
                return client.search(
                    query=query,
                    max_results=max_results,
                    include_raw_content=False
                )
            
            response = await loop.run_in_executor(None, do_search)
            
            results = []
            
            for i, r in enumerate(response.get("results", [])):
                results.append(WebSearchResult(
                    title=r.get("title", ""),
                    url=r.get("url", ""),
                    snippet=r.get("content", ""),
                    source="tavily",
                    score=r.get("score", 1.0 - (i * 0.1)),
                    published_date=r.get("published_date"),
                    metadata={"raw_relevance_score": r.get("relevance_score")}
                ))
            
            return results
            
        except ImportError:
            raise WebRetrieverError("tavily-python is required. Install with: pip install tavily-python")
        except Exception as e:
            logger.error(f"Tavily search failed: {str(e)}")
            raise WebRetrieverError(f"Tavily search failed: {str(e)}")


class WebRetriever:
    """
    Web retriever for online information search.
    
    Supports multiple providers:
    - DuckDuckGo (free, no API key required)
    - Serper.dev (Google search API)
    - Tavily (AI-focused search API)
    """
    
    def __init__(self, config: Optional[WebRetrieverConfig] = None):
        """
        Initialize the web retriever.
        
        Args:
            config: Web retriever configuration
        """
        self.config = config or WebRetrieverConfig()
        self._searcher = None
        self._initialize_searcher()
    
    def _initialize_searcher(self) -> None:
        """Initialize the search provider."""
        provider = self.config.provider.lower()
        
        if provider == "duckduckgo":
            self._searcher = DuckDuckGoSearcher(timeout=self.config.timeout)
        elif provider == "serper":
            if not self.config.api_key:
                raise WebRetrieverError("Serper API key is required")
            self._searcher = SerperSearcher(
                api_key=self.config.api_key,
                timeout=self.config.timeout
            )
        elif provider == "tavily":
            if not self.config.api_key:
                raise WebRetrieverError("Tavily API key is required")
            self._searcher = TavilySearcher(
                api_key=self.config.api_key,
                timeout=self.config.timeout
            )
        else:
            raise WebRetrieverError(f"Unknown provider: {provider}")
        
        logger.info(f"Web retriever initialized with provider: {provider}")
    
    async def search(
        self,
        query: str,
        max_results: Optional[int] = None
    ) -> List[WebSearchResult]:
        """
        Search the web for information.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            List of WebSearchResult objects
        """
        if not query or not query.strip():
            return []
        
        max_results = max_results or self.config.max_results
        
        try:
            if isinstance(self._searcher, DuckDuckGoSearcher):
                results = await self._searcher.search(
                    query,
                    max_results=max_results,
                    region=self.config.region
                )
            else:
                results = await self._searcher.search(query, max_results=max_results)
            
            logger.info(f"Web search returned {len(results)} results for: {query[:50]}...")
            return results
            
        except WebRetrieverError:
            raise
        except Exception as e:
            logger.error(f"Web search failed: {str(e)}")
            return []
    
    def search_sync(
        self,
        query: str,
        max_results: Optional[int] = None
    ) -> List[WebSearchResult]:
        """
        Synchronous wrapper for search.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            List of WebSearchResult objects
        """
        return run_async_task(self.search(query, max_results))
    
    async def search_multiple(
        self,
        queries: List[str],
        max_results_per_query: Optional[int] = None
    ) -> Dict[str, List[WebSearchResult]]:
        """
        Search multiple queries concurrently.
        
        Args:
            queries: List of search queries
            max_results_per_query: Results per query
            
        Returns:
            Dictionary mapping queries to results
        """
        max_results = max_results_per_query or self.config.max_results
        
        tasks = [self.search(q, max_results) for q in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        output = {}
        for query, result in zip(queries, results):
            if isinstance(result, Exception):
                logger.error(f"Search failed for '{query}': {result}")
                output[query] = []
            else:
                output[query] = result
        
        return output
    
    def format_results_as_context(
        self,
        results: List[WebSearchResult],
        include_urls: bool = True
    ) -> str:
        """
        Format search results as context for LLM.
        
        Args:
            results: Search results
            include_urls: Whether to include URLs
            
        Returns:
            Formatted context string
        """
        if not results:
            return "No web search results found."
        
        context_parts = ["Web Search Results:\n"]
        
        for i, result in enumerate(results, 1):
            parts = [f"{i}. {result.title}"]
            
            if include_urls:
                parts.append(f"   URL: {result.url}")
            
            parts.append(f"   {result.snippet}")
            
            if result.published_date:
                parts.append(f"   Published: {result.published_date}")
            
            context_parts.append("\n".join(parts))
        
        return "\n\n".join(context_parts)


def create_web_retriever(
    provider: str = "duckduckgo",
    api_key: Optional[str] = None,
    **kwargs
) -> WebRetriever:
    """
    Factory function to create a web retriever.
    
    Args:
        provider: Search provider name
        api_key: API key (if required)
        **kwargs: Additional configuration
        
    Returns:
        Configured WebRetriever instance
    """
    config = WebRetrieverConfig(provider=provider, api_key=api_key, **kwargs)
    return WebRetriever(config)

