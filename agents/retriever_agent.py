import logging
from typing import Any, Dict, List, Optional

from crewai import Agent
from .utils import run_async_task

logger = logging.getLogger(__name__)


class RetrieverAgent:
    """
    Retriever agent for information retrieval.
    
    Responsibilities:
    - Search document databases (ChromaDB)
    - Perform web searches when needed
    - Combine and rank results from multiple sources
    - Provide relevant context for answer generation
    """
    
    SYSTEM_PROMPT = """You are an expert Information Retriever in a multi-agent RAG system.

Your responsibilities:
1. Search through document databases (ChromaDB) to find relevant information
2. Perform web searches when internal documents are insufficient
3. Combine results from multiple sources using hybrid retrieval
4. Filter and rank results based on relevance

When retrieving information:
- Use appropriate search strategies (semantic, keyword, hybrid)
- Consider the context and intent from the supervisor's analysis
- Retrieve sufficient context while avoiding information overload
- Flag when information might be outdated or insufficient

Always provide source attribution for retrieved information."""
    
    def __init__(
        self,
        llm: Optional[Any] = None,
        verbose: bool = True,
        tools: Optional[List[Any]] = None,
        chroma_retriever: Optional[Any] = None,
        web_retriever: Optional[Any] = None,
        hybrid_retriever: Optional[Any] = None
    ):
        """
        Initialize the retriever agent.
        
        Args:
            llm: LLM instance
            verbose: Enable verbose output
            tools: List of tools
            chroma_retriever: ChromaRetriever instance
            web_retriever: WebRetriever instance
            hybrid_retriever: HybridRetriever instance
        """
        self.llm = llm
        self.verbose = verbose
        self.tools = tools or []
        self._chroma_retriever = chroma_retriever
        self._web_retriever = web_retriever
        self._hybrid_retriever = hybrid_retriever
        self._agent: Optional[Agent] = None
    
    @property
    def agent(self) -> Agent:
        """Get or create the CrewAI agent."""
        if self._agent is None:
            self._agent = self._create_agent()
        return self._agent
    
    def _create_agent(self) -> Agent:
        """Create the CrewAI retriever agent."""
        return Agent(
            role="Information Retriever",
            goal="Retrieve the most relevant information from knowledge bases and web sources",
            backstory=self.SYSTEM_PROMPT,
            verbose=self.verbose,
            allow_delegation=False,
            tools=self.tools,
            llm=self.llm
        )
    
    def _initialize_retrievers(self) -> None:
        """Initialize retrievers if not provided."""
        if self._hybrid_retriever is None:
            try:
                from ..retriever import HybridRetriever
                self._hybrid_retriever = HybridRetriever()
                logger.info("Initialized HybridRetriever")
            except Exception as e:
                logger.warning(f"Could not initialize HybridRetriever: {e}")
        
        if self._chroma_retriever is None and self._hybrid_retriever:
            self._chroma_retriever = self._hybrid_retriever._chroma_retriever
        
        if self._web_retriever is None and self._hybrid_retriever:
            self._web_retriever = self._hybrid_retriever._web_retriever
    
    def retrieve(
        self,
        query: str,
        search_queries: Optional[List[str]] = None,
        use_documents: bool = True,
        use_web: bool = False,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Retrieve information for a query.
        
        Args:
            query: Main query
            search_queries: Additional search queries
            use_documents: Search local documents
            use_web: Search the web
            top_k: Number of results
            filters: Metadata filters
            
        Returns:
            Retrieval results dictionary
        """
        self._initialize_retrievers()
        
        all_queries = [query]
        if search_queries:
            all_queries.extend(search_queries)
        
        results = {
            "query": query,
            "local_results": [],
            "web_results": [],
            "combined_context": "",
            "sources": [],
            "success": True,
            "error": None
        }
        
        try:
            if self._hybrid_retriever:
                # Use hybrid retriever for combined search
                hybrid_results = self._hybrid_retriever.search_sync(
                    query=query,
                    top_k=top_k,
                    include_web=use_web,
                    filters=filters
                )
                
                for r in hybrid_results:
                    if r.source_type == "local":
                        results["local_results"].append({
                            "id": r.id,
                            "content": r.content,
                            "score": r.score,
                            "source": r.source,
                            "metadata": r.metadata
                        })
                    else:
                        results["web_results"].append({
                            "title": r.title,
                            "url": r.url,
                            "content": r.content,
                            "score": r.score
                        })
                
                # Build combined context
                results["combined_context"] = self._hybrid_retriever.format_results_as_context(
                    hybrid_results,
                    include_sources=True
                )
                
                # Collect sources
                for r in hybrid_results:
                    source = r.url if r.url else r.source
                    if source and source not in results["sources"]:
                        results["sources"].append(source)
            
            else:
                # Fallback to individual retrievers
                if use_documents and self._chroma_retriever:
                    local_results = self._chroma_retriever.search(
                        query=query,
                        top_k=top_k,
                        filters=filters
                    )
                    
                    for r in local_results:
                        results["local_results"].append({
                            "id": r.id,
                            "content": r.content,
                            "score": r.score,
                            "source": r.source,
                            "metadata": r.metadata
                        })
                
                if use_web and self._web_retriever:
                    web_results = run_async_task(
                        self._web_retriever.search(query, max_results=5)
                    )
                    
                    for r in web_results:
                        results["web_results"].append({
                            "title": r.title,
                            "url": r.url,
                            "content": r.snippet,
                            "score": r.score
                        })
                
                # Build combined context
                results["combined_context"] = self._build_context(results)
            
            logger.info(
                f"Retrieved {len(results['local_results'])} local + "
                f"{len(results['web_results'])} web results"
            )
            
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            results["success"] = False
            results["error"] = str(e)
        
        return results
    
    def _build_context(self, results: Dict[str, Any]) -> str:
        """Build combined context from results."""
        context_parts = []
        
        if results["local_results"]:
            context_parts.append("=== Local Documents ===\n")
            for i, r in enumerate(results["local_results"], 1):
                context_parts.append(f"[{i}] (Score: {r['score']:.3f})")
                context_parts.append(f"Source: {r['source']}")
                context_parts.append(r["content"])
                context_parts.append("")
        
        if results["web_results"]:
            context_parts.append("\n=== Web Results ===\n")
            for i, r in enumerate(results["web_results"], 1):
                context_parts.append(f"[{i}] {r['title']}")
                context_parts.append(f"URL: {r['url']}")
                context_parts.append(r["content"])
                context_parts.append("")
        
        return "\n".join(context_parts)
    
    def search_documents(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search only local documents.
        
        Args:
            query: Search query
            top_k: Number of results
            filters: Metadata filters
            
        Returns:
            List of document results
        """
        result = self.retrieve(
            query=query,
            use_documents=True,
            use_web=False,
            top_k=top_k,
            filters=filters
        )
        return result["local_results"]
    
    def search_web(
        self,
        query: str,
        max_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search only the web.
        
        Args:
            query: Search query
            max_results: Maximum results
            
        Returns:
            List of web results
        """
        result = self.retrieve(
            query=query,
            use_documents=False,
            use_web=True,
            top_k=max_results
        )
        return result["web_results"]
    
    def get_context_for_generation(
        self,
        query: str,
        analysis: Optional[Dict[str, Any]] = None,
        max_context_length: int = 4000
    ) -> str:
        """
        Get formatted context for answer generation.
        
        Args:
            query: User query
            analysis: Query analysis from supervisor
            max_context_length: Maximum context length
            
        Returns:
            Formatted context string
        """
        # Determine search strategy
        use_web = False
        search_queries = [query]
        
        if analysis:
            use_web = analysis.get("search_strategy", {}).get("use_web_search", False)
            search_queries = analysis.get("search_strategy", {}).get("search_queries", [query])
        
        # Retrieve information
        results = self.retrieve(
            query=query,
            search_queries=search_queries,
            use_documents=True,
            use_web=use_web
        )
        
        context = results["combined_context"]
        
        # Truncate if needed
        if len(context) > max_context_length:
            context = context[:max_context_length] + "\n\n[Context truncated...]"
        
        return context


def create_retriever_agent(
    llm: Optional[Any] = None,
    verbose: bool = True,
    tools: Optional[List[Any]] = None,
    hybrid_retriever: Optional[Any] = None
) -> RetrieverAgent:
    """
    Factory function to create a retriever agent.
    
    Args:
        llm: LLM instance
        verbose: Enable verbose output
        tools: Available tools
        hybrid_retriever: HybridRetriever instance
        
    Returns:
        Configured RetrieverAgent instance
    """
    return RetrieverAgent(
        llm=llm,
        verbose=verbose,
        tools=tools,
        hybrid_retriever=hybrid_retriever
    )

