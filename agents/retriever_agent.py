"""
Retriever Agent - Production Grade with Multi-Modal Support

Features:
- Multi-strategy retrieval (dense, BM25, hybrid)
- Cross-reference expansion
- Hierarchical chunk navigation
- Multi-modal retrieval (text, tables, images)
- Source attribution
"""

import logging
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field

from crewai import Agent
from .utils import run_async_task

logger = logging.getLogger(__name__)


@dataclass
class RetrievalContext:
    """Structured retrieval results."""
    local_results: List[Dict[str, Any]] = field(default_factory=list)
    web_results: List[Dict[str, Any]] = field(default_factory=list)
    combined_context: str = ""
    sources: List[str] = field(default_factory=list)
    
    # Multi-modal results
    text_chunks: List[Dict[str, Any]] = field(default_factory=list)
    table_chunks: List[Dict[str, Any]] = field(default_factory=list)
    image_chunks: List[Dict[str, Any]] = field(default_factory=list)
    
    # Metadata
    total_results: int = 0
    retrieval_methods: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "local_results": self.local_results,
            "web_results": self.web_results,
            "combined_context": self.combined_context,
            "sources": self.sources,
            "text_chunks": self.text_chunks,
            "table_chunks": self.table_chunks,
            "image_chunks": self.image_chunks,
            "total_results": self.total_results,
            "retrieval_methods": self.retrieval_methods
        }


class RetrieverAgent:
    """
    Production-grade Retriever Agent for multi-modal document retrieval.
    
    Features:
    - HNSW-powered semantic search via Milvus
    - RRF fusion of multiple retrieval strategies
    - Cross-encoder re-ranking
    - Multi-modal content retrieval
    - Hierarchical context expansion
    """
    
    SYSTEM_PROMPT = """You are an expert Information Retrieval Specialist in a production RAG system.

CORE RESPONSIBILITIES:
1. RETRIEVE the most relevant information from all available sources
2. RANK results by relevance and reliability
3. EXPAND context using hierarchical relationships
4. LINK related content (tables, images) to text
5. ATTRIBUTE sources accurately

RETRIEVAL STRATEGIES:
- Semantic Search: HNSW vector similarity for conceptual matching
- Keyword Search: BM25 for exact term matching
- Hybrid: RRF fusion of multiple strategies
- Multi-Hop: Sequential retrieval for complex queries

CONTENT TYPES:
- Text: Main document paragraphs and sections
- Tables: Structured data requiring special formatting
- Images: Descriptions and OCR-extracted content

QUALITY CRITERIA:
- Relevance: Content directly addresses the query
- Accuracy: Information is factual and current
- Completeness: Sufficient context for understanding
- Source Diversity: Multiple sources for validation

Always prioritize the most authoritative and relevant sources."""

    def __init__(
        self,
        llm: Optional[Any] = None,
        verbose: bool = True,
        tools: Optional[List[Any]] = None,
        hybrid_retriever: Optional[Any] = None
    ):
        """
        Initialize the retriever agent.
        
        Args:
            llm: OpenAI LLM instance
            verbose: Enable verbose output
            tools: Available tools
            hybrid_retriever: Pre-configured retriever instance
        """
        self.llm = llm
        self.verbose = verbose
        self.tools = tools or []
        self._hybrid_retriever = hybrid_retriever
        self._agent: Optional[Agent] = None
        
        # Configuration
        self.default_top_k = 10
        self.rerank_top_k = 5
        self.min_score_threshold = 0.3
        self.max_context_length = 8000
    
    @property
    def agent(self) -> Agent:
        """Get or create the CrewAI agent."""
        if self._agent is None:
            self._agent = self._create_agent()
        return self._agent
    
    def _create_agent(self) -> Agent:
        """Create the CrewAI retriever agent."""
        return Agent(
            role="Information Retrieval Specialist",
            goal="Retrieve the most relevant and comprehensive information from all sources",
            backstory=self.SYSTEM_PROMPT,
            verbose=self.verbose,
            tools=self.tools,
            llm=self.llm,
            max_iter=2,
            memory=True
        )
    
    def _initialize_retriever(self) -> None:
        """Initialize the hybrid retriever if not provided."""
        if self._hybrid_retriever is None:
            try:
                from retriever import AdvancedRetriever
                self._hybrid_retriever = AdvancedRetriever()
                logger.info("Initialized AdvancedRetriever")
            except Exception as e:
                logger.error(f"Failed to initialize retriever: {e}")
                raise
    
    def retrieve(
        self,
        query: str,
        search_queries: Optional[List[str]] = None,
        use_documents: bool = True,
        use_web: bool = False,
        content_types: Optional[List[str]] = None,
        expand_context: bool = True,
        top_k: Optional[int] = None
    ) -> RetrievalContext:
        """
        Perform multi-modal retrieval.
        
        Args:
            query: Main query
            search_queries: Additional search variants
            use_documents: Search local documents
            use_web: Include web search
            content_types: Filter by content types
            expand_context: Include linked elements
            top_k: Number of results
            
        Returns:
            RetrievalContext with all results
        """
        self._initialize_retriever()
        
        search_queries = search_queries or [query]
        top_k = top_k or self.default_top_k
        content_types = content_types or ["text", "table", "image"]
        
        context = RetrievalContext()
        context.retrieval_methods = []
        
        try:
            # Local document retrieval
            if use_documents:
                local_results = self._retrieve_local(
                    search_queries, 
                    top_k,
                    content_types
                )
                context.local_results = local_results
                context.retrieval_methods.append("local_milvus")
                
                # Categorize by content type
                for result in local_results:
                    c_type = result.get("content_type", "text")
                    if c_type == "table":
                        context.table_chunks.append(result)
                    elif c_type == "image":
                        context.image_chunks.append(result)
                    else:
                        context.text_chunks.append(result)
                    
                    # Add source
                    source = result.get("source") or result.get("file_name", "Unknown")
                    if source not in context.sources:
                        context.sources.append(source)
            
            # Web retrieval
            if use_web:
                web_results = self._retrieve_web(query, top_k // 2)
                context.web_results = web_results
                context.retrieval_methods.append("web_search")
                
                for result in web_results:
                    source = result.get("url") or result.get("source", "Web")
                    if source not in context.sources:
                        context.sources.append(source)
            
            # Expand context with linked elements
            if expand_context:
                context = self._expand_with_links(context)
            
            # Build combined context
            context.combined_context = self._build_combined_context(context)
            context.total_results = len(context.local_results) + len(context.web_results)
            
            logger.info(
                f"Retrieved {context.total_results} results "
                f"(text: {len(context.text_chunks)}, "
                f"tables: {len(context.table_chunks)}, "
                f"images: {len(context.image_chunks)})"
            )
            
            return context
            
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return context
    
    def _retrieve_local(
        self,
        queries: List[str],
        top_k: int,
        content_types: List[str]
    ) -> List[Dict[str, Any]]:
        """Retrieve from local Milvus store."""
        all_results = []
        
        for query in queries:
            try:
                results = self._hybrid_retriever.search(
                    query=query,
                    top_k=top_k,
                    methods=["dense", "bm25"]
                )
                
                for result in results:
                    result_dict = {
                        "id": getattr(result, "id", ""),
                        "content": getattr(result, "content", str(result)),
                        "score": getattr(result, "score", 0),
                        "content_type": getattr(result, "content_type", "text"),
                        "source": getattr(result, "source", ""),
                        "metadata": getattr(result, "metadata", {}),
                        "linked_elements": getattr(result, "linked_elements", []),
                        "parent_id": getattr(result, "parent_id", None)
                    }
                    
                    # Filter by content type
                    if result_dict["content_type"] in content_types:
                        if result_dict["score"] >= self.min_score_threshold:
                            all_results.append(result_dict)
                            
            except Exception as e:
                logger.warning(f"Local retrieval failed for query: {e}")
        
        # Deduplicate by content
        seen_content = set()
        unique_results = []
        for r in all_results:
            content_hash = hash(r["content"][:100])
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_results.append(r)
        
        # Sort by score
        unique_results.sort(key=lambda x: x["score"], reverse=True)
        
        return unique_results[:top_k]
    
    def _retrieve_web(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Retrieve from web sources."""
        try:
            if hasattr(self._hybrid_retriever, 'search_web'):
                results = self._hybrid_retriever.search_web(query, top_k)
                return [
                    {
                        "content": r.get("content", ""),
                        "url": r.get("url", ""),
                        "title": r.get("title", ""),
                        "source": "web"
                    }
                    for r in results
                ]
        except Exception as e:
            logger.warning(f"Web retrieval failed: {e}")
        
        return []
    
    def _expand_with_links(self, context: RetrievalContext) -> RetrievalContext:
        """Expand results with linked elements."""
        try:
            expanded_tables = []
            expanded_images = []
            
            for result in context.local_results:
                linked = result.get("linked_elements", [])
                
                for link_id in linked[:3]:  # Limit expansion
                    try:
                        if hasattr(self._hybrid_retriever, 'get'):
                            linked_results = self._hybrid_retriever.get(ids=[link_id])
                            for lr in linked_results:
                                lr_type = getattr(lr, "content_type", "text")
                                lr_dict = {
                                    "id": getattr(lr, "id", ""),
                                    "content": getattr(lr, "content", ""),
                                    "content_type": lr_type,
                                    "linked_from": result.get("id")
                                }
                                
                                if lr_type == "table":
                                    expanded_tables.append(lr_dict)
                                elif lr_type == "image":
                                    expanded_images.append(lr_dict)
                    except:
                        pass
            
            context.table_chunks.extend(expanded_tables)
            context.image_chunks.extend(expanded_images)
            
        except Exception as e:
            logger.warning(f"Link expansion failed: {e}")
        
        return context
    
    def _build_combined_context(self, context: RetrievalContext) -> str:
        """Build combined context string from all results."""
        parts = []
        
        # Add text chunks
        if context.text_chunks:
            parts.append("=== TEXT CONTENT ===")
            for i, chunk in enumerate(context.text_chunks[:7], 1):
                source = chunk.get("source", "Unknown")
                content = chunk.get("content", "")
                parts.append(f"\n[Source: {source}]\n{content}")
        
        # Add table chunks
        if context.table_chunks:
            parts.append("\n\n=== TABLES ===")
            for i, chunk in enumerate(context.table_chunks[:3], 1):
                content = chunk.get("content", "")
                parts.append(f"\n[Table {i}]\n{content}")
        
        # Add image chunks
        if context.image_chunks:
            parts.append("\n\n=== IMAGE DESCRIPTIONS ===")
            for i, chunk in enumerate(context.image_chunks[:3], 1):
                content = chunk.get("content", "")
                parts.append(f"\n[Image {i}]\n{content}")
        
        # Add web results
        if context.web_results:
            parts.append("\n\n=== WEB SOURCES ===")
            for i, result in enumerate(context.web_results[:3], 1):
                title = result.get("title", "")
                content = result.get("content", "")[:500]
                url = result.get("url", "")
                parts.append(f"\n[{title}]\n{content}...\nSource: {url}")
        
        combined = "\n".join(parts)
        
        # Truncate if too long
        if len(combined) > self.max_context_length:
            combined = combined[:self.max_context_length] + "\n\n[Truncated...]"
        
        return combined
    
    def get_sources(self, context: RetrievalContext) -> List[str]:
        """Extract unique sources from context."""
        return context.sources
    
    def retrieve_by_type(
        self,
        query: str,
        content_type: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Retrieve specific content type."""
        self._initialize_retriever()
        
        try:
            if hasattr(self._hybrid_retriever, 'search_by_type'):
                results = self._hybrid_retriever.search_by_type(
                    query=query,
                    content_type=content_type,
                    top_k=top_k
                )
                return [
                    {
                        "content": getattr(r, "content", ""),
                        "score": getattr(r, "score", 0),
                        "metadata": getattr(r, "metadata", {})
                    }
                    for r in results
                ]
        except Exception as e:
            logger.error(f"Type-specific retrieval failed: {e}")
        
        return []


def create_retriever_agent(
    llm: Optional[Any] = None,
    verbose: bool = True,
    tools: Optional[List[Any]] = None,
    retriever: Optional[Any] = None
) -> RetrieverAgent:
    """Factory function to create retriever agent."""
    return RetrieverAgent(
        llm=llm,
        verbose=verbose,
        tools=tools,
        hybrid_retriever=retriever
    )
