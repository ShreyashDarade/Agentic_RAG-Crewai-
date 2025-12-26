"""
Milvus Search Tool - Production Grade CrewAI Tool

Features:
- HNSW-powered semantic search
- Multi-modal search (text, tables, images)
- Cross-reference retrieval
- Hierarchical chunk navigation
- Advanced filtering
"""

import logging
from typing import Any, Dict, List, Optional, Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class MilvusSearchInput(BaseModel):
    """Input schema for Milvus search tool."""
    query: str = Field(
        ...,
        description="The search query to find relevant documents"
    )
    top_k: int = Field(
        default=5,
        description="Number of results to return",
        ge=1,
        le=20
    )
    content_type: Optional[str] = Field(
        default=None,
        description="Filter by content type: 'text', 'table', or 'image'"
    )
    include_linked: bool = Field(
        default=True,
        description="Include linked elements (tables/images) with text results"
    )


class MilvusSearchTool(BaseTool):
    """
    Production-grade Milvus search tool for CrewAI agents.
    
    Features:
    - HNSW vector search with high recall
    - Content type filtering
    - Cross-reference expansion
    - Relevance scoring
    """
    
    name: str = "milvus_search"
    description: str = """
    Search the document database using production-grade Milvus vector search.
    
    Use this tool to:
    - Find relevant information from ingested documents
    - Search for specific content types (text, tables, images)
    - Retrieve related tables and images linked to text
    
    The search uses HNSW indexing for optimal performance and accuracy.
    Provide clear, specific queries for best results.
    """
    args_schema: Type[BaseModel] = MilvusSearchInput
    
    _retriever: Any = None
    _embedder: Any = None
    
    def __init__(self, retriever: Any = None, embedder: Any = None, **kwargs):
        """
        Initialize the Milvus search tool.
        
        Args:
            retriever: AdvancedRetriever or MilvusCloudStore instance
            embedder: OpenAI embedder instance
            **kwargs: Additional arguments
        """
        super().__init__(**kwargs)
        self._retriever = retriever
        self._embedder = embedder
    
    def _initialize_components(self) -> None:
        """Initialize retriever and embedder if not provided."""
        if self._retriever is None:
            try:
                from retriever import AdvancedRetriever
                self._retriever = AdvancedRetriever()
                logger.info("Initialized AdvancedRetriever")
            except Exception as e:
                logger.error(f"Failed to initialize retriever: {e}")
                raise
        
        if self._embedder is None:
            try:
                from embeddings import OpenAIEmbedder
                self._embedder = OpenAIEmbedder()
                logger.info("Initialized OpenAI embedder")
            except Exception as e:
                logger.warning(f"Embedder initialization failed: {e}")
    
    def _run(
        self,
        query: str,
        top_k: int = 5,
        content_type: Optional[str] = None,
        include_linked: bool = True
    ) -> str:
        """
        Execute Milvus search.
        
        Args:
            query: Search query
            top_k: Number of results
            content_type: Content type filter
            include_linked: Include linked elements
            
        Returns:
            Formatted search results
        """
        self._initialize_components()
        
        try:
            logger.info(f"Searching Milvus: '{query[:50]}...' (top_k={top_k})")
            
            # Perform search
            if hasattr(self._retriever, 'search'):
                results = self._retriever.search(
                    query=query,
                    top_k=top_k,
                    methods=["dense", "bm25"]
                )
            else:
                # Direct vector store search
                query_embedding = self._embedder.embed_query(query)
                results = self._retriever.search(
                    query_embedding=query_embedding,
                    top_k=top_k,
                    content_type=content_type
                )
            
            if not results:
                return "No relevant documents found for this query."
            
            # Filter by content type if specified
            if content_type:
                results = [
                    r for r in results 
                    if getattr(r, 'content_type', 'text') == content_type
                ]
            
            # Format results
            output_parts = [f"Found {len(results)} relevant documents:\n"]
            
            for i, result in enumerate(results, 1):
                score = getattr(result, 'score', 0) or getattr(result, 'rerank_score', 0)
                content = getattr(result, 'content', str(result))
                metadata = getattr(result, 'metadata', {})
                c_type = getattr(result, 'content_type', 'text')
                
                output_parts.append(f"\n{'='*60}")
                output_parts.append(f"📄 Result {i} | Score: {score:.3f} | Type: {c_type}")
                output_parts.append(f"{'='*60}")
                
                # Source info
                source = metadata.get("source") or metadata.get("file_name", "Unknown")
                output_parts.append(f"📁 Source: {source}")
                
                # Chunk info
                chunk_idx = metadata.get("chunk_index", 0)
                parent_id = getattr(result, 'parent_id', None) or metadata.get("parent_id")
                if parent_id:
                    output_parts.append(f"🔗 Parent: {parent_id} | Chunk: {chunk_idx}")
                
                # Linked elements
                linked = getattr(result, 'linked_elements', []) or metadata.get("linked_elements", [])
                if linked and include_linked:
                    output_parts.append(f"📎 Linked: {', '.join(linked[:3])}")
                
                # Content
                output_parts.append(f"\n📝 Content:\n{content}")
                
                # Add linked element details if available
                if include_linked and linked and hasattr(self._retriever, 'get_linked_elements'):
                    try:
                        linked_results = self._retriever.get_linked_elements(result.id)
                        for lr in linked_results[:2]:
                            lr_type = getattr(lr, 'content_type', 'unknown')
                            lr_content = getattr(lr, 'content', '')[:200]
                            output_parts.append(f"\n  ↳ [{lr_type.upper()}]: {lr_content}...")
                    except:
                        pass
            
            output_parts.append(f"\n{'='*60}")
            return "\n".join(output_parts)
            
        except Exception as e:
            error_msg = f"Milvus search failed: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    async def _arun(
        self,
        query: str,
        top_k: int = 5,
        content_type: Optional[str] = None,
        include_linked: bool = True
    ) -> str:
        """Async version of search."""
        return self._run(query, top_k, content_type, include_linked)


class MilvusMultiModalSearchInput(BaseModel):
    """Input for multi-modal search."""
    query: str = Field(..., description="Search query")
    search_text: bool = Field(default=True, description="Search text content")
    search_tables: bool = Field(default=True, description="Search tables")
    search_images: bool = Field(default=True, description="Search image descriptions")
    top_k_per_type: int = Field(default=3, description="Results per content type")


class MilvusMultiModalSearchTool(BaseTool):
    """
    Multi-modal search across text, tables, and images.
    """
    
    name: str = "milvus_multimodal_search"
    description: str = """
    Search across different content types (text, tables, images) simultaneously.
    
    Use this when you need comprehensive information that might include:
    - Text paragraphs
    - Data tables
    - Image descriptions
    
    Returns organized results grouped by content type.
    """
    args_schema: Type[BaseModel] = MilvusMultiModalSearchInput
    
    _retriever: Any = None
    _embedder: Any = None
    
    def __init__(self, retriever: Any = None, embedder: Any = None, **kwargs):
        super().__init__(**kwargs)
        self._retriever = retriever
        self._embedder = embedder
    
    def _initialize(self) -> None:
        if self._retriever is None:
            from retriever import AdvancedRetriever
            self._retriever = AdvancedRetriever()
        if self._embedder is None:
            from embeddings import OpenAIEmbedder
            self._embedder = OpenAIEmbedder()
    
    def _run(
        self,
        query: str,
        search_text: bool = True,
        search_tables: bool = True,
        search_images: bool = True,
        top_k_per_type: int = 3
    ) -> str:
        """Execute multi-modal search."""
        self._initialize()
        
        try:
            content_types = []
            if search_text:
                content_types.append("text")
            if search_tables:
                content_types.append("table")
            if search_images:
                content_types.append("image")
            
            # Search each content type
            all_results = {}
            query_embedding = self._embedder.embed_query(query)
            
            for c_type in content_types:
                if hasattr(self._retriever, 'search_by_type'):
                    results = self._retriever.search_by_type(
                        query=query,
                        content_type=c_type,
                        top_k=top_k_per_type
                    )
                else:
                    results = self._retriever.search(
                        query_embedding=query_embedding,
                        top_k=top_k_per_type,
                        content_type=c_type
                    )
                all_results[c_type] = results
            
            # Format output
            output_parts = [f"Multi-modal search results for: '{query[:50]}...'\n"]
            
            for c_type, results in all_results.items():
                emoji = {"text": "📝", "table": "📊", "image": "🖼️"}.get(c_type, "📄")
                output_parts.append(f"\n{emoji} {c_type.upper()} RESULTS ({len(results)}):")
                output_parts.append("-" * 40)
                
                if not results:
                    output_parts.append("  No results found")
                    continue
                
                for i, r in enumerate(results, 1):
                    content = getattr(r, 'content', str(r))[:300]
                    score = getattr(r, 'score', 0)
                    output_parts.append(f"\n  [{i}] Score: {score:.3f}")
                    output_parts.append(f"      {content}...")
            
            return "\n".join(output_parts)
            
        except Exception as e:
            return f"Multi-modal search failed: {str(e)}"
    
    async def _arun(self, **kwargs) -> str:
        return self._run(**kwargs)


class MilvusHierarchicalSearchInput(BaseModel):
    """Input for hierarchical search."""
    query: str = Field(..., description="Search query")
    get_parent_context: bool = Field(default=True, description="Include parent chunk for context")
    get_children: bool = Field(default=False, description="Include child chunks")
    top_k: int = Field(default=5, description="Number of results")


class MilvusHierarchicalSearchTool(BaseTool):
    """
    Hierarchical search with parent-child chunk navigation.
    """
    
    name: str = "milvus_hierarchical_search"
    description: str = """
    Search with hierarchical chunk context.
    
    Use when you need:
    - More context around a specific match
    - The broader section containing a detail
    - All details within a broader section
    
    Returns results with their parent/child relationships.
    """
    args_schema: Type[BaseModel] = MilvusHierarchicalSearchInput
    
    _retriever: Any = None
    _vector_store: Any = None
    
    def __init__(self, retriever: Any = None, vector_store: Any = None, **kwargs):
        super().__init__(**kwargs)
        self._retriever = retriever
        self._vector_store = vector_store
    
    def _run(
        self,
        query: str,
        get_parent_context: bool = True,
        get_children: bool = False,
        top_k: int = 5
    ) -> str:
        """Execute hierarchical search."""
        try:
            if self._retriever is None:
                from retriever import AdvancedRetriever
                self._retriever = AdvancedRetriever()
            
            # Initial search
            results = self._retriever.search(query=query, top_k=top_k)
            
            if not results:
                return "No results found."
            
            output_parts = [f"Hierarchical search results:\n"]
            
            for i, r in enumerate(results, 1):
                content = getattr(r, 'content', str(r))
                parent_id = getattr(r, 'parent_id', None)
                metadata = getattr(r, 'metadata', {})
                
                output_parts.append(f"\n{'='*50}")
                output_parts.append(f"Result {i}:")
                output_parts.append(f"Content: {content[:400]}...")
                
                # Get parent context
                if get_parent_context and parent_id and self._vector_store:
                    try:
                        parent_results = self._vector_store.get(ids=[parent_id])
                        if parent_results:
                            parent = parent_results[0]
                            parent_content = getattr(parent, 'content', '')[:500]
                            output_parts.append(f"\n📦 PARENT CONTEXT:")
                            output_parts.append(f"   {parent_content}...")
                    except:
                        pass
                
                # Get children
                if get_children and self._vector_store:
                    try:
                        children = self._vector_store.get(
                            filter_expr=f'parent_id == "{r.id}"'
                        )
                        if children:
                            output_parts.append(f"\n📎 CHILD CHUNKS ({len(children)}):")
                            for child in children[:3]:
                                child_content = getattr(child, 'content', '')[:200]
                                output_parts.append(f"   • {child_content}...")
                    except:
                        pass
            
            return "\n".join(output_parts)
            
        except Exception as e:
            return f"Hierarchical search failed: {str(e)}"
    
    async def _arun(self, **kwargs) -> str:
        return self._run(**kwargs)


# Factory functions
def create_milvus_search_tool(retriever: Any = None, embedder: Any = None) -> MilvusSearchTool:
    """Create Milvus search tool."""
    return MilvusSearchTool(retriever=retriever, embedder=embedder)


def create_multimodal_search_tool(retriever: Any = None, embedder: Any = None) -> MilvusMultiModalSearchTool:
    """Create multi-modal search tool."""
    return MilvusMultiModalSearchTool(retriever=retriever, embedder=embedder)


def create_hierarchical_search_tool(retriever: Any = None, vector_store: Any = None) -> MilvusHierarchicalSearchTool:
    """Create hierarchical search tool."""
    return MilvusHierarchicalSearchTool(retriever=retriever, vector_store=vector_store)
