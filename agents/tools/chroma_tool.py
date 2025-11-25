import logging
from typing import Any, Dict, List, Optional, Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ChromaSearchInput(BaseModel):
    """Input schema for ChromaDB search tool."""
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
    filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional metadata filters for the search"
    )


class ChromaSearchTool(BaseTool):
    """
    Tool for searching documents in ChromaDB.
    
    This tool performs semantic search on the document database
    using embeddings and optional keyword matching.
    """
    
    name: str = "chroma_search"
    description: str = """
    Search the document database for relevant information.
    Use this tool when you need to find information from ingested documents.
    Provide a clear, specific query for best results.
    Returns relevant document chunks with their content and metadata.
    """
    args_schema: Type[BaseModel] = ChromaSearchInput
    
    # Internal retriever reference
    _retriever: Any = None
    
    def __init__(self, retriever: Any = None, **kwargs):
        """
        Initialize the ChromaDB search tool.
        
        Args:
            retriever: ChromaRetriever or HybridRetriever instance
            **kwargs: Additional arguments
        """
        super().__init__(**kwargs)
        self._retriever = retriever
    
    def _initialize_retriever(self) -> None:
        """Initialize retriever if not provided."""
        if self._retriever is None:
            try:
                from ...retriever import ChromaRetriever
                self._retriever = ChromaRetriever()
                logger.info("Initialized default ChromaRetriever")
            except Exception as e:
                logger.error(f"Failed to initialize retriever: {e}")
                raise
    
    def _run(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Execute the ChromaDB search.
        
        Args:
            query: Search query
            top_k: Number of results
            filters: Metadata filters
            
        Returns:
            Formatted search results
        """
        self._initialize_retriever()
        
        try:
            logger.info(f"Searching ChromaDB: {query[:50]}... (top_k={top_k})")
            
            results = self._retriever.search(
                query=query,
                top_k=top_k,
                filters=filters
            )
            
            if not results:
                return "No relevant documents found for this query."
            
            # Format results
            output_parts = [f"Found {len(results)} relevant documents:\n"]
            
            for i, result in enumerate(results, 1):
                output_parts.append(f"\n--- Result {i} (Score: {result.score:.3f}) ---")
                
                # Add source info
                source = result.metadata.get("source") or result.metadata.get("file_name", "Unknown")
                output_parts.append(f"Source: {source}")
                
                # Add chunk info
                chunk_idx = result.metadata.get("chunk_index", 0)
                total_chunks = result.metadata.get("total_chunks", 1)
                output_parts.append(f"Chunk: {chunk_idx + 1}/{total_chunks}")
                
                # Add content
                output_parts.append(f"\nContent:\n{result.content}")
                output_parts.append("")
            
            return "\n".join(output_parts)
            
        except Exception as e:
            error_msg = f"ChromaDB search failed: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    async def _arun(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> str:
        """Async version of the search."""
        # For now, just run sync version
        return self._run(query, top_k, filters)


class ChromaFilteredSearchInput(BaseModel):
    """Input schema for filtered ChromaDB search."""
    query: str = Field(
        ...,
        description="The search query"
    )
    source_filter: Optional[str] = Field(
        default=None,
        description="Filter by source file name or path"
    )
    top_k: int = Field(
        default=5,
        description="Number of results to return"
    )


class ChromaFilteredSearchTool(BaseTool):
    """
    Tool for filtered document search.
    
    Allows searching with source file filters.
    """
    
    name: str = "chroma_filtered_search"
    description: str = """
    Search documents with optional source filtering.
    Use when you need to search within specific documents or sources.
    """
    args_schema: Type[BaseModel] = ChromaFilteredSearchInput
    
    _retriever: Any = None
    
    def __init__(self, retriever: Any = None, **kwargs):
        super().__init__(**kwargs)
        self._retriever = retriever
    
    def _initialize_retriever(self) -> None:
        if self._retriever is None:
            from ...retriever import ChromaRetriever
            self._retriever = ChromaRetriever()
    
    def _run(
        self,
        query: str,
        source_filter: Optional[str] = None,
        top_k: int = 5
    ) -> str:
        """Execute filtered search."""
        self._initialize_retriever()
        
        try:
            filters = None
            if source_filter:
                filters = {"source": {"$contains": source_filter}}
            
            results = self._retriever.search(
                query=query,
                top_k=top_k,
                filters=filters
            )
            
            if not results:
                return "No documents found matching the criteria."
            
            output_parts = [f"Found {len(results)} documents:\n"]
            
            for i, result in enumerate(results, 1):
                output_parts.append(f"\n[{i}] {result.content[:200]}...")
                output_parts.append(f"    (Score: {result.score:.3f}, Source: {result.source})")
            
            return "\n".join(output_parts)
            
        except Exception as e:
            return f"Search failed: {str(e)}"
    
    async def _arun(self, **kwargs) -> str:
        return self._run(**kwargs)


def create_chroma_search_tool(retriever: Any = None) -> ChromaSearchTool:
    """
    Factory function to create a ChromaDB search tool.
    
    Args:
        retriever: Optional retriever instance
        
    Returns:
        Configured ChromaSearchTool
    """
    return ChromaSearchTool(retriever=retriever)

