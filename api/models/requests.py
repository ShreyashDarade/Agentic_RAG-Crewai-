from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, validator, ConfigDict


class QueryRequest(BaseModel):
    """Request model for query endpoint."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "What is the capital of France?",
                "include_sources": True,
                "include_trace": False,
                "max_results": 10
            }
        }
    )
    
    query: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="The user query to process"
    )
    context: Optional[str] = Field(
        default=None,
        max_length=10000,
        description="Optional context from previous interactions"
    )
    include_sources: bool = Field(
        default=True,
        description="Whether to include source references in response"
    )
    include_trace: bool = Field(
        default=False,
        description="Whether to include execution trace"
    )
    use_web_search: Optional[bool] = Field(
        default=None,
        description="Override web search setting (None = auto-detect)"
    )
    max_results: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum retrieval results"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional metadata for the query"
    )
    
    @validator("query")
    def validate_query(cls, v):
        """Validate query is not empty after stripping."""
        if not v or not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()
    

class IngestRequest(BaseModel):
    """Request model for directory ingestion."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "directory": "./data/raw",
                "force": False,
                "recursive": True,
                "batch_size": 10
            }
        }
    )
    
    directory: Optional[str] = Field(
        default=None,
        description="Directory path to ingest (uses default if not provided)"
    )
    force: bool = Field(
        default=False,
        description="Force reprocessing of already ingested files"
    )
    recursive: bool = Field(
        default=True,
        description="Recursively scan subdirectories"
    )
    batch_size: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of files to process per batch"
    )
    file_extensions: Optional[List[str]] = Field(
        default=None,
        description="Filter by file extensions (e.g., ['.pdf', '.docx'])"
    )
    collection_name: Optional[str] = Field(
        default=None,
        description="Target collection name"
    )
    

class IngestFileRequest(BaseModel):
    """Request model for single file or directory ingestion."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "file_path": "./data/raw/",
                "force": False,
                "batch_size": 25
            }
        }
    )
    
    file_path: Optional[str] = Field(
        default=None,
        description="Path to the file or directory to ingest. When omitted the pipeline raw directory is used."
    )
    force: bool = Field(
        default=False,
        description="Force reprocessing if already ingested"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional metadata for the file"
    )
    batch_size: Optional[int] = Field(
        default=None,
        ge=1,
        le=100,
        description="Override batch size when ingesting a directory"
    )
    
    @validator("file_path")
    def validate_file_path(cls, v):
        """Normalize file path input."""
        if v is None:
            return None
        stripped = v.strip()
        return stripped or None
    

class SearchRequest(BaseModel):
    """Request model for direct search."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "Multi-agent coordination best practices",
                "top_k": 5,
                "search_type": "hybrid"
            }
        }
    )
    
    query: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Search query"
    )
    top_k: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Number of results"
    )
    search_type: str = Field(
        default="hybrid",
        description="Search type: 'local', 'web', or 'hybrid'"
    )
    filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Metadata filters"
    )
    
    @validator("search_type")
    def validate_search_type(cls, v):
        """Validate search type."""
        valid_types = ["local", "web", "hybrid"]
        if v.lower() not in valid_types:
            raise ValueError(f"search_type must be one of {valid_types}")
        return v.lower()


class FeedbackRequest(BaseModel):
    """Request model for feedback on responses."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "trace_id": "abc12345",
                "rating": 4,
                "comment": "Good response but could include more details"
            }
        }
    )
    
    trace_id: str = Field(
        ...,
        description="Trace ID of the response"
    )
    rating: int = Field(
        ...,
        ge=1,
        le=5,
        description="Rating from 1 to 5"
    )
    comment: Optional[str] = Field(
        default=None,
        max_length=1000,
        description="Optional feedback comment"
    )
    

