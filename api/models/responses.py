from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime


class Source(BaseModel):
    """Source reference in response."""
    
    id: Optional[str] = None
    title: Optional[str] = None
    source: str
    url: Optional[str] = None
    score: float = 0.0
    snippet: Optional[str] = None


class TraceStep(BaseModel):
    """Single step in execution trace."""
    
    agent: str
    action: str
    status: str
    duration_ms: Optional[float] = None
    timestamp: str


class TraceResponse(BaseModel):
    """Execution trace response."""
    
    trace_id: str
    query: str
    status: str
    total_duration_ms: float
    steps: List[TraceStep]
    tool_chain: List[Dict[str, Any]]


class QueryResponse(BaseModel):
    """Response model for query endpoint."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "query": "What is the capital of France?",
                "response": "The capital of France is Paris.",
                "sources": [
                    {
                        "source": "geography.pdf",
                        "score": 0.95,
                        "snippet": "Paris is the capital city of France..."
                    }
                ],
                "confidence": 0.92,
                "execution_time_ms": 1234.56,
                "trace": None,
                "metadata": {}
            }
        }
    )
    
    success: bool = Field(
        ...,
        description="Whether the query was processed successfully"
    )
    query: str = Field(
        ...,
        description="The original query"
    )
    response: str = Field(
        ...,
        description="The generated response"
    )
    sources: List[Source] = Field(
        default_factory=list,
        description="Sources used in generating the response"
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence score of the response"
    )
    execution_time_ms: float = Field(
        default=0.0,
        description="Total execution time in milliseconds"
    )
    trace: Optional[TraceResponse] = Field(
        default=None,
        description="Execution trace (if requested)"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if query failed"
    )
    

class FileIngestionResult(BaseModel):
    """Result of single file ingestion."""
    
    file_path: str
    file_name: str
    success: bool
    chunks_created: int = 0
    error: Optional[str] = None
    skipped: bool = False


class IngestResponse(BaseModel):
    """Response model for ingestion endpoint."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "message": "Processed 5 files",
                "stats": {
                    "files_discovered": 10,
                    "files_processed": 5,
                    "files_skipped": 5,
                    "chunks_created": 50
                },
                "files_processed": [
                    {
                        "file_path": "./data/raw/doc.pdf",
                        "file_name": "doc.pdf",
                        "success": True,
                        "chunks_created": 10
                    }
                ]
            }
        }
    )
    
    success: bool = Field(
        ...,
        description="Whether ingestion was successful"
    )
    message: str = Field(
        ...,
        description="Status message"
    )
    stats: Dict[str, int] = Field(
        default_factory=dict,
        description="Ingestion statistics"
    )
    files_processed: List[FileIngestionResult] = Field(
        default_factory=list,
        description="Results per file"
    )
    errors: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description="List of errors if any"
    )
    

class HealthResponse(BaseModel):
    """Response model for health check endpoint."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "timestamp": "2024-01-15T10:30:00",
                "components": {
                    "llm": {"status": "healthy"},
                    "vector_store": {"status": "healthy", "document_count": 1000},
                    "embedder": {"status": "healthy"}
                }
            }
        }
    )
    
    status: str = Field(
        ...,
        description="Service status"
    )
    version: str = Field(
        ...,
        description="API version"
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="Current timestamp"
    )
    components: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Component health status"
    )
    

class ErrorResponse(BaseModel):
    """Response model for errors."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": False,
                "error": "Query processing failed",
                "error_type": "processing_error",
                "detail": "LLM timeout after 60 seconds",
                "trace_id": "abc12345"
            }
        }
    )
    
    success: bool = Field(
        default=False,
        description="Always false for errors"
    )
    error: str = Field(
        ...,
        description="Error message"
    )
    error_type: str = Field(
        default="unknown",
        description="Type of error"
    )
    detail: Optional[str] = Field(
        default=None,
        description="Detailed error information"
    )
    trace_id: Optional[str] = Field(
        default=None,
        description="Trace ID for debugging"
    )
    

class StatusResponse(BaseModel):
    """Response model for status endpoint."""
    
    status: str
    uptime_seconds: float
    total_queries: int
    successful_queries: int
    failed_queries: int
    avg_response_time_ms: float
    document_count: int
    memory_entries: int


class CollectionInfo(BaseModel):
    """Information about a vector store collection."""
    
    name: str
    document_count: int
    embedding_dimension: int
    distance_metric: str


class ListCollectionsResponse(BaseModel):
    """Response for listing collections."""
    
    collections: List[CollectionInfo]
    total: int

