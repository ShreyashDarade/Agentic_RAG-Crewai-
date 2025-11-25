import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from fastapi.responses import StreamingResponse

from api.models.requests import QueryRequest, SearchRequest, FeedbackRequest
from api.models.responses import (
    QueryResponse,
    TraceResponse,
    TraceStep,
    Source,
    ErrorResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Query"])

# Global crew manager instance (set by main.py)
_crew_manager = None


def get_crew_manager():
    """Dependency to get crew manager."""
    global _crew_manager
    if _crew_manager is None:
        from orchestrator import CrewManager
        _crew_manager = CrewManager()
        _crew_manager.initialize()
    return _crew_manager


def set_crew_manager(manager):
    """Set the crew manager instance."""
    global _crew_manager
    _crew_manager = manager


@router.post(
    "/agent_query",
    response_model=QueryResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    },
    summary="Process a query through the multi-agent system",
    description="Submit a query to be processed by the multi-agent RAG system. "
                "Returns the generated response along with sources and execution trace."
)
async def agent_query(
    request: QueryRequest,
    crew_manager = Depends(get_crew_manager)
):
    """
    Process a query through the multi-agent pipeline.
    
    This endpoint:
    1. Analyzes the query (Supervisor Agent)
    2. Retrieves relevant information (Retriever Agent)
    3. Generates a response (Generator Agent)
    4. Validates and improves the response (Feedback Agent)
    
    Returns the final response along with sources and optional execution trace.
    """
    try:
        # Execute query
        result = crew_manager.execute_query(
            query=request.query,
            context=request.context,
            metadata=request.metadata
        )
        
        # Build sources
        sources = []
        if request.include_sources and result.sources:
            for i, source in enumerate(result.sources[:10]):
                if isinstance(source, str):
                    sources.append(Source(
                        id=f"source_{i}",
                        source=source,
                        score=0.0
                    ))
                elif isinstance(source, dict):
                    sources.append(Source(
                        id=source.get("id", f"source_{i}"),
                        title=source.get("title"),
                        source=source.get("source", "unknown"),
                        url=source.get("url"),
                        score=source.get("score", 0.0),
                        snippet=source.get("snippet")
                    ))
        
        # Build trace if requested
        trace = None
        if request.include_trace and result.metadata.get("trace_id"):
            trace_data = crew_manager.get_execution_trace(result.metadata["trace_id"])
            if trace_data:
                trace = TraceResponse(
                    trace_id=trace_data.get("trace_id", ""),
                    query=trace_data.get("query", request.query),
                    status=trace_data.get("status", "completed"),
                    total_duration_ms=trace_data.get("total_duration_ms", 0),
                    steps=[
                        TraceStep(
                            agent=s.get("agent", ""),
                            action=s.get("action", ""),
                            status=s.get("status", ""),
                            duration_ms=s.get("duration_ms"),
                            timestamp=s.get("timestamp", "")
                        )
                        for s in trace_data.get("steps", [])
                    ],
                    tool_chain=trace_data.get("tool_chain", [])
                )
        
        return QueryResponse(
            success=result.success,
            query=request.query,
            response=result.response,
            sources=sources,
            confidence=result.confidence,
            execution_time_ms=result.execution_time * 1000,
            trace=trace,
            metadata=result.metadata,
            error=result.error
        )
        
    except Exception as e:
        logger.error(f"Query processing failed: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error": str(e),
                "error_type": "processing_error"
            }
        )


@router.post(
    "/search",
    response_model=QueryResponse,
    summary="Direct search without full agent pipeline",
    description="Perform a direct search without going through the full agent pipeline."
)
async def search(
    request: SearchRequest,
    crew_manager = Depends(get_crew_manager)
):
    """
    Perform direct search.
    
    Bypasses the full agent pipeline for faster results.
    Useful for simple retrieval tasks.
    """
    try:
        # Get retriever from crew manager
        if not crew_manager._initialized:
            crew_manager.initialize()
        
        retriever = crew_manager._retriever
        if not retriever:
            raise HTTPException(
                status_code=500,
                detail="Retriever not initialized"
            )
        
        # Perform search
        use_web = request.search_type in ["web", "hybrid"]
        use_local = request.search_type in ["local", "hybrid"]
        
        results = retriever.retrieve(
            query=request.query,
            use_documents=use_local,
            use_web=use_web,
            top_k=request.top_k,
            filters=request.filters
        )
        
        # Build sources
        sources = []
        for r in results.get("local_results", [])[:10]:
            sources.append(Source(
                id=r.get("id", ""),
                source=r.get("source", "local"),
                score=r.get("score", 0.0),
                snippet=r.get("content", "")[:200]
            ))
        
        for r in results.get("web_results", [])[:5]:
            sources.append(Source(
                title=r.get("title"),
                source="web",
                url=r.get("url"),
                score=r.get("score", 0.0),
                snippet=r.get("content", "")[:200]
            ))
        
        return QueryResponse(
            success=results.get("success", True),
            query=request.query,
            response=results.get("combined_context", "No results found"),
            sources=sources,
            confidence=0.8 if sources else 0.0,
            execution_time_ms=0,
            error=results.get("error")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@router.get(
    "/trace/{trace_id}",
    response_model=TraceResponse,
    summary="Get execution trace",
    description="Retrieve the execution trace for a previous query."
)
async def get_trace(
    trace_id: str,
    crew_manager = Depends(get_crew_manager)
):
    """Get execution trace by ID."""
    try:
        trace_data = crew_manager.get_execution_trace(trace_id)
        
        if not trace_data:
            raise HTTPException(
                status_code=404,
                detail=f"Trace {trace_id} not found"
            )
        
        return TraceResponse(
            trace_id=trace_data.get("trace_id", trace_id),
            query=trace_data.get("query", ""),
            status=trace_data.get("status", "unknown"),
            total_duration_ms=trace_data.get("total_duration_ms", 0),
            steps=[
                TraceStep(
                    agent=s.get("agent", ""),
                    action=s.get("action", ""),
                    status=s.get("status", ""),
                    duration_ms=s.get("duration_ms"),
                    timestamp=s.get("timestamp", "")
                )
                for s in trace_data.get("steps", [])
            ],
            tool_chain=trace_data.get("tool_chain", [])
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get trace: {e}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@router.get(
    "/history",
    summary="Get conversation history",
    description="Retrieve recent conversation history."
)
async def get_history(
    limit: int = Query(default=10, ge=1, le=100),
    crew_manager = Depends(get_crew_manager)
):
    """Get recent conversation history."""
    try:
        history = crew_manager.get_conversation_history(limit=limit)
        return {
            "success": True,
            "history": history,
            "count": len(history)
        }
    except Exception as e:
        logger.error(f"Failed to get history: {e}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@router.delete(
    "/history",
    summary="Clear conversation history",
    description="Clear all conversation history from memory."
)
async def clear_history(
    crew_manager = Depends(get_crew_manager)
):
    """Clear conversation history."""
    try:
        crew_manager.clear_memory()
        return {
            "success": True,
            "message": "History cleared"
        }
    except Exception as e:
        logger.error(f"Failed to clear history: {e}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@router.post(
    "/feedback",
    summary="Submit feedback on a response",
    description="Submit user feedback for a query response."
)
async def submit_feedback(
    request: FeedbackRequest,
    background_tasks: BackgroundTasks
):
    """Submit feedback for a response."""
    try:
        # Log feedback (could be stored in a database)
        logger.info(
            f"Feedback received: trace_id={request.trace_id}, "
            f"rating={request.rating}, comment={request.comment}"
        )
        
        return {
            "success": True,
            "message": "Feedback submitted successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to submit feedback: {e}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

