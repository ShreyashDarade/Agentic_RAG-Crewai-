"""
Multi-Agent RAG API - Production Grade

Features:
- OpenAI LLM integration
- Milvus Cloud vector store
- Advanced retrieval with RRF fusion
- CrewAI multi-agent orchestration
- Comprehensive health monitoring
"""

import logging
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# CrewAI settings
CREW_SETTINGS_PATH = PROJECT_ROOT / "config" / "crew_settings.json"
if not os.environ.get("CREWAI_SETTINGS_PATH"):
    CREW_SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not CREW_SETTINGS_PATH.exists():
        CREW_SETTINGS_PATH.write_text("{}\n", encoding="utf-8")
    os.environ["CREWAI_SETTINGS_PATH"] = str(CREW_SETTINGS_PATH)

from api.routes.query import router as query_router, set_crew_manager
from api.routes.ingest import router as ingest_router, set_ingestion_pipeline
from api.models.responses import HealthResponse, ErrorResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global instances
_crew_manager = None
_ingestion_pipeline = None
_start_time = None


def validate_environment() -> bool:
    """Validate required environment variables."""
    errors = []
    
    if not os.getenv("OPENAI_API_KEY"):
        errors.append("OPENAI_API_KEY is required")
    
    if not os.getenv("MILVUS_URI") and not os.getenv("ZILLIZ_URI"):
        errors.append("MILVUS_URI or ZILLIZ_URI is required for Milvus Cloud")
    
    if errors:
        for error in errors:
            logger.error(error)
        return False
    
    return True


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    global _crew_manager, _ingestion_pipeline, _start_time
    
    logger.info("=" * 60)
    logger.info("Starting Multi-Agent RAG API (Production)")
    logger.info("=" * 60)
    _start_time = datetime.now()
    
    # Validate environment
    if not validate_environment():
        logger.error("Environment validation failed - API will start in degraded mode")
    
    # Initialize components
    try:
        from orchestrator import CrewManager
        from data_pipeline import IngestionPipeline
        
        # Initialize crew manager with OpenAI + Milvus
        logger.info("Initializing CrewManager...")
        _crew_manager = CrewManager()
        _crew_manager.initialize()
        set_crew_manager(_crew_manager)
        logger.info("✓ CrewManager initialized")
        
        # Initialize ingestion pipeline
        logger.info("Initializing Ingestion Pipeline...")
        _ingestion_pipeline = IngestionPipeline()
        set_ingestion_pipeline(_ingestion_pipeline)
        logger.info("✓ Ingestion Pipeline initialized")
        
        logger.info("=" * 60)
        logger.info("API Ready - All components initialized")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        logger.warning("API starting in degraded mode")
    
    yield
    
    # Cleanup
    logger.info("Shutting down Multi-Agent RAG API...")
    if _crew_manager and hasattr(_crew_manager, '_llm') and _crew_manager._llm:
        try:
            import asyncio
            await _crew_manager._llm.close()
        except Exception:
            pass


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    
    app = FastAPI(
        title="Multi-Agent RAG API",
        description="""
        Production-grade Multi-Agent RAG (Retrieval-Augmented Generation) system.
        
        ## Features
        
        - **OpenAI GPT-4o-mini** for LLM
        - **OpenAI text-embedding-3-small** for embeddings
        - **Milvus Cloud** with HNSW indexing for vector storage
        - **EasyOCR** multilingual OCR with spaCy NLP
        - **Advanced Chunking** with cross-references (text ↔ tables ↔ images)
        - **RRF Fusion** combining dense + BM25 search
        - **Cross-encoder Re-ranking** for improved relevance
        - **MMR Diversity** for varied results
        - **CrewAI Multi-Agent** orchestration
        
        ## Architecture
        
        - **Supervisor Agent**: Query analysis and planning
        - **Retriever Agent**: Multi-modal document retrieval
        - **Generator Agent**: Context-aware response synthesis
        - **Feedback Agent**: Response validation and improvement
        
        ## Endpoints
        
        ### Query Processing
        - `POST /api/v1/agent_query` - Process query through multi-agent system
        - `POST /api/v1/search` - Direct search without agent pipeline
        - `GET /api/v1/trace/{trace_id}` - Get execution trace
        - `GET /api/v1/history` - Get conversation history
        
        ### Document Ingestion
        - `POST /api/v1/ingest` - Ingest documents from directory
        - `POST /api/v1/ingest/file` - Ingest a single file
        - `POST /api/v1/ingest/upload` - Upload and ingest a file
        - `GET /api/v1/ingest/status` - Get ingestion status
        """,
        version="2.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include routers
    app.include_router(query_router)
    app.include_router(ingest_router)
    
    # Exception handlers
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        errors = []
        for error in exc.errors():
            errors.append({
                "field": ".".join(str(x) for x in error["loc"]),
                "message": error["msg"]
            })
        
        return JSONResponse(
            status_code=422,
            content={
                "success": False,
                "error": "Validation error",
                "error_type": "validation_error",
                "detail": errors
            }
        )
    
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "success": False,
                "error": str(exc.detail),
                "error_type": "http_error"
            }
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unexpected error: {exc}", exc_info=True)
        
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "Internal server error",
                "error_type": "internal_error",
                "detail": str(exc) if os.getenv("DEBUG", "false").lower() == "true" else None
            }
        )
    
    # Health check
    @app.get(
        "/health",
        response_model=HealthResponse,
        tags=["Health"],
        summary="Health check",
        description="Check API health and component status"
    )
    async def health_check():
        components = {}
        overall_status = "healthy"
        
        # Check LLM (OpenAI)
        try:
            if _crew_manager and _crew_manager._llm:
                components["llm"] = {
                    "status": "healthy",
                    "provider": "openai",
                    "model": "gpt-4o-mini"
                }
            else:
                components["llm"] = {"status": "not_initialized"}
                overall_status = "degraded"
        except Exception as e:
            components["llm"] = {"status": "unhealthy", "error": str(e)}
            overall_status = "unhealthy"
        
        # Check Vector Store (Milvus Cloud)
        try:
            if _crew_manager and _crew_manager._advanced_retriever:
                doc_count = _crew_manager._advanced_retriever.document_count
                components["vector_store"] = {
                    "status": "healthy",
                    "provider": "milvus_cloud",
                    "index_type": "HNSW",
                    "document_count": doc_count
                }
            else:
                components["vector_store"] = {"status": "not_initialized"}
        except Exception as e:
            components["vector_store"] = {"status": "unhealthy", "error": str(e)}
            overall_status = "unhealthy"
        
        # Check Ingestion Pipeline
        try:
            if _ingestion_pipeline:
                status = _ingestion_pipeline.get_ingestion_status()
                components["ingestion"] = {
                    "status": "healthy",
                    "documents_count": status.get("total_documents", 0)
                }
            else:
                components["ingestion"] = {"status": "not_initialized"}
        except Exception as e:
            components["ingestion"] = {"status": "unhealthy", "error": str(e)}
        
        return HealthResponse(
            status=overall_status,
            version="2.0.0",
            components=components
        )
    
    # Root endpoint
    @app.get("/", tags=["Root"])
    async def root():
        return {
            "name": "Multi-Agent RAG API",
            "version": "2.0.0",
            "description": "Production-grade RAG with OpenAI + Milvus Cloud",
            "docs": "/docs",
            "health": "/health",
            "status": "running",
            "features": {
                "llm": "OpenAI GPT-4o-mini",
                "embeddings": "OpenAI text-embedding-3-small",
                "vector_store": "Milvus Cloud (HNSW)",
                "ocr": "EasyOCR + spaCy",
                "retrieval": "RRF + Re-ranking + MMR",
                "agents": "CrewAI Multi-Agent"
            }
        }
    
    # Status endpoint
    @app.get("/status", tags=["Health"])
    async def status():
        global _start_time
        
        uptime = (datetime.now() - _start_time).total_seconds() if _start_time else 0
        
        stats = {
            "status": "running",
            "uptime_seconds": uptime,
            "uptime_human": f"{int(uptime // 3600)}h {int((uptime % 3600) // 60)}m",
            "environment": os.getenv("APP_ENV", "development"),
        }
        
        # Add trace stats
        if _crew_manager and _crew_manager._trace_logger:
            try:
                trace_stats = _crew_manager._trace_logger.get_stats()
                stats["traces"] = trace_stats
            except:
                pass
        
        # Add memory stats
        if _crew_manager and _crew_manager._memory_store:
            try:
                memory_stats = _crew_manager._memory_store.get_stats()
                stats["memory"] = memory_stats
            except:
                pass
        
        return stats
    
    return app


# Create app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
