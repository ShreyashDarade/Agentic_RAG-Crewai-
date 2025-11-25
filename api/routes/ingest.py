import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import JSONResponse

from api.models.requests import IngestRequest, IngestFileRequest
from api.models.responses import IngestResponse, FileIngestionResult, ErrorResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Ingestion"])

# Global ingestion pipeline instance
_ingestion_pipeline = None


def get_ingestion_pipeline():
    """Dependency to get ingestion pipeline."""
    global _ingestion_pipeline
    if _ingestion_pipeline is None:
        from data_pipeline import IngestionPipeline
        _ingestion_pipeline = IngestionPipeline()
    return _ingestion_pipeline


def set_ingestion_pipeline(pipeline):
    """Set the ingestion pipeline instance."""
    global _ingestion_pipeline
    _ingestion_pipeline = pipeline


@router.post(
    "/ingest",
    response_model=IngestResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    },
    summary="Ingest documents from a directory",
    description="Process and ingest all supported documents from a directory into the vector store."
)
async def ingest_directory(
    request: IngestRequest,
    background_tasks: BackgroundTasks,
    pipeline = Depends(get_ingestion_pipeline)
):
    """
    Ingest documents from a directory.
    
    This endpoint:
    1. Discovers supported files in the directory
    2. Parses and extracts text content
    3. Applies OCR if needed
    4. Chunks the content
    5. Generates embeddings
    6. Stores in ChromaDB
    
    Files that have already been ingested are skipped unless force=True.
    """
    try:
        # Run ingestion
        result = pipeline.ingest_directory(
            directory=request.directory,
            force=request.force,
            batch_size=request.batch_size
        )
        
        # Build file results
        files_processed = []
        # Note: The pipeline doesn't return per-file results by default
        # This would need to be enhanced for detailed per-file tracking
        
        return IngestResponse(
            success=result.get("success", False),
            message=result.get("message", "Ingestion completed"),
            stats=result.get("stats", {}),
            files_processed=files_processed,
            errors=result.get("errors")
        )
        
    except Exception as e:
        logger.error(f"Directory ingestion failed: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error": str(e),
                "error_type": "ingestion_error"
            }
        )


@router.post(
    "/ingest/file",
    response_model=IngestResponse,
    summary="Ingest a single file",
    description="Process and ingest a single document file."
)
async def ingest_file(
    request: IngestFileRequest,
    pipeline = Depends(get_ingestion_pipeline)
):
    """
    Ingest a single file or all files within a directory.
    
    Processes a single document through the ingestion pipeline. When the provided
    path points to a directory (or is omitted), the pipeline ingests every supported
    file inside that directory using the directory ingestion workflow.
    """
    try:
        target_path = request.file_path or pipeline.config.raw_documents_dir
        path_obj = Path(target_path)

        if path_obj.is_dir():
            summary = pipeline.ingest_directory(
                directory=str(path_obj),
                force=request.force,
                batch_size=request.batch_size
            )
            return IngestResponse(
                success=summary.get("success", False),
                message=summary.get("message", f"Ingested directory: {path_obj}"),
                stats=summary.get("stats", {}),
                files_processed=[],
                errors=summary.get("errors")
            )
        
        if not path_obj.exists():
            raise HTTPException(
                status_code=404,
                detail=f"File not found: {target_path}"
            )
        
        if not path_obj.is_file():
            raise HTTPException(
                status_code=400,
                detail=f"Path is not a file: {target_path}"
            )

        result = pipeline.process_file(
            file_path=str(path_obj),
            force=request.force
        )
        
        return IngestResponse(
            success=result.get("success", False),
            message=f"File {'ingested' if result.get('success') else 'failed'}",
            stats={
                "files_processed": 1 if result.get("success") else 0,
                "chunks_created": result.get("chunks_created", 0)
            },
            files_processed=[
                FileIngestionResult(
                    file_path=result.get("file_path", str(path_obj)),
                    file_name=result.get("file_name", path_obj.name),
                    success=result.get("success", False),
                    chunks_created=result.get("chunks_created", 0),
                    error=result.get("error"),
                    skipped=result.get("skipped", False)
                )
            ],
            errors=[{"file": str(path_obj), "error": result.get("error")}] if result.get("error") else None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File ingestion failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@router.post(
    "/ingest/upload",
    response_model=IngestResponse,
    summary="Upload and ingest a file",
    description="Upload a file and ingest it into the vector store."
)
async def upload_and_ingest(
    file: UploadFile = File(...),
    force: bool = Form(default=False),
    pipeline = Depends(get_ingestion_pipeline)
):
    """
    Upload and ingest a file.
    
    Accepts file upload and processes it through the ingestion pipeline.
    """
    try:
        # Save uploaded file
        raw_dir = Path(pipeline.config.raw_documents_dir)
        raw_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = raw_dir / file.filename
        
        # Write file
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        
        logger.info(f"Uploaded file saved: {file_path}")
        
        # Process file
        result = pipeline.process_file(
            file_path=str(file_path),
            force=force
        )
        
        return IngestResponse(
            success=result.get("success", False),
            message=f"File {'uploaded and ingested' if result.get('success') else 'upload succeeded but ingestion failed'}",
            stats={
                "files_processed": 1 if result.get("success") else 0,
                "chunks_created": result.get("chunks_created", 0)
            },
            files_processed=[
                FileIngestionResult(
                    file_path=str(file_path),
                    file_name=file.filename,
                    success=result.get("success", False),
                    chunks_created=result.get("chunks_created", 0),
                    error=result.get("error"),
                    skipped=result.get("skipped", False)
                )
            ]
        )
        
    except Exception as e:
        logger.error(f"Upload and ingest failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@router.get(
    "/ingest/status",
    summary="Get ingestion status",
    description="Get the current status of the ingestion pipeline."
)
async def get_ingestion_status(
    pipeline = Depends(get_ingestion_pipeline)
):
    """Get ingestion pipeline status."""
    try:
        status = pipeline.get_ingestion_status()
        return {
            "success": True,
            "status": status
        }
    except Exception as e:
        logger.error(f"Failed to get status: {e}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@router.get(
    "/ingest/files",
    summary="List ingested files",
    description="Get a list of all ingested files."
)
async def list_ingested_files(
    pipeline = Depends(get_ingestion_pipeline)
):
    """List all ingested files."""
    try:
        all_docs = pipeline.state.get_all_documents()
        
        files = []
        for file_hash, doc_state in all_docs.items():
            files.append({
                "file_path": doc_state.file_path,
                "file_name": doc_state.file_name,
                "file_hash": doc_state.file_hash,
                "status": doc_state.status,
                "chunk_count": doc_state.chunk_count,
                "ingested_at": doc_state.ingested_at,
                "error": doc_state.error_message
            })
        
        return {
            "success": True,
            "files": files,
            "total": len(files)
        }
        
    except Exception as e:
        logger.error(f"Failed to list files: {e}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@router.delete(
    "/ingest/reset",
    summary="Reset ingestion state",
    description="Reset the ingestion state and optionally clear the vector store."
)
async def reset_ingestion(
    clear_vector_store: bool = True,
    pipeline = Depends(get_ingestion_pipeline)
):
    """Reset ingestion state."""
    try:
        pipeline.reset(clear_vector_store=clear_vector_store)
        return {
            "success": True,
            "message": "Ingestion state reset",
            "vector_store_cleared": clear_vector_store
        }
    except Exception as e:
        logger.error(f"Failed to reset: {e}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@router.post(
    "/ingest/discover",
    summary="Discover files without ingesting",
    description="Scan a directory and return discovered files without processing them."
)
async def discover_files(
    request: IngestRequest,
    pipeline = Depends(get_ingestion_pipeline)
):
    """Discover files in a directory."""
    try:
        files = pipeline.discover_files(
            directory=request.directory,
            recursive=request.recursive
        )
        
        return {
            "success": True,
            "files": files,
            "total": len(files),
            "already_processed": sum(1 for f in files if f.get("already_processed")),
            "pending": sum(1 for f in files if not f.get("already_processed"))
        }
        
    except Exception as e:
        logger.error(f"Failed to discover files: {e}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

