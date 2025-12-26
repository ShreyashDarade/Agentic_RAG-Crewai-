"""
Ingestion Pipeline - Production Grade with Milvus Cloud

Features:
- Multi-format document ingestion
- EasyOCR for scanned documents
- Advanced chunking with cross-references
- OpenAI embeddings
- Milvus Cloud vector storage
- Batch processing with retry logic
"""

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class IngestionConfig:
    """Configuration for the ingestion pipeline."""
    input_directory: str = "./data/raw"
    processed_directory: str = "./data/processed"
    
    batch_size: int = 10
    max_file_size_mb: int = 50
    
    supported_extensions: List[str] = field(default_factory=lambda: [
        ".pdf", ".docx", ".doc", ".txt", ".md",
        ".html", ".htm", ".xlsx", ".xls", ".csv",
        ".pptx", ".ppt", ".json", ".xml",
        ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp"
    ])
    
    # OCR settings
    enable_ocr: bool = True
    ocr_languages: List[str] = field(default_factory=lambda: ["en"])
    
    # Chunking settings
    chunk_strategy: str = "semantic"
    chunk_size: int = 512
    chunk_overlap: int = 100
    enable_cross_reference: bool = True
    enable_hierarchy: bool = True
    
    # Embedding settings (OpenAI only)
    embedding_model: str = "text-embedding-3-small"
    
    # Vector store settings (Milvus Cloud only)
    collection_name: str = "documents"
    
    # Deduplication
    enable_dedup: bool = True


@dataclass
class IngestionResult:
    """Result of an ingestion operation."""
    total_files: int
    processed_files: int
    failed_files: int
    total_chunks: int
    total_embeddings: int
    processing_time: float
    errors: List[Dict[str, str]]
    file_results: List[Dict[str, Any]]
    
    @property
    def success_rate(self) -> float:
        if self.total_files == 0:
            return 0.0
        return self.processed_files / self.total_files
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_files": self.total_files,
            "processed_files": self.processed_files,
            "failed_files": self.failed_files,
            "total_chunks": self.total_chunks,
            "total_embeddings": self.total_embeddings,
            "processing_time": self.processing_time,
            "success_rate": self.success_rate,
            "errors": self.errors,
            "file_results": self.file_results
        }


class IngestionError(Exception):
    """Exception for ingestion errors."""
    pass


class IngestionPipeline:
    """
    Production-grade ingestion pipeline.
    
    Handles:
    1. Document loading (multi-format)
    2. OCR for images/scanned PDFs
    3. Cross-reference chunking
    4. OpenAI embedding generation
    5. Milvus Cloud storage
    """
    
    def __init__(self, config: Optional[IngestionConfig] = None):
        self.config = config or IngestionConfig()
        
        self._file_loader = None
        self._ocr_processor = None
        self._chunker = None
        self._embedder = None
        self._vector_store = None
        
        self._processed_hashes: set = set()
        
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize pipeline components."""
        try:
            # File loader
            from .file_loader import FileLoader
            self._file_loader = FileLoader()
            
            # OCR processor (EasyOCR + spaCy)
            if self.config.enable_ocr:
                from .ocr_processor import EnhancedOCRProcessor, EnhancedOCRConfig
                ocr_config = EnhancedOCRConfig(
                    languages=self.config.ocr_languages,
                    nlp_enabled=True
                )
                self._ocr_processor = EnhancedOCRProcessor(ocr_config)
            
            # Advanced chunker
            from .chunker import AdvancedChunker, AdvancedChunkerConfig
            chunker_config = AdvancedChunkerConfig(
                strategy=self.config.chunk_strategy,
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                enable_cross_reference=self.config.enable_cross_reference,
                enable_hierarchy=self.config.enable_hierarchy
            )
            self._chunker = AdvancedChunker(chunker_config)
            
            # OpenAI embedder
            self._initialize_embedder()
            
            # Milvus Cloud store
            self._initialize_vector_store()
            
            logger.info("Ingestion pipeline initialized successfully")
            
        except Exception as e:
            logger.error(f"Pipeline initialization failed: {e}")
            raise IngestionError(f"Failed to initialize pipeline: {str(e)}")
    
    def _initialize_embedder(self) -> None:
        """Initialize OpenAI embedder."""
        from embeddings import OpenAIEmbedder
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise IngestionError("OPENAI_API_KEY not set")
        
        self._embedder = OpenAIEmbedder(api_key=api_key)
        logger.info("OpenAI embedder initialized")
    
    def _initialize_vector_store(self) -> None:
        """Initialize Milvus Cloud store."""
        from embeddings import MilvusCloudStore, MilvusCloudConfig
        
        config = MilvusCloudConfig(
            collection_name=self.config.collection_name
        )
        self._vector_store = MilvusCloudStore(config)
        logger.info("Milvus Cloud store initialized")
    
    def _compute_hash(self, content: str) -> str:
        """Compute content hash."""
        return hashlib.md5(content.encode()).hexdigest()
    
    def _is_duplicate(self, content: str) -> bool:
        """Check for duplicate content."""
        if not self.config.enable_dedup:
            return False
        
        content_hash = self._compute_hash(content)
        if content_hash in self._processed_hashes:
            return True
        
        self._processed_hashes.add(content_hash)
        return False
    
    def ingest_file(
        self,
        file_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Ingest a single file.
        
        Args:
            file_path: Path to the file
            metadata: Additional metadata
            
        Returns:
            Ingestion result for this file
        """
        path = Path(file_path)
        result = {
            "file": str(path),
            "file_name": path.name,
            "success": False,
            "chunks": 0,
            "error": None
        }
        
        try:
            # Check file size
            file_size_mb = path.stat().st_size / (1024 * 1024)
            if file_size_mb > self.config.max_file_size_mb:
                result["error"] = f"File too large: {file_size_mb:.1f}MB"
                return result
            
            # Load document
            loaded_doc = self._file_loader.load(str(path))
            content = loaded_doc.content
            
            # Apply OCR if needed
            if self._ocr_processor:
                ocr_result = self._ocr_processor.process_document(
                    str(path),
                    existing_content=content
                )
                content = ocr_result.get("combined_content", content)
                
                if metadata is None:
                    metadata = {}
                metadata["entities"] = ocr_result.get("entities", [])
                metadata["keywords"] = ocr_result.get("keywords", [])
                metadata["ocr_applied"] = ocr_result.get("ocr_applied", False)
            
            # Check for duplicates
            if self._is_duplicate(content):
                result["error"] = "Duplicate content"
                result["success"] = True
                return result
            
            # Build base metadata
            base_metadata = {
                "source": str(path),
                "file_name": path.name,
                "file_type": path.suffix,
                "ingested_at": datetime.now().isoformat(),
                **(metadata or {})
            }
            
            # Extract tables and images for cross-referencing
            tables = loaded_doc.metadata.get("tables", [])
            images = loaded_doc.metadata.get("images", [])
            
            # Chunk with cross-references
            chunks = self._chunker.chunk_with_tables_and_images(
                content,
                tables=tables,
                images=images,
                metadata=base_metadata
            )
            
            if not chunks:
                result["error"] = "No chunks generated"
                return result
            
            # Generate embeddings
            chunk_texts = [chunk.content for chunk in chunks]
            embeddings = self._embedder.embed_documents(chunk_texts)
            
            # Prepare for Milvus
            chunk_ids = [f"{path.stem}_{chunk.id}" for chunk in chunks]
            chunk_metadatas = []
            content_types = []
            
            for chunk in chunks:
                chunk_meta = {
                    **base_metadata,
                    "chunk_index": chunk.index,
                    "parent_id": chunk.parent_id or "",
                    "linked_elements": chunk.linked_elements,
                    "content_type": chunk.content_type.value,
                    **chunk.metadata
                }
                chunk_metadatas.append(chunk_meta)
                content_types.append(chunk.content_type.value)
            
            # Store in Milvus Cloud
            self._vector_store.add(
                ids=chunk_ids,
                embeddings=embeddings,
                documents=chunk_texts,
                metadatas=chunk_metadatas,
                content_types=content_types
            )
            
            result["success"] = True
            result["chunks"] = len(chunks)
            result["content_length"] = len(content)
            result["tables_extracted"] = len(tables)
            result["images_extracted"] = len(images)
            
            logger.info(f"Ingested {path.name}: {len(chunks)} chunks")
            
        except Exception as e:
            logger.error(f"Failed to ingest {path}: {e}")
            result["error"] = str(e)
        
        return result
    
    def ingest_directory(
        self,
        directory: Optional[str] = None,
        recursive: bool = True
    ) -> IngestionResult:
        """
        Ingest all files in a directory.
        
        Args:
            directory: Directory path
            recursive: Process subdirectories
            
        Returns:
            IngestionResult with statistics
        """
        directory = directory or self.config.input_directory
        dir_path = Path(directory)
        
        if not dir_path.exists():
            raise IngestionError(f"Directory does not exist: {directory}")
        
        start_time = datetime.now()
        
        # Find files
        files = []
        for ext in self.config.supported_extensions:
            if recursive:
                files.extend(dir_path.rglob(f"*{ext}"))
            else:
                files.extend(dir_path.glob(f"*{ext}"))
        
        total_files = len(files)
        processed_files = 0
        failed_files = 0
        total_chunks = 0
        errors = []
        file_results = []
        
        logger.info(f"Found {total_files} files to process")
        
        # Process in batches
        for i in range(0, len(files), self.config.batch_size):
            batch = files[i:i + self.config.batch_size]
            
            for file_path in batch:
                result = self.ingest_file(str(file_path))
                file_results.append(result)
                
                if result["success"]:
                    processed_files += 1
                    total_chunks += result.get("chunks", 0)
                else:
                    failed_files += 1
                    if result.get("error"):
                        errors.append({
                            "file": str(file_path),
                            "error": result["error"]
                        })
            
            logger.info(f"Batch {i//self.config.batch_size + 1}: {processed_files}/{total_files}")
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return IngestionResult(
            total_files=total_files,
            processed_files=processed_files,
            failed_files=failed_files,
            total_chunks=total_chunks,
            total_embeddings=total_chunks,
            processing_time=processing_time,
            errors=errors,
            file_results=file_results
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return {
            "vector_store_count": self._vector_store.count() if self._vector_store else 0,
            "processed_hashes": len(self._processed_hashes),
            "config": {
                "chunk_strategy": self.config.chunk_strategy,
                "chunk_size": self.config.chunk_size,
                "collection_name": self.config.collection_name,
                "embedding_model": self.config.embedding_model
            }
        }
    
    def get_ingestion_status(self) -> Dict[str, Any]:
        """Get ingestion status for API."""
        return {
            "total_documents": self._vector_store.count() if self._vector_store else 0,
            "collection": self.config.collection_name,
            "ready": True
        }
    
    def reset(self) -> None:
        """Reset the pipeline."""
        if self._vector_store:
            self._vector_store.reset()
        self._processed_hashes.clear()
        logger.info("Pipeline reset complete")


def create_ingestion_pipeline(
    input_directory: str = "./data/raw",
    **kwargs
) -> IngestionPipeline:
    """Factory function."""
    config = IngestionConfig(input_directory=input_directory, **kwargs)
    return IngestionPipeline(config)
