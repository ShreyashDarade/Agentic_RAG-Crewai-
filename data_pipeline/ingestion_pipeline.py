import hashlib
import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class IngestionConfig:
    """Configuration for the ingestion pipeline."""
    data_directory: str = "./data"
    raw_documents_dir: str = "./data/raw"
    processed_dir: str = "./data/processed"
    ingestion_state_file: str = "./data/ingestion_state.json"
    
    # Processing options
    enable_ocr: bool = True
    enable_filtering: bool = True
    enable_chunking: bool = True
    
    # Chunking settings
    chunk_size: int = 512
    chunk_overlap: int = 50
    chunking_strategy: str = "recursive"
    
    # Batch processing
    batch_size: int = 10
    
    # Collection settings
    collection_name: str = "documents"
    
    def __post_init__(self):
        """Create directories if they don't exist."""
        for dir_path in [self.data_directory, self.raw_documents_dir, self.processed_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)


@dataclass
class DocumentState:
    """Tracks the state of a processed document."""
    file_path: str
    file_hash: str
    file_name: str
    ingested_at: str
    chunk_count: int
    status: str  # pending, processing, completed, failed
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentState":
        return cls(**data)


class IngestionState:
    """
    Manages ingestion state persistence.
    Tracks which files have been processed to avoid reprocessing.
    """
    
    def __init__(self, state_file: str):
        """
        Initialize ingestion state.
        
        Args:
            state_file: Path to the state JSON file
        """
        self.state_file = Path(state_file)
        self._documents: Dict[str, DocumentState] = {}
        self._load_state()
    
    def _load_state(self) -> None:
        """Load state from file."""
        if self.state_file.exists():
            try:
                with open(self.state_file, "r") as f:
                    data = json.load(f)
                
                for file_hash, doc_data in data.get("documents", {}).items():
                    self._documents[file_hash] = DocumentState.from_dict(doc_data)
                
                logger.info(f"Loaded ingestion state: {len(self._documents)} documents tracked")
                
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to load state file: {e}. Starting fresh.")
                self._documents = {}
            except Exception as e:
                logger.error(f"Error loading state: {e}")
                self._documents = {}
    
    def _save_state(self) -> None:
        """Save state to file."""
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                "last_updated": datetime.now().isoformat(),
                "document_count": len(self._documents),
                "documents": {
                    hash_: doc.to_dict() 
                    for hash_, doc in self._documents.items()
                }
            }
            
            with open(self.state_file, "w") as f:
                json.dump(data, f, indent=2, default=str)
            
            logger.debug(f"Saved ingestion state to {self.state_file}")
            
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
    
    def is_processed(self, file_hash: str) -> bool:
        """Check if a file has been processed."""
        doc = self._documents.get(file_hash)
        return doc is not None and doc.status == "completed"
    
    def get_document_state(self, file_hash: str) -> Optional[DocumentState]:
        """Get the state of a specific document."""
        return self._documents.get(file_hash)
    
    def add_document(self, doc_state: DocumentState) -> None:
        """Add or update a document's state."""
        self._documents[doc_state.file_hash] = doc_state
        self._save_state()
    
    def update_status(
        self,
        file_hash: str,
        status: str,
        error_message: Optional[str] = None,
        chunk_count: int = 0
    ) -> None:
        """Update a documents status."""
        if file_hash in self._documents:
            self._documents[file_hash].status = status
            self._documents[file_hash].error_message = error_message
            if chunk_count > 0:
                self._documents[file_hash].chunk_count = chunk_count
            self._save_state()
    
    def get_all_documents(self) -> Dict[str, DocumentState]:
        """Get all tracked documents."""
        return self._documents.copy()
    
    def get_pending_documents(self) -> List[DocumentState]:
        """Get documents that need processing."""
        return [
            doc for doc in self._documents.values()
            if doc.status in ["pending", "failed"]
        ]
    
    def remove_document(self, file_hash: str) -> None:
        """Remove a document from tracking."""
        if file_hash in self._documents:
            del self._documents[file_hash]
            self._save_state()
    
    def clear(self) -> None:
        """Clear all state."""
        self._documents = {}
        self._save_state()


class IngestionPipelineError(Exception):
    """Exception raised for ingestion pipeline errors."""
    pass


class IngestionPipeline:
    """
    Main ingestion pipeline for document processing.
    
    Pipeline stages:
    1. File discovery and validation
    2. Document loading and parsing
    3. OCR processing (if needed)
    4. Metadata filtering
    5. Text chunking
    6. Embedding generation
    7. Vector store insertion
    """
    
    def __init__(self, config: Optional[IngestionConfig] = None):
        """
        Initialize the ingestion pipeline.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config or IngestionConfig()
        
        # Initialize state tracker
        self.state = IngestionState(self.config.ingestion_state_file)
        
        # Initialize components (lazy loading)
        self._file_loader = None
        self._ocr_processor = None
        self._metadata_filter = None
        self._chunker = None
        self._embedder = None
        self._vector_store = None
        
        # Statistics
        self._stats = {
            "files_discovered": 0,
            "files_skipped": 0,
            "files_processed": 0,
            "files_failed": 0,
            "chunks_created": 0,
            "embeddings_generated": 0,
        }
        
        logger.info(f"Ingestion pipeline initialized. Data dir: {self.config.data_directory}")
    
    @property
    def file_loader(self):
        """Lazy load file loader."""
        if self._file_loader is None:
            from .file_loader import FileLoader
            self._file_loader = FileLoader()
        return self._file_loader
    
    @property
    def ocr_processor(self):
        """Lazy load OCR processor."""
        if self._ocr_processor is None:
            from .ocr_processor import OCRProcessor
            self._ocr_processor = OCRProcessor()
        return self._ocr_processor
    
    @property
    def metadata_filter(self):
        """Lazy load metadata filter."""
        if self._metadata_filter is None:
            from .metadata_filter import MetadataFilter, FilterConfig
            filter_config = FilterConfig(max_duplicate_ratio=self.config.__dict__.get("max_duplicate_ratio", 0.85))
            self._metadata_filter = MetadataFilter(filter_config)
        return self._metadata_filter
    
    @property
    def chunker(self):
        """Lazy load chunker."""
        if self._chunker is None:
            from .chunker import Chunker, ChunkerConfig
            config = ChunkerConfig(
                strategy=self.config.chunking_strategy,
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap
            )
            self._chunker = Chunker(config)
        return self._chunker
    
    @property
    def embedder(self):
        """Lazy load embedder."""
        if self._embedder is None:
            from embeddings import Embedder
            self._embedder = Embedder()
        return self._embedder
    
    @property
    def vector_store(self):
        """Lazy load vector store."""
        if self._vector_store is None:
            from embeddings import VectorStore, VectorStoreConfig
            config = VectorStoreConfig(
                collection_name=self.config.collection_name,
                persist_directory=os.path.join(self.config.data_directory, "chromadb")
            )
            self._vector_store = VectorStore(config)
        return self._vector_store
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate MD5 hash of a file."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def discover_files(
        self,
        directory: Optional[str] = None,
        recursive: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Discover files to process.
        
        Args:
            directory: Directory to scan (uses config if None)
            recursive: Whether to scan subdirectories
            
        Returns:
            List of file info dictionaries
        """
        directory = directory or self.config.raw_documents_dir
        dir_path = Path(directory)
        
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {directory}")
            return []
        
        files = []
        supported_extensions = set(self.file_loader.config.supported_extensions)
        
        pattern = "**/*" if recursive else "*"
        
        for file_path in dir_path.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                file_hash = self._calculate_file_hash(str(file_path))
                
                files.append({
                    "path": str(file_path),
                    "name": file_path.name,
                    "hash": file_hash,
                    "extension": file_path.suffix.lower(),
                    "size": file_path.stat().st_size,
                    "already_processed": self.state.is_processed(file_hash)
                })
        
        self._stats["files_discovered"] = len(files)
        logger.info(f"Discovered {len(files)} files in {directory}")
        
        return files
    
    def process_file(
        self,
        file_path: str,
        force: bool = False
    ) -> Dict[str, Any]:
        """
        Process a single file through the pipeline.
        
        Args:
            file_path: Path to the file
            force: Force reprocessing even if already done
            
        Returns:
            Processing result dictionary
        """
        path_obj = Path(file_path).resolve()
        result_path = str(path_obj)
        file_name = path_obj.name
        result = {
            "file_path": result_path,
            "file_name": file_name,
            "file_hash": "",
            "success": False,
            "chunks_created": 0,
            "error": None,
            "skipped": False
        }

        if not path_obj.exists():
            result["error"] = f"File not found: {result_path}"
            return result

        if not path_obj.is_file():
            result["error"] = f"Path is not a file: {result_path}"
            return result

        file_hash = self._calculate_file_hash(str(path_obj))
        result["file_hash"] = file_hash
        
        result["file_path"] = result_path
        
        # Check if already processed
        if not force and self.state.is_processed(file_hash):
            result["skipped"] = True
            result["success"] = True
            result["message"] = "Already processed"
            self._stats["files_skipped"] += 1
            logger.info(f"Skipping already processed file: {file_name}")
            return result
        
        # Track document state
        doc_state = DocumentState(
            file_path=result_path,
            file_hash=file_hash,
            file_name=file_name,
            ingested_at=datetime.now().isoformat(),
            chunk_count=0,
            status="processing"
        )
        self.state.add_document(doc_state)
        
        try:
            # Stage 1: Load document
            logger.info(f"Loading document: {file_name}")
            loaded_doc = self.file_loader.load(result_path)
            content = loaded_doc.content
            metadata = loaded_doc.metadata
            
            # Stage 2: OCR processing (if enabled and needed)
            if self.config.enable_ocr:
                ocr_result = self.ocr_processor.process_document(result_path, content)
                if ocr_result.get("ocr_applied"):
                    content = ocr_result.get("combined_content", content)
                    metadata["ocr_applied"] = True
            
            # Stage 3: Filter and clean
            if self.config.enable_filtering:
                filter_result = self.metadata_filter.filter_document(
                    content, metadata, file_hash
                )
                if not filter_result.passed:
                    result["error"] = f"Filtered: {filter_result.reason}"
                    self.state.update_status(file_hash, "failed", result["error"])
                    self._stats["files_failed"] += 1
                    return result
                
                content = filter_result.document["content"]
                metadata = filter_result.metadata_cleaned
            
            # Stage 4: Chunk document
            if self.config.enable_chunking:
                chunks = self.chunker.chunk(content, metadata)
            else:
                from .chunker import Chunk
                chunks = [Chunk(
                    content=content,
                    index=0,
                    start_char=0,
                    end_char=len(content),
                    metadata=metadata
                )]
            
            if not chunks:
                result["error"] = "No chunks created"
                self.state.update_status(file_hash, "failed", result["error"])
                self._stats["files_failed"] += 1
                return result
            
            # Stage 5: Generate embeddings
            chunk_texts = [chunk.content for chunk in chunks]
            embeddings = self.embedder.embed_documents(chunk_texts)
            
            # Stage 6: Store in vector database
            chunk_ids = [f"{file_hash}_{i}" for i in range(len(chunks))]
            chunk_metadatas = []
            
            for chunk in chunks:
                chunk_meta = {
                    "source": result_path,
                    "file_name": file_name,
                    "file_hash": file_hash,
                    "chunk_index": chunk.index,
                    "total_chunks": len(chunks),
                    **{k: v for k, v in chunk.metadata.items() if isinstance(v, (str, int, float, bool))}
                }
                chunk_metadatas.append(chunk_meta)
            
            self.vector_store.add(
                ids=chunk_ids,
                embeddings=embeddings,
                documents=chunk_texts,
                metadatas=chunk_metadatas
            )
            
            # Update state
            result["success"] = True
            result["chunks_created"] = len(chunks)
            result["embeddings_generated"] = len(embeddings)
            
            self.state.update_status(file_hash, "completed", chunk_count=len(chunks))
            self._stats["files_processed"] += 1
            self._stats["chunks_created"] += len(chunks)
            self._stats["embeddings_generated"] += len(embeddings)
            
            logger.info(f"Successfully processed {file_name}: {len(chunks)} chunks")
            
        except Exception as e:
            result["error"] = str(e)
            self.state.update_status(file_hash, "failed", str(e))
            self._stats["files_failed"] += 1
            logger.error(f"Failed to process {file_name}: {e}")
        
        return result
    
    def ingest_directory(
        self,
        directory: Optional[str] = None,
        force: bool = False,
        batch_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Ingest all documents from a directory.
        
        Args:
            directory: Directory to ingest (uses config if None)
            force: Force reprocessing of all files
            batch_size: Number of files per batch
            
        Returns:
            Ingestion results summary
        """
        directory = directory or self.config.raw_documents_dir
        batch_size = batch_size or self.config.batch_size
        
        # Reset stats
        self._stats = {
            "files_discovered": 0,
            "files_skipped": 0,
            "files_processed": 0,
            "files_failed": 0,
            "chunks_created": 0,
            "embeddings_generated": 0,
        }
        
        # Discover files
        files = self.discover_files(directory)
        
        if not files:
            return {
                "success": True,
                "message": "No files to process",
                "stats": self._stats
            }
        
        # Filter to unprocessed files (unless force)
        if not force:
            files = [f for f in files if not f["already_processed"]]
        
        if not files:
            return {
                "success": True,
                "message": "All files already processed",
                "stats": self._stats
            }
        
        results = []
        errors = []
        
        # Process in batches
        for i in range(0, len(files), batch_size):
            batch = files[i:i + batch_size]
            logger.info(f"Processing batch {i // batch_size + 1}: {len(batch)} files")
            
            for file_info in batch:
                result = self.process_file(file_info["path"], force=force)
                results.append(result)
                
                if not result["success"] and not result.get("skipped"):
                    errors.append({
                        "file": file_info["path"],
                        "error": result.get("error")
                    })
        
        return {
            "success": len(errors) == 0,
            "message": f"Processed {self._stats['files_processed']} files",
            "stats": self._stats,
            "errors": errors if errors else None
        }
    
    def get_ingestion_status(self) -> Dict[str, Any]:
        """Get current ingestion status."""
        all_docs = self.state.get_all_documents()
        
        status_counts = {
            "completed": 0,
            "failed": 0,
            "processing": 0,
            "pending": 0
        }
        
        for doc in all_docs.values():
            status_counts[doc.status] = status_counts.get(doc.status, 0) + 1
        
        return {
            "total_documents": len(all_docs),
            "status_counts": status_counts,
            "vector_store_count": self.vector_store.count(),
            "last_stats": self._stats
        }
    
    def reset(self, clear_vector_store: bool = True) -> None:
        """
        Reset the pipeline state.
        
        Args:
            clear_vector_store: Also clear the vector store
        """
        self.state.clear()
        
        if clear_vector_store:
            self.vector_store.reset()
        
        self._stats = {
            "files_discovered": 0,
            "files_skipped": 0,
            "files_processed": 0,
            "files_failed": 0,
            "chunks_created": 0,
            "embeddings_generated": 0,
        }
        
        logger.info("Pipeline state reset")


def create_ingestion_pipeline(
    data_directory: str = "./data",
    **kwargs
) -> IngestionPipeline:
    """
    Factory function to create an ingestion pipeline.
    
    Args:
        data_directory: Base data directory
        **kwargs: Additional configuration options
        
    Returns:
        Configured IngestionPipeline instance
    """
    config = IngestionConfig(data_directory=data_directory, **kwargs)
    return IngestionPipeline(config)

