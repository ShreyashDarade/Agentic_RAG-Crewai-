"""
Data Pipeline Module - Production Grade

Provides:
- Advanced chunking with cross-references
- Enhanced OCR with EasyOCR + spaCy
- Document loading and processing
- Ingestion pipeline
"""

from .chunker import (
    AdvancedChunker,
    AdvancedChunkerConfig,
    AdvancedChunkerError,
    EnhancedChunk,
    ContentElement,
    ContentType,
    create_advanced_chunker,
)
from .ocr_processor import (
    EnhancedOCRProcessor,
    EnhancedOCRConfig,
    EnhancedOCRError,
    create_enhanced_ocr_processor,
)
from .file_loader import (
    FileLoader,
    FileLoaderConfig,
    LoadedDocument,
    FileLoaderError,
)
from .ingestion_pipeline import (
    IngestionPipeline,
    IngestionConfig,
    IngestionResult,
    IngestionError,
)

__all__ = [
    # Chunker
    "AdvancedChunker",
    "AdvancedChunkerConfig",
    "AdvancedChunkerError",
    "EnhancedChunk",
    "ContentElement",
    "ContentType",
    "create_advanced_chunker",
    # OCR
    "EnhancedOCRProcessor",
    "EnhancedOCRConfig",
    "EnhancedOCRError",
    "create_enhanced_ocr_processor",
    # File Loader
    "FileLoader",
    "FileLoaderConfig",
    "LoadedDocument",
    "FileLoaderError",
    # Ingestion
    "IngestionPipeline",
    "IngestionConfig",
    "IngestionResult",
    "IngestionError",
]
