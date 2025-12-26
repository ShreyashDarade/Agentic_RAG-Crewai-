"""
Advanced Chunker - Production Grade with Cross-References

Features:
- Multiple chunking strategies (recursive, semantic, agentic)
- Cross-reference linkage between text, tables, and images
- Parent-child hierarchical chunking
- Context preservation
- Metadata enrichment
"""

import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class ContentType(Enum):
    """Types of content that can be chunked."""
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"
    CODE = "code"
    HEADER = "header"
    LIST = "list"


@dataclass
class AdvancedChunkerConfig:
    """Configuration for advanced chunking."""
    strategy: str = "semantic"  # recursive, semantic, agentic
    chunk_size: int = 512
    chunk_overlap: int = 100
    min_chunk_size: int = 100
    max_chunk_size: int = 1500
    
    # Semantic chunking
    semantic_threshold: float = 0.5
    
    # Cross-reference settings
    enable_cross_reference: bool = True
    link_tables_to_text: bool = True
    link_images_to_text: bool = True
    context_window: int = 2  # paragraphs before/after
    
    # Hierarchical chunking
    enable_hierarchy: bool = True
    parent_chunk_size: int = 2000
    parent_overlap: int = 200
    
    # Structure preservation
    preserve_structure: bool = True
    add_context: bool = True
    
    separators: List[str] = field(default_factory=lambda: [
        "\n\n",  # Paragraphs
        "\n",    # Lines
        ". ",    # Sentences
        "! ",    # Exclamations
        "? ",    # Questions
        "; ",    # Semicolons
        ", ",    # Commas
        " ",     # Words
    ])


@dataclass
class ContentElement:
    """Represents an extracted content element (text, table, image)."""
    id: str
    content_type: ContentType
    content: str
    position: int  # Position in document
    page: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    bounding_box: Optional[Tuple[float, float, float, float]] = None  # x1, y1, x2, y2


@dataclass
class EnhancedChunk:
    """Represents an enhanced text chunk with cross-references."""
    id: str
    content: str
    content_type: ContentType
    index: int
    start_char: int
    end_char: int
    
    # Hierarchical references
    parent_id: Optional[str] = None
    child_ids: List[str] = field(default_factory=list)
    
    # Cross-references
    linked_elements: List[str] = field(default_factory=list)  # IDs of related tables/images
    context_before: str = ""
    context_after: str = ""
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def length(self) -> int:
        """Get chunk length."""
        return len(self.content)
    
    @property
    def word_count(self) -> int:
        """Get word count."""
        return len(self.content.split())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "content": self.content,
            "content_type": self.content_type.value,
            "index": self.index,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "parent_id": self.parent_id,
            "child_ids": self.child_ids,
            "linked_elements": self.linked_elements,
            "context_before": self.context_before,
            "context_after": self.context_after,
            "metadata": self.metadata,
        }


class AdvancedChunkerError(Exception):
    """Exception raised for chunking errors."""
    pass


class AdvancedChunker:
    """
    Advanced text chunker with cross-reference support.
    
    Features:
    - Multiple chunking strategies
    - Hierarchical parent-child relationships
    - Cross-references between text, tables, and images
    - Context preservation
    """
    
    def __init__(self, config: Optional[AdvancedChunkerConfig] = None):
        """
        Initialize the advanced chunker.
        
        Args:
            config: Chunker configuration
        """
        self.config = config or AdvancedChunkerConfig()
        self._validate_config()
        self._embedder = None
    
    def _validate_config(self) -> None:
        """Validate configuration."""
        if self.config.chunk_size < self.config.min_chunk_size:
            raise AdvancedChunkerError(
                f"chunk_size ({self.config.chunk_size}) must be >= "
                f"min_chunk_size ({self.config.min_chunk_size})"
            )
        
        if self.config.chunk_overlap >= self.config.chunk_size:
            raise AdvancedChunkerError(
                f"chunk_overlap ({self.config.chunk_overlap}) must be < "
                f"chunk_size ({self.config.chunk_size})"
            )
    
    def _generate_id(self) -> str:
        """Generate unique chunk ID."""
        return str(uuid.uuid4())[:8]
    
    def chunk(
        self,
        text: str,
        content_elements: Optional[List[ContentElement]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[EnhancedChunk]:
        """
        Chunk text with cross-references to content elements.
        
        Args:
            text: Text to chunk
            content_elements: Tables, images, and other elements
            metadata: Optional metadata to include in chunks
            
        Returns:
            List of EnhancedChunk objects
        """
        if not text or not text.strip():
            return []
        
        content_elements = content_elements or []
        
        # Choose chunking strategy
        strategy = self.config.strategy.lower()
        
        if strategy == "recursive":
            chunks = self._recursive_chunk(text)
        elif strategy == "semantic":
            chunks = self._semantic_chunk(text)
        elif strategy == "agentic":
            chunks = self._agentic_chunk(text)
        else:
            raise AdvancedChunkerError(f"Unknown chunking strategy: {strategy}")
        
        # Add metadata
        if metadata:
            for chunk in chunks:
                chunk.metadata.update(metadata)
        
        # Add context
        if self.config.add_context:
            chunks = self._add_context_to_chunks(chunks, text)
        
        # Create cross-references
        if self.config.enable_cross_reference and content_elements:
            chunks = self._create_cross_references(chunks, content_elements, text)
        
        # Create hierarchical structure
        if self.config.enable_hierarchy:
            chunks = self._create_hierarchy(chunks, text)
        
        logger.info(f"Created {len(chunks)} chunks using {strategy} strategy")
        
        return chunks
    
    def _recursive_chunk(self, text: str) -> List[EnhancedChunk]:
        """
        Recursively split text on multiple separators.
        """
        return self._split_recursive(
            text,
            self.config.separators,
            self.config.chunk_size,
            self.config.chunk_overlap
        )
    
    def _split_recursive(
        self,
        text: str,
        separators: List[str],
        chunk_size: int,
        overlap: int
    ) -> List[EnhancedChunk]:
        """Recursive splitting implementation."""
        chunks = []
        
        # If text fits in one chunk, return it
        if len(text) <= chunk_size:
            if len(text) >= self.config.min_chunk_size:
                return [EnhancedChunk(
                    id=self._generate_id(),
                    content=text.strip(),
                    content_type=ContentType.TEXT,
                    index=0,
                    start_char=0,
                    end_char=len(text)
                )]
            return []
        
        # Find the best separator to use
        separator = self._find_best_separator(text, separators, chunk_size)
        
        if separator is None:
            # No good separator, use fixed chunking
            return self._fixed_chunk(text)
        
        # Split on the separator
        parts = text.split(separator)
        
        current_chunk = ""
        chunk_start = 0
        current_pos = 0
        
        for part in parts:
            potential_chunk = current_chunk + separator + part if current_chunk else part
            
            if len(potential_chunk) <= chunk_size:
                current_chunk = potential_chunk
            else:
                # Save current chunk if large enough
                if current_chunk and len(current_chunk) >= self.config.min_chunk_size:
                    chunks.append(EnhancedChunk(
                        id=self._generate_id(),
                        content=current_chunk.strip(),
                        content_type=ContentType.TEXT,
                        index=len(chunks),
                        start_char=chunk_start,
                        end_char=chunk_start + len(current_chunk)
                    ))
                    
                    # Handle overlap
                    if overlap > 0 and len(current_chunk) > overlap:
                        overlap_text = current_chunk[-overlap:]
                        chunk_start += len(current_chunk) - overlap
                        current_chunk = overlap_text + separator + part
                    else:
                        chunk_start += len(current_chunk) + len(separator)
                        current_chunk = part
                else:
                    current_chunk = part
                    chunk_start = current_pos
                
                # If single part is too large, recursively split
                if len(current_chunk) > chunk_size:
                    remaining_separators = separators[separators.index(separator)+1:] if separator in separators else separators[1:]
                    
                    if remaining_separators:
                        sub_chunks = self._split_recursive(
                            current_chunk,
                            remaining_separators,
                            chunk_size,
                            overlap
                        )
                        for sub_chunk in sub_chunks:
                            sub_chunk.index = len(chunks)
                            sub_chunk.start_char += chunk_start
                            sub_chunk.end_char += chunk_start
                            chunks.append(sub_chunk)
                        current_chunk = ""
                    else:
                        # Force split
                        for i in range(0, len(current_chunk), chunk_size - overlap):
                            sub_text = current_chunk[i:i + chunk_size]
                            if len(sub_text) >= self.config.min_chunk_size:
                                chunks.append(EnhancedChunk(
                                    id=self._generate_id(),
                                    content=sub_text.strip(),
                                    content_type=ContentType.TEXT,
                                    index=len(chunks),
                                    start_char=chunk_start + i,
                                    end_char=chunk_start + i + len(sub_text)
                                ))
                        current_chunk = ""
        
        # Don't forget the last chunk
        if current_chunk and len(current_chunk) >= self.config.min_chunk_size:
            chunks.append(EnhancedChunk(
                id=self._generate_id(),
                content=current_chunk.strip(),
                content_type=ContentType.TEXT,
                index=len(chunks),
                start_char=chunk_start,
                end_char=chunk_start + len(current_chunk)
            ))
        
        # Re-index chunks
        for i, chunk in enumerate(chunks):
            chunk.index = i
        
        return chunks
    
    def _find_best_separator(
        self,
        text: str,
        separators: List[str],
        chunk_size: int
    ) -> Optional[str]:
        """Find the best separator that creates reasonable chunk sizes."""
        for separator in separators:
            if separator in text:
                parts = text.split(separator)
                avg_size = len(text) / len(parts) if parts else len(text)
                
                if avg_size < chunk_size * 2:
                    return separator
        
        return None
    
    def _semantic_chunk(self, text: str) -> List[EnhancedChunk]:
        """
        Chunk text while preserving semantic units.
        
        Uses sentence boundaries and paragraph structure.
        """
        chunks = []
        
        # Split into paragraphs first
        paragraphs = re.split(r'\n\n+', text)
        
        current_chunk = ""
        chunk_start = 0
        current_pos = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                current_pos += 2
                continue
            
            # Detect content type
            content_type = self._detect_content_type(para)
            
            # Split paragraph into sentences
            sentences = re.split(r'(?<=[.!?])\s+', para)
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                
                potential = current_chunk + " " + sentence if current_chunk else sentence
                
                if len(potential) <= self.config.chunk_size:
                    current_chunk = potential
                else:
                    # Save current chunk
                    if current_chunk and len(current_chunk) >= self.config.min_chunk_size:
                        chunks.append(EnhancedChunk(
                            id=self._generate_id(),
                            content=current_chunk.strip(),
                            content_type=content_type,
                            index=len(chunks),
                            start_char=chunk_start,
                            end_char=chunk_start + len(current_chunk),
                            metadata={"type": "semantic"}
                        ))
                    
                    chunk_start = current_pos
                    current_chunk = sentence
                
                current_pos += len(sentence) + 1
            
            current_pos += 2  # Paragraph break
        
        # Last chunk
        if current_chunk and len(current_chunk) >= self.config.min_chunk_size:
            chunks.append(EnhancedChunk(
                id=self._generate_id(),
                content=current_chunk.strip(),
                content_type=ContentType.TEXT,
                index=len(chunks),
                start_char=chunk_start,
                end_char=chunk_start + len(current_chunk),
                metadata={"type": "semantic"}
            ))
        
        return chunks
    
    def _agentic_chunk(self, text: str) -> List[EnhancedChunk]:
        """
        Agentic chunking using propositions.
        
        Attempts to create semantically independent chunks.
        """
        # First, do semantic chunking
        chunks = self._semantic_chunk(text)
        
        # Then merge very small adjacent chunks if they're semantically related
        merged_chunks = []
        current_merged = None
        
        for chunk in chunks:
            if current_merged is None:
                current_merged = chunk
            elif len(current_merged.content) + len(chunk.content) + 1 <= self.config.chunk_size:
                # Merge if combined size is acceptable
                current_merged.content += " " + chunk.content
                current_merged.end_char = chunk.end_char
            else:
                merged_chunks.append(current_merged)
                current_merged = chunk
        
        if current_merged:
            merged_chunks.append(current_merged)
        
        # Re-index
        for i, chunk in enumerate(merged_chunks):
            chunk.index = i
        
        return merged_chunks
    
    def _fixed_chunk(self, text: str) -> List[EnhancedChunk]:
        """Simple fixed-size chunking."""
        chunks = []
        chunk_size = self.config.chunk_size
        overlap = self.config.chunk_overlap
        
        step = chunk_size - overlap
        
        for i in range(0, len(text), step):
            chunk_text = text[i:i + chunk_size]
            
            if len(chunk_text) >= self.config.min_chunk_size:
                chunks.append(EnhancedChunk(
                    id=self._generate_id(),
                    content=chunk_text.strip(),
                    content_type=ContentType.TEXT,
                    index=len(chunks),
                    start_char=i,
                    end_char=i + len(chunk_text),
                    metadata={"type": "fixed"}
                ))
        
        return chunks
    
    def _detect_content_type(self, text: str) -> ContentType:
        """Detect content type from text pattern."""
        text = text.strip()
        
        # Check for headers
        if re.match(r'^#{1,6}\s+.+$', text) or re.match(r'^[A-Z][A-Za-z\s]+:?\s*$', text):
            return ContentType.HEADER
        
        # Check for lists
        if re.match(r'^[\-\*\•]\s+', text) or re.match(r'^\d+\.\s+', text):
            return ContentType.LIST
        
        # Check for code blocks
        if text.startswith('```') or re.match(r'^    .+', text):
            return ContentType.CODE
        
        # Check for tables (pipe-separated)
        if '|' in text and text.count('|') >= 2:
            return ContentType.TABLE
        
        return ContentType.TEXT
    
    def _add_context_to_chunks(
        self,
        chunks: List[EnhancedChunk],
        original_text: str
    ) -> List[EnhancedChunk]:
        """Add contextual information to chunks."""
        total_chunks = len(chunks)
        
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["total_chunks"] = total_chunks
            chunk.metadata["position"] = "start" if i == 0 else ("end" if i == total_chunks - 1 else "middle")
            chunk.metadata["char_position"] = f"{chunk.start_char}-{chunk.end_char}"
            
            # Add neighboring context
            if i > 0:
                chunk.context_before = chunks[i-1].content[-100:]
            if i < total_chunks - 1:
                chunk.context_after = chunks[i+1].content[:100]
        
        return chunks
    
    def _create_cross_references(
        self,
        chunks: List[EnhancedChunk],
        content_elements: List[ContentElement],
        original_text: str
    ) -> List[EnhancedChunk]:
        """
        Create cross-references between text chunks and content elements.
        
        Links tables and images to their surrounding text context.
        """
        if not content_elements:
            return chunks
        
        for element in content_elements:
            # Find chunks that overlap with or are near this element
            for chunk in chunks:
                # Check position overlap
                if self._positions_overlap(
                    (chunk.start_char, chunk.end_char),
                    (element.position, element.position + len(element.content))
                ):
                    chunk.linked_elements.append(element.id)
                    chunk.metadata[f"has_{element.content_type.value}"] = True
                    
                    # Add reference in chunk content
                    if element.content_type == ContentType.TABLE:
                        chunk.metadata["table_reference"] = element.id
                    elif element.content_type == ContentType.IMAGE:
                        chunk.metadata["image_reference"] = element.id
        
        return chunks
    
    def _positions_overlap(
        self,
        range1: Tuple[int, int],
        range2: Tuple[int, int],
        buffer: int = 500  # Character buffer for "nearness"
    ) -> bool:
        """Check if two position ranges overlap or are near each other."""
        start1, end1 = range1
        start2, end2 = range2
        
        # Expand ranges by buffer
        start1 -= buffer
        end1 += buffer
        
        return not (end1 < start2 or end2 < start1)
    
    def _create_hierarchy(
        self,
        chunks: List[EnhancedChunk],
        original_text: str
    ) -> List[EnhancedChunk]:
        """
        Create parent-child hierarchical structure.
        
        Larger parent chunks contain multiple smaller child chunks.
        """
        if not self.config.enable_hierarchy or len(chunks) < 2:
            return chunks
        
        # Create parent chunks
        parent_chunks = self._fixed_chunk_with_config(
            original_text,
            self.config.parent_chunk_size,
            self.config.parent_overlap
        )
        
        # Assign children to parents
        for parent in parent_chunks:
            parent.metadata["is_parent"] = True
            
            for child in chunks:
                # Check if child is within parent's range
                if (child.start_char >= parent.start_char and 
                    child.end_char <= parent.end_char):
                    child.parent_id = parent.id
                    parent.child_ids.append(child.id)
        
        # Return both parent and child chunks
        all_chunks = parent_chunks + chunks
        
        # Re-index
        for i, chunk in enumerate(all_chunks):
            chunk.index = i
        
        return all_chunks
    
    def _fixed_chunk_with_config(
        self,
        text: str,
        chunk_size: int,
        overlap: int
    ) -> List[EnhancedChunk]:
        """Create fixed chunks with specific size and overlap."""
        chunks = []
        step = chunk_size - overlap
        
        for i in range(0, len(text), step):
            chunk_text = text[i:i + chunk_size]
            
            if len(chunk_text) >= self.config.min_chunk_size:
                chunks.append(EnhancedChunk(
                    id=self._generate_id(),
                    content=chunk_text.strip(),
                    content_type=ContentType.TEXT,
                    index=len(chunks),
                    start_char=i,
                    end_char=i + len(chunk_text),
                    metadata={"type": "parent", "is_parent": True}
                ))
        
        return chunks
    
    def chunk_with_tables_and_images(
        self,
        text: str,
        tables: Optional[List[Dict[str, Any]]] = None,
        images: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[EnhancedChunk]:
        """
        Chunk text with explicit table and image references.
        
        Args:
            text: Document text
            tables: List of table dictionaries with content and position
            images: List of image dictionaries with description and position
            metadata: Optional metadata
            
        Returns:
            List of enhanced chunks with cross-references
        """
        content_elements = []
        
        # Convert tables to ContentElements
        if tables:
            for i, table in enumerate(tables):
                content_elements.append(ContentElement(
                    id=f"table_{i}",
                    content_type=ContentType.TABLE,
                    content=table.get("content", ""),
                    position=table.get("position", 0),
                    page=table.get("page"),
                    metadata=table.get("metadata", {})
                ))
        
        # Convert images to ContentElements
        if images:
            for i, image in enumerate(images):
                content_elements.append(ContentElement(
                    id=f"image_{i}",
                    content_type=ContentType.IMAGE,
                    content=image.get("description", ""),
                    position=image.get("position", 0),
                    page=image.get("page"),
                    metadata=image.get("metadata", {}),
                    bounding_box=image.get("bounding_box")
                ))
        
        return self.chunk(text, content_elements, metadata)


def create_advanced_chunker(
    strategy: str = "semantic",
    chunk_size: int = 512,
    enable_cross_reference: bool = True,
    **kwargs
) -> AdvancedChunker:
    """
    Factory function to create an advanced chunker.
    
    Args:
        strategy: Chunking strategy (recursive, semantic, agentic)
        chunk_size: Target chunk size
        enable_cross_reference: Enable cross-reference linking
        **kwargs: Additional configuration
        
    Returns:
        Configured AdvancedChunker instance
    """
    config = AdvancedChunkerConfig(
        strategy=strategy,
        chunk_size=chunk_size,
        enable_cross_reference=enable_cross_reference,
        **kwargs
    )
    return AdvancedChunker(config)
