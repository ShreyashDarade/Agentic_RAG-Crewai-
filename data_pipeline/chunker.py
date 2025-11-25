import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ChunkerConfig:
    """Configuration for chunking."""
    strategy: str = "recursive"  # recursive, semantic, fixed
    chunk_size: int = 512
    chunk_overlap: int = 50
    min_chunk_size: int = 100
    max_chunk_size: int = 1500
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
    preserve_structure: bool = True
    add_context: bool = True


@dataclass
class Chunk:
    """Represents a text chunk."""
    content: str
    index: int
    start_char: int
    end_char: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def length(self) -> int:
        """Get chunk length."""
        return len(self.content)
    
    @property
    def word_count(self) -> int:
        """Get word count."""
        return len(self.content.split())


class ChunkerError(Exception):
    """Exception raised for chunking errors."""
    pass


class Chunker:
    """
    Text chunker with multiple strategies.
    
    Supports:
    - Recursive chunking (splits on multiple separators)
    - Semantic chunking (tries to preserve meaning units)
    - Fixed chunking (simple character-based splits)
    """
    
    def __init__(self, config: Optional[ChunkerConfig] = None):
        """
        Initialize the chunker.
        
        Args:
            config: Chunker configuration
        """
        self.config = config or ChunkerConfig()
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate configuration."""
        if self.config.chunk_size < self.config.min_chunk_size:
            raise ChunkerError(
                f"chunk_size ({self.config.chunk_size}) must be >= "
                f"min_chunk_size ({self.config.min_chunk_size})"
            )
        
        if self.config.chunk_overlap >= self.config.chunk_size:
            raise ChunkerError(
                f"chunk_overlap ({self.config.chunk_overlap}) must be < "
                f"chunk_size ({self.config.chunk_size})"
            )
    
    def chunk(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        """
        Chunk text using configured strategy.
        
        Args:
            text: Text to chunk
            metadata: Optional metadata to include in chunks
            
        Returns:
            List of Chunk objects
        """
        if not text or not text.strip():
            return []
        
        strategy = self.config.strategy.lower()
        
        if strategy == "recursive":
            chunks = self._recursive_chunk(text)
        elif strategy == "semantic":
            chunks = self._semantic_chunk(text)
        elif strategy == "fixed":
            chunks = self._fixed_chunk(text)
        else:
            raise ChunkerError(f"Unknown chunking strategy: {strategy}")
        
        # Add metadata to chunks
        if metadata:
            for chunk in chunks:
                chunk.metadata.update(metadata)
        
        # Add context if enabled
        if self.config.add_context:
            chunks = self._add_context_to_chunks(chunks, text)
        
        logger.info(f"Created {len(chunks)} chunks using {strategy} strategy")
        
        return chunks
    
    def _recursive_chunk(self, text: str) -> List[Chunk]:
        """
        Recursively split text on multiple separators.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of chunks
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
    ) -> List[Chunk]:
        """
        Recursive splitting implementation.
        """
        chunks = []
        current_pos = 0
        
        # If text fits in one chunk, return it
        if len(text) <= chunk_size:
            if len(text) >= self.config.min_chunk_size:
                return [Chunk(
                    content=text.strip(),
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
        
        for part in parts:
            # Check if adding this part would exceed chunk size
            potential_chunk = current_chunk + separator + part if current_chunk else part
            
            if len(potential_chunk) <= chunk_size:
                current_chunk = potential_chunk
            else:
                # Save current chunk if it's large enough
                if current_chunk and len(current_chunk) >= self.config.min_chunk_size:
                    chunks.append(Chunk(
                        content=current_chunk.strip(),
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
                    # Try with remaining separators
                    remaining_separators = separators[separators.index(separator)+1:] if separator in separators else separators[1:]
                    
                    if remaining_separators:
                        sub_chunks = self._split_recursive(
                            current_chunk,
                            remaining_separators,
                            chunk_size,
                            overlap
                        )
                        # Adjust indices
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
                                chunks.append(Chunk(
                                    content=sub_text.strip(),
                                    index=len(chunks),
                                    start_char=chunk_start + i,
                                    end_char=chunk_start + i + len(sub_text)
                                ))
                        current_chunk = ""
        
        # Don't forget the last chunk
        if current_chunk and len(current_chunk) >= self.config.min_chunk_size:
            chunks.append(Chunk(
                content=current_chunk.strip(),
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
        """
        Find the best separator that creates reasonable chunk sizes.
        """
        for separator in separators:
            if separator in text:
                parts = text.split(separator)
                avg_size = len(text) / len(parts) if parts else len(text)
                
                # Good separator if average part size is reasonable
                if avg_size < chunk_size * 2:
                    return separator
        
        return None
    
    def _semantic_chunk(self, text: str) -> List[Chunk]:
        """
        Chunk text while trying to preserve semantic units.
        
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
                current_pos += 2  # Account for \n\n
                continue
            
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
                        chunks.append(Chunk(
                            content=current_chunk.strip(),
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
            chunks.append(Chunk(
                content=current_chunk.strip(),
                index=len(chunks),
                start_char=chunk_start,
                end_char=chunk_start + len(current_chunk),
                metadata={"type": "semantic"}
            ))
        
        return chunks
    
    def _fixed_chunk(self, text: str) -> List[Chunk]:
        """
        Simple fixed-size chunking.
        """
        chunks = []
        chunk_size = self.config.chunk_size
        overlap = self.config.chunk_overlap
        
        step = chunk_size - overlap
        
        for i in range(0, len(text), step):
            chunk_text = text[i:i + chunk_size]
            
            if len(chunk_text) >= self.config.min_chunk_size:
                chunks.append(Chunk(
                    content=chunk_text.strip(),
                    index=len(chunks),
                    start_char=i,
                    end_char=i + len(chunk_text),
                    metadata={"type": "fixed"}
                ))
        
        return chunks
    
    def _add_context_to_chunks(
        self,
        chunks: List[Chunk],
        original_text: str
    ) -> List[Chunk]:
        """
        Add contextual information to chunks.
        """
        total_chunks = len(chunks)
        
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["total_chunks"] = total_chunks
            chunk.metadata["position"] = "start" if i == 0 else ("end" if i == total_chunks - 1 else "middle")
            chunk.metadata["char_position"] = f"{chunk.start_char}-{chunk.end_char}"
            
            # Add neighboring context hints
            if i > 0:
                chunk.metadata["previous_chunk_end"] = chunks[i-1].content[-50:]
            if i < total_chunks - 1:
                chunk.metadata["next_chunk_start"] = chunks[i+1].content[:50]
        
        return chunks
    
    def chunk_with_headers(
        self,
        text: str,
        header_pattern: str = r'^#+\s+.+$|^[A-Z][A-Za-z\s]+:?\s*$'
    ) -> List[Chunk]:
        """
        Chunk text while preserving headers with their content.
        
        Args:
            text: Text to chunk
            header_pattern: Regex pattern for headers
            
        Returns:
            List of chunks with preserved headers
        """
        chunks = []
        lines = text.split('\n')
        
        current_header = ""
        current_content = []
        chunk_start = 0
        current_pos = 0
        
        header_regex = re.compile(header_pattern, re.MULTILINE)
        
        for line in lines:
            if header_regex.match(line.strip()):
                # Save previous section
                if current_content:
                    section_text = current_header + "\n" + "\n".join(current_content) if current_header else "\n".join(current_content)
                    
                    # Chunk the section
                    section_chunks = self.chunk(section_text)
                    
                    for sc in section_chunks:
                        sc.start_char += chunk_start
                        sc.end_char += chunk_start
                        sc.metadata["header"] = current_header.strip() if current_header else None
                        chunks.append(sc)
                
                current_header = line
                current_content = []
                chunk_start = current_pos
            else:
                current_content.append(line)
            
            current_pos += len(line) + 1
        
        # Don't forget the last section
        if current_content:
            section_text = current_header + "\n" + "\n".join(current_content) if current_header else "\n".join(current_content)
            section_chunks = self.chunk(section_text)
            
            for sc in section_chunks:
                sc.start_char += chunk_start
                sc.end_char += chunk_start
                sc.metadata["header"] = current_header.strip() if current_header else None
                chunks.append(sc)
        
        # Re-index
        for i, chunk in enumerate(chunks):
            chunk.index = i
        
        return chunks


def create_chunker(
    strategy: str = "recursive",
    chunk_size: int = 512,
    **kwargs
) -> Chunker:
    """
    Factory function to create a chunker.
    
    Args:
        strategy: Chunking strategy
        chunk_size: Target chunk size
        **kwargs: Additional configuration
        
    Returns:
        Configured Chunker instance
    """
    config = ChunkerConfig(strategy=strategy, chunk_size=chunk_size, **kwargs)
    return Chunker(config)

