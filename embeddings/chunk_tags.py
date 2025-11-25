import hashlib
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set
from enum import Enum

logger = logging.getLogger(__name__)


class ChunkType(Enum):
    """Types of chunks based on content."""
    TEXT = "text"
    CODE = "code"
    TABLE = "table"
    LIST = "list"
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    QUOTE = "quote"
    MIXED = "mixed"


class ContentCategory(Enum):
    """Content categories for semantic tagging."""
    GENERAL = "general"
    TECHNICAL = "technical"
    LEGAL = "legal"
    FINANCIAL = "financial"
    MEDICAL = "medical"
    SCIENTIFIC = "scientific"
    INSTRUCTIONAL = "instructional"
    REFERENCE = "reference"


@dataclass
class TaggedChunk:
    """
    Represents a chunk with associated tags and metadata.
    """
    id: str
    content: str
    chunk_type: ChunkType
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Position information
    chunk_index: int = 0
    total_chunks: int = 0
    start_char: int = 0
    end_char: int = 0
    
    # Relevance information
    importance_score: float = 0.5
    category: ContentCategory = ContentCategory.GENERAL
    
    # Timestamps
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "content": self.content,
            "chunk_type": self.chunk_type.value,
            "tags": list(self.tags),
            "metadata": self.metadata,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "importance_score": self.importance_score,
            "category": self.category.value,
            "created_at": self.created_at,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaggedChunk":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            content=data["content"],
            chunk_type=ChunkType(data.get("chunk_type", "text")),
            tags=set(data.get("tags", [])),
            metadata=data.get("metadata", {}),
            chunk_index=data.get("chunk_index", 0),
            total_chunks=data.get("total_chunks", 0),
            start_char=data.get("start_char", 0),
            end_char=data.get("end_char", 0),
            importance_score=data.get("importance_score", 0.5),
            category=ContentCategory(data.get("category", "general")),
            created_at=data.get("created_at", datetime.now().isoformat()),
        )


class ChunkTagger:
    """
    Adds semantic tags to chunks for improved relevance tracking.
    
    Features:
    - Automatic chunk type detection
    - Keyword-based tagging
    - Category classification
    - Importance scoring
    """
    
    # Keyword patterns for tagging
    TECHNICAL_KEYWORDS = {
        "api", "function", "class", "method", "variable", "parameter",
        "database", "server", "client", "http", "json", "xml",
        "algorithm", "data structure", "interface", "module"
    }
    
    LEGAL_KEYWORDS = {
        "agreement", "contract", "liability", "warranty", "terms",
        "conditions", "clause", "party", "parties", "hereby",
        "pursuant", "jurisdiction", "indemnify"
    }
    
    FINANCIAL_KEYWORDS = {
        "revenue", "profit", "loss", "balance", "asset", "liability",
        "equity", "dividend", "interest", "investment", "budget",
        "fiscal", "quarter", "annual"
    }
    
    INSTRUCTIONAL_KEYWORDS = {
        "step", "steps", "how to", "guide", "tutorial", "instruction",
        "example", "note", "tip", "warning", "important"
    }
    
    # Code patterns
    CODE_PATTERNS = [
        r"```[\s\S]*?```",  # Markdown code blocks
        r"def\s+\w+\s*\(",  # Python functions
        r"function\s+\w+\s*\(",  # JavaScript functions
        r"class\s+\w+",  # Class definitions
        r"import\s+[\w.]+",  # Import statements
        r"from\s+[\w.]+\s+import",  # Python imports
    ]
    
    def __init__(self, custom_tags: Optional[Dict[str, Set[str]]] = None):
        """
        Initialize the chunk tagger.
        
        Args:
            custom_tags: Custom keyword-to-tag mappings
        """
        self.custom_tags = custom_tags or {}
        self._compile_patterns()
    
    def _compile_patterns(self) -> None:
        """Compile regex patterns for efficiency."""
        self._code_patterns = [re.compile(p, re.IGNORECASE) for p in self.CODE_PATTERNS]
    
    def generate_chunk_id(self, content: str, source: str, index: int) -> str:
        """
        Generate a unique ID for a chunk.
        
        Args:
            content: Chunk content
            source: Source document identifier
            index: Chunk index within document
            
        Returns:
            Unique chunk ID
        """
        hash_input = f"{source}:{index}:{content[:100]}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:16]
    
    def detect_chunk_type(self, content: str) -> ChunkType:
        """
        Detect the type of content in a chunk.
        
        Args:
            content: Chunk content
            
        Returns:
            Detected ChunkType
        """
        content_lower = content.lower().strip()
        
        # Check for code
        for pattern in self._code_patterns:
            if pattern.search(content):
                return ChunkType.CODE
        
        # Check for tables (simple heuristic)
        if "|" in content and content.count("|") > 3:
            return ChunkType.TABLE
        
        # Check for lists
        list_patterns = [r"^\s*[-*•]\s", r"^\s*\d+\.\s"]
        for pattern in list_patterns:
            if re.search(pattern, content, re.MULTILINE):
                return ChunkType.LIST
        
        # Check for headings
        if content_lower.startswith("#") or re.match(r"^[A-Z][A-Za-z\s]+:?\s*$", content.strip()):
            return ChunkType.HEADING
        
        # Check for quotes
        if content.startswith(">") or content.startswith('"'):
            return ChunkType.QUOTE
        
        return ChunkType.PARAGRAPH
    
    def extract_tags(self, content: str) -> Set[str]:
        """
        Extract relevant tags from content.
        
        Args:
            content: Chunk content
            
        Returns:
            Set of extracted tags
        """
        tags = set()
        content_lower = content.lower()
        words = set(re.findall(r'\b\w+\b', content_lower))
        
        # Check technical keywords
        tech_matches = words.intersection(self.TECHNICAL_KEYWORDS)
        if tech_matches:
            tags.add("technical")
            tags.update(tech_matches)
        
        # Check legal keywords
        legal_matches = words.intersection(self.LEGAL_KEYWORDS)
        if legal_matches:
            tags.add("legal")
        
        # Check financial keywords
        financial_matches = words.intersection(self.FINANCIAL_KEYWORDS)
        if financial_matches:
            tags.add("financial")
        
        # Check instructional keywords
        instructional_matches = words.intersection(self.INSTRUCTIONAL_KEYWORDS)
        if instructional_matches:
            tags.add("instructional")
        
        # Apply custom tags
        for tag, keywords in self.custom_tags.items():
            if words.intersection(keywords):
                tags.add(tag)
        
        # Extract entities (simple capitalized words)
        entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', content)
        for entity in entities[:5]:  # Limit entities
            if len(entity) > 2:
                tags.add(f"entity:{entity.lower()}")
        
        return tags
    
    def classify_category(self, content: str, tags: Set[str]) -> ContentCategory:
        """
        Classify the content category.
        
        Args:
            content: Chunk content
            tags: Already extracted tags
            
        Returns:
            Content category
        """
        if "technical" in tags or "code" in tags:
            return ContentCategory.TECHNICAL
        if "legal" in tags:
            return ContentCategory.LEGAL
        if "financial" in tags:
            return ContentCategory.FINANCIAL
        if "instructional" in tags:
            return ContentCategory.INSTRUCTIONAL
        
        return ContentCategory.GENERAL
    
    def calculate_importance(
        self,
        content: str,
        chunk_type: ChunkType,
        chunk_index: int,
        total_chunks: int
    ) -> float:
        """
        Calculate importance score for a chunk.
        
        Args:
            content: Chunk content
            chunk_type: Type of chunk
            chunk_index: Position in document
            total_chunks: Total chunks in document
            
        Returns:
            Importance score (0-1)
        """
        score = 0.5  # Base score
        
        # Content length factor
        content_length = len(content)
        if 200 <= content_length <= 1000:
            score += 0.1
        elif content_length > 1000:
            score += 0.05
        
        # Position factor (beginning and end are often more important)
        if total_chunks > 1:
            position_ratio = chunk_index / (total_chunks - 1)
            if position_ratio < 0.2 or position_ratio > 0.8:
                score += 0.1
        
        # Chunk type factor
        type_scores = {
            ChunkType.HEADING: 0.15,
            ChunkType.CODE: 0.1,
            ChunkType.TABLE: 0.1,
            ChunkType.LIST: 0.05,
        }
        score += type_scores.get(chunk_type, 0)
        
        # Keyword density factor
        words = content.split()
        if words:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio > 0.7:
                score += 0.05
        
        return min(1.0, max(0.0, score))
    
    def tag_chunk(
        self,
        content: str,
        source: str,
        chunk_index: int,
        total_chunks: int,
        start_char: int = 0,
        end_char: int = 0,
        extra_metadata: Optional[Dict[str, Any]] = None
    ) -> TaggedChunk:
        """
        Create a fully tagged chunk.
        
        Args:
            content: Chunk content
            source: Source document identifier
            chunk_index: Position in document
            total_chunks: Total chunks
            start_char: Start position in original document
            end_char: End position in original document
            extra_metadata: Additional metadata to include
            
        Returns:
            TaggedChunk with all tags and metadata
        """
        # Generate ID
        chunk_id = self.generate_chunk_id(content, source, chunk_index)
        
        # Detect type
        chunk_type = self.detect_chunk_type(content)
        
        # Extract tags
        tags = self.extract_tags(content)
        tags.add(chunk_type.value)
        
        # Classify category
        category = self.classify_category(content, tags)
        
        # Calculate importance
        importance = self.calculate_importance(
            content, chunk_type, chunk_index, total_chunks
        )
        
        # Build metadata
        metadata = {
            "source": source,
            "word_count": len(content.split()),
            "char_count": len(content),
        }
        if extra_metadata:
            metadata.update(extra_metadata)
        
        return TaggedChunk(
            id=chunk_id,
            content=content,
            chunk_type=chunk_type,
            tags=tags,
            metadata=metadata,
            chunk_index=chunk_index,
            total_chunks=total_chunks,
            start_char=start_char,
            end_char=end_char,
            importance_score=importance,
            category=category,
        )
    
    def tag_chunks(
        self,
        chunks: List[str],
        source: str,
        extra_metadata: Optional[Dict[str, Any]] = None
    ) -> List[TaggedChunk]:
        """
        Tag multiple chunks from a document.
        
        Args:
            chunks: List of chunk contents
            source: Source document identifier
            extra_metadata: Additional metadata for all chunks
            
        Returns:
            List of TaggedChunk objects
        """
        total_chunks = len(chunks)
        tagged_chunks = []
        
        current_pos = 0
        for i, content in enumerate(chunks):
            end_pos = current_pos + len(content)
            
            tagged = self.tag_chunk(
                content=content,
                source=source,
                chunk_index=i,
                total_chunks=total_chunks,
                start_char=current_pos,
                end_char=end_pos,
                extra_metadata=extra_metadata
            )
            
            tagged_chunks.append(tagged)
            current_pos = end_pos
        
        logger.info(f"Tagged {len(tagged_chunks)} chunks from {source}")
        return tagged_chunks
    
    def filter_by_tags(
        self,
        chunks: List[TaggedChunk],
        required_tags: Optional[Set[str]] = None,
        excluded_tags: Optional[Set[str]] = None,
        min_importance: float = 0.0
    ) -> List[TaggedChunk]:
        """
        Filter chunks based on tags and importance.
        
        Args:
            chunks: List of tagged chunks
            required_tags: Tags that must be present
            excluded_tags: Tags that must not be present
            min_importance: Minimum importance score
            
        Returns:
            Filtered list of chunks
        """
        filtered = []
        
        for chunk in chunks:
            # Check importance
            if chunk.importance_score < min_importance:
                continue
            
            # Check required tags
            if required_tags and not required_tags.issubset(chunk.tags):
                continue
            
            # Check excluded tags
            if excluded_tags and excluded_tags.intersection(chunk.tags):
                continue
            
            filtered.append(chunk)
        
        return filtered

