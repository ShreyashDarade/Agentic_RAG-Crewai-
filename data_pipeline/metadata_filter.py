import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set, Union

logger = logging.getLogger(__name__)


@dataclass
class FilterConfig:
    """Configuration for metadata filtering."""
    # Size filters
    min_content_length: int = 50
    max_content_length: int = 1000000
    
    # Date filters
    min_date: Optional[str] = None  # ISO format
    max_date: Optional[str] = None  # ISO format
    
    # Author filters
    allowed_authors: Optional[Set[str]] = None
    blocked_authors: Optional[Set[str]] = None
    
    # Source filters
    allowed_sources: Optional[Set[str]] = None
    blocked_sources: Optional[Set[str]] = None
    
    # Extension filters
    allowed_extensions: Optional[Set[str]] = None
    blocked_extensions: Optional[Set[str]] = None
    
    # Content filters
    required_keywords: Optional[Set[str]] = None
    blocked_keywords: Optional[Set[str]] = None
    
    # Quality filters
    min_word_count: int = 10
    max_duplicate_ratio: float = 0.85
    
    # Custom filters
    custom_filters: List[Callable] = field(default_factory=list)


@dataclass
class FilterResult:
    """Result of document filtering."""
    passed: bool
    document: Optional[Any]
    reason: str = ""
    filters_applied: List[str] = field(default_factory=list)
    metadata_cleaned: Dict[str, Any] = field(default_factory=dict)


class MetadataFilterError(Exception):
    """Exception raised for metadata filter errors."""
    pass


class MetadataFilter:
    """
    Filters and cleanses documents based on metadata.
    
    Features:
    - Content length filtering
    - Date range filtering
    - Author/source filtering
    - Keyword filtering
    - Duplicate detection
    - Custom filter support
    """
    
    def __init__(self, config: Optional[FilterConfig] = None):
        """
        Initialize the metadata filter.
        
        Args:
            config: Filter configuration
        """
        self.config = config or FilterConfig()
        self._seen_hashes: Set[str] = set()
    
    def _check_content_length(self, content: str) -> tuple:
        """Check content length constraints."""
        length = len(content)
        
        if length < self.config.min_content_length:
            return False, f"Content too short: {length} < {self.config.min_content_length}"
        
        if length > self.config.max_content_length:
            return False, f"Content too long: {length} > {self.config.max_content_length}"
        
        return True, ""
    
    def _check_word_count(self, content: str) -> tuple:
        """Check minimum word count."""
        word_count = len(content.split())
        
        if word_count < self.config.min_word_count:
            return False, f"Word count too low: {word_count} < {self.config.min_word_count}"
        
        return True, ""
    
    def _check_date_range(self, metadata: Dict[str, Any]) -> tuple:
        """Check date constraints."""
        date_fields = ["created_time", "modified_time", "date", "created_at"]
        
        doc_date = None
        for field in date_fields:
            if field in metadata and metadata[field]:
                try:
                    if isinstance(metadata[field], str):
                        doc_date = datetime.fromisoformat(metadata[field].replace("Z", "+00:00"))
                    elif isinstance(metadata[field], datetime):
                        doc_date = metadata[field]
                    break
                except ValueError:
                    continue
        
        if not doc_date:
            return True, ""  # No date to check
        
        if self.config.min_date:
            min_date = datetime.fromisoformat(self.config.min_date)
            if doc_date < min_date:
                return False, f"Document date {doc_date} before min_date {min_date}"
        
        if self.config.max_date:
            max_date = datetime.fromisoformat(self.config.max_date)
            if doc_date > max_date:
                return False, f"Document date {doc_date} after max_date {max_date}"
        
        return True, ""
    
    def _check_author(self, metadata: Dict[str, Any]) -> tuple:
        """Check author constraints."""
        author = metadata.get("author", "").lower().strip()
        
        if not author:
            return True, ""
        
        if self.config.allowed_authors:
            allowed = {a.lower() for a in self.config.allowed_authors}
            if author not in allowed:
                return False, f"Author '{author}' not in allowed list"
        
        if self.config.blocked_authors:
            blocked = {a.lower() for a in self.config.blocked_authors}
            if author in blocked:
                return False, f"Author '{author}' is blocked"
        
        return True, ""
    
    def _check_source(self, metadata: Dict[str, Any]) -> tuple:
        """Check source constraints."""
        source = metadata.get("source", "").lower().strip()
        file_path = metadata.get("file_path", "").lower()
        
        check_value = source or file_path
        if not check_value:
            return True, ""
        
        if self.config.allowed_sources:
            allowed = {s.lower() for s in self.config.allowed_sources}
            if not any(a in check_value for a in allowed):
                return False, f"Source not in allowed list"
        
        if self.config.blocked_sources:
            blocked = {s.lower() for s in self.config.blocked_sources}
            if any(b in check_value for b in blocked):
                return False, f"Source is blocked"
        
        return True, ""
    
    def _check_extension(self, metadata: Dict[str, Any]) -> tuple:
        """Check file extension constraints."""
        extension = metadata.get("file_extension", "").lower()
        
        if not extension:
            return True, ""
        
        if self.config.allowed_extensions:
            if extension not in self.config.allowed_extensions:
                return False, f"Extension '{extension}' not allowed"
        
        if self.config.blocked_extensions:
            if extension in self.config.blocked_extensions:
                return False, f"Extension '{extension}' is blocked"
        
        return True, ""
    
    def _check_keywords(self, content: str) -> tuple:
        """Check keyword constraints."""
        content_lower = content.lower()
        
        if self.config.required_keywords:
            for keyword in self.config.required_keywords:
                if keyword.lower() not in content_lower:
                    return False, f"Required keyword '{keyword}' not found"
        
        if self.config.blocked_keywords:
            for keyword in self.config.blocked_keywords:
                if keyword.lower() in content_lower:
                    return False, f"Blocked keyword '{keyword}' found"
        
        return True, ""
    
    def _check_duplicate(self, content: str, file_hash: Optional[str] = None) -> tuple:
        """Check for duplicate content."""
        import hashlib
        
        # Use provided hash or calculate from content
        if file_hash:
            content_hash = file_hash
        else:
            content_hash = hashlib.md5(content.encode()).hexdigest()
        
        if content_hash in self._seen_hashes:
            return False, "Duplicate content detected"
        
        return True, content_hash
    
    def _check_duplicate_ratio(self, content: str) -> tuple:
        """Check for high ratio of repeated content."""
        words = content.lower().split()
        if not words:
            return True, ""
        
        unique_words = set(words)
        unique_ratio = len(unique_words) / len(words)
        duplicate_ratio = 1 - unique_ratio
        
        if duplicate_ratio > self.config.max_duplicate_ratio:
            return False, f"High duplicate ratio: {duplicate_ratio:.2f} > {self.config.max_duplicate_ratio}"
        
        return True, ""
    
    def clean_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean and normalize metadata.
        
        Args:
            metadata: Raw metadata dictionary
            
        Returns:
            Cleaned metadata dictionary
        """
        cleaned = {}
        
        for key, value in metadata.items():
            # Skip None values
            if value is None:
                continue
            
            # Clean string values
            if isinstance(value, str):
                value = value.strip()
                if not value:
                    continue
            
            # Normalize key names
            clean_key = key.lower().replace(" ", "_").replace("-", "_")
            
            # Handle specific fields
            if clean_key in ["created_time", "modified_time", "date"]:
                try:
                    if isinstance(value, str):
                        # Ensure ISO format
                        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
                        value = dt.isoformat()
                except ValueError:
                    pass
            
            cleaned[clean_key] = value
        
        return cleaned
    
    def clean_content(self, content: str) -> str:
        """
        Clean and normalize content.
        
        Args:
            content: Raw content string
            
        Returns:
            Cleaned content string
        """
        if not content:
            return ""
        
        # Remove excessive whitespace
        content = re.sub(r"[ \t]+", " ", content)
        content = re.sub(r"\n{3,}", "\n\n", content)
        
        # Remove control characters (except newlines and tabs)
        content = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", content)
        
        # Strip leading/trailing whitespace
        content = content.strip()
        
        return content
    
    def filter_document(
        self,
        content: str,
        metadata: Dict[str, Any],
        file_hash: Optional[str] = None,
        track_duplicates: bool = True
    ) -> FilterResult:
        """
        Apply all filters to a document.
        
        Args:
            content: Document content
            metadata: Document metadata
            file_hash: Optional pre-calculated file hash
            track_duplicates: Whether to track this document for duplicate detection
            
        Returns:
            FilterResult with pass/fail status and details
        """
        filters_applied = []
        
        # Clean content and metadata
        clean_content_text = self.clean_content(content)
        clean_meta = self.clean_metadata(metadata)
        
        # Apply filters
        checks = [
            ("content_length", lambda: self._check_content_length(clean_content_text)),
            ("word_count", lambda: self._check_word_count(clean_content_text)),
            ("date_range", lambda: self._check_date_range(clean_meta)),
            ("author", lambda: self._check_author(clean_meta)),
            ("source", lambda: self._check_source(clean_meta)),
            ("extension", lambda: self._check_extension(clean_meta)),
            ("keywords", lambda: self._check_keywords(clean_content_text)),
            ("duplicate_ratio", lambda: self._check_duplicate_ratio(clean_content_text)),
        ]
        
        for filter_name, check_func in checks:
            passed, reason = check_func()
            filters_applied.append(filter_name)
            
            if not passed:
                return FilterResult(
                    passed=False,
                    document=None,
                    reason=reason,
                    filters_applied=filters_applied,
                    metadata_cleaned=clean_meta
                )
        
        # Check duplicate (handled separately for hash tracking)
        passed, result = self._check_duplicate(clean_content_text, file_hash)
        filters_applied.append("duplicate")
        
        if not passed:
            return FilterResult(
                passed=False,
                document=None,
                reason=result,
                filters_applied=filters_applied,
                metadata_cleaned=clean_meta
            )
        
        # Track hash if requested
        if track_duplicates:
            self._seen_hashes.add(result)
        
        # Apply custom filters
        for i, custom_filter in enumerate(self.config.custom_filters):
            filter_name = f"custom_{i}"
            filters_applied.append(filter_name)
            
            try:
                passed = custom_filter(clean_content_text, clean_meta)
                if not passed:
                    return FilterResult(
                        passed=False,
                        document=None,
                        reason=f"Custom filter {i} rejected document",
                        filters_applied=filters_applied,
                        metadata_cleaned=clean_meta
                    )
            except Exception as e:
                logger.warning(f"Custom filter {i} error: {str(e)}")
        
        return FilterResult(
            passed=True,
            document={"content": clean_content_text, "metadata": clean_meta},
            reason="All filters passed",
            filters_applied=filters_applied,
            metadata_cleaned=clean_meta
        )
    
    def filter_documents(
        self,
        documents: List[Dict[str, Any]]
    ) -> tuple:
        """
        Filter a list of documents.
        
        Args:
            documents: List of document dictionaries with 'content' and 'metadata'
            
        Returns:
            Tuple of (passed_documents, rejected_documents)
        """
        passed = []
        rejected = []
        
        for doc in documents:
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})
            file_hash = doc.get("file_hash")
            
            result = self.filter_document(content, metadata, file_hash)
            
            if result.passed:
                passed.append(result.document)
            else:
                rejected.append({
                    "original": doc,
                    "reason": result.reason
                })
        
        logger.info(f"Filtered documents: {len(passed)} passed, {len(rejected)} rejected")
        
        return passed, rejected
    
    def reset_duplicate_tracking(self) -> None:
        """Reset the duplicate tracking set."""
        self._seen_hashes.clear()
        logger.info("Duplicate tracking reset")
    
    def add_custom_filter(self, filter_func: Callable[[str, Dict], bool]) -> None:
        """
        Add a custom filter function.
        
        Args:
            filter_func: Function that takes (content, metadata) and returns bool
        """
        self.config.custom_filters.append(filter_func)


def create_metadata_filter(**kwargs) -> MetadataFilter:
    """
    Factory function to create a metadata filter.
    
    Args:
        **kwargs: Configuration options
        
    Returns:
        Configured MetadataFilter instance
    """
    config = FilterConfig(**kwargs)
    return MetadataFilter(config)

