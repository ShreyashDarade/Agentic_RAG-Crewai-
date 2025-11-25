import hashlib
import json
import logging
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class MemoryConfig:
    """Configuration for memory store."""
    max_items: int = 100
    ttl_seconds: int = 3600  # 1 hour
    persist_path: Optional[str] = "./data/memory"
    enable_persistence: bool = True
    enable_embedding_memory: bool = False


@dataclass
class MemoryEntry:
    """Represents a single memory entry."""
    id: str
    query: str
    response: str
    context: str
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryEntry":
        return cls(**data)


class MemoryStoreError(Exception):
    """Exception raised for memory store errors."""
    pass


class MemoryStore:
    """
    Memory store for conversation history and context.
    
    Features:
    - Short-term conversation memory
    - Context retrieval for follow-up queries
    - Optional persistence
    - TTL-based expiration
    """
    
    def __init__(self, config: Optional[MemoryConfig] = None):
        """
        Initialize the memory store.
        
        Args:
            config: Memory configuration
        """
        self.config = config or MemoryConfig()
        self._memory: deque = deque(maxlen=self.config.max_items)
        self._memory_index: Dict[str, MemoryEntry] = {}
        self._entity_memory: Dict[str, List[str]] = {}  # entity -> [memory_ids]
        
        if self.config.enable_persistence:
            self._load_from_disk()
    
    def _generate_id(self, query: str, timestamp: str) -> str:
        """Generate unique ID for memory entry."""
        content = f"{query}:{timestamp}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def store(
        self,
        query: str,
        response: str,
        context: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        entities: Optional[List[str]] = None
    ) -> str:
        """
        Store a new memory entry.
        
        Args:
            query: User query
            response: Generated response
            context: Retrieved context
            metadata: Additional metadata
            entities: Extracted entities for indexing
            
        Returns:
            Memory entry ID
        """
        timestamp = datetime.now().isoformat()
        memory_id = self._generate_id(query, timestamp)
        
        entry = MemoryEntry(
            id=memory_id,
            query=query,
            response=response,
            context=context,
            timestamp=timestamp,
            metadata=metadata or {}
        )
        
        # Add to memory
        self._memory.append(entry)
        self._memory_index[memory_id] = entry
        
        # Index entities
        if entities:
            for entity in entities:
                entity_lower = entity.lower()
                if entity_lower not in self._entity_memory:
                    self._entity_memory[entity_lower] = []
                self._entity_memory[entity_lower].append(memory_id)
        
        # Persist if enabled
        if self.config.enable_persistence:
            self._save_to_disk()
        
        logger.debug(f"Stored memory entry: {memory_id}")
        return memory_id
    
    def get(self, memory_id: str) -> Optional[MemoryEntry]:
        """
        Get a specific memory entry.
        
        Args:
            memory_id: Memory entry ID
            
        Returns:
            MemoryEntry or None
        """
        return self._memory_index.get(memory_id)
    
    def get_recent(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent memory entries.
        
        Args:
            limit: Maximum entries to return
            
        Returns:
            List of memory entry dictionaries
        """
        recent = list(self._memory)[-limit:]
        return [entry.to_dict() for entry in reversed(recent)]
    
    def search_by_query(
        self,
        query: str,
        top_k: int = 5
    ) -> List[MemoryEntry]:
        """
        Search memory by query similarity.
        
        Args:
            query: Search query
            top_k: Number of results
            
        Returns:
            List of matching memory entries
        """
        # Simple keyword-based search
        query_words = set(query.lower().split())
        
        scored_entries = []
        for entry in self._memory:
            entry_words = set(entry.query.lower().split())
            overlap = len(query_words.intersection(entry_words))
            if overlap > 0:
                score = overlap / max(len(query_words), len(entry_words))
                scored_entries.append((entry, score))
        
        # Sort by score
        scored_entries.sort(key=lambda x: x[1], reverse=True)
        
        return [entry for entry, score in scored_entries[:top_k]]
    
    def search_by_entity(self, entity: str) -> List[MemoryEntry]:
        """
        Search memory by entity.
        
        Args:
            entity: Entity to search for
            
        Returns:
            List of memory entries mentioning the entity
        """
        entity_lower = entity.lower()
        memory_ids = self._entity_memory.get(entity_lower, [])
        
        entries = []
        for mid in memory_ids:
            entry = self._memory_index.get(mid)
            if entry:
                entries.append(entry)
        
        return entries
    
    def get_context_for_query(
        self,
        query: str,
        max_context_length: int = 2000
    ) -> str:
        """
        Get relevant context from memory for a query.
        
        Args:
            query: Current query
            max_context_length: Maximum context length
            
        Returns:
            Formatted context string
        """
        # Get recent entries
        recent = list(self._memory)[-5:]
        
        # Get related entries
        related = self.search_by_query(query, top_k=3)
        
        # Combine and deduplicate
        all_entries = []
        seen_ids = set()
        
        for entry in reversed(recent):
            if entry.id not in seen_ids:
                all_entries.append(entry)
                seen_ids.add(entry.id)
        
        for entry in related:
            if entry.id not in seen_ids:
                all_entries.append(entry)
                seen_ids.add(entry.id)
        
        # Build context string
        context_parts = ["Previous conversation context:"]
        current_length = len(context_parts[0])
        
        for entry in all_entries[:5]:
            entry_text = f"\nQ: {entry.query[:200]}\nA: {entry.response[:300]}"
            
            if current_length + len(entry_text) > max_context_length:
                break
            
            context_parts.append(entry_text)
            current_length += len(entry_text)
        
        return "\n".join(context_parts) if len(context_parts) > 1 else ""
    
    def clear(self) -> None:
        """Clear all memory."""
        self._memory.clear()
        self._memory_index.clear()
        self._entity_memory.clear()
        
        if self.config.enable_persistence:
            self._save_to_disk()
        
        logger.info("Memory cleared")
    
    def _save_to_disk(self) -> None:
        """Save memory to disk."""
        if not self.config.persist_path:
            return
        
        try:
            persist_path = Path(self.config.persist_path)
            persist_path.mkdir(parents=True, exist_ok=True)
            
            memory_file = persist_path / "memory.json"
            
            data = {
                "entries": [entry.to_dict() for entry in self._memory],
                "entity_index": self._entity_memory,
                "saved_at": datetime.now().isoformat()
            }
            
            with open(memory_file, "w") as f:
                json.dump(data, f, indent=2, default=str)
            
            logger.debug(f"Memory saved to {memory_file}")
            
        except Exception as e:
            logger.warning(f"Failed to save memory: {e}")
    
    def _load_from_disk(self) -> None:
        """Load memory from disk."""
        if not self.config.persist_path:
            return
        
        try:
            persist_path = Path(self.config.persist_path)
            memory_file = persist_path / "memory.json"
            
            if not memory_file.exists():
                return
            
            with open(memory_file, "r") as f:
                data = json.load(f)
            
            # Load entries
            for entry_data in data.get("entries", []):
                entry = MemoryEntry.from_dict(entry_data)
                self._memory.append(entry)
                self._memory_index[entry.id] = entry
            
            # Load entity index
            self._entity_memory = data.get("entity_index", {})
            
            logger.info(f"Loaded {len(self._memory)} memory entries from disk")
            
        except Exception as e:
            logger.warning(f"Failed to load memory: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return {
            "total_entries": len(self._memory),
            "max_entries": self.config.max_items,
            "entities_tracked": len(self._entity_memory),
            "oldest_entry": self._memory[0].timestamp if self._memory else None,
            "newest_entry": self._memory[-1].timestamp if self._memory else None,
            "persistence_enabled": self.config.enable_persistence
        }
    
    @property
    def size(self) -> int:
        """Get number of entries in memory."""
        return len(self._memory)


class SharedState:
    """
    Shared state for agent communication.
    
    Allows agents to share data during execution.
    """
    
    def __init__(self):
        self._state: Dict[str, Any] = {}
        self._history: List[Dict[str, Any]] = []
    
    def set(self, key: str, value: Any) -> None:
        """Set a state value."""
        old_value = self._state.get(key)
        self._state[key] = value
        
        self._history.append({
            "action": "set",
            "key": key,
            "old_value": old_value,
            "new_value": value,
            "timestamp": datetime.now().isoformat()
        })
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a state value."""
        return self._state.get(key, default)
    
    def delete(self, key: str) -> None:
        """Delete a state value."""
        if key in self._state:
            old_value = self._state.pop(key)
            self._history.append({
                "action": "delete",
                "key": key,
                "old_value": old_value,
                "timestamp": datetime.now().isoformat()
            })
    
    def clear(self) -> None:
        """Clear all state."""
        self._state.clear()
        self._history.clear()
    
    def get_all(self) -> Dict[str, Any]:
        """Get all state."""
        return self._state.copy()
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get state history."""
        return self._history.copy()


def create_memory_store(
    max_items: int = 100,
    enable_persistence: bool = True,
    **kwargs
) -> MemoryStore:
    """
    Factory function to create a memory store.
    
    Args:
        max_items: Maximum memory entries
        enable_persistence: Enable disk persistence
        **kwargs: Additional configuration
        
    Returns:
        Configured MemoryStore instance
    """
    config = MemoryConfig(max_items=max_items, enable_persistence=enable_persistence, **kwargs)
    return MemoryStore(config)

