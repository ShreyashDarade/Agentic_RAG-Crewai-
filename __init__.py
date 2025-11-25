"""
A production-ready multi-agent RAG (Retrieval-Augmented Generation) system
using CrewAI, Groq, ChromaDB, and FastAPI.

Modules:
    - agents: CrewAI-based agent classes
    - orchestrator: Multi-agent coordination
    - llm: LLM abstraction layer
    - embeddings: Vector embeddings and storage
    - retriever: Information retrieval
    - data_pipeline: Document processing
    - api: FastAPI application
    - config: Configuration management
"""

__version__ = "1.0.0"
__author__ = "Your Name"

from .orchestrator import CrewManager, MemoryStore, TraceLogger
from .data_pipeline import IngestionPipeline

__all__ = [
    "CrewManager",
    "MemoryStore",
    "TraceLogger",
    "IngestionPipeline",
]

