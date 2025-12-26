"""
LLM Module - Production Grade (OpenAI Only)

Provides:
- OpenAI GPT client with function calling, streaming, and tools API
- Base LLM classes and utilities
"""

from .base_llm import (
    BaseLLM,
    LLMConfig,
    LLMMessage,
    LLMResponse,
    LLMProvider,
    LLMRole,
    LLMError,
)
from .openai_client import (
    OpenAIClient,
    OpenAIError,
    create_openai_client,
)

__all__ = [
    # Base classes
    "BaseLLM",
    "LLMConfig",
    "LLMMessage",
    "LLMResponse",
    "LLMProvider",
    "LLMRole",
    "LLMError",
    # OpenAI Client
    "OpenAIClient",
    "OpenAIError",
    "create_openai_client",
]
