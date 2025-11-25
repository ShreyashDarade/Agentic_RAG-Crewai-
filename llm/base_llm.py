import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from enum import Enum

logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """Supported LLM providers."""
    GROQ = "groq"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


@dataclass
class LLMConfig:
    """Configuration for LLM clients."""
    provider: LLMProvider
    model: str
    api_key: str
    temperature: float = 0.1
    max_tokens: int = 4096
    timeout: int = 60
    retry_attempts: int = 3
    retry_delay: float = 2.0
    stop_sequences: Optional[List[str]] = None
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.api_key:
            raise ValueError(f"API key is required for {self.provider.value}")
        if self.temperature < 0 or self.temperature > 2:
            raise ValueError("Temperature must be between 0 and 2")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")


@dataclass
class LLMMessage:
    """Represents a message in a conversation."""
    role: str  # "system", "user", "assistant"
    content: str
    name: Optional[str] = None
    
    def to_dict(self) -> Dict[str, str]:
        """Convert message to dictionary format."""
        msg = {"role": self.role, "content": self.content}
        if self.name:
            msg["name"] = self.name
        return msg


@dataclass
class LLMResponse:
    """Standardized response from LLM."""
    content: str
    model: str
    provider: str
    usage: Dict[str, int] = field(default_factory=dict)
    finish_reason: Optional[str] = None
    raw_response: Optional[Any] = None
    latency_ms: float = 0.0
    
    @property
    def prompt_tokens(self) -> int:
        """Get prompt token count."""
        return self.usage.get("prompt_tokens", 0)
    
    @property
    def completion_tokens(self) -> int:
        """Get completion token count."""
        return self.usage.get("completion_tokens", 0)
    
    @property
    def total_tokens(self) -> int:
        """Get total token count."""
        return self.usage.get("total_tokens", 0)


class LLMError(Exception):
    """Base exception for LLM errors."""
    
    def __init__(self, message: str, provider: str, original_error: Optional[Exception] = None):
        self.message = message
        self.provider = provider
        self.original_error = original_error
        super().__init__(f"[{provider}] {message}")


class RateLimitError(LLMError):
    """Raised when rate limit is exceeded."""
    pass


class AuthenticationError(LLMError):
    """Raised when authentication fails."""
    pass


class InvalidRequestError(LLMError):
    """Raised when request is invalid."""
    pass


class BaseLLM(ABC):
    """
    Abstract base class for LLM implementations.
    
    All LLM providers should inherit from this class and implement
    the required abstract methods.
    """
    
    def __init__(self, config: LLMConfig):
        """
        Initialize the LLM client.
        
        Args:
            config: LLM configuration object
        """
        self.config = config
        self._validate_config()
        logger.info(f"Initialized {self.provider_name} LLM with model: {config.model}")
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the name of the LLM provider."""
        pass
    
    @abstractmethod
    def _validate_config(self) -> None:
        """Validate provider-specific configuration."""
        pass
    
    @abstractmethod
    async def generate(
        self,
        messages: List[LLMMessage],
        **kwargs
    ) -> LLMResponse:
        """
        Generate a response from the LLM.
        
        Args:
            messages: List of conversation messages
            **kwargs: Additional provider-specific parameters
            
        Returns:
            LLMResponse object containing the generated response
        """
        pass
    
    @abstractmethod
    async def generate_stream(
        self,
        messages: List[LLMMessage],
        **kwargs
    ):
        """
        Generate a streaming response from the LLM.
        
        Args:
            messages: List of conversation messages
            **kwargs: Additional provider-specific parameters
            
        Yields:
            Chunks of the generated response
        """
        pass
    
    def create_messages(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None
    ) -> List[LLMMessage]:
        """
        Create a list of messages for the LLM.
        
        Args:
            user_prompt: The user's input prompt
            system_prompt: Optional system prompt
            history: Optional conversation history
            
        Returns:
            List of LLMMessage objects
        """
        messages = []
        
        if system_prompt:
            messages.append(LLMMessage(role="system", content=system_prompt))
        
        if history:
            for msg in history:
                messages.append(LLMMessage(
                    role=msg.get("role", "user"),
                    content=msg.get("content", "")
                ))
        
        messages.append(LLMMessage(role="user", content=user_prompt))
        
        return messages
    
    async def simple_generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Simple interface to generate a response.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            **kwargs: Additional parameters
            
        Returns:
            Generated text content
        """
        messages = self.create_messages(prompt, system_prompt)
        response = await self.generate(messages, **kwargs)
        return response.content
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(provider={self.provider_name}, model={self.config.model})"

