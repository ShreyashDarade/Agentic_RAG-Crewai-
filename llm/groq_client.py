

import asyncio
import logging
import time
from typing import Any, AsyncGenerator, Dict, List, Optional

from groq import AsyncGroq, APIError, RateLimitError as GroqRateLimitError, AuthenticationError as GroqAuthError

from .base_llm import (
    BaseLLM,
    LLMConfig,
    LLMMessage,
    LLMResponse,
    LLMError,
    RateLimitError,
    AuthenticationError,
    InvalidRequestError,
)

logger = logging.getLogger(__name__)


class GroqClient(BaseLLM):
    """
    Groq LLM client implementation.
    
    Provides access to Groq-hosted models like Llama, Mixtral, etc.
    """
    
    SUPPORTED_MODELS = ["openai/gpt-oss-120b"]
    
    def __init__(self, config: LLMConfig):
        """
        Initialize Groq client.
        
        Args:
            config: LLM configuration with Groq API key
        """
        super().__init__(config)
        self._client = AsyncGroq(api_key=config.api_key)
    
    @property
    def provider_name(self) -> str:
        """Return provider name."""
        return "groq"
    
    def _validate_config(self) -> None:
        """Validate Groq-specific configuration."""
        if not self.config.api_key:
            raise ValueError("Groq API key is required")
        
        if self.config.model not in self.SUPPORTED_MODELS:
            logger.warning(
                f"Model {self.config.model} may not be supported. "
                f"Supported models: {self.SUPPORTED_MODELS}"
            )
    
    def _handle_error(self, error: Exception) -> None:
        """
        Handle Groq API errors and convert to standard exceptions.
        
        Args:
            error: The original exception
            
        Raises:
            Appropriate LLMError subclass
        """
        if isinstance(error, GroqRateLimitError):
            raise RateLimitError(
                "Rate limit exceeded. Please try again later.",
                self.provider_name,
                error
            )
        elif isinstance(error, GroqAuthError):
            raise AuthenticationError(
                "Authentication failed. Check your API key.",
                self.provider_name,
                error
            )
        elif isinstance(error, APIError):
            raise InvalidRequestError(
                f"API error: {str(error)}",
                self.provider_name,
                error
            )
        else:
            raise LLMError(
                f"Unexpected error: {str(error)}",
                self.provider_name,
                error
            )
    
    async def _execute_with_retry(
        self,
        operation,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute an operation with retry logic.
        
        Args:
            operation: Async function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Result of the operation
        """
        last_error = None
        
        for attempt in range(self.config.retry_attempts):
            try:
                return await operation(*args, **kwargs)
            except GroqRateLimitError as e:
                last_error = e
                wait_time = self.config.retry_delay * (2 ** attempt)
                logger.warning(
                    f"Rate limit hit, retrying in {wait_time}s "
                    f"(attempt {attempt + 1}/{self.config.retry_attempts})"
                )
                await asyncio.sleep(wait_time)
            except (APIError, Exception) as e:
                last_error = e
                if attempt < self.config.retry_attempts - 1:
                    logger.warning(
                        f"Error occurred, retrying "
                        f"(attempt {attempt + 1}/{self.config.retry_attempts}): {str(e)}"
                    )
                    await asyncio.sleep(self.config.retry_delay)
                else:
                    break
        
        self._handle_error(last_error)
    
    async def generate(
        self,
        messages: List[LLMMessage],
        **kwargs
    ) -> LLMResponse:
        """
        Generate a response from Groq.
        
        Args:
            messages: List of conversation messages
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
            
        Returns:
            LLMResponse with generated content
        """
        start_time = time.time()
        
        # Prepare messages in Groq format
        formatted_messages = [msg.to_dict() for msg in messages]
        
        # Merge config with kwargs
        params = {
            "model": kwargs.get("model", self.config.model),
            "messages": formatted_messages,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "top_p": kwargs.get("top_p", self.config.top_p),
            "stream": False,
        }
        
        if self.config.stop_sequences:
            params["stop"] = self.config.stop_sequences
        
        logger.debug(f"Generating response with params: {params}")
        
        async def _make_request():
            return await self._client.chat.completions.create(**params)
        
        response = await self._execute_with_retry(_make_request)
        
        latency_ms = (time.time() - start_time) * 1000
        
        return LLMResponse(
            content=response.choices[0].message.content,
            model=response.model,
            provider=self.provider_name,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
            finish_reason=response.choices[0].finish_reason,
            raw_response=response,
            latency_ms=latency_ms,
        )
    
    async def generate_stream(
        self,
        messages: List[LLMMessage],
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Generate a streaming response from Groq.
        
        Args:
            messages: List of conversation messages
            **kwargs: Additional parameters
            
        Yields:
            Chunks of generated text
        """
        # Prepare messages in Groq format
        formatted_messages = [msg.to_dict() for msg in messages]
        
        params = {
            "model": kwargs.get("model", self.config.model),
            "messages": formatted_messages,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "top_p": kwargs.get("top_p", self.config.top_p),
            "stream": True,
        }
        
        if self.config.stop_sequences:
            params["stop"] = self.config.stop_sequences
        
        try:
            stream = await self._client.chat.completions.create(**params)
            
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            self._handle_error(e)
    
    async def generate_with_json(
        self,
        messages: List[LLMMessage],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a JSON response from Groq.
        
        Args:
            messages: List of conversation messages
            **kwargs: Additional parameters
            
        Returns:
            Parsed JSON response
        """
        import json
        
        # Add instruction for JSON output
        json_instruction = LLMMessage(
            role="system",
            content="You must respond with valid JSON only. No additional text or explanation."
        )
        
        messages_with_json = [json_instruction] + list(messages)
        
        response = await self.generate(messages_with_json, **kwargs)
        
        try:
            # Try to extract JSON from the response
            content = response.content.strip()
            
            # Handle markdown code blocks
            if content.startswith("```json"):
                content = content[7:]
            elif content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            
            return json.loads(content.strip())
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {response.content}")
            raise InvalidRequestError(
                f"Failed to parse JSON response: {str(e)}",
                self.provider_name,
                e
            )
    
    async def count_tokens(self, text: str) -> int:
        """
        Estimate token count for text.
        
        Note: This is an approximation. For exact counts,
        use the response usage data.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Estimated token count
        """
        # Rough approximation: ~4 characters per token
        return len(text) // 4
    
    async def close(self) -> None:
        """Close the client connection."""
        await self._client.close()
        logger.info("Groq client closed")


def create_groq_client(
    api_key: str,
    model: str = "llama-3.1-70b-versatile",
    **kwargs
) -> GroqClient:
    """
    Factory function to create a Groq client.
    
    Args:
        api_key: Groq API key
        model: Model name to use
        **kwargs: Additional configuration options
        
    Returns:
        Configured GroqClient instance
    """
    from .base_llm import LLMProvider
    
    config = LLMConfig(
        provider=LLMProvider.GROQ,
        model=model,
        api_key=api_key,
        **kwargs
    )
    
    return GroqClient(config)

