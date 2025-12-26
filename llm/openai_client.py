"""
OpenAI LLM Client - Production Grade Implementation

Provides access to OpenAI GPT models with:
- Async/streaming support
- Retry logic with exponential backoff
- Token counting
- JSON mode
- Function calling
"""

import asyncio
import logging
import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Callable
import json

from openai import AsyncOpenAI, APIError, RateLimitError as OpenAIRateLimitError
from openai import AuthenticationError as OpenAIAuthError, BadRequestError

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


class OpenAIClient(BaseLLM):
    """
    OpenAI LLM client implementation.
    
    Provides access to GPT-4o, GPT-4-turbo, GPT-3.5-turbo models
    with advanced features like JSON mode and function calling.
    """
    
    SUPPORTED_MODELS = [
        "gpt-4o-mini"
    ]
    
    EMBEDDING_MODELS = [
        "text-embedding-3-small"
    ]
    
    def __init__(self, config: LLMConfig):
        """
        Initialize OpenAI client.
        
        Args:
            config: LLM configuration with OpenAI API key
        """
        super().__init__(config)
        self._client = AsyncOpenAI(api_key=config.api_key)
        self._tokenizer = None
    
    @property
    def provider_name(self) -> str:
        """Return provider name."""
        return "openai"
    
    def _validate_config(self) -> None:
        """Validate OpenAI-specific configuration."""
        if not self.config.api_key:
            raise ValueError("OpenAI API key is required")
        
        if self.config.model not in self.SUPPORTED_MODELS:
            logger.warning(
                f"Model {self.config.model} may not be supported. "
                f"Supported models: {self.SUPPORTED_MODELS}"
            )
    
    def _handle_error(self, error: Exception) -> None:
        """
        Handle OpenAI API errors and convert to standard exceptions.
        
        Args:
            error: The original exception
            
        Raises:
            Appropriate LLMError subclass
        """
        if isinstance(error, OpenAIRateLimitError):
            raise RateLimitError(
                "Rate limit exceeded. Please try again later.",
                self.provider_name,
                error
            )
        elif isinstance(error, OpenAIAuthError):
            raise AuthenticationError(
                "Authentication failed. Check your API key.",
                self.provider_name,
                error
            )
        elif isinstance(error, BadRequestError):
            raise InvalidRequestError(
                f"Bad request: {str(error)}",
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
        operation: Callable,
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
            except OpenAIRateLimitError as e:
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
        Generate a response from OpenAI.
        
        Args:
            messages: List of conversation messages
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
            
        Returns:
            LLMResponse with generated content
        """
        start_time = time.time()
        
        # Prepare messages in OpenAI format
        formatted_messages = [msg.to_dict() for msg in messages]
        
        # Merge config with kwargs
        params = {
            "model": kwargs.get("model", self.config.model),
            "messages": formatted_messages,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "top_p": kwargs.get("top_p", self.config.top_p),
        }
        
        # Add optional parameters
        if self.config.stop_sequences:
            params["stop"] = self.config.stop_sequences
        
        if kwargs.get("json_mode"):
            params["response_format"] = {"type": "json_object"}
        
        if kwargs.get("functions"):
            params["functions"] = kwargs["functions"]
        
        if kwargs.get("function_call"):
            params["function_call"] = kwargs["function_call"]
        
        if kwargs.get("tools"):
            params["tools"] = kwargs["tools"]
        
        if kwargs.get("tool_choice"):
            params["tool_choice"] = kwargs["tool_choice"]
        
        logger.debug(f"Generating response with model: {params['model']}")
        
        async def _make_request():
            return await self._client.chat.completions.create(**params)
        
        response = await self._execute_with_retry(_make_request)
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Handle function calls
        function_call = None
        tool_calls = None
        
        if response.choices[0].message.function_call:
            function_call = {
                "name": response.choices[0].message.function_call.name,
                "arguments": response.choices[0].message.function_call.arguments
            }
        
        if response.choices[0].message.tool_calls:
            tool_calls = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                }
                for tc in response.choices[0].message.tool_calls
            ]
        
        return LLMResponse(
            content=response.choices[0].message.content or "",
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
            function_call=function_call,
            tool_calls=tool_calls
        )
    
    async def generate_stream(
        self,
        messages: List[LLMMessage],
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Generate a streaming response from OpenAI.
        
        Args:
            messages: List of conversation messages
            **kwargs: Additional parameters
            
        Yields:
            Chunks of generated text
        """
        # Prepare messages in OpenAI format
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
        schema: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a JSON response from OpenAI using JSON mode.
        
        Args:
            messages: List of conversation messages
            schema: Optional JSON schema for validation
            **kwargs: Additional parameters
            
        Returns:
            Parsed JSON response
        """
        # Add instruction for JSON output
        json_instruction = LLMMessage(
            role="system",
            content="You must respond with valid JSON only. Ensure the JSON is properly formatted."
        )
        
        messages_with_json = [json_instruction] + list(messages)
        
        # Enable JSON mode
        kwargs["json_mode"] = True
        
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
            
            parsed = json.loads(content.strip())
            
            # Validate against schema if provided
            if schema:
                # Basic schema validation could be added here
                pass
            
            return parsed
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {response.content}")
            raise InvalidRequestError(
                f"Failed to parse JSON response: {str(e)}",
                self.provider_name,
                e
            )
    
    async def generate_with_functions(
        self,
        messages: List[LLMMessage],
        functions: List[Dict[str, Any]],
        function_call: Optional[str] = "auto",
        **kwargs
    ) -> LLMResponse:
        """
        Generate a response with function calling.
        
        Args:
            messages: List of conversation messages
            functions: List of function definitions
            function_call: Function call mode ("auto", "none", or function name)
            **kwargs: Additional parameters
            
        Returns:
            LLMResponse with potential function call
        """
        kwargs["functions"] = functions
        kwargs["function_call"] = function_call
        
        return await self.generate(messages, **kwargs)
    
    async def generate_with_tools(
        self,
        messages: List[LLMMessage],
        tools: List[Dict[str, Any]],
        tool_choice: Optional[str] = "auto",
        **kwargs
    ) -> LLMResponse:
        """
        Generate a response with tool calling (newer API).
        
        Args:
            messages: List of conversation messages
            tools: List of tool definitions
            tool_choice: Tool choice mode ("auto", "none", or specific tool)
            **kwargs: Additional parameters
            
        Returns:
            LLMResponse with potential tool calls
        """
        kwargs["tools"] = tools
        kwargs["tool_choice"] = tool_choice
        
        return await self.generate(messages, **kwargs)
    
    async def count_tokens(self, text: str) -> int:
        """
        Count tokens for text using tiktoken.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Token count
        """
        try:
            import tiktoken
            
            if self._tokenizer is None:
                try:
                    self._tokenizer = tiktoken.encoding_for_model(self.config.model)
                except KeyError:
                    # Fallback to cl100k_base for newer models
                    self._tokenizer = tiktoken.get_encoding("cl100k_base")
            
            return len(self._tokenizer.encode(text))
            
        except ImportError:
            # Fallback: rough approximation
            return len(text) // 4
    
    async def count_messages_tokens(self, messages: List[LLMMessage]) -> int:
        """
        Count tokens for a list of messages.
        
        Args:
            messages: List of messages
            
        Returns:
            Total token count
        """
        total = 0
        for msg in messages:
            # Account for message overhead
            total += 4  # Every message follows <im_start>{role/name}\n{content}<im_end>\n
            total += await self.count_tokens(msg.content)
            if msg.role:
                total += await self.count_tokens(msg.role)
        total += 2  # Every reply is primed with <im_start>assistant
        return total
    
    async def embed(
        self,
        texts: List[str],
        model: Optional[str] = None
    ) -> List[List[float]]:
        """
        Generate embeddings for texts.
        
        Args:
            texts: List of texts to embed
            model: Embedding model to use
            
        Returns:
            List of embedding vectors
        """
        model = model or "text-embedding-3-small"
        
        try:
            response = await self._client.embeddings.create(
                input=texts,
                model=model
            )
            
            return [item.embedding for item in response.data]
            
        except Exception as e:
            self._handle_error(e)
    
    async def close(self) -> None:
        """Close the client connection."""
        await self._client.close()
        logger.info("OpenAI client closed")


def create_openai_client(
    api_key: str,
    model: str = "gpt-4o-mini",
    **kwargs
) -> OpenAIClient:
    """
    Factory function to create an OpenAI client.
    
    Args:
        api_key: OpenAI API key
        model: Model name to use
        **kwargs: Additional configuration options
        
    Returns:
        Configured OpenAIClient instance
    """
    from .base_llm import LLMProvider
    
    config = LLMConfig(
        provider=LLMProvider.OPENAI,
        model=model,
        api_key=api_key,
        **kwargs
    )
    
    return OpenAIClient(config)
