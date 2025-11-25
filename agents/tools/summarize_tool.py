import logging
from typing import Any, Optional, Type

from crewai.tools import BaseTool
from ..utils import run_async_task
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class SummarizeInput(BaseModel):
    """Input schema for summarize tool."""
    text: str = Field(
        ...,
        description="The text content to summarize"
    )
    max_length: int = Field(
        default=200,
        description="Maximum length of the summary in words",
        ge=50,
        le=1000
    )
    style: str = Field(
        default="concise",
        description="Summary style: 'concise', 'detailed', 'bullet_points'"
    )


class SummarizeTool(BaseTool):
    """
    Tool for summarizing text content.
    
    Uses LLM to generate concise summaries of longer texts.
    """
    
    name: str = "summarize"
    description: str = """
    Summarize long text content into a concise summary.
    Use this tool when:
    - Retrieved content is too long to process
    - You need to extract key points from text
    - Multiple documents need to be condensed
    Supports different summary styles: concise, detailed, or bullet_points.
    """
    args_schema: Type[BaseModel] = SummarizeInput
    
    _llm_client: Any = None
    _api_key: Optional[str] = None
    _model: str = "openai/gpt-oss-120b"
    
    def __init__(
        self,
        llm_client: Any = None,
        api_key: Optional[str] = None,
        model: str = "openai/gpt-oss-120b",
        **kwargs
    ):
        """
        Initialize the summarize tool.
        
        Args:
            llm_client: LLM client instance
            api_key: API key for LLM
            model: Model name to use
            **kwargs: Additional arguments
        """
        super().__init__(**kwargs)
        self._llm_client = llm_client
        self._api_key = api_key
        self._model = model
    
    def _initialize_llm(self) -> None:
        """Initialize LLM client if not provided."""
        if self._llm_client is None:
            try:
                import os
                from ...llm import GroqClient, LLMConfig, LLMProvider
                
                api_key = self._api_key or os.getenv("GROQ_API_KEY")
                if not api_key:
                    raise ValueError("GROQ_API_KEY not found")
                
                config = LLMConfig(
                    provider=LLMProvider.GROQ,
                    model=self._model,
                    api_key=api_key,
                    temperature=0.3,
                    max_tokens=1000
                )
                self._llm_client = GroqClient(config)
                logger.info(f"Initialized LLM client with model: {self._model}")
                
            except Exception as e:
                logger.error(f"Failed to initialize LLM: {e}")
                raise
    
    def _get_summary_prompt(self, text: str, max_length: int, style: str) -> str:
        """Generate the summarization prompt."""
        style_instructions = {
            "concise": f"Create a concise summary in {max_length} words or less. Focus on the main points.",
            "detailed": f"Create a detailed summary in {max_length} words or less. Include important details and context.",
            "bullet_points": f"Create a bullet-point summary with {max_length // 20} key points maximum."
        }
        
        instruction = style_instructions.get(style, style_instructions["concise"])
        
        return f"""Summarize the following text. {instruction}

Text to summarize:
{text}

Summary:"""
    
    def _run(
        self,
        text: str,
        max_length: int = 200,
        style: str = "concise"
    ) -> str:
        """
        Execute the summarization.
        
        Args:
            text: Text to summarize
            max_length: Maximum summary length
            style: Summary style
            
        Returns:
            Summary text
        """
        if not text or not text.strip():
            return "No text provided for summarization."
        
        # If text is short, return as is
        if len(text.split()) <= max_length:
            return f"Text is already concise:\n{text}"
        
        self._initialize_llm()
        
        try:
            logger.info(f"Summarizing text ({len(text)} chars) in {style} style")
            
            prompt = self._get_summary_prompt(text, max_length, style)
            
            response = run_async_task(
                self._llm_client.simple_generate(
                    prompt=prompt,
                    system_prompt="You are a helpful assistant that creates clear, accurate summaries."
                )
            )
            
            return response.strip()
            
        except Exception as e:
            error_msg = f"Summarization failed: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    async def _arun(
        self,
        text: str,
        max_length: int = 200,
        style: str = "concise"
    ) -> str:
        """Async version of summarization."""
        if not text or not text.strip():
            return "No text provided for summarization."
        
        if len(text.split()) <= max_length:
            return f"Text is already concise:\n{text}"
        
        self._initialize_llm()
        
        try:
            prompt = self._get_summary_prompt(text, max_length, style)
            
            response = await self._llm_client.simple_generate(
                prompt=prompt,
                system_prompt="You are a helpful assistant that creates clear, accurate summaries."
            )
            
            return response.strip()
            
        except Exception as e:
            return f"Summarization failed: {str(e)}"


class KeyPointExtractionInput(BaseModel):
    """Input schema for key point extraction."""
    text: str = Field(
        ...,
        description="The text to extract key points from"
    )
    num_points: int = Field(
        default=5,
        description="Number of key points to extract",
        ge=1,
        le=15
    )


class KeyPointExtractionTool(BaseTool):
    """
    Tool for extracting key points from text.
    """
    
    name: str = "extract_key_points"
    description: str = """
    Extract key points from text content.
    Returns a list of the most important points in the text.
    """
    args_schema: Type[BaseModel] = KeyPointExtractionInput
    
    _llm_client: Any = None
    _api_key: Optional[str] = None
    
    def __init__(
        self,
        llm_client: Any = None,
        api_key: Optional[str] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self._llm_client = llm_client
        self._api_key = api_key
    
    def _initialize_llm(self) -> None:
        if self._llm_client is None:
            import os
            from ...llm import GroqClient, LLMConfig, LLMProvider
            
            api_key = self._api_key or os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY not found")
            
            config = LLMConfig(
                provider=LLMProvider.GROQ,
                model="openai/gpt-oss-120b",
                api_key=api_key,
                temperature=0.2
            )
            self._llm_client = GroqClient(config)
    
    def _run(self, text: str, num_points: int = 5) -> str:
        """Extract key points from text."""
        if not text or not text.strip():
            return "No text provided."
        
        self._initialize_llm()
        
        try:
            prompt = f"""Extract exactly {num_points} key points from the following text.
Format each point as a bullet point starting with "• ".

Text:
{text}

Key Points:"""
            
            response = run_async_task(
                self._llm_client.simple_generate(
                    prompt=prompt,
                    system_prompt="You are a helpful assistant that extracts key information."
                )
            )
            
            return response.strip()
            
        except Exception as e:
            return f"Key point extraction failed: {str(e)}"
    
    async def _arun(self, text: str, num_points: int = 5) -> str:
        """Async version."""
        if not text:
            return "No text provided."
        
        self._initialize_llm()
        
        try:
            prompt = f"""Extract exactly {num_points} key points from the following text.
Format each point as a bullet point starting with "• ".

Text:
{text}

Key Points:"""
            
            response = await self._llm_client.simple_generate(
                prompt=prompt,
                system_prompt="You are a helpful assistant that extracts key information."
            )
            
            return response.strip()
            
        except Exception as e:
            return f"Key point extraction failed: {str(e)}"


class CompareDocumentsInput(BaseModel):
    """Input schema for document comparison."""
    text1: str = Field(
        ...,
        description="First text for comparison"
    )
    text2: str = Field(
        ...,
        description="Second text for comparison"
    )
    comparison_type: str = Field(
        default="differences",
        description="Type of comparison: 'differences', 'similarities', or 'both'"
    )


class CompareDocumentsTool(BaseTool):
    """
    Tool for comparing two text documents.
    """
    
    name: str = "compare_documents"
    description: str = """
    Compare two text documents to find similarities or differences.
    Useful for identifying conflicts or complementary information.
    """
    args_schema: Type[BaseModel] = CompareDocumentsInput
    
    _llm_client: Any = None
    _api_key: Optional[str] = None
    
    def __init__(self, llm_client: Any = None, api_key: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self._llm_client = llm_client
        self._api_key = api_key
    
    def _initialize_llm(self) -> None:
        if self._llm_client is None:
            import os
            from ...llm import GroqClient, LLMConfig, LLMProvider
            
            api_key = self._api_key or os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY not found")
            
            config = LLMConfig(
                provider=LLMProvider.GROQ,
                model="openai/gpt-oss-120b",
                api_key=api_key,
                temperature=0.2
            )
            self._llm_client = GroqClient(config)
    
    def _run(
        self,
        text1: str,
        text2: str,
        comparison_type: str = "differences"
    ) -> str:
        """Compare two documents."""
        if not text1 or not text2:
            return "Both texts are required for comparison."
        
        self._initialize_llm()
        
        try:
            type_instructions = {
                "differences": "Identify the key differences between these two texts.",
                "similarities": "Identify the key similarities between these two texts.",
                "both": "Identify both the similarities and differences between these two texts."
            }
            
            instruction = type_instructions.get(comparison_type, type_instructions["differences"])
            
            prompt = f"""{instruction}

Text 1:
{text1[:2000]}

Text 2:
{text2[:2000]}

Analysis:"""
            
            response = run_async_task(
                self._llm_client.simple_generate(
                    prompt=prompt,
                    system_prompt="You are a helpful assistant that analyzes and compares texts."
                )
            )
            
            return response.strip()
            
        except Exception as e:
            return f"Comparison failed: {str(e)}"
    
    async def _arun(self, **kwargs) -> str:
        return self._run(**kwargs)


def create_summarize_tool(
    llm_client: Any = None,
    api_key: Optional[str] = None
) -> SummarizeTool:
    """
    Factory function to create a summarize tool.
    
    Args:
        llm_client: LLM client instance
        api_key: API key
        
    Returns:
        Configured SummarizeTool
    """
    return SummarizeTool(llm_client=llm_client, api_key=api_key)

