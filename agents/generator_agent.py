import logging
from typing import Any, Dict, List, Optional

from crewai import Agent
from .utils import run_async_task

logger = logging.getLogger(__name__)


class GeneratorAgent:
    """
    Generator agent for answer synthesis.
    
    Responsibilities:
    - Synthesize retrieved information into coherent responses
    - Structure responses clearly and logically
    - Cite sources appropriately
    - Handle incomplete or conflicting information
    """
    
    SYSTEM_PROMPT = """You are an expert Answer Generator in a multi-agent RAG system.

Your responsibilities:
1. Synthesize retrieved information into clear, accurate, and comprehensive answers
2. Structure responses clearly and logically
3. Cite sources appropriately
4. Handle cases where information is incomplete or conflicting

When generating responses:
- Base your answer primarily on the retrieved context
- Clearly distinguish between facts from sources and your reasoning
- If information is insufficient, acknowledge limitations honestly
- Format responses appropriately (lists, paragraphs, code blocks, etc.)
- Be concise but comprehensive

Never fabricate information not present in the context."""
    
    def __init__(
        self,
        llm: Optional[Any] = None,
        verbose: bool = True,
        tools: Optional[List[Any]] = None
    ):
        """
        Initialize the generator agent.
        
        Args:
            llm: LLM instance
            verbose: Enable verbose output
            tools: List of tools
        """
        self.llm = llm
        self.verbose = verbose
        self.tools = tools or []
        self._agent: Optional[Agent] = None
    
    @property
    def agent(self) -> Agent:
        """Get or create the CrewAI agent."""
        if self._agent is None:
            self._agent = self._create_agent()
        return self._agent
    
    def _create_agent(self) -> Agent:
        """Create the CrewAI generator agent."""
        return Agent(
            role="Answer Generator",
            goal="Synthesize retrieved information into clear, accurate, and comprehensive answers",
            backstory=self.SYSTEM_PROMPT,
            verbose=self.verbose,
            allow_delegation=False,
            tools=self.tools,
            llm=self.llm
        )
    
    def _initialize_llm(self) -> None:
        """Initialize LLM if not provided."""
        if self.llm is None:
            try:
                import os
                from ..llm import GroqClient, LLMConfig, LLMProvider
                
                api_key = os.getenv("GROQ_API_KEY")
                if not api_key:
                    raise ValueError("GROQ_API_KEY not found")
                
                config = LLMConfig(
                    provider=LLMProvider.GROQ,
                    model="openai/gpt-oss-120b",
                    api_key=api_key,
                    temperature=0.2,
                    max_tokens=2048
                )
                self.llm = GroqClient(config)
                logger.info("Initialized LLM for generator agent")
                
            except Exception as e:
                logger.error(f"Failed to initialize LLM: {e}")
                raise
    
    def generate(
        self,
        query: str,
        context: str,
        analysis: Optional[Dict[str, Any]] = None,
        sources: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate an answer for the query.
        
        Args:
            query: User's question
            context: Retrieved context
            analysis: Query analysis from supervisor
            sources: List of source references
            
        Returns:
            Generated response dictionary
        """
        self._initialize_llm()
        
        result = {
            "query": query,
            "response": "",
            "sources_used": sources or [],
            "confidence": 0.0,
            "success": True,
            "error": None
        }
        
        try:
            # Build generation prompt
            prompt = self._build_generation_prompt(query, context, analysis)
            
            response = run_async_task(
                self.llm.simple_generate(
                    prompt=prompt,
                    system_prompt=self.SYSTEM_PROMPT
                )
            )
            
            result["response"] = response.strip()
            result["confidence"] = self._estimate_confidence(context, response)
            
            logger.info(f"Generated response ({len(response)} chars)")
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            result["success"] = False
            result["error"] = str(e)
            result["response"] = f"I apologize, but I encountered an error while generating the response: {str(e)}"
        
        return result
    
    def _build_generation_prompt(
        self,
        query: str,
        context: str,
        analysis: Optional[Dict[str, Any]] = None
    ) -> str:
        """Build the generation prompt."""
        prompt_parts = [
            "Based on the following context, answer the user's question accurately and comprehensively.",
            "",
            "=== User Question ===",
            query,
            ""
        ]
        
        if analysis:
            prompt_parts.extend([
                "=== Query Analysis ===",
                f"Type: {analysis.get('query_type', 'unknown')}",
                f"Intent: {analysis.get('intent', 'Answer question')}",
                f"Complexity: {analysis.get('complexity', 'moderate')}",
                ""
            ])
        
        prompt_parts.extend([
            "=== Retrieved Context ===",
            context if context else "No context available.",
            "",
            "=== Instructions ===",
            "1. Answer based on the provided context",
            "2. If the context doesn't contain enough information, say so clearly",
            "3. Cite sources when making specific claims (use [1], [2], etc.)",
            "4. Structure your response clearly",
            "5. Be concise but complete",
            "",
            "=== Your Response ==="
        ])
        
        return "\n".join(prompt_parts)
    
    def _estimate_confidence(self, context: str, response: str) -> float:
        """
        Estimate confidence in the generated response.
        
        Args:
            context: Retrieved context
            response: Generated response
            
        Returns:
            Confidence score (0-1)
        """
        if not context or not response:
            return 0.0
        
        # Simple heuristic-based confidence estimation
        confidence = 0.5  # Base confidence
        
        # More context = higher confidence
        context_length = len(context)
        if context_length > 1000:
            confidence += 0.2
        elif context_length > 500:
            confidence += 0.1
        
        # Response length relative to context
        if len(response) < context_length * 0.5:
            confidence += 0.1  # Concise responses are often more confident
        
        # Check for uncertainty phrases
        uncertainty_phrases = [
            "i'm not sure", "unclear", "might be", "possibly",
            "don't have enough", "cannot determine", "no information"
        ]
        response_lower = response.lower()
        for phrase in uncertainty_phrases:
            if phrase in response_lower:
                confidence -= 0.15
        
        return max(0.0, min(1.0, confidence))
    
    def generate_with_citations(
        self,
        query: str,
        context: str,
        sources: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate a response with proper source citations.
        
        Args:
            query: User's question
            context: Retrieved context
            sources: Source information for citations
            
        Returns:
            Response with citations
        """
        self._initialize_llm()
        
        # Build source reference
        source_refs = "\n".join([
            f"[{i+1}] {s.get('title', s.get('source', 'Unknown'))}"
            for i, s in enumerate(sources[:10])
        ])
        
        prompt = f"""Answer the following question based on the provided context.
Include source citations using [1], [2], etc. format when making specific claims.

Question: {query}

Context:
{context}

Available Sources:
{source_refs}

Provide a well-structured answer with citations:"""
        
        try:
            response = run_async_task(
                self.llm.simple_generate(
                    prompt=prompt,
                    system_prompt=self.SYSTEM_PROMPT
                )
            )
            
            return {
                "response": response.strip(),
                "sources": sources,
                "success": True
            }
            
        except Exception as e:
            return {
                "response": f"Error generating response: {str(e)}",
                "sources": [],
                "success": False,
                "error": str(e)
            }
    
    def summarize_context(self, context: str, max_length: int = 500) -> str:
        """
        Summarize long context before generation.
        
        Args:
            context: Context to summarize
            max_length: Maximum summary length
            
        Returns:
            Summarized context
        """
        if len(context) <= max_length * 4:
            return context
        
        self._initialize_llm()
        
        try:
            import asyncio
            
            prompt = f"""Summarize the following information, keeping the most important facts:

{context[:8000]}

Summary (max {max_length} words):"""
            
            summary = run_async_task(
                self.llm.simple_generate(
                    prompt=prompt,
                    system_prompt="You are a helpful assistant that creates concise summaries."
                )
            )
            
            return summary.strip()
            
        except Exception as e:
            logger.warning(f"Summarization failed: {e}")
            return context[:max_length * 4] + "..."


def create_generator_agent(
    llm: Optional[Any] = None,
    verbose: bool = True,
    tools: Optional[List[Any]] = None
) -> GeneratorAgent:
    """
    Factory function to create a generator agent.
    
    Args:
        llm: LLM instance
        verbose: Enable verbose output
        tools: Available tools
        
    Returns:
        Configured GeneratorAgent instance
    """
    return GeneratorAgent(llm=llm, verbose=verbose, tools=tools)

