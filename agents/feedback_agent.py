import json
import logging
from typing import Any, Dict, List, Optional

from crewai import Agent
from .utils import run_async_task

logger = logging.getLogger(__name__)


class FeedbackAgent:
    """
    Feedback agent for quality assurance.
    
    Responsibilities:
    - Evaluate generated responses for accuracy
    - Identify errors, gaps, or inconsistencies
    - Rerank retrieved results
    - Suggest improvements
    - Validate responses against source context
    """
    
    SYSTEM_PROMPT = """You are an expert Quality Assurance Specialist in a multi-agent RAG system.

Your responsibilities:
1. Evaluate generated responses for accuracy and completeness
2. Identify potential errors, gaps, or inconsistencies
3. Suggest improvements when needed
4. Validate that responses properly address the original query

When evaluating responses:
- Check factual accuracy against retrieved sources
- Verify logical consistency and reasoning
- Assess completeness relative to the query
- Identify any unsupported claims
- Evaluate clarity and readability

Provide specific, actionable feedback for improvements."""
    
    def __init__(
        self,
        llm: Optional[Any] = None,
        verbose: bool = True,
        tools: Optional[List[Any]] = None
    ):
        """
        Initialize the feedback agent.
        
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
        """Create the CrewAI feedback agent."""
        return Agent(
            role="Quality Assurance Specialist",
            goal="Evaluate, critique, and improve generated responses for accuracy and completeness",
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
                from ..llm import OpenAIClient, LLMConfig, LLMProvider
                
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OPENAI_API_KEY not found")
                
                config = LLMConfig(
                    provider=LLMProvider.OPENAI,
                    model="gpt-4o-mini",
                    api_key=api_key,
                    temperature=0.1,
                    max_tokens=2048
                )
                self.llm = OpenAIClient(config)
                logger.info("Initialized OpenAI LLM for feedback agent")
                
            except Exception as e:
                logger.error(f"Failed to initialize LLM: {e}")
                raise
    
    def validate_response(
        self,
        query: str,
        response: str,
        context: str
    ) -> Dict[str, Any]:
        """
        Validate a generated response.
        
        Args:
            query: Original user query
            response: Generated response
            context: Source context used for generation
            
        Returns:
            Validation result dictionary
        """
        self._initialize_llm()
        
        result = {
            "query": query,
            "is_valid": True,
            "overall_score": 0.0,
            "accuracy_score": 0.0,
            "completeness_score": 0.0,
            "clarity_score": 0.0,
            "issues": [],
            "suggestions": [],
            "revision_needed": False,
            "revision_instructions": None
        }
        
        try:
            validation_prompt = f"""Evaluate the following response for quality and accuracy.

Original Query: {query}

Generated Response: {response}

Source Context: {context[:3000]}

Evaluate the response on these criteria and provide scores (1-10):
1. Accuracy: Does it correctly reflect the source information?
2. Completeness: Does it fully address the query?
3. Clarity: Is it well-structured and easy to understand?

Provide your evaluation in this exact JSON format:
{{
    "overall_score": 8,
    "accuracy_score": 8,
    "completeness_score": 7,
    "clarity_score": 9,
    "issues": ["list of specific issues if any"],
    "suggestions": ["list of improvement suggestions"],
    "approved": true,
    "revision_needed": false,
    "revision_instructions": "specific instructions if revision needed or null"
}}

Respond ONLY with the JSON."""
            
            evaluation = run_async_task(
                self.llm.simple_generate(
                    prompt=validation_prompt,
                    system_prompt="You are a quality evaluator. Respond only with valid JSON."
                )
            )
            
            # Parse JSON response
            evaluation = evaluation.strip()
            if evaluation.startswith("```json"):
                evaluation = evaluation[7:]
            if evaluation.startswith("```"):
                evaluation = evaluation[3:]
            if evaluation.endswith("```"):
                evaluation = evaluation[:-3]
            
            eval_data = json.loads(evaluation.strip())
            
            result["overall_score"] = eval_data.get("overall_score", 5) / 10
            result["accuracy_score"] = eval_data.get("accuracy_score", 5) / 10
            result["completeness_score"] = eval_data.get("completeness_score", 5) / 10
            result["clarity_score"] = eval_data.get("clarity_score", 5) / 10
            result["issues"] = eval_data.get("issues", [])
            result["suggestions"] = eval_data.get("suggestions", [])
            result["is_valid"] = eval_data.get("approved", True)
            result["revision_needed"] = eval_data.get("revision_needed", False)
            result["revision_instructions"] = eval_data.get("revision_instructions")
            
            logger.info(
                f"Validation complete: score={result['overall_score']:.2f}, "
                f"valid={result['is_valid']}"
            )
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse validation JSON: {e}")
            result = self._simple_validation(query, response, context)
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            result["issues"].append(f"Validation error: {str(e)}")
        
        return result
    
    def _simple_validation(
        self,
        query: str,
        response: str,
        context: str
    ) -> Dict[str, Any]:
        """Simple fallback validation."""
        result = {
            "query": query,
            "is_valid": True,
            "overall_score": 0.6,
            "accuracy_score": 0.6,
            "completeness_score": 0.6,
            "clarity_score": 0.6,
            "issues": [],
            "suggestions": [],
            "revision_needed": False,
            "revision_instructions": None
        }
        
        # Basic checks
        if not response or len(response) < 20:
            result["issues"].append("Response is too short")
            result["is_valid"] = False
            result["overall_score"] = 0.2
        
        if not context:
            result["issues"].append("No context was provided")
            result["accuracy_score"] = 0.3
        
        # Check for refusal phrases
        refusal_phrases = ["i cannot", "i'm unable", "i don't have access"]
        if any(phrase in response.lower() for phrase in refusal_phrases):
            result["issues"].append("Response contains refusal language")
        
        return result
    
    def rerank_results(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Rerank search results by relevance.
        
        Args:
            query: Original query
            results: List of search results
            top_k: Number of results to return
            
        Returns:
            Reranked results list
        """
        if not results:
            return []
        
        if len(results) <= top_k:
            return results
        
        self._initialize_llm()
        
        try:
            # Build passages text
            passages_text = "\n\n".join([
                f"[{i}] {r.get('content', r.get('snippet', ''))[:500]}"
                for i, r in enumerate(results[:15])
            ])
            
            rerank_prompt = f"""Given the user query and a list of retrieved passages, rerank them by relevance.

User Query: {query}

Passages:
{passages_text}

Return the indices of the top {top_k} most relevant passages in order from most to least relevant.
Format: comma-separated indices, e.g., "3, 0, 5, 2, 7"

Most relevant passage indices:"""
            
            response = run_async_task(
                self.llm.simple_generate(
                    prompt=rerank_prompt,
                    system_prompt="You are a relevance ranking expert. Return only the indices."
                )
            )
            
            # Parse indices
            indices_str = response.strip().replace("[", "").replace("]", "")
            indices = [int(i.strip()) for i in indices_str.split(",") if i.strip().isdigit()]
            
            # Reorder results
            reranked = []
            seen = set()
            for idx in indices[:top_k]:
                if 0 <= idx < len(results) and idx not in seen:
                    reranked.append(results[idx])
                    seen.add(idx)
            
            # Add remaining results if needed
            for i, r in enumerate(results):
                if len(reranked) >= top_k:
                    break
                if i not in seen:
                    reranked.append(r)
            
            logger.info(f"Reranked {len(results)} results to top {len(reranked)}")
            return reranked
            
        except Exception as e:
            logger.warning(f"Reranking failed: {e}, returning original order")
            return results[:top_k]
    
    def critique_and_improve(
        self,
        query: str,
        response: str,
        context: str,
        validation: Dict[str, Any]
    ) -> str:
        """
        Critique and improve a response based on validation feedback.
        
        Args:
            query: Original query
            response: Generated response
            context: Source context
            validation: Validation results
            
        Returns:
            Improved response
        """
        if not validation.get("revision_needed", False):
            return response
        
        self._initialize_llm()
        
        try:
            issues_text = "\n".join([f"- {i}" for i in validation.get("issues", [])])
            suggestions_text = "\n".join([f"- {s}" for s in validation.get("suggestions", [])])
            
            improve_prompt = f"""Improve the following response based on the feedback provided.

Original Query: {query}

Current Response: {response}

Issues Identified:
{issues_text}

Suggestions:
{suggestions_text}

Additional Instructions: {validation.get('revision_instructions', 'Improve the response')}

Source Context for reference:
{context[:2000]}

Provide an improved response:"""
            
            improved = run_async_task(
                self.llm.simple_generate(
                    prompt=improve_prompt,
                    system_prompt="You are an expert at improving responses. Provide a better version."
                )
            )
            
            logger.info("Response improved based on feedback")
            return improved.strip()
            
        except Exception as e:
            logger.warning(f"Improvement failed: {e}")
            return response
    
    def fact_check(
        self,
        response: str,
        context: str
    ) -> Dict[str, Any]:
        """
        Check facts in response against context.
        
        Args:
            response: Response to check
            context: Source context
            
        Returns:
            Fact check results
        """
        self._initialize_llm()
        
        result = {
            "verified_claims": [],
            "unverified_claims": [],
            "contradictions": [],
            "overall_factual": True
        }
        
        try:
            check_prompt = f"""Analyze the following response for factual accuracy against the provided context.

Response to check:
{response}

Source Context:
{context[:3000]}

Identify:
1. Claims that are supported by the context
2. Claims that cannot be verified from the context
3. Any contradictions between the response and context

Format your response as:
VERIFIED: [list of verified claims]
UNVERIFIED: [list of unverified claims]
CONTRADICTIONS: [list of contradictions]"""
            
            analysis = run_async_task(
                self.llm.simple_generate(
                    prompt=check_prompt,
                    system_prompt="You are a fact-checking expert. Be thorough and accurate."
                )
            )
            
            # Parse the analysis
            lines = analysis.strip().split("\n")
            current_section = None
            
            for line in lines:
                line = line.strip()
                if line.startswith("VERIFIED:"):
                    current_section = "verified"
                    content = line[9:].strip()
                    if content:
                        result["verified_claims"].append(content)
                elif line.startswith("UNVERIFIED:"):
                    current_section = "unverified"
                    content = line[11:].strip()
                    if content:
                        result["unverified_claims"].append(content)
                elif line.startswith("CONTRADICTIONS:"):
                    current_section = "contradictions"
                    content = line[15:].strip()
                    if content:
                        result["contradictions"].append(content)
                elif line.startswith("-") or line.startswith("•"):
                    content = line[1:].strip()
                    if current_section == "verified":
                        result["verified_claims"].append(content)
                    elif current_section == "unverified":
                        result["unverified_claims"].append(content)
                    elif current_section == "contradictions":
                        result["contradictions"].append(content)
            
            result["overall_factual"] = len(result["contradictions"]) == 0
            
        except Exception as e:
            logger.warning(f"Fact check failed: {e}")
        
        return result


def create_feedback_agent(
    llm: Optional[Any] = None,
    verbose: bool = True,
    tools: Optional[List[Any]] = None
) -> FeedbackAgent:
    """
    Factory function to create a feedback agent.
    
    Args:
        llm: LLM instance
        verbose: Enable verbose output
        tools: Available tools
        
    Returns:
        Configured FeedbackAgent instance
    """
    return FeedbackAgent(llm=llm, verbose=verbose, tools=tools)

