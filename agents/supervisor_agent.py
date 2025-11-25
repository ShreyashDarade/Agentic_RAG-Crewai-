import json
import logging
from typing import Any, Dict, List, Optional

from crewai import Agent
from .utils import run_async_task

logger = logging.getLogger(__name__)


class SupervisorAgent:
    """
    Supervisor agent for query analysis and task planning.
    
    Responsibilities:
    - Analyze incoming queries
    - Determine optimal execution strategy
    - Select appropriate tools and agents
    - Coordinate workflow execution
    """
    
    SYSTEM_PROMPT = """You are an expert Query Planning Supervisor in a multi-agent RAG system.

Your responsibilities:
1. Analyze incoming user queries to understand intent and requirements
2. Determine if the query needs document retrieval, web search, or direct response
3. Plan the execution strategy for other agents
4. Coordinate the flow of information between agents

When analyzing a query, consider:
- Query type: factual, analytical, creative, or multi-part
- Information sources needed: internal documents, web search, or both
- Complexity level: simple lookup vs. complex reasoning
- Any specific requirements or constraints mentioned

Output your analysis in a structured format that other agents can use.
Always be thorough but efficient in your planning."""
    
    def __init__(
        self,
        llm: Optional[Any] = None,
        verbose: bool = True,
        allow_delegation: bool = True,
        tools: Optional[List[Any]] = None
    ):
        """
        Initialize the supervisor agent.
        
        Args:
            llm: LLM instance to use
            verbose: Enable verbose output
            allow_delegation: Allow task delegation
            tools: List of tools available to the agent
        """
        self.llm = llm
        self.verbose = verbose
        self.allow_delegation = allow_delegation
        self.tools = tools or []
        self._agent: Optional[Agent] = None
    
    @property
    def agent(self) -> Agent:
        """Get or create the CrewAI agent."""
        if self._agent is None:
            self._agent = self._create_agent()
        return self._agent
    
    def _create_agent(self) -> Agent:
        """Create the CrewAI supervisor agent."""
        return Agent(
            role="Query Planning Supervisor",
            goal="Analyze user queries, create execution plans, and coordinate other agents to provide comprehensive answers",
            backstory=self.SYSTEM_PROMPT,
            verbose=self.verbose,
            allow_delegation=self.allow_delegation,
            tools=self.tools,
            llm=self.llm
        )
    
    def analyze_query(self, query: str, context: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze a user query and create an execution plan.
        
        Args:
            query: The user's query
            context: Optional context from previous interactions
            
        Returns:
            Analysis and execution plan dictionary
        """
        analysis_prompt = f"""Analyze the following user query and provide a structured execution plan.

User Query: {query}

Previous Context: {context if context else "None"}

Provide your analysis in the following JSON format:
{{
    "query_type": "factual|analytical|creative|multi_part",
    "intent": "brief description of user intent",
    "key_entities": ["list", "of", "key", "entities"],
    "search_strategy": {{
        "use_documents": true|false,
        "use_web_search": true|false,
        "search_queries": ["optimized", "search", "queries"]
    }},
    "complexity": "simple|moderate|complex",
    "special_requirements": ["any", "special", "requirements"],
    "execution_plan": [
        {{"step": 1, "action": "description", "agent": "agent_name"}}
    ]
}}

Respond ONLY with the JSON, no additional text."""
        
        try:
            # Use the agent to analyze
            if self.llm:
                import asyncio
                
                response = run_async_task(
                    self.llm.simple_generate(
                        prompt=analysis_prompt,
                        system_prompt="You are a query analysis expert. Respond only with valid JSON."
                    )
                )
                
                # Parse JSON response
                response = response.strip()
                if response.startswith("```json"):
                    response = response[7:]
                if response.startswith("```"):
                    response = response[3:]
                if response.endswith("```"):
                    response = response[:-3]
                
                analysis = json.loads(response.strip())
                
                logger.info(f"Query analyzed: {analysis.get('query_type')} - {analysis.get('complexity')}")
                return analysis
            else:
                # Fallback to simple analysis
                return self._simple_analysis(query)
                
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse analysis JSON: {e}")
            return self._simple_analysis(query)
        except Exception as e:
            logger.error(f"Query analysis failed: {e}")
            return self._simple_analysis(query)
    
    def _simple_analysis(self, query: str) -> Dict[str, Any]:
        """
        Simple fallback query analysis.
        
        Args:
            query: The user's query
            
        Returns:
            Basic analysis dictionary
        """
        # Simple keyword-based analysis
        query_lower = query.lower()
        
        # Determine query type
        if any(w in query_lower for w in ["what is", "define", "explain"]):
            query_type = "factual"
        elif any(w in query_lower for w in ["compare", "analyze", "evaluate"]):
            query_type = "analytical"
        elif any(w in query_lower for w in ["create", "write", "generate"]):
            query_type = "creative"
        elif " and " in query_lower or ";" in query:
            query_type = "multi_part"
        else:
            query_type = "factual"
        
        # Determine if web search is needed
        needs_web = any(w in query_lower for w in [
            "latest", "recent", "current", "today", "news", "2024", "2025"
        ])
        
        # Determine complexity
        word_count = len(query.split())
        if word_count > 30 or query_type == "multi_part":
            complexity = "complex"
        elif word_count > 15:
            complexity = "moderate"
        else:
            complexity = "simple"
        
        return {
            "query_type": query_type,
            "intent": "Answer user query",
            "key_entities": self._extract_entities(query),
            "search_strategy": {
                "use_documents": True,
                "use_web_search": needs_web,
                "search_queries": [query]
            },
            "complexity": complexity,
            "special_requirements": [],
            "execution_plan": [
                {"step": 1, "action": "Search documents", "agent": "retriever"},
                {"step": 2, "action": "Generate response", "agent": "generator"},
                {"step": 3, "action": "Validate response", "agent": "feedback"}
            ]
        }
    
    def _extract_entities(self, query: str) -> List[str]:
        """Extract potential entities from query."""
        import re
        
        # Find capitalized words (potential entities)
        entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query)
        
        # Find quoted strings
        quoted = re.findall(r'"([^"]+)"', query)
        entities.extend(quoted)
        
        return list(set(entities))[:5]  # Limit to 5 entities
    
    def create_search_queries(self, analysis: Dict[str, Any]) -> List[str]:
        """
        Create optimized search queries from analysis.
        
        Args:
            analysis: Query analysis dictionary
            
        Returns:
            List of search queries
        """
        queries = []
        
        # Use search queries from analysis
        if "search_strategy" in analysis:
            queries.extend(analysis["search_strategy"].get("search_queries", []))
        
        # Add entity-based queries
        if "key_entities" in analysis:
            for entity in analysis["key_entities"]:
                queries.append(f"information about {entity}")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_queries = []
        for q in queries:
            if q.lower() not in seen:
                seen.add(q.lower())
                unique_queries.append(q)
        
        return unique_queries
    
    def should_use_web_search(self, analysis: Dict[str, Any]) -> bool:
        """
        Determine if web search should be used.
        
        Args:
            analysis: Query analysis dictionary
            
        Returns:
            True if web search should be used
        """
        return analysis.get("search_strategy", {}).get("use_web_search", False)
    
    def get_execution_plan(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Get the execution plan from analysis.
        
        Args:
            analysis: Query analysis dictionary
            
        Returns:
            List of execution steps
        """
        return analysis.get("execution_plan", [
            {"step": 1, "action": "Search", "agent": "retriever"},
            {"step": 2, "action": "Generate", "agent": "generator"}
        ])


def create_supervisor_agent(
    llm: Optional[Any] = None,
    verbose: bool = True,
    tools: Optional[List[Any]] = None
) -> SupervisorAgent:
    """
    Factory function to create a supervisor agent.
    
    Args:
        llm: LLM instance
        verbose: Enable verbose output
        tools: Available tools
        
    Returns:
        Configured SupervisorAgent instance
    """
    return SupervisorAgent(llm=llm, verbose=verbose, tools=tools)

