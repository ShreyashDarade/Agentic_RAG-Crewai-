"""
Supervisor Agent - Production Grade with Advanced Planning

Features:
- Multi-step query decomposition
- Intent classification with confidence
- Dynamic execution planning
- Resource allocation optimization
- Fallback strategy generation
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from crewai import Agent
from .utils import run_async_task

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Classification of query types."""
    FACTUAL = "factual"
    ANALYTICAL = "analytical"
    CREATIVE = "creative"
    COMPARATIVE = "comparative"
    PROCEDURAL = "procedural"
    MULTI_PART = "multi_part"
    CONVERSATIONAL = "conversational"


class SearchStrategy(Enum):
    """Search strategy types."""
    DOCUMENTS_ONLY = "documents_only"
    WEB_ONLY = "web_only"
    HYBRID = "hybrid"
    MULTI_HOP = "multi_hop"


@dataclass
class QueryAnalysis:
    """Structured query analysis result."""
    original_query: str
    query_type: QueryType
    intent: str
    confidence: float
    key_entities: List[str]
    search_strategy: SearchStrategy
    search_queries: List[str]
    complexity: str  # simple, moderate, complex
    requires_reasoning: bool
    requires_synthesis: bool
    content_types_needed: List[str]  # text, table, image
    execution_plan: List[Dict[str, Any]]
    fallback_strategies: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_query": self.original_query,
            "query_type": self.query_type.value,
            "intent": self.intent,
            "confidence": self.confidence,
            "key_entities": self.key_entities,
            "search_strategy": self.search_strategy.value,
            "search_queries": self.search_queries,
            "complexity": self.complexity,
            "requires_reasoning": self.requires_reasoning,
            "requires_synthesis": self.requires_synthesis,
            "content_types_needed": self.content_types_needed,
            "execution_plan": self.execution_plan,
            "fallback_strategies": self.fallback_strategies,
            "metadata": self.metadata
        }


class SupervisorAgent:
    """
    Production-grade Supervisor Agent for query orchestration.
    
    Responsibilities:
    - Deep query analysis and intent classification
    - Multi-step query decomposition
    - Dynamic execution planning
    - Agent coordination and delegation
    - Fallback strategy management
    """
    
    SYSTEM_PROMPT = """You are an expert Query Planning Supervisor in a production-grade multi-agent RAG system.

CORE RESPONSIBILITIES:
1. ANALYZE incoming queries with deep understanding of user intent
2. CLASSIFY query types and determine optimal execution strategies
3. DECOMPOSE complex queries into manageable sub-tasks
4. PLAN execution with resource optimization
5. COORDINATE agents for maximum efficiency
6. PREPARE fallback strategies for robustness

ANALYSIS DIMENSIONS:
- Query Type: factual, analytical, creative, comparative, procedural, multi-part
- Information Sources: internal documents, web search, or hybrid
- Content Types: text, tables, images, code
- Complexity: simple (direct lookup), moderate (requires reasoning), complex (multi-hop)
- Temporal Aspects: historical, current, or predictive

DECISION CRITERIA FOR WEB SEARCH:
- Current events, news, latest updates
- Information likely not in documents (general knowledge)
- Time-sensitive queries (dates, deadlines)
- Real-time data requirements

OPTIMIZATION GOALS:
- Minimize latency while maximizing accuracy
- Efficient resource utilization
- Graceful degradation under failures

Always provide structured, actionable analysis that downstream agents can execute."""

    ANALYSIS_PROMPT_TEMPLATE = """Analyze this query and provide a comprehensive execution plan.

USER QUERY: {query}

PREVIOUS CONTEXT: {context}

Provide analysis in this exact JSON format:
{{
    "query_type": "factual|analytical|creative|comparative|procedural|multi_part",
    "intent": "Clear description of what the user wants to achieve",
    "confidence": 0.0-1.0,
    "key_entities": ["entity1", "entity2"],
    "search_strategy": "documents_only|web_only|hybrid|multi_hop",
    "search_queries": ["optimized query 1", "optimized query 2"],
    "complexity": "simple|moderate|complex",
    "requires_reasoning": true|false,
    "requires_synthesis": true|false,
    "content_types_needed": ["text", "table", "image"],
    "sub_questions": ["decomposed question 1", "decomposed question 2"],
    "execution_plan": [
        {{"step": 1, "action": "description", "agent": "retriever|generator|feedback", "priority": "high|medium|low"}}
    ],
    "fallback_strategies": ["strategy 1 if main approach fails", "strategy 2"],
    "estimated_complexity_score": 1-10
}}

IMPORTANT: Respond with ONLY valid JSON, no additional text."""

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
            llm: OpenAI LLM instance
            verbose: Enable verbose output
            allow_delegation: Allow task delegation
            tools: Tools available to the agent
        """
        self.llm = llm
        self.verbose = verbose
        self.allow_delegation = allow_delegation
        self.tools = tools or []
        self._agent: Optional[Agent] = None
        
        # Configuration
        self.max_search_queries = 5
        self.confidence_threshold = 0.7
    
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
            goal="Analyze queries comprehensively, create optimal execution plans, and coordinate agents for accurate, efficient responses",
            backstory=self.SYSTEM_PROMPT,
            verbose=self.verbose,
            allow_delegation=self.allow_delegation,
            tools=self.tools,
            llm=self.llm,
            max_iter=3,
            memory=True
        )
    
    def analyze_query(
        self,
        query: str,
        context: Optional[str] = None
    ) -> QueryAnalysis:
        """
        Perform deep query analysis.
        
        Args:
            query: User's query
            context: Optional conversation context
            
        Returns:
            Structured QueryAnalysis object
        """
        try:
            if self.llm:
                analysis_dict = self._llm_analysis(query, context)
            else:
                analysis_dict = self._rule_based_analysis(query, context)
            
            # Convert to QueryAnalysis object
            analysis = QueryAnalysis(
                original_query=query,
                query_type=QueryType(analysis_dict.get("query_type", "factual")),
                intent=analysis_dict.get("intent", "Answer user query"),
                confidence=float(analysis_dict.get("confidence", 0.8)),
                key_entities=analysis_dict.get("key_entities", []),
                search_strategy=SearchStrategy(
                    analysis_dict.get("search_strategy", "documents_only")
                ),
                search_queries=analysis_dict.get("search_queries", [query])[:self.max_search_queries],
                complexity=analysis_dict.get("complexity", "moderate"),
                requires_reasoning=analysis_dict.get("requires_reasoning", False),
                requires_synthesis=analysis_dict.get("requires_synthesis", False),
                content_types_needed=analysis_dict.get("content_types_needed", ["text"]),
                execution_plan=analysis_dict.get("execution_plan", self._default_plan()),
                fallback_strategies=analysis_dict.get("fallback_strategies", []),
                metadata={
                    "sub_questions": analysis_dict.get("sub_questions", []),
                    "complexity_score": analysis_dict.get("estimated_complexity_score", 5)
                }
            )
            
            logger.info(
                f"Query analyzed: type={analysis.query_type.value}, "
                f"complexity={analysis.complexity}, confidence={analysis.confidence:.2f}"
            )
            
            return analysis
            
        except Exception as e:
            logger.error(f"Query analysis failed: {e}")
            return self._fallback_analysis(query)
    
    def _llm_analysis(self, query: str, context: Optional[str]) -> Dict[str, Any]:
        """Perform LLM-based query analysis."""
        prompt = self.ANALYSIS_PROMPT_TEMPLATE.format(
            query=query,
            context=context or "None"
        )
        
        response = run_async_task(
            self.llm.simple_generate(
                prompt=prompt,
                system_prompt="You are a query analysis expert. Respond with valid JSON only."
            )
        )
        
        # Clean and parse JSON
        response = response.strip()
        if response.startswith("```"):
            response = re.sub(r'^```\w*\n?', '', response)
            response = re.sub(r'\n?```$', '', response)
        
        return json.loads(response.strip())
    
    def _rule_based_analysis(self, query: str, context: Optional[str]) -> Dict[str, Any]:
        """Fallback rule-based analysis."""
        query_lower = query.lower()
        
        # Determine query type
        query_type = self._classify_query_type(query_lower)
        
        # Determine search strategy
        search_strategy = self._determine_search_strategy(query_lower)
        
        # Extract entities
        entities = self._extract_entities(query)
        
        # Determine complexity
        complexity = self._assess_complexity(query, query_type)
        
        # Generate optimized queries
        search_queries = self._generate_search_queries(query, entities)
        
        # Determine content types
        content_types = self._determine_content_types(query_lower)
        
        return {
            "query_type": query_type,
            "intent": "Answer user query",
            "confidence": 0.75,
            "key_entities": entities,
            "search_strategy": search_strategy,
            "search_queries": search_queries,
            "complexity": complexity,
            "requires_reasoning": complexity != "simple",
            "requires_synthesis": query_type in ["analytical", "comparative", "multi_part"],
            "content_types_needed": content_types,
            "execution_plan": self._default_plan(),
            "fallback_strategies": ["Direct LLM response if retrieval fails"]
        }
    
    def _classify_query_type(self, query: str) -> str:
        """Classify query type based on patterns."""
        patterns = {
            "factual": ["what is", "who is", "when", "where", "define", "explain"],
            "analytical": ["why", "analyze", "evaluate", "assess", "examine"],
            "creative": ["create", "write", "generate", "design", "draft"],
            "comparative": ["compare", "contrast", "difference", "versus", "vs"],
            "procedural": ["how to", "steps", "process", "procedure", "guide"],
        }
        
        for qtype, keywords in patterns.items():
            if any(kw in query for kw in keywords):
                return qtype
        
        if " and " in query or ";" in query or "?" in query.split()[-1:]:
            return "multi_part"
        
        return "factual"
    
    def _determine_search_strategy(self, query: str) -> str:
        """Determine optimal search strategy."""
        web_indicators = [
            "latest", "recent", "current", "today", "news", "2024", "2025",
            "update", "trending", "live", "real-time", "breaking"
        ]
        
        if any(ind in query for ind in web_indicators):
            return "hybrid"
        
        multi_hop_indicators = ["relationship between", "how does", "impact of", "connection"]
        if any(ind in query for ind in multi_hop_indicators):
            return "multi_hop"
        
        return "documents_only"
    
    def _extract_entities(self, query: str) -> List[str]:
        """Extract key entities from query."""
        entities = []
        
        # Capitalized words
        entities.extend(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query))
        
        # Quoted strings
        entities.extend(re.findall(r'"([^"]+)"', query))
        entities.extend(re.findall(r"'([^']+)'", query))
        
        # Technical terms (camelCase, snake_case)
        entities.extend(re.findall(r'\b[a-z]+(?:[A-Z][a-z]+)+\b', query))
        entities.extend(re.findall(r'\b[a-z]+(?:_[a-z]+)+\b', query))
        
        return list(set(entities))[:10]
    
    def _assess_complexity(self, query: str, query_type: str) -> str:
        """Assess query complexity."""
        word_count = len(query.split())
        
        # Complex patterns
        complex_patterns = ["relationship", "impact", "multiple", "all", "every", "comprehensive"]
        has_complex = any(p in query.lower() for p in complex_patterns)
        
        if word_count > 30 or query_type == "multi_part" or has_complex:
            return "complex"
        elif word_count > 15 or query_type in ["analytical", "comparative"]:
            return "moderate"
        return "simple"
    
    def _generate_search_queries(self, query: str, entities: List[str]) -> List[str]:
        """Generate optimized search queries."""
        queries = [query]
        
        # Remove question marks for keyword search
        if "?" in query:
            queries.append(query.replace("?", "").strip())
        
        # Entity-focused queries
        for entity in entities[:3]:
            queries.append(f"{entity} information")
        
        # Remove stop words for focused search
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "be", "have", "has", "what", "how"}
        keywords = [w for w in query.lower().split() if w not in stop_words]
        if len(keywords) >= 2:
            queries.append(" ".join(keywords))
        
        return list(dict.fromkeys(queries))[:self.max_search_queries]
    
    def _determine_content_types(self, query: str) -> List[str]:
        """Determine needed content types."""
        types = ["text"]
        
        table_indicators = ["table", "data", "statistics", "numbers", "comparison", "list"]
        if any(ind in query for ind in table_indicators):
            types.append("table")
        
        image_indicators = ["image", "picture", "diagram", "chart", "graph", "visual"]
        if any(ind in query for ind in image_indicators):
            types.append("image")
        
        return types
    
    def _default_plan(self) -> List[Dict[str, Any]]:
        """Generate default execution plan."""
        return [
            {"step": 1, "action": "Retrieve relevant documents", "agent": "retriever", "priority": "high"},
            {"step": 2, "action": "Generate response", "agent": "generator", "priority": "high"},
            {"step": 3, "action": "Validate and improve", "agent": "feedback", "priority": "medium"}
        ]
    
    def _fallback_analysis(self, query: str) -> QueryAnalysis:
        """Provide fallback analysis on errors."""
        return QueryAnalysis(
            original_query=query,
            query_type=QueryType.FACTUAL,
            intent="Answer user query",
            confidence=0.5,
            key_entities=[],
            search_strategy=SearchStrategy.DOCUMENTS_ONLY,
            search_queries=[query],
            complexity="moderate",
            requires_reasoning=True,
            requires_synthesis=False,
            content_types_needed=["text"],
            execution_plan=self._default_plan(),
            fallback_strategies=["Direct LLM fallback"],
            metadata={"fallback": True}
        )
    
    def create_search_queries(self, analysis: QueryAnalysis) -> List[str]:
        """Extract search queries from analysis."""
        return analysis.search_queries
    
    def should_use_web_search(self, analysis: QueryAnalysis) -> bool:
        """Determine if web search should be used."""
        return analysis.search_strategy in [SearchStrategy.WEB_ONLY, SearchStrategy.HYBRID]
    
    def get_execution_plan(self, analysis: QueryAnalysis) -> List[Dict[str, Any]]:
        """Get execution plan from analysis."""
        return analysis.execution_plan
    
    def decompose_query(self, analysis: QueryAnalysis) -> List[str]:
        """Get sub-questions for complex queries."""
        return analysis.metadata.get("sub_questions", [analysis.original_query])


def create_supervisor_agent(
    llm: Optional[Any] = None,
    verbose: bool = True,
    tools: Optional[List[Any]] = None
) -> SupervisorAgent:
    """Factory function to create supervisor agent."""
    return SupervisorAgent(llm=llm, verbose=verbose, tools=tools)
