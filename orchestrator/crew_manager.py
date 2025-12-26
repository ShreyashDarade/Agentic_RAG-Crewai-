"""
Crew Manager - Production Grade with Advanced CrewAI Features

Features:
- OpenAI LLM integration
- Milvus Cloud vector store
- Advanced retrieval with RRF
- Hierarchical agent process
- Enhanced memory management
- Comprehensive tracing
"""

import logging
import os
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable

logger = logging.getLogger(__name__)


@dataclass
class CrewConfig:
    """Configuration for the crew manager."""
    process: str = "hierarchical"
    verbose: bool = True
    
    # Memory
    memory: bool = True
    memory_type: str = "hybrid"
    
    # Iteration limits
    max_iterations: int = 5
    max_rpm: int = 10
    
    # Agent toggles
    enable_supervisor: bool = True
    enable_retriever: bool = True
    enable_generator: bool = True
    enable_feedback: bool = True
    
    # Fallback
    fallback_enabled: bool = True
    max_retries: int = 2
    
    # Callbacks
    enable_callbacks: bool = True
    log_agent_steps: bool = True
    
    # Caching
    cache_responses: bool = True


@dataclass
class QueryResult:
    """Result of a query execution."""
    query: str
    response: str
    sources: List[str]
    confidence: float
    execution_time: float
    steps_executed: List[Dict[str, Any]]
    success: bool
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "response": self.response,
            "sources": self.sources,
            "confidence": self.confidence,
            "execution_time": self.execution_time,
            "steps_executed": self.steps_executed,
            "success": self.success,
            "error": self.error,
            "metadata": self.metadata
        }


class CrewManagerError(Exception):
    """Exception for crew manager errors."""
    pass


class AgentCallback:
    """Callback handler for agent events."""
    
    def __init__(self, log_steps: bool = True):
        self.log_steps = log_steps
        self.steps: List[Dict[str, Any]] = []
    
    def on_agent_start(self, agent_name: str, task: str) -> None:
        if self.log_steps:
            logger.info(f"Agent '{agent_name}' starting: {task[:50]}...")
        self.steps.append({
            "event": "start",
            "agent": agent_name,
            "task": task,
            "timestamp": datetime.now().isoformat()
        })
    
    def on_agent_finish(self, agent_name: str, result: Any) -> None:
        if self.log_steps:
            logger.info(f"Agent '{agent_name}' finished")
        self.steps.append({
            "event": "finish",
            "agent": agent_name,
            "result_length": len(str(result)) if result else 0,
            "timestamp": datetime.now().isoformat()
        })
    
    def on_agent_error(self, agent_name: str, error: Exception) -> None:
        logger.error(f"Agent '{agent_name}' error: {error}")
        self.steps.append({
            "event": "error",
            "agent": agent_name,
            "error": str(error),
            "timestamp": datetime.now().isoformat()
        })
    
    def get_steps(self) -> List[Dict[str, Any]]:
        return self.steps
    
    def clear(self) -> None:
        self.steps = []


class CrewManager:
    """
    Production-grade Crew Manager.
    
    Features:
    - OpenAI GPT-4o-mini LLM
    - Milvus Cloud vector store
    - Advanced retrieval with RRF fusion
    - Multi-agent orchestration
    """
    
    def __init__(self, config: Optional[CrewConfig] = None):
        self.config = config or CrewConfig()
        self._ensure_crewai_settings()
        
        # Agents
        self._supervisor = None
        self._retriever = None
        self._generator = None
        self._feedback = None
        
        # Components
        self._llm = None
        self._advanced_retriever = None
        self._memory_store = None
        self._trace_logger = None
        self._callback = None
        
        self._initialized = False
    
    def _ensure_crewai_settings(self) -> None:
        """Ensure CrewAI settings file exists."""
        existing_path = os.environ.get("CREWAI_SETTINGS_PATH")
        if existing_path:
            return
        
        default_path = Path(__file__).resolve().parent.parent / "config" / "crew_settings.json"
        default_path.parent.mkdir(parents=True, exist_ok=True)
        if not default_path.exists():
            default_path.write_text("{}\n", encoding="utf-8")
        
        os.environ["CREWAI_SETTINGS_PATH"] = str(default_path)
    
    def initialize(self) -> None:
        """Initialize all components."""
        if self._initialized:
            return
        
        logger.info("Initializing crew manager...")
        
        try:
            if self.config.enable_callbacks:
                self._callback = AgentCallback(self.config.log_agent_steps)
            
            self._initialize_llm()
            self._initialize_retrievers()
            self._initialize_agents()
            self._initialize_support()
            
            self._initialized = True
            logger.info("Crew manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Crew manager initialization failed: {e}")
            raise CrewManagerError(f"Initialization failed: {str(e)}")
    
    def _initialize_llm(self) -> None:
        """Initialize OpenAI LLM client."""
        from llm import OpenAIClient, LLMConfig, LLMProvider
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        config = LLMConfig(
            provider=LLMProvider.OPENAI,
            model="gpt-4o-mini",
            api_key=api_key,
            temperature=0.2,
            max_tokens=4096
        )
        self._llm = OpenAIClient(config)
        logger.info("OpenAI LLM initialized")
    
    def _initialize_retrievers(self) -> None:
        """Initialize Milvus-based retriever."""
        try:
            from retriever import AdvancedRetriever, AdvancedRetrieverConfig
            
            config = AdvancedRetrieverConfig(
                collection_name="documents",
                enable_rerank=True,
                enable_diversity=True,
                enable_bm25=True,
                fusion_method="rrf"
            )
            self._advanced_retriever = AdvancedRetriever(config)
            logger.info("Advanced retriever initialized")
            
        except Exception as e:
            logger.error(f"Retriever initialization failed: {e}")
            raise
    
    def _initialize_agents(self) -> None:
        """Initialize agent instances."""
        from agents import SupervisorAgent, RetrieverAgent, GeneratorAgent, FeedbackAgent
        
        if self.config.enable_supervisor:
            self._supervisor = SupervisorAgent(
                llm=self._llm,
                verbose=self.config.verbose
            )
        
        if self.config.enable_retriever:
            self._retriever = RetrieverAgent(
                llm=self._llm,
                verbose=self.config.verbose,
                hybrid_retriever=self._advanced_retriever
            )
        
        if self.config.enable_generator:
            self._generator = GeneratorAgent(
                llm=self._llm,
                verbose=self.config.verbose
            )
        
        if self.config.enable_feedback:
            self._feedback = FeedbackAgent(
                llm=self._llm,
                verbose=self.config.verbose
            )
        
        logger.info("Agents initialized")
    
    def _initialize_support(self) -> None:
        """Initialize support components."""
        try:
            from .memory_store import MemoryStore
            from .trace_logger import TraceLogger
            
            if self.config.memory:
                self._memory_store = MemoryStore()
            
            self._trace_logger = TraceLogger()
        except ImportError as e:
            logger.warning(f"Support components not available: {e}")
    
    def execute_query(
        self,
        query: str,
        context: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> QueryResult:
        """
        Execute a query through the multi-agent pipeline.
        
        Args:
            query: User query
            context: Optional context
            metadata: Additional metadata
            
        Returns:
            QueryResult with response and details
        """
        if not self._initialized:
            self.initialize()
        
        start_time = datetime.now()
        steps_executed = []
        
        if self._callback:
            self._callback.clear()
        
        result = QueryResult(
            query=query,
            response="",
            sources=[],
            confidence=0.0,
            execution_time=0.0,
            steps_executed=[],
            success=False,
            metadata=metadata or {}
        )
        
        try:
            trace_id = None
            if self._trace_logger:
                trace_id = self._trace_logger.start_trace(query)
            
            # Step 1: Query Analysis
            analysis = None
            if self._supervisor:
                if self._callback:
                    self._callback.on_agent_start("supervisor", "analyze_query")
                
                analysis = self._supervisor.analyze_query(query, context)
                
                if self._callback:
                    self._callback.on_agent_finish("supervisor", analysis)
                
                steps_executed.append({
                    "step": 1,
                    "agent": "supervisor",
                    "action": "analyze_query",
                    "result": {
                        "query_type": analysis.query_type.value if hasattr(analysis, 'query_type') else "unknown",
                        "complexity": getattr(analysis, 'complexity', 'moderate')
                    }
                })
            
            # Step 2: Information Retrieval
            retrieval_result = None
            if self._retriever:
                if self._callback:
                    self._callback.on_agent_start("retriever", "retrieve")
                
                use_web = False
                search_queries = [query]
                
                if analysis:
                    use_web = self._supervisor.should_use_web_search(analysis)
                    search_queries = self._supervisor.create_search_queries(analysis)
                    content_types = getattr(analysis, 'content_types_needed', ['text'])
                else:
                    content_types = ['text']
                
                retrieval_result = self._retriever.retrieve(
                    query=query,
                    search_queries=search_queries,
                    use_documents=True,
                    use_web=use_web,
                    content_types=content_types
                )
                
                if self._callback:
                    self._callback.on_agent_finish("retriever", retrieval_result)
                
                steps_executed.append({
                    "step": 2,
                    "agent": "retriever",
                    "action": "retrieve",
                    "result": {
                        "local_count": len(retrieval_result.local_results),
                        "web_count": len(retrieval_result.web_results),
                        "total": retrieval_result.total_results
                    }
                })
                
                result.sources = retrieval_result.sources
            
            # Step 3: Answer Generation
            generation_result = None
            if self._generator and retrieval_result:
                if self._callback:
                    self._callback.on_agent_start("generator", "generate")
                
                context_text = retrieval_result.combined_context
                
                generation_result = self._generator.generate(
                    query=query,
                    context=context_text,
                    analysis=analysis.to_dict() if hasattr(analysis, 'to_dict') else analysis,
                    sources=result.sources
                )
                
                if self._callback:
                    self._callback.on_agent_finish("generator", generation_result)
                
                steps_executed.append({
                    "step": 3,
                    "agent": "generator",
                    "action": "generate",
                    "result": {
                        "response_length": len(generation_result.get("response", "")),
                        "confidence": generation_result.get("confidence", 0)
                    }
                })
                
                result.response = generation_result.get("response", "")
                result.confidence = generation_result.get("confidence", 0)
            
            # Step 4: Response Validation
            if self._feedback and generation_result:
                if self._callback:
                    self._callback.on_agent_start("feedback", "validate")
                
                validation = self._feedback.validate_response(
                    query=query,
                    response=result.response,
                    context=retrieval_result.combined_context if retrieval_result else ""
                )
                
                if self._callback:
                    self._callback.on_agent_finish("feedback", validation)
                
                steps_executed.append({
                    "step": 4,
                    "agent": "feedback",
                    "action": "validate",
                    "result": {
                        "is_valid": validation.get("is_valid"),
                        "score": validation.get("overall_score")
                    }
                })
                
                # Improve if needed
                if validation.get("revision_needed"):
                    improved = self._feedback.critique_and_improve(
                        query=query,
                        response=result.response,
                        context=retrieval_result.combined_context if retrieval_result else "",
                        validation=validation
                    )
                    result.response = improved
                    steps_executed.append({
                        "step": 5,
                        "agent": "feedback",
                        "action": "improve",
                        "result": {"improved": True}
                    })
                
                result.confidence = validation.get("overall_score", result.confidence)
            
            # Store in memory
            if self._memory_store and retrieval_result:
                self._memory_store.store(
                    query=query,
                    response=result.response,
                    context=retrieval_result.combined_context,
                    metadata={"trace_id": trace_id}
                )
            
            # Complete trace
            if self._trace_logger and trace_id:
                self._trace_logger.end_trace(trace_id, success=True)
            
            result.success = True
            result.steps_executed = steps_executed
            result.execution_time = (datetime.now() - start_time).total_seconds()
            
            if trace_id:
                result.metadata["trace_id"] = trace_id
            
            if self._callback:
                result.metadata["agent_steps"] = self._callback.get_steps()
            
            logger.info(
                f"Query executed in {result.execution_time:.2f}s "
                f"(confidence: {result.confidence:.2f})"
            )
            
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            
            if self._callback:
                self._callback.on_agent_error("pipeline", e)
            
            result.error = str(e)
            result.success = False
            result.execution_time = (datetime.now() - start_time).total_seconds()
            
            if self.config.fallback_enabled:
                fallback = self._execute_fallback(query)
                if fallback:
                    result.response = fallback
                    result.success = True
                    result.metadata["fallback_used"] = True
        
        return result
    
    def _execute_fallback(self, query: str) -> Optional[str]:
        """Execute fallback strategy."""
        logger.info("Executing fallback...")
        
        try:
            import asyncio
            
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            response = loop.run_until_complete(
                self._llm.simple_generate(
                    prompt=f"Please answer: {query}",
                    system_prompt="You are a helpful assistant."
                )
            )
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Fallback failed: {e}")
            return None
    
    def get_execution_trace(self, trace_id: str) -> Dict[str, Any]:
        if self._trace_logger:
            return self._trace_logger.get_trace(trace_id)
        return {}
    
    def get_conversation_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        if self._memory_store:
            return self._memory_store.get_recent(limit)
        return []
    
    def clear_memory(self) -> None:
        if self._memory_store:
            self._memory_store.clear()
    
    @property
    def is_initialized(self) -> bool:
        return self._initialized


def create_crew_manager(**kwargs) -> CrewManager:
    """Factory function."""
    config = CrewConfig(**kwargs)
    return CrewManager(config)
