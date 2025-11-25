import logging
import os
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class CrewConfig:
    """Configuration for the crew manager."""
    process: str = "sequential"  # sequential, hierarchical
    verbose: bool = True
    memory: bool = True
    max_iterations: int = 5
    max_rpm: int = 10
    
    # Component settings
    enable_supervisor: bool = True
    enable_retriever: bool = True
    enable_generator: bool = True
    enable_feedback: bool = True
    
    # Fallback settings
    fallback_enabled: bool = True
    max_retries: int = 2


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
    """Exception raised for crew manager errors."""
    pass


class CrewManager:
    """
    Manager for coordinating multi-agent workflows.
    
    Handles:
    - Agent instantiation and configuration
    - Workflow execution
    - Error handling and fallbacks
    - Result aggregation
    """
    
    def __init__(self, config: Optional[CrewConfig] = None):
        """
        Initialize the crew manager.
        
        Args:
            config: Crew configuration
        """
        self.config = config or CrewConfig()
        self._ensure_crewai_settings()
        
        # Agent instances
        self._supervisor = None
        self._retriever = None
        self._generator = None
        self._feedback = None
        
        # Shared components
        self._llm = None
        self._hybrid_retriever = None
        self._memory_store = None
        self._trace_logger = None
        
        self._initialized = False
    
    def initialize(self) -> None:
        """Initialize all components."""
        if self._initialized:
            return
        
        logger.info("Initializing crew manager...")
        
        try:
            # Initialize LLM
            self._initialize_llm()
            
            # Initialize retrievers
            self._initialize_retrievers()
            
            # Initialize agents
            self._initialize_agents()
            
            # Initialize memory and trace
            self._initialize_support()
            
            self._initialized = True
            logger.info("Crew manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize crew manager: {e}")
            raise CrewManagerError(f"Initialization failed: {str(e)}")
    
    def _initialize_llm(self) -> None:
        """Initialize LLM client."""
        try:
            from llm import GroqClient, LLMConfig, LLMProvider
            
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY environment variable not set")
            
            config = LLMConfig(
                provider=LLMProvider.GROQ,
                model="openai/gpt-oss-120b",
                api_key=api_key,
                temperature=0.2,
                max_tokens=4096
            )
            self._llm = GroqClient(config)
            logger.info("LLM initialized")
            
        except Exception as e:
            logger.error(f"LLM initialization failed: {e}")
            raise

    def _ensure_crewai_settings(self) -> None:
        """Ensure CrewAI settings file exists to prevent noisy CLI warnings."""
        existing_path = os.environ.get("CREWAI_SETTINGS_PATH")
        if existing_path:
            return

        default_path = Path(__file__).resolve().parent.parent / "config" / "crew_settings.json"
        default_path.parent.mkdir(parents=True, exist_ok=True)
        if not default_path.exists():
            default_path.write_text("{}\n", encoding="utf-8")

        os.environ["CREWAI_SETTINGS_PATH"] = str(default_path)
    
    def _initialize_retrievers(self) -> None:
        """Initialize retrieval components."""
        try:
            from retriever import HybridRetriever
            
            self._hybrid_retriever = HybridRetriever()
            logger.info("Hybrid retriever initialized")
            
        except Exception as e:
            logger.warning(f"Hybrid retriever initialization failed: {e}")
    
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
                hybrid_retriever=self._hybrid_retriever
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
        from .memory_store import MemoryStore
        from .trace_logger import TraceLogger
        
        if self.config.memory:
            self._memory_store = MemoryStore()
        
        self._trace_logger = TraceLogger()
    
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
            context: Optional context from previous interactions
            metadata: Additional metadata
            
        Returns:
            QueryResult with response and execution details
        """
        if not self._initialized:
            self.initialize()
        
        start_time = datetime.now()
        steps_executed = []
        
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
            # Start trace
            trace_id = self._trace_logger.start_trace(query)
            
            # Step 1: Query Analysis (Supervisor)
            analysis = None
            if self._supervisor:
                self._trace_logger.log_step(trace_id, "supervisor", "analyze_query", "started")
                analysis = self._supervisor.analyze_query(query, context)
                self._trace_logger.log_step(trace_id, "supervisor", "analyze_query", "completed", analysis)
                
                steps_executed.append({
                    "step": 1,
                    "agent": "supervisor",
                    "action": "analyze_query",
                    "result": analysis
                })
            
            # Step 2: Information Retrieval
            retrieval_result = None
            if self._retriever:
                self._trace_logger.log_step(trace_id, "retriever", "retrieve", "started")
                
                use_web = self._supervisor.should_use_web_search(analysis) if analysis else False
                search_queries = self._supervisor.create_search_queries(analysis) if analysis else [query]
                
                retrieval_result = self._retriever.retrieve(
                    query=query,
                    search_queries=search_queries,
                    use_documents=True,
                    use_web=use_web
                )
                
                self._trace_logger.log_step(trace_id, "retriever", "retrieve", "completed", {
                    "local_count": len(retrieval_result.get("local_results", [])),
                    "web_count": len(retrieval_result.get("web_results", []))
                })
                
                steps_executed.append({
                    "step": 2,
                    "agent": "retriever",
                    "action": "retrieve",
                    "result": {
                        "local_results": len(retrieval_result.get("local_results", [])),
                        "web_results": len(retrieval_result.get("web_results", []))
                    }
                })
                
                result.sources = retrieval_result.get("sources", [])
            
            # Step 3: Answer Generation
            generation_result = None
            if self._generator and retrieval_result:
                self._trace_logger.log_step(trace_id, "generator", "generate", "started")
                
                context_text = retrieval_result.get("combined_context", "")
                
                generation_result = self._generator.generate(
                    query=query,
                    context=context_text,
                    analysis=analysis,
                    sources=result.sources
                )
                
                self._trace_logger.log_step(trace_id, "generator", "generate", "completed", {
                    "response_length": len(generation_result.get("response", "")),
                    "confidence": generation_result.get("confidence", 0)
                })
                
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
            
            # Step 4: Response Validation and Improvement
            if self._feedback and generation_result:
                self._trace_logger.log_step(trace_id, "feedback", "validate", "started")
                
                validation = self._feedback.validate_response(
                    query=query,
                    response=result.response,
                    context=retrieval_result.get("combined_context", "")
                )
                
                self._trace_logger.log_step(trace_id, "feedback", "validate", "completed", {
                    "is_valid": validation.get("is_valid"),
                    "score": validation.get("overall_score")
                })
                
                steps_executed.append({
                    "step": 4,
                    "agent": "feedback",
                    "action": "validate",
                    "result": {
                        "is_valid": validation.get("is_valid"),
                        "score": validation.get("overall_score"),
                        "issues": validation.get("issues", [])
                    }
                })
                
                # Improve if needed
                if validation.get("revision_needed"):
                    self._trace_logger.log_step(trace_id, "feedback", "improve", "started")
                    
                    improved_response = self._feedback.critique_and_improve(
                        query=query,
                        response=result.response,
                        context=retrieval_result.get("combined_context", ""),
                        validation=validation
                    )
                    
                    result.response = improved_response
                    
                    self._trace_logger.log_step(trace_id, "feedback", "improve", "completed")
                    
                    steps_executed.append({
                        "step": 5,
                        "agent": "feedback",
                        "action": "improve",
                        "result": {"improved": True}
                    })
                
                # Update confidence based on validation
                result.confidence = validation.get("overall_score", result.confidence)
            
            # Store in memory
            if self._memory_store:
                self._memory_store.store(
                    query=query,
                    response=result.response,
                    context=retrieval_result.get("combined_context", "") if retrieval_result else "",
                    metadata={"trace_id": trace_id}
                )
            
            # Complete trace
            self._trace_logger.end_trace(trace_id, success=True)
            
            result.success = True
            result.steps_executed = steps_executed
            result.execution_time = (datetime.now() - start_time).total_seconds()
            result.metadata["trace_id"] = trace_id
            
            logger.info(
                f"Query executed successfully in {result.execution_time:.2f}s "
                f"(confidence: {result.confidence:.2f})"
            )
            
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            result.error = str(e)
            result.success = False
            result.execution_time = (datetime.now() - start_time).total_seconds()
            
            # Attempt fallback
            if self.config.fallback_enabled:
                fallback_result = self._execute_fallback(query)
                if fallback_result:
                    result.response = fallback_result
                    result.success = True
                    result.metadata["fallback_used"] = True
        
        return result
    
    def _execute_fallback(self, query: str) -> Optional[str]:
        """
        Execute fallback strategy when main pipeline fails.
        
        Args:
            query: Original query
            
        Returns:
            Fallback response or None
        """
        logger.info("Executing fallback strategy...")
        
        try:
            import asyncio
            
            # Simple direct LLM response
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            response = loop.run_until_complete(
                self._llm.simple_generate(
                    prompt=f"Please answer the following question to the best of your ability:\n\n{query}",
                    system_prompt="You are a helpful assistant. If you don't have enough information, say so clearly."
                )
            )
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Fallback also failed: {e}")
            return None
    
    def get_execution_trace(self, trace_id: str) -> Dict[str, Any]:
        """
        Get execution trace for a query.
        
        Args:
            trace_id: Trace identifier
            
        Returns:
            Trace data dictionary
        """
        if self._trace_logger:
            return self._trace_logger.get_trace(trace_id)
        return {}
    
    def get_conversation_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent conversation history.
        
        Args:
            limit: Maximum number of entries
            
        Returns:
            List of conversation entries
        """
        if self._memory_store:
            return self._memory_store.get_recent(limit)
        return []
    
    def clear_memory(self) -> None:
        """Clear conversation memory."""
        if self._memory_store:
            self._memory_store.clear()
    
    @property
    def is_initialized(self) -> bool:
        """Check if manager is initialized."""
        return self._initialized


def create_crew_manager(
    verbose: bool = True,
    memory: bool = True,
    **kwargs
) -> CrewManager:
    """
    Factory function to create a crew manager.
    
    Args:
        verbose: Enable verbose output
        memory: Enable memory
        **kwargs: Additional configuration
        
    Returns:
        Configured CrewManager instance
    """
    config = CrewConfig(verbose=verbose, memory=memory, **kwargs)
    manager = CrewManager(config)
    return manager

