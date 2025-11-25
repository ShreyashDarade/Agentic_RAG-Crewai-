import json
import logging
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class TraceConfig:
    """Configuration for trace logging."""
    enabled: bool = True
    persist_traces: bool = True
    persist_path: str = "./data/traces"
    max_traces: int = 1000
    include_payloads: bool = True


@dataclass
class TraceStep:
    """Represents a single step in execution trace."""
    step_id: str
    agent: str
    action: str
    status: str  # started, completed, failed
    timestamp: str
    duration_ms: Optional[float] = None
    input_data: Optional[Dict[str, Any]] = None
    output_data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExecutionTrace:
    """Complete execution trace for a query."""
    trace_id: str
    query: str
    started_at: str
    completed_at: Optional[str] = None
    total_duration_ms: float = 0.0
    status: str = "running"  # running, completed, failed
    steps: List[TraceStep] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "query": self.query,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "total_duration_ms": self.total_duration_ms,
            "status": self.status,
            "steps": [step.to_dict() for step in self.steps],
            "metadata": self.metadata
        }


class TraceLoggerError(Exception):
    """Exception raised for trace logger errors."""
    pass


class TraceLogger:
    """
    Logger for tracking agent execution traces.
    
    Features:
    - Step-by-step execution tracking
    - Duration measurement
    - Error capture
    - Trace persistence
    - API-friendly output format
    """
    
    def __init__(self, config: Optional[TraceConfig] = None):
        """
        Initialize the trace logger.
        
        Args:
            config: Trace configuration
        """
        self.config = config or TraceConfig()
        self._traces: Dict[str, ExecutionTrace] = {}
        self._step_start_times: Dict[str, datetime] = {}
        
        if self.config.persist_traces:
            Path(self.config.persist_path).mkdir(parents=True, exist_ok=True)
    
    def start_trace(
        self,
        query: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Start a new execution trace.
        
        Args:
            query: User query
            metadata: Additional metadata
            
        Returns:
            Trace ID
        """
        if not self.config.enabled:
            return ""
        
        trace_id = str(uuid.uuid4())[:8]
        
        trace = ExecutionTrace(
            trace_id=trace_id,
            query=query,
            started_at=datetime.now().isoformat(),
            metadata=metadata or {}
        )
        
        self._traces[trace_id] = trace
        
        logger.debug(f"Started trace: {trace_id}")
        return trace_id
    
    def log_step(
        self,
        trace_id: str,
        agent: str,
        action: str,
        status: str,
        data: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
    ) -> str:
        """
        Log an execution step.
        
        Args:
            trace_id: Trace ID
            agent: Agent name
            action: Action being performed
            status: Step status
            data: Input or output data
            error: Error message if failed
            
        Returns:
            Step ID
        """
        if not self.config.enabled or trace_id not in self._traces:
            return ""
        
        step_id = f"{trace_id}_{len(self._traces[trace_id].steps)}"
        timestamp = datetime.now().isoformat()
        
        # Calculate duration if completing a step
        duration_ms = None
        step_key = f"{trace_id}:{agent}:{action}"
        
        if status == "started":
            self._step_start_times[step_key] = datetime.now()
        elif status in ["completed", "failed"] and step_key in self._step_start_times:
            start_time = self._step_start_times.pop(step_key)
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000
        
        step = TraceStep(
            step_id=step_id,
            agent=agent,
            action=action,
            status=status,
            timestamp=timestamp,
            duration_ms=duration_ms,
            input_data=data if status == "started" and self.config.include_payloads else None,
            output_data=data if status in ["completed", "failed"] and self.config.include_payloads else None,
            error=error
        )
        
        self._traces[trace_id].steps.append(step)
        
        logger.debug(f"Logged step: {agent}.{action} -> {status}")
        return step_id
    
    def end_trace(
        self,
        trace_id: str,
        success: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        End an execution trace.
        
        Args:
            trace_id: Trace ID
            success: Whether execution was successful
            metadata: Additional metadata to add
        """
        if not self.config.enabled or trace_id not in self._traces:
            return
        
        trace = self._traces[trace_id]
        trace.completed_at = datetime.now().isoformat()
        trace.status = "completed" if success else "failed"
        
        # Calculate total duration
        start_time = datetime.fromisoformat(trace.started_at)
        end_time = datetime.fromisoformat(trace.completed_at)
        trace.total_duration_ms = (end_time - start_time).total_seconds() * 1000
        
        if metadata:
            trace.metadata.update(metadata)
        
        # Persist trace
        if self.config.persist_traces:
            self._persist_trace(trace)
        
        logger.debug(f"Ended trace: {trace_id} ({trace.status}, {trace.total_duration_ms:.2f}ms)")
    
    def get_trace(self, trace_id: str) -> Dict[str, Any]:
        """
        Get a trace by ID.
        
        Args:
            trace_id: Trace ID
            
        Returns:
            Trace dictionary
        """
        if trace_id in self._traces:
            return self._traces[trace_id].to_dict()
        
        # Try to load from disk
        if self.config.persist_traces:
            trace = self._load_trace(trace_id)
            if trace:
                return trace
        
        return {}
    
    def get_trace_summary(self, trace_id: str) -> Dict[str, Any]:
        """
        Get a summary of a trace.
        
        Args:
            trace_id: Trace ID
            
        Returns:
            Summary dictionary
        """
        trace = self._traces.get(trace_id)
        if not trace:
            return {}
        
        step_summary = []
        for step in trace.steps:
            step_summary.append({
                "agent": step.agent,
                "action": step.action,
                "status": step.status,
                "duration_ms": step.duration_ms
            })
        
        return {
            "trace_id": trace.trace_id,
            "query": trace.query[:100],
            "status": trace.status,
            "total_duration_ms": trace.total_duration_ms,
            "step_count": len(trace.steps),
            "steps": step_summary
        }
    
    def get_recent_traces(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent traces.
        
        Args:
            limit: Maximum number of traces
            
        Returns:
            List of trace summaries
        """
        traces = list(self._traces.values())
        traces.sort(key=lambda t: t.started_at, reverse=True)
        
        return [self.get_trace_summary(t.trace_id) for t in traces[:limit]]
    
    def _persist_trace(self, trace: ExecutionTrace) -> None:
        """Save trace to disk."""
        try:
            trace_file = Path(self.config.persist_path) / f"{trace.trace_id}.json"
            
            with open(trace_file, "w") as f:
                json.dump(trace.to_dict(), f, indent=2, default=str)
            
        except Exception as e:
            logger.warning(f"Failed to persist trace: {e}")
    
    def _load_trace(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """Load trace from disk."""
        try:
            trace_file = Path(self.config.persist_path) / f"{trace_id}.json"
            
            if trace_file.exists():
                with open(trace_file, "r") as f:
                    return json.load(f)
            
        except Exception as e:
            logger.warning(f"Failed to load trace: {e}")
        
        return None
    
    def format_for_api(self, trace_id: str) -> Dict[str, Any]:
        """
        Format trace for API response.
        
        Args:
            trace_id: Trace ID
            
        Returns:
            API-friendly trace format
        """
        trace = self._traces.get(trace_id)
        if not trace:
            return {"error": "Trace not found"}
        
        # Build tool chain
        tool_chain = []
        for step in trace.steps:
            tool_chain.append({
                "tool": f"{step.agent}.{step.action}",
                "status": step.status,
                "duration_ms": step.duration_ms,
                "timestamp": step.timestamp
            })
        
        return {
            "trace_id": trace.trace_id,
            "query": trace.query,
            "execution": {
                "started_at": trace.started_at,
                "completed_at": trace.completed_at,
                "total_duration_ms": trace.total_duration_ms,
                "status": trace.status
            },
            "tool_chain": tool_chain,
            "step_count": len(trace.steps),
            "metadata": trace.metadata
        }
    
    def clear(self) -> None:
        """Clear all traces from memory."""
        self._traces.clear()
        self._step_start_times.clear()
        logger.info("Traces cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get trace statistics."""
        if not self._traces:
            return {
                "total_traces": 0,
                "completed": 0,
                "failed": 0,
                "running": 0,
                "avg_duration_ms": 0
            }
        
        status_counts = {"completed": 0, "failed": 0, "running": 0}
        durations = []
        
        for trace in self._traces.values():
            status_counts[trace.status] = status_counts.get(trace.status, 0) + 1
            if trace.total_duration_ms > 0:
                durations.append(trace.total_duration_ms)
        
        return {
            "total_traces": len(self._traces),
            "completed": status_counts.get("completed", 0),
            "failed": status_counts.get("failed", 0),
            "running": status_counts.get("running", 0),
            "avg_duration_ms": sum(durations) / len(durations) if durations else 0
        }


def create_trace_logger(
    enabled: bool = True,
    persist_traces: bool = True,
    **kwargs
) -> TraceLogger:
    """
    Factory function to create a trace logger.
    
    Args:
        enabled: Enable tracing
        persist_traces: Enable trace persistence
        **kwargs: Additional configuration
        
    Returns:
        Configured TraceLogger instance
    """
    config = TraceConfig(enabled=enabled, persist_traces=persist_traces, **kwargs)
    return TraceLogger(config)

