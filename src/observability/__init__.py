"""
Observability & Tracing - Comprehensive Logging for RAG Operations

This module provides structured observability for all RAG operations:
1. Trace every retrieval, LLM call, tool execution
2. Track latencies, token counts, costs
3. Structured logging with correlation IDs
4. Metrics export (Prometheus-compatible)
5. Span-based tracing (OpenTelemetry-compatible)

Usage:
    from src.observability import Tracer, trace_operation

    with trace_operation("retrieval", {"query": query}) as span:
        result = retriever.retrieve(query)
        span.set_attribute("chunk_count", len(result.chunks))
"""

from __future__ import annotations

import json
import logging
import time
import uuid
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union
from collections import defaultdict
import statistics

logger = logging.getLogger(__name__)


# =============================================================================
# SPAN & TRACE TYPES
# =============================================================================

class SpanKind(Enum):
    """Type of operation being traced."""
    RETRIEVAL = "retrieval"
    EMBEDDING = "embedding"
    LLM_CALL = "llm_call"
    TOOL_CALL = "tool_call"
    DATABASE = "database"
    CACHE = "cache"
    VALIDATION = "validation"
    GENERATION = "generation"


class SpanStatus(Enum):
    """Status of a completed span."""
    OK = "ok"
    ERROR = "error"
    TIMEOUT = "timeout"


@dataclass
class Span:
    """
    A single traced operation (span).
    
    Spans can be nested to form a trace tree.
    """
    span_id: str
    trace_id: str
    parent_span_id: Optional[str]
    name: str
    kind: SpanKind
    
    # Timing
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    
    # Status
    status: SpanStatus = SpanStatus.OK
    error_message: Optional[str] = None
    
    # Attributes (key-value pairs)
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    # Metrics
    token_count_input: int = 0
    token_count_output: int = 0
    cost_usd: float = 0.0
    
    # Events within the span
    events: List[Dict[str, Any]] = field(default_factory=list)
    
    def set_attribute(self, key: str, value: Any):
        """Set an attribute on the span."""
        self.attributes[key] = value
    
    def add_event(self, name: str, attributes: Optional[Dict] = None):
        """Add an event to the span."""
        self.events.append({
            "name": name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "attributes": attributes or {},
        })
    
    def set_error(self, error: Exception):
        """Mark span as error."""
        self.status = SpanStatus.ERROR
        self.error_message = str(error)
    
    def end(self):
        """End the span and calculate duration."""
        self.end_time = datetime.now(timezone.utc)
        self.duration_ms = (self.end_time - self.start_time).total_seconds() * 1000
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/export."""
        return {
            "span_id": self.span_id,
            "trace_id": self.trace_id,
            "parent_span_id": self.parent_span_id,
            "name": self.name,
            "kind": self.kind.value,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "status": self.status.value,
            "error_message": self.error_message,
            "attributes": self.attributes,
            "token_count_input": self.token_count_input,
            "token_count_output": self.token_count_output,
            "cost_usd": self.cost_usd,
            "events": self.events,
        }
    
    def to_log_line(self) -> str:
        """Format as single log line."""
        status_emoji = "✓" if self.status == SpanStatus.OK else "✗"
        duration = f"{self.duration_ms:.0f}ms" if self.duration_ms else "?"
        
        attrs = " ".join(f"{k}={v}" for k, v in list(self.attributes.items())[:5])
        
        return (
            f"{status_emoji} [{self.kind.value}] {self.name} "
            f"({duration}) trace={self.trace_id[:8]} {attrs}"
        )


@dataclass
class Trace:
    """
    A complete trace consisting of multiple spans.
    
    Represents an end-to-end operation like a full RAG pipeline.
    """
    trace_id: str
    name: str
    spans: List[Span] = field(default_factory=list)
    
    # Aggregate metrics
    total_duration_ms: float = 0.0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    
    # Metadata
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    def add_span(self, span: Span):
        """Add a span to the trace."""
        self.spans.append(span)
        
        if not self.started_at or span.start_time < self.started_at:
            self.started_at = span.start_time
        
        if span.end_time:
            if not self.completed_at or span.end_time > self.completed_at:
                self.completed_at = span.end_time
        
        self.total_tokens += span.token_count_input + span.token_count_output
        self.total_cost_usd += span.cost_usd
    
    def finalize(self):
        """Finalize trace and calculate aggregates."""
        if self.started_at and self.completed_at:
            self.total_duration_ms = (self.completed_at - self.started_at).total_seconds() * 1000
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "trace_id": self.trace_id,
            "name": self.name,
            "spans": [s.to_dict() for s in self.spans],
            "total_duration_ms": self.total_duration_ms,
            "total_tokens": self.total_tokens,
            "total_cost_usd": self.total_cost_usd,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


# =============================================================================
# TRACER
# =============================================================================

class Tracer:
    """
    Main tracer for creating and managing spans/traces.
    
    Thread-safe and supports nested spans via context.
    
    Usage:
        tracer = Tracer()
        
        with tracer.start_trace("rag_pipeline") as trace:
            with tracer.start_span("retrieval", SpanKind.RETRIEVAL) as span:
                result = retriever.retrieve(query)
                span.set_attribute("chunk_count", len(result.chunks))
    """
    
    def __init__(
        self,
        service_name: str = "compliance-rag",
        export_logs: bool = True,
        export_callback: Optional[Callable[[Span], None]] = None,
    ):
        self.service_name = service_name
        self.export_logs = export_logs
        self.export_callback = export_callback
        
        # Thread-local storage for current context
        self._local = threading.local()
        
        # Storage for completed traces (for analysis)
        self._completed_traces: List[Trace] = []
        self._lock = threading.Lock()
    
    @property
    def _current_trace(self) -> Optional[Trace]:
        return getattr(self._local, 'current_trace', None)
    
    @_current_trace.setter
    def _current_trace(self, trace: Optional[Trace]):
        self._local.current_trace = trace
    
    @property
    def _current_span(self) -> Optional[Span]:
        return getattr(self._local, 'current_span', None)
    
    @_current_span.setter
    def _current_span(self, span: Optional[Span]):
        self._local.current_span = span
    
    @property
    def _span_stack(self) -> List[Span]:
        if not hasattr(self._local, 'span_stack'):
            self._local.span_stack = []
        return self._local.span_stack
    
    @contextmanager
    def start_trace(self, name: str, attributes: Optional[Dict] = None):
        """
        Start a new trace.
        
        Usage:
            with tracer.start_trace("rag_pipeline") as trace:
                # ... operations ...
        """
        trace_id = uuid.uuid4().hex
        trace = Trace(trace_id=trace_id, name=name)
        
        old_trace = self._current_trace
        self._current_trace = trace
        
        try:
            yield trace
        finally:
            trace.finalize()
            self._current_trace = old_trace
            
            with self._lock:
                self._completed_traces.append(trace)
                # Keep only last 1000 traces
                if len(self._completed_traces) > 1000:
                    self._completed_traces = self._completed_traces[-1000:]
    
    @contextmanager
    def start_span(
        self,
        name: str,
        kind: SpanKind,
        attributes: Optional[Dict] = None,
    ):
        """
        Start a new span within current trace.
        
        Usage:
            with tracer.start_span("embedding", SpanKind.EMBEDDING) as span:
                embedding = embedder.embed(text)
                span.set_attribute("dimension", len(embedding))
        """
        span_id = uuid.uuid4().hex[:16]
        trace_id = self._current_trace.trace_id if self._current_trace else uuid.uuid4().hex
        parent_span_id = self._current_span.span_id if self._current_span else None
        
        span = Span(
            span_id=span_id,
            trace_id=trace_id,
            parent_span_id=parent_span_id,
            name=name,
            kind=kind,
            start_time=datetime.now(timezone.utc),
            attributes=attributes or {},
        )
        
        # Push to stack
        self._span_stack.append(span)
        old_span = self._current_span
        self._current_span = span
        
        try:
            yield span
        except Exception as e:
            span.set_error(e)
            raise
        finally:
            span.end()
            self._span_stack.pop()
            self._current_span = old_span
            
            # Add to trace
            if self._current_trace:
                self._current_trace.add_span(span)
            
            # Export
            self._export_span(span)
    
    def _export_span(self, span: Span):
        """Export span to configured destinations."""
        if self.export_logs:
            logger.info(span.to_log_line())
        
        if self.export_callback:
            try:
                self.export_callback(span)
            except Exception as e:
                logger.warning(f"Span export callback failed: {e}")
    
    def get_recent_traces(self, limit: int = 100) -> List[Trace]:
        """Get recent completed traces."""
        with self._lock:
            return self._completed_traces[-limit:]


# =============================================================================
# METRICS COLLECTOR
# =============================================================================

@dataclass
class MetricsBucket:
    """Aggregated metrics for a time window."""
    count: int = 0
    total_duration_ms: float = 0.0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    errors: int = 0
    latencies: List[float] = field(default_factory=list)
    
    def record(self, duration_ms: float, tokens: int = 0, cost: float = 0.0, error: bool = False):
        self.count += 1
        self.total_duration_ms += duration_ms
        self.total_tokens += tokens
        self.total_cost_usd += cost
        if error:
            self.errors += 1
        self.latencies.append(duration_ms)
    
    @property
    def mean_latency(self) -> float:
        return self.total_duration_ms / self.count if self.count else 0
    
    @property
    def p50_latency(self) -> float:
        if not self.latencies:
            return 0
        sorted_latencies = sorted(self.latencies)
        return sorted_latencies[len(sorted_latencies) // 2]
    
    @property
    def p95_latency(self) -> float:
        if not self.latencies:
            return 0
        sorted_latencies = sorted(self.latencies)
        idx = int(len(sorted_latencies) * 0.95)
        return sorted_latencies[min(idx, len(sorted_latencies) - 1)]
    
    @property
    def p99_latency(self) -> float:
        if not self.latencies:
            return 0
        sorted_latencies = sorted(self.latencies)
        idx = int(len(sorted_latencies) * 0.99)
        return sorted_latencies[min(idx, len(sorted_latencies) - 1)]
    
    @property
    def error_rate(self) -> float:
        return self.errors / self.count if self.count else 0


class MetricsCollector:
    """
    Collects and aggregates metrics for monitoring.
    
    Supports Prometheus-style metrics export.
    """
    
    def __init__(self):
        self._metrics: Dict[str, MetricsBucket] = defaultdict(MetricsBucket)
        self._lock = threading.Lock()
    
    def record(
        self,
        metric_name: str,
        duration_ms: float,
        tokens: int = 0,
        cost_usd: float = 0.0,
        error: bool = False,
        labels: Optional[Dict[str, str]] = None,
    ):
        """Record a metric observation."""
        # Create key with labels
        key = metric_name
        if labels:
            label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
            key = f"{metric_name}{{{label_str}}}"
        
        with self._lock:
            self._metrics[key].record(duration_ms, tokens, cost_usd, error)
    
    def get_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get all metrics as dictionary."""
        with self._lock:
            return {
                name: {
                    "count": bucket.count,
                    "mean_latency_ms": bucket.mean_latency,
                    "p50_latency_ms": bucket.p50_latency,
                    "p95_latency_ms": bucket.p95_latency,
                    "p99_latency_ms": bucket.p99_latency,
                    "total_tokens": bucket.total_tokens,
                    "total_cost_usd": bucket.total_cost_usd,
                    "error_rate": bucket.error_rate,
                }
                for name, bucket in self._metrics.items()
            }
    
    def to_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        
        with self._lock:
            for name, bucket in self._metrics.items():
                # Parse name and labels
                if "{" in name:
                    base_name = name.split("{")[0]
                    labels = name.split("{")[1].rstrip("}")
                else:
                    base_name = name
                    labels = ""
                
                label_str = f"{{{labels}}}" if labels else ""
                
                lines.append(f"# HELP {base_name}_total Total count")
                lines.append(f"{base_name}_total{label_str} {bucket.count}")
                
                lines.append(f"# HELP {base_name}_latency_ms Latency in milliseconds")
                lines.append(f'{base_name}_latency_ms{{quantile="0.5"}}{label_str} {bucket.p50_latency:.2f}')
                lines.append(f'{base_name}_latency_ms{{quantile="0.95"}}{label_str} {bucket.p95_latency:.2f}')
                lines.append(f'{base_name}_latency_ms{{quantile="0.99"}}{label_str} {bucket.p99_latency:.2f}')
                
                lines.append(f"# HELP {base_name}_tokens_total Total tokens processed")
                lines.append(f"{base_name}_tokens_total{label_str} {bucket.total_tokens}")
                
                lines.append(f"# HELP {base_name}_errors_total Total errors")
                lines.append(f"{base_name}_errors_total{label_str} {bucket.errors}")
        
        return "\n".join(lines)
    
    def reset(self):
        """Reset all metrics."""
        with self._lock:
            self._metrics.clear()


# =============================================================================
# CONVENIENCE DECORATORS
# =============================================================================

# Global tracer instance
_global_tracer: Optional[Tracer] = None
_global_metrics: Optional[MetricsCollector] = None


def get_tracer() -> Tracer:
    """Get or create global tracer."""
    global _global_tracer
    if _global_tracer is None:
        _global_tracer = Tracer()
    return _global_tracer


def get_metrics() -> MetricsCollector:
    """Get or create global metrics collector."""
    global _global_metrics
    if _global_metrics is None:
        _global_metrics = MetricsCollector()
    return _global_metrics


F = TypeVar('F', bound=Callable[..., Any])


def trace_function(
    name: Optional[str] = None,
    kind: SpanKind = SpanKind.TOOL_CALL,
    record_args: bool = True,
) -> Callable[[F], F]:
    """
    Decorator to trace a function.
    
    Usage:
        @trace_function("retrieve_context", SpanKind.RETRIEVAL)
        def retrieve_context(query: str) -> List[Chunk]:
            ...
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            span_name = name or func.__name__
            tracer = get_tracer()
            
            attributes = {}
            if record_args:
                # Record first few args
                for i, arg in enumerate(args[:3]):
                    attributes[f"arg_{i}"] = str(arg)[:100]
                for k, v in list(kwargs.items())[:3]:
                    attributes[f"kwarg_{k}"] = str(v)[:100]
            
            with tracer.start_span(span_name, kind, attributes) as span:
                result = func(*args, **kwargs)
                
                # Try to add result info
                if hasattr(result, '__len__'):
                    span.set_attribute("result_length", len(result))
                
                return result
        
        return wrapper  # type: ignore
    return decorator


@contextmanager
def trace_operation(
    name: str,
    kind: SpanKind = SpanKind.TOOL_CALL,
    attributes: Optional[Dict] = None,
):
    """
    Context manager for tracing an operation.
    
    Usage:
        with trace_operation("embedding", SpanKind.EMBEDDING, {"text_length": len(text)}) as span:
            embedding = embedder.embed(text)
            span.set_attribute("dimension", len(embedding))
    """
    tracer = get_tracer()
    with tracer.start_span(name, kind, attributes) as span:
        yield span


# =============================================================================
# SPECIALIZED LOGGERS
# =============================================================================

class RAGLogger:
    """
    Specialized logger for RAG operations.
    
    Provides structured logging with automatic metrics collection.
    """
    
    def __init__(self, tracer: Optional[Tracer] = None, metrics: Optional[MetricsCollector] = None):
        self.tracer = tracer or get_tracer()
        self.metrics = metrics or get_metrics()
    
    def log_retrieval(
        self,
        query: str,
        chunk_count: int,
        latency_ms: float,
        confidence: float,
        control_type: Optional[str] = None,
        success: bool = True,
        error: Optional[str] = None,
    ):
        """Log a retrieval operation."""
        self.metrics.record(
            "rag_retrieval",
            duration_ms=latency_ms,
            error=not success,
            labels={"control_type": control_type or "unknown"},
        )
        
        log_data = {
            "event": "retrieval",
            "query_preview": query[:50],
            "chunk_count": chunk_count,
            "latency_ms": round(latency_ms, 2),
            "confidence": round(confidence, 3),
            "control_type": control_type,
            "success": success,
        }
        
        if error:
            log_data["error"] = error
            logger.warning(f"Retrieval failed: {json.dumps(log_data)}")
        else:
            logger.info(f"Retrieval: {json.dumps(log_data)}")
    
    def log_llm_call(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float,
        success: bool = True,
        error: Optional[str] = None,
        cost_usd: float = 0.0,
    ):
        """Log an LLM call."""
        total_tokens = prompt_tokens + completion_tokens
        
        self.metrics.record(
            "llm_call",
            duration_ms=latency_ms,
            tokens=total_tokens,
            cost_usd=cost_usd,
            error=not success,
            labels={"model": model},
        )
        
        log_data = {
            "event": "llm_call",
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "latency_ms": round(latency_ms, 2),
            "cost_usd": round(cost_usd, 6),
            "success": success,
        }
        
        if error:
            log_data["error"] = error
            logger.warning(f"LLM call failed: {json.dumps(log_data)}")
        else:
            logger.info(f"LLM call: {json.dumps(log_data)}")
    
    def log_embedding(
        self,
        text_count: int,
        dimension: int,
        latency_ms: float,
        success: bool = True,
        cached: bool = False,
    ):
        """Log an embedding operation."""
        self.metrics.record(
            "embedding",
            duration_ms=latency_ms,
            error=not success,
            labels={"cached": str(cached).lower()},
        )
        
        log_data = {
            "event": "embedding",
            "text_count": text_count,
            "dimension": dimension,
            "latency_ms": round(latency_ms, 2),
            "cached": cached,
            "success": success,
        }
        
        logger.debug(f"Embedding: {json.dumps(log_data)}")
    
    def log_validation(
        self,
        validation_type: str,
        passed: bool,
        errors: List[str],
        latency_ms: float,
    ):
        """Log a validation operation."""
        self.metrics.record(
            "validation",
            duration_ms=latency_ms,
            error=not passed,
            labels={"type": validation_type},
        )
        
        log_data = {
            "event": "validation",
            "type": validation_type,
            "passed": passed,
            "error_count": len(errors),
            "latency_ms": round(latency_ms, 2),
        }
        
        if not passed:
            log_data["errors"] = errors[:5]  # Limit error list
            logger.warning(f"Validation failed: {json.dumps(log_data)}")
        else:
            logger.debug(f"Validation: {json.dumps(log_data)}")


# Convenience instance
rag_logger = RAGLogger()


# =============================================================================
# IMPORTS FROM SUBMODULES
# =============================================================================

from src.observability.caching import (
    CacheConfig,
    QueryCache,
    LRUCache,
    BatchEmbedder,
    ContextWindowConfig,
    ContextWindowOptimizer,
)

__all__ = [
    # Tracing
    "Span",
    "SpanKind",
    "SpanStatus",
    "Trace",
    "Tracer",
    "get_tracer",
    "trace_function",
    "trace_operation",
    # Metrics
    "MetricsBucket",
    "MetricsCollector",
    "get_metrics",
    # Logging
    "RAGLogger",
    "rag_logger",
    # Caching
    "CacheConfig",
    "QueryCache",
    "LRUCache",
    "BatchEmbedder",
    "ContextWindowConfig",
    "ContextWindowOptimizer",
]
