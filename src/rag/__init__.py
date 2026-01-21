"""
RAG (Retrieval-Augmented Generation) Pipeline

This module provides true RAG functionality:
1. Embed and store policy documents in pgvector
2. Retrieve relevant context based on control results
3. Ground LLM generation in actual policy text
4. Safety verification (hallucination detection, output validation)
5. Production monitoring and evaluation
"""

from .vector_store import VectorStore, PolicyChunk
from .embedder import LocalEmbedder, embed_policies
from .retriever import RAGRetriever

# Safety components
from .safety import (
    SafeRAGPipeline,
    SafeRAGResult,
    HallucinationDetector,
    HallucinationCheckResult,
    OutputValidator,
    OutputValidationResult,
    CircuitBreaker,
    GracefulDegradation,
    SafetyThresholds,
    RiskLevel,
    PRODUCTION_THRESHOLDS,
    DEVELOPMENT_THRESHOLDS,
)

# Evaluation components
from .evaluation import (
    RetrievalEvaluator,
    RetrievalMetrics,
    AggregateRetrievalMetrics,
    QualityThresholds,
    PRODUCTION_THRESHOLDS as EVAL_PRODUCTION_THRESHOLDS,
    GoldenDatasetBuilder,
    get_default_golden_dataset,
    ProductionMonitor,
    RegressionTester,
    RegressionTestResult,
)

__all__ = [
    # Core RAG
    "VectorStore",
    "PolicyChunk", 
    "LocalEmbedder",
    "embed_policies",
    "RAGRetriever",
    # Safety
    "SafeRAGPipeline",
    "SafeRAGResult",
    "HallucinationDetector",
    "HallucinationCheckResult",
    "OutputValidator",
    "OutputValidationResult",
    "CircuitBreaker",
    "GracefulDegradation",
    "SafetyThresholds",
    "RiskLevel",
    "PRODUCTION_THRESHOLDS",
    "DEVELOPMENT_THRESHOLDS",
    # Evaluation
    "RetrievalEvaluator",
    "RetrievalMetrics",
    "AggregateRetrievalMetrics",
    "QualityThresholds",
    "GoldenDatasetBuilder",
    "get_default_golden_dataset",
    "ProductionMonitor",
    "RegressionTester",
    "RegressionTestResult",
]
