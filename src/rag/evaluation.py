"""
RAG Evaluation Framework - Metrics & Monitoring for Production

This module provides comprehensive evaluation capabilities:
1. Retrieval Quality Metrics - Hit rate, MRR, NDCG
2. Generation Quality Metrics - Faithfulness, relevance
3. Golden Dataset Testing - Regression testing
4. Production Monitoring - Latency, error rates, drift detection
5. A/B Testing Support - Compare retrieval strategies

SEC EXAMINATION CRITICAL:
- All evaluations are logged with timestamps
- Regression tests run before each production deployment
- Quality degradation triggers alerts
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set, Callable
from collections import defaultdict
import uuid

logger = logging.getLogger(__name__)


# =============================================================================
# RETRIEVAL QUALITY METRICS
# =============================================================================

@dataclass
class RetrievalMetrics:
    """Metrics for a single retrieval evaluation."""
    query_id: str
    query: str
    
    # Hit metrics
    hit_at_1: bool
    hit_at_3: bool
    hit_at_5: bool
    hit_at_10: bool
    
    # Ranking metrics
    mrr: float  # Mean Reciprocal Rank
    ndcg_at_5: float  # Normalized Discounted Cumulative Gain
    
    # Context quality
    retrieved_count: int
    relevant_count: int
    precision: float
    recall: float
    f1: float
    
    # Performance
    latency_ms: float
    
    # Metadata
    evaluated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class AggregateRetrievalMetrics:
    """Aggregated metrics across multiple queries."""
    total_queries: int
    
    # Hit rates
    hit_rate_at_1: float
    hit_rate_at_3: float
    hit_rate_at_5: float
    hit_rate_at_10: float
    
    # Ranking
    mean_mrr: float
    mean_ndcg: float
    
    # Precision/Recall
    mean_precision: float
    mean_recall: float
    mean_f1: float
    
    # Performance
    mean_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    
    # Metadata
    evaluation_id: str
    evaluated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/storage."""
        return {
            "evaluation_id": self.evaluation_id,
            "total_queries": self.total_queries,
            "hit_rate_at_1": round(self.hit_rate_at_1, 4),
            "hit_rate_at_3": round(self.hit_rate_at_3, 4),
            "hit_rate_at_5": round(self.hit_rate_at_5, 4),
            "mean_mrr": round(self.mean_mrr, 4),
            "mean_ndcg": round(self.mean_ndcg, 4),
            "mean_precision": round(self.mean_precision, 4),
            "mean_recall": round(self.mean_recall, 4),
            "mean_f1": round(self.mean_f1, 4),
            "mean_latency_ms": round(self.mean_latency_ms, 2),
            "p95_latency_ms": round(self.p95_latency_ms, 2),
            "p99_latency_ms": round(self.p99_latency_ms, 2),
            "evaluated_at": self.evaluated_at.isoformat(),
        }
    
    def passes_threshold(self, thresholds: "QualityThresholds") -> Tuple[bool, List[str]]:
        """Check if metrics pass quality thresholds."""
        failures = []
        
        if self.hit_rate_at_3 < thresholds.min_hit_rate_at_3:
            failures.append(f"hit_rate@3 {self.hit_rate_at_3:.2%} < {thresholds.min_hit_rate_at_3:.2%}")
        
        if self.mean_mrr < thresholds.min_mrr:
            failures.append(f"MRR {self.mean_mrr:.3f} < {thresholds.min_mrr:.3f}")
        
        if self.mean_precision < thresholds.min_precision:
            failures.append(f"precision {self.mean_precision:.2%} < {thresholds.min_precision:.2%}")
        
        if self.p95_latency_ms > thresholds.max_p95_latency_ms:
            failures.append(f"P95 latency {self.p95_latency_ms:.0f}ms > {thresholds.max_p95_latency_ms:.0f}ms")
        
        return len(failures) == 0, failures


@dataclass(frozen=True)
class QualityThresholds:
    """Quality thresholds for production deployment."""
    min_hit_rate_at_3: float = 0.80
    min_mrr: float = 0.60
    min_precision: float = 0.70
    min_recall: float = 0.60
    max_p95_latency_ms: float = 2000.0
    max_p99_latency_ms: float = 5000.0
    max_error_rate: float = 0.01


PRODUCTION_THRESHOLDS = QualityThresholds()
STAGING_THRESHOLDS = QualityThresholds(
    min_hit_rate_at_3=0.70,
    min_mrr=0.50,
    min_precision=0.60,
    max_p95_latency_ms=3000.0,
)


class RetrievalEvaluator:
    """
    Evaluates retrieval quality against golden datasets.
    
    Golden dataset format:
    [
        {
            "query": "technology sector concentration limit",
            "relevant_doc_ids": ["POL-CONC-001", "POL-CONC-002"],
            "relevant_chunk_ids": ["concentration_limits.md:3.1:abc123"],
        },
        ...
    ]
    """
    
    def __init__(self, retriever):
        self.retriever = retriever
    
    def evaluate_single(
        self,
        query: str,
        relevant_doc_ids: Set[str],
        relevant_chunk_ids: Optional[Set[str]] = None,
        k: int = 10,
    ) -> RetrievalMetrics:
        """Evaluate a single query."""
        start_time = time.time()
        
        # Retrieve
        try:
            from .retriever import RetrievedContext
            result = self.retriever.retrieve_for_control(
                control_name=query,
                control_type="evaluation",
                status="pass",
                calculated_value=0.0,
                threshold=0.0,
                limit=k,
            )
            chunks = result.chunks if hasattr(result, 'chunks') else []
        except Exception as e:
            logger.error(f"Retrieval failed for evaluation: {e}")
            chunks = []
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Calculate metrics
        retrieved_ids = set()
        for chunk in chunks:
            retrieved_ids.add(chunk.document_id)
            retrieved_ids.add(chunk.chunk_id)
        
        # Hits
        hits = [False] * k
        for i, chunk in enumerate(chunks[:k]):
            is_relevant = (
                chunk.document_id in relevant_doc_ids or
                chunk.chunk_id in (relevant_chunk_ids or set())
            )
            if is_relevant:
                hits[i] = True
        
        hit_at_1 = any(hits[:1])
        hit_at_3 = any(hits[:3])
        hit_at_5 = any(hits[:5])
        hit_at_10 = any(hits[:10])
        
        # MRR
        mrr = 0.0
        for i, is_hit in enumerate(hits):
            if is_hit:
                mrr = 1.0 / (i + 1)
                break
        
        # NDCG@5
        dcg = 0.0
        for i, is_hit in enumerate(hits[:5]):
            if is_hit:
                dcg += 1.0 / math.log2(i + 2)
        
        # Ideal DCG (all relevant in top positions)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant_doc_ids), 5)))
        ndcg = dcg / idcg if idcg > 0 else 0.0
        
        # Precision/Recall
        relevant_retrieved = sum(hits)
        precision = relevant_retrieved / len(chunks) if chunks else 0.0
        recall = relevant_retrieved / len(relevant_doc_ids) if relevant_doc_ids else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return RetrievalMetrics(
            query_id=hashlib.md5(query.encode()).hexdigest()[:8],
            query=query,
            hit_at_1=hit_at_1,
            hit_at_3=hit_at_3,
            hit_at_5=hit_at_5,
            hit_at_10=hit_at_10,
            mrr=mrr,
            ndcg_at_5=ndcg,
            retrieved_count=len(chunks),
            relevant_count=len(relevant_doc_ids),
            precision=precision,
            recall=recall,
            f1=f1,
            latency_ms=latency_ms,
        )
    
    def evaluate_dataset(
        self,
        golden_dataset: List[Dict[str, Any]],
    ) -> AggregateRetrievalMetrics:
        """Evaluate against a golden dataset."""
        evaluation_id = str(uuid.uuid4())[:8]
        results = []
        
        for item in golden_dataset:
            query = item["query"]
            relevant_docs = set(item.get("relevant_doc_ids", []))
            relevant_chunks = set(item.get("relevant_chunk_ids", []))
            
            if not relevant_docs and not relevant_chunks:
                logger.warning(f"Skipping query with no relevant docs: {query}")
                continue
            
            result = self.evaluate_single(
                query=query,
                relevant_doc_ids=relevant_docs,
                relevant_chunk_ids=relevant_chunks,
            )
            results.append(result)
        
        if not results:
            raise ValueError("No valid evaluation results")
        
        # Aggregate
        latencies = [r.latency_ms for r in results]
        latencies.sort()
        
        return AggregateRetrievalMetrics(
            total_queries=len(results),
            hit_rate_at_1=sum(r.hit_at_1 for r in results) / len(results),
            hit_rate_at_3=sum(r.hit_at_3 for r in results) / len(results),
            hit_rate_at_5=sum(r.hit_at_5 for r in results) / len(results),
            hit_rate_at_10=sum(r.hit_at_10 for r in results) / len(results),
            mean_mrr=statistics.mean(r.mrr for r in results),
            mean_ndcg=statistics.mean(r.ndcg_at_5 for r in results),
            mean_precision=statistics.mean(r.precision for r in results),
            mean_recall=statistics.mean(r.recall for r in results),
            mean_f1=statistics.mean(r.f1 for r in results),
            mean_latency_ms=statistics.mean(latencies),
            p50_latency_ms=latencies[len(latencies) // 2],
            p95_latency_ms=latencies[int(len(latencies) * 0.95)] if len(latencies) >= 20 else latencies[-1],
            p99_latency_ms=latencies[int(len(latencies) * 0.99)] if len(latencies) >= 100 else latencies[-1],
            evaluation_id=evaluation_id,
        )


# =============================================================================
# GOLDEN DATASET MANAGEMENT
# =============================================================================

@dataclass
class GoldenDatasetEntry:
    """A single entry in the golden dataset."""
    query: str
    relevant_doc_ids: List[str]
    relevant_chunk_ids: List[str] = field(default_factory=list)
    control_type: str = "general"
    difficulty: str = "normal"  # easy, normal, hard
    notes: str = ""


class GoldenDatasetBuilder:
    """
    Build and manage golden datasets for evaluation.
    
    Usage:
        builder = GoldenDatasetBuilder()
        builder.add_entry(
            query="technology sector concentration",
            relevant_doc_ids=["concentration_limits.md"],
            control_type="concentration",
        )
        builder.save("golden_dataset.json")
    """
    
    def __init__(self):
        self.entries: List[GoldenDatasetEntry] = []
    
    def add_entry(
        self,
        query: str,
        relevant_doc_ids: List[str],
        relevant_chunk_ids: Optional[List[str]] = None,
        control_type: str = "general",
        difficulty: str = "normal",
        notes: str = "",
    ):
        """Add an entry to the golden dataset."""
        self.entries.append(GoldenDatasetEntry(
            query=query,
            relevant_doc_ids=relevant_doc_ids,
            relevant_chunk_ids=relevant_chunk_ids or [],
            control_type=control_type,
            difficulty=difficulty,
            notes=notes,
        ))
    
    def to_list(self) -> List[Dict[str, Any]]:
        """Convert to list format for evaluation."""
        return [
            {
                "query": e.query,
                "relevant_doc_ids": e.relevant_doc_ids,
                "relevant_chunk_ids": e.relevant_chunk_ids,
                "control_type": e.control_type,
                "difficulty": e.difficulty,
            }
            for e in self.entries
        ]
    
    def save(self, path: Path):
        """Save golden dataset to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_list(), f, indent=2)
        logger.info(f"Saved golden dataset with {len(self.entries)} entries to {path}")
    
    @classmethod
    def load(cls, path: Path) -> "GoldenDatasetBuilder":
        """Load golden dataset from JSON file."""
        builder = cls()
        with open(path) as f:
            data = json.load(f)
        
        for item in data:
            builder.add_entry(
                query=item["query"],
                relevant_doc_ids=item["relevant_doc_ids"],
                relevant_chunk_ids=item.get("relevant_chunk_ids", []),
                control_type=item.get("control_type", "general"),
                difficulty=item.get("difficulty", "normal"),
            )
        
        return builder


# Default compliance golden dataset
def get_default_golden_dataset() -> List[Dict[str, Any]]:
    """
    Default golden dataset for compliance RAG.
    
    These are curated query-document pairs that represent
    typical compliance questions and their expected sources.
    """
    return [
        {
            "query": "technology sector concentration limit percentage",
            "relevant_doc_ids": ["concentration_limits.md", "investment_guidelines.md"],
            "control_type": "concentration",
            "difficulty": "easy",
        },
        {
            "query": "single issuer exposure maximum threshold NAV",
            "relevant_doc_ids": ["concentration_limits.md"],
            "control_type": "concentration",
            "difficulty": "easy",
        },
        {
            "query": "liquidity bucket T+1 minimum cash requirement",
            "relevant_doc_ids": ["liquidity_policy.md"],
            "control_type": "liquidity",
            "difficulty": "normal",
        },
        {
            "query": "breach escalation notification compliance officer",
            "relevant_doc_ids": ["exception_management.md"],
            "control_type": "exception",
            "difficulty": "normal",
        },
        {
            "query": "exception approval waiver authorized limit increase",
            "relevant_doc_ids": ["exception_management.md"],
            "control_type": "exception",
            "difficulty": "normal",
        },
        {
            "query": "gross exposure leverage limit hedge fund",
            "relevant_doc_ids": ["exposure_limits.md", "investment_guidelines.md"],
            "control_type": "exposure",
            "difficulty": "normal",
        },
        {
            "query": "net exposure calculation methodology long short",
            "relevant_doc_ids": ["exposure_limits.md"],
            "control_type": "exposure",
            "difficulty": "hard",
        },
        {
            "query": "counterparty prime broker exposure limit OTC",
            "relevant_doc_ids": ["investment_guidelines.md", "exposure_limits.md"],
            "control_type": "counterparty",
            "difficulty": "hard",
        },
        {
            "query": "SEC Form PF filing requirements disclosure",
            "relevant_doc_ids": ["sec_compliance.md"],
            "control_type": "regulatory",
            "difficulty": "normal",
        },
        {
            "query": "commodity trading derivatives position limits CFTC",
            "relevant_doc_ids": ["commodity_trading.md"],
            "control_type": "commodity",
            "difficulty": "normal",
        },
        {
            "query": "remediation plan cure period breach resolution timeline",
            "relevant_doc_ids": ["exception_management.md"],
            "control_type": "exception",
            "difficulty": "hard",
        },
        {
            "query": "geographic concentration country exposure limit",
            "relevant_doc_ids": ["concentration_limits.md", "investment_guidelines.md"],
            "control_type": "concentration",
            "difficulty": "hard",
        },
    ]


# =============================================================================
# PRODUCTION MONITORING
# =============================================================================

@dataclass
class MonitoringEvent:
    """A single monitoring event."""
    event_type: str
    timestamp: datetime
    operation_id: str
    metrics: Dict[str, Any]
    tags: Dict[str, str] = field(default_factory=dict)


class ProductionMonitor:
    """
    Real-time monitoring for RAG system in production.
    
    Tracks:
    - Retrieval latency (P50, P95, P99)
    - Error rates
    - Confidence distributions
    - Quality drift detection
    """
    
    def __init__(
        self,
        alert_callback: Optional[Callable[[str, Dict], None]] = None,
        window_size_minutes: int = 60,
    ):
        self.alert_callback = alert_callback
        self.window_size = timedelta(minutes=window_size_minutes)
        self.events: List[MonitoringEvent] = []
        self._thresholds = PRODUCTION_THRESHOLDS
    
    def record_retrieval(
        self,
        operation_id: str,
        latency_ms: float,
        confidence: float,
        chunk_count: int,
        success: bool,
        error: Optional[str] = None,
    ):
        """Record a retrieval operation."""
        self.events.append(MonitoringEvent(
            event_type="retrieval",
            timestamp=datetime.now(timezone.utc),
            operation_id=operation_id,
            metrics={
                "latency_ms": latency_ms,
                "confidence": confidence,
                "chunk_count": chunk_count,
                "success": success,
                "error": error,
            },
        ))
        
        # Check for issues
        self._check_latency_alert(latency_ms)
        self._check_confidence_alert(confidence)
        
        # Prune old events
        self._prune_old_events()
    
    def record_generation(
        self,
        operation_id: str,
        latency_ms: float,
        verified: bool,
        risk_level: str,
        requires_review: bool,
    ):
        """Record a generation operation."""
        self.events.append(MonitoringEvent(
            event_type="generation",
            timestamp=datetime.now(timezone.utc),
            operation_id=operation_id,
            metrics={
                "latency_ms": latency_ms,
                "verified": verified,
                "risk_level": risk_level,
                "requires_review": requires_review,
            },
        ))
    
    def get_current_stats(self) -> Dict[str, Any]:
        """Get current monitoring statistics."""
        now = datetime.now(timezone.utc)
        cutoff = now - self.window_size
        
        recent = [e for e in self.events if e.timestamp > cutoff]
        retrievals = [e for e in recent if e.event_type == "retrieval"]
        generations = [e for e in recent if e.event_type == "generation"]
        
        if not retrievals:
            return {"window_minutes": self.window_size.total_seconds() / 60, "no_data": True}
        
        latencies = sorted(e.metrics["latency_ms"] for e in retrievals)
        confidences = [e.metrics["confidence"] for e in retrievals]
        successes = [e.metrics["success"] for e in retrievals]
        
        return {
            "window_minutes": self.window_size.total_seconds() / 60,
            "retrieval_count": len(retrievals),
            "generation_count": len(generations),
            "retrieval_latency_p50": latencies[len(latencies) // 2],
            "retrieval_latency_p95": latencies[int(len(latencies) * 0.95)] if len(latencies) >= 20 else latencies[-1],
            "retrieval_latency_p99": latencies[int(len(latencies) * 0.99)] if len(latencies) >= 100 else latencies[-1],
            "mean_confidence": statistics.mean(confidences) if confidences else 0,
            "min_confidence": min(confidences) if confidences else 0,
            "success_rate": sum(successes) / len(successes) if successes else 0,
            "error_rate": 1 - (sum(successes) / len(successes)) if successes else 0,
            "verification_rate": (
                sum(1 for e in generations if e.metrics.get("verified", False)) / len(generations)
                if generations else 0
            ),
            "review_required_rate": (
                sum(1 for e in generations if e.metrics.get("requires_review", False)) / len(generations)
                if generations else 0
            ),
        }
    
    def detect_drift(self, baseline: Dict[str, float]) -> List[str]:
        """
        Detect quality drift compared to baseline.
        
        Returns list of drift warnings.
        """
        current = self.get_current_stats()
        
        if current.get("no_data"):
            return []
        
        warnings = []
        
        # Check for significant degradation (>20% worse)
        if "mean_confidence" in baseline and current["mean_confidence"] < baseline["mean_confidence"] * 0.8:
            warnings.append(
                f"Confidence drift: {current['mean_confidence']:.2%} vs baseline {baseline['mean_confidence']:.2%}"
            )
        
        if "success_rate" in baseline and current["success_rate"] < baseline["success_rate"] * 0.9:
            warnings.append(
                f"Success rate drift: {current['success_rate']:.2%} vs baseline {baseline['success_rate']:.2%}"
            )
        
        if "retrieval_latency_p95" in baseline and current["retrieval_latency_p95"] > baseline["retrieval_latency_p95"] * 1.5:
            warnings.append(
                f"Latency drift: P95 {current['retrieval_latency_p95']:.0f}ms vs baseline {baseline['retrieval_latency_p95']:.0f}ms"
            )
        
        return warnings
    
    def _check_latency_alert(self, latency_ms: float):
        """Check if latency exceeds threshold."""
        if latency_ms > self._thresholds.max_p95_latency_ms:
            self._send_alert("high_latency", {
                "latency_ms": latency_ms,
                "threshold_ms": self._thresholds.max_p95_latency_ms,
            })
    
    def _check_confidence_alert(self, confidence: float):
        """Check if confidence is critically low."""
        if confidence < 0.3:
            self._send_alert("low_confidence", {
                "confidence": confidence,
                "threshold": 0.3,
            })
    
    def _send_alert(self, alert_type: str, data: Dict[str, Any]):
        """Send an alert."""
        logger.warning(f"ALERT [{alert_type}]: {data}")
        if self.alert_callback:
            try:
                self.alert_callback(alert_type, data)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
    
    def _prune_old_events(self):
        """Remove events outside the monitoring window."""
        cutoff = datetime.now(timezone.utc) - self.window_size * 2
        self.events = [e for e in self.events if e.timestamp > cutoff]


# =============================================================================
# REGRESSION TESTING
# =============================================================================

@dataclass
class RegressionTestResult:
    """Result of a regression test run."""
    test_id: str
    passed: bool
    current_metrics: AggregateRetrievalMetrics
    baseline_metrics: Optional[AggregateRetrievalMetrics]
    regressions: List[str]
    improvements: List[str]
    run_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_report(self) -> str:
        """Generate human-readable report."""
        lines = [
            f"Regression Test Report: {self.test_id}",
            f"Run at: {self.run_at.isoformat()}",
            f"Result: {'PASSED' if self.passed else 'FAILED'}",
            "",
            "Current Metrics:",
            f"  Hit Rate @3: {self.current_metrics.hit_rate_at_3:.2%}",
            f"  MRR: {self.current_metrics.mean_mrr:.3f}",
            f"  Precision: {self.current_metrics.mean_precision:.2%}",
            f"  P95 Latency: {self.current_metrics.p95_latency_ms:.0f}ms",
        ]
        
        if self.baseline_metrics:
            lines.extend([
                "",
                "Baseline Comparison:",
                f"  Hit Rate @3: {self.baseline_metrics.hit_rate_at_3:.2%} → {self.current_metrics.hit_rate_at_3:.2%}",
                f"  MRR: {self.baseline_metrics.mean_mrr:.3f} → {self.current_metrics.mean_mrr:.3f}",
            ])
        
        if self.regressions:
            lines.extend(["", "Regressions:"])
            for r in self.regressions:
                lines.append(f"  ⚠️  {r}")
        
        if self.improvements:
            lines.extend(["", "Improvements:"])
            for i in self.improvements:
                lines.append(f"  ✅ {i}")
        
        return "\n".join(lines)


class RegressionTester:
    """
    Run regression tests before production deployment.
    
    Usage:
        tester = RegressionTester(retriever)
        result = tester.run_full_suite()
        
        if not result.passed:
            raise Exception("Regression test failed - blocking deployment")
    """
    
    def __init__(
        self,
        retriever,
        golden_dataset: Optional[List[Dict]] = None,
        thresholds: QualityThresholds = PRODUCTION_THRESHOLDS,
        baseline_path: Optional[Path] = None,
    ):
        self.retriever = retriever
        self.golden_dataset = golden_dataset or get_default_golden_dataset()
        self.thresholds = thresholds
        self.baseline_path = baseline_path
        self.evaluator = RetrievalEvaluator(retriever)
    
    def run_full_suite(self) -> RegressionTestResult:
        """Run full regression test suite."""
        test_id = f"regression_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Starting regression test {test_id} with {len(self.golden_dataset)} queries")
        
        # Evaluate current
        current_metrics = self.evaluator.evaluate_dataset(self.golden_dataset)
        
        # Load baseline if available
        baseline_metrics = self._load_baseline()
        
        # Check thresholds
        passes_threshold, threshold_failures = current_metrics.passes_threshold(self.thresholds)
        
        # Compare to baseline
        regressions = []
        improvements = []
        
        if baseline_metrics:
            regressions, improvements = self._compare_to_baseline(current_metrics, baseline_metrics)
        
        # Combine failures
        all_regressions = threshold_failures + regressions
        
        passed = len(all_regressions) == 0
        
        result = RegressionTestResult(
            test_id=test_id,
            passed=passed,
            current_metrics=current_metrics,
            baseline_metrics=baseline_metrics,
            regressions=all_regressions,
            improvements=improvements,
        )
        
        # Log result
        if passed:
            logger.info(f"Regression test {test_id} PASSED")
        else:
            logger.error(f"Regression test {test_id} FAILED: {all_regressions}")
        
        return result
    
    def save_as_baseline(self, metrics: AggregateRetrievalMetrics):
        """Save current metrics as new baseline."""
        if self.baseline_path:
            with open(self.baseline_path, 'w') as f:
                json.dump(metrics.to_dict(), f, indent=2)
            logger.info(f"Saved new baseline to {self.baseline_path}")
    
    def _load_baseline(self) -> Optional[AggregateRetrievalMetrics]:
        """Load baseline metrics from file."""
        if not self.baseline_path or not self.baseline_path.exists():
            return None
        
        try:
            with open(self.baseline_path) as f:
                data = json.load(f)
            
            return AggregateRetrievalMetrics(
                total_queries=data["total_queries"],
                hit_rate_at_1=data["hit_rate_at_1"],
                hit_rate_at_3=data["hit_rate_at_3"],
                hit_rate_at_5=data["hit_rate_at_5"],
                hit_rate_at_10=data.get("hit_rate_at_10", 0),
                mean_mrr=data["mean_mrr"],
                mean_ndcg=data["mean_ndcg"],
                mean_precision=data["mean_precision"],
                mean_recall=data["mean_recall"],
                mean_f1=data["mean_f1"],
                mean_latency_ms=data["mean_latency_ms"],
                p50_latency_ms=data.get("p50_latency_ms", data["mean_latency_ms"]),
                p95_latency_ms=data["p95_latency_ms"],
                p99_latency_ms=data["p99_latency_ms"],
                evaluation_id="baseline",
            )
        except Exception as e:
            logger.warning(f"Failed to load baseline: {e}")
            return None
    
    def _compare_to_baseline(
        self,
        current: AggregateRetrievalMetrics,
        baseline: AggregateRetrievalMetrics,
    ) -> Tuple[List[str], List[str]]:
        """Compare current metrics to baseline."""
        regressions = []
        improvements = []
        
        # Hit rate (5% tolerance for regression)
        if current.hit_rate_at_3 < baseline.hit_rate_at_3 * 0.95:
            regressions.append(
                f"Hit rate @3 regressed: {baseline.hit_rate_at_3:.2%} → {current.hit_rate_at_3:.2%}"
            )
        elif current.hit_rate_at_3 > baseline.hit_rate_at_3 * 1.05:
            improvements.append(
                f"Hit rate @3 improved: {baseline.hit_rate_at_3:.2%} → {current.hit_rate_at_3:.2%}"
            )
        
        # MRR
        if current.mean_mrr < baseline.mean_mrr * 0.95:
            regressions.append(
                f"MRR regressed: {baseline.mean_mrr:.3f} → {current.mean_mrr:.3f}"
            )
        elif current.mean_mrr > baseline.mean_mrr * 1.05:
            improvements.append(
                f"MRR improved: {baseline.mean_mrr:.3f} → {current.mean_mrr:.3f}"
            )
        
        # Latency (20% tolerance)
        if current.p95_latency_ms > baseline.p95_latency_ms * 1.2:
            regressions.append(
                f"P95 latency regressed: {baseline.p95_latency_ms:.0f}ms → {current.p95_latency_ms:.0f}ms"
            )
        elif current.p95_latency_ms < baseline.p95_latency_ms * 0.8:
            improvements.append(
                f"P95 latency improved: {baseline.p95_latency_ms:.0f}ms → {current.p95_latency_ms:.0f}ms"
            )
        
        return regressions, improvements
