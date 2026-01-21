"""
Performance and Stress Tests

Comprehensive performance testing suite for the compliance system:
- Throughput benchmarks
- Latency measurements
- Stress testing under load
- Memory profiling
- Scalability tests

Test Categories:
- TestRetrievalPerformance: Search latency tests
- TestDatabasePerformance: Database operation benchmarks
- TestAgentPerformance: Agent execution timing
- TestScalability: System scalability tests
- TestStressTests: High-load stress tests
"""

from __future__ import annotations

import pytest
import sys
import time
import statistics
import gc
import json
import hashlib
from typing import List, Dict, Any, Callable
from unittest.mock import MagicMock
from dataclasses import dataclass
from contextlib import contextmanager
from pathlib import Path
from datetime import date

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

pytestmark = [pytest.mark.performance]


# =============================================================================
# PERFORMANCE TEST UTILITIES
# =============================================================================

@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    name: str
    iterations: int
    total_time_ms: float
    avg_time_ms: float
    min_time_ms: float
    max_time_ms: float
    std_dev_ms: float
    throughput_per_sec: float
    
    def __str__(self) -> str:
        return (
            f"{self.name}:\n"
            f"  Iterations: {self.iterations}\n"
            f"  Total: {self.total_time_ms:.2f}ms\n"
            f"  Avg: {self.avg_time_ms:.2f}ms\n"
            f"  Min: {self.min_time_ms:.2f}ms\n"
            f"  Max: {self.max_time_ms:.2f}ms\n"
            f"  StdDev: {self.std_dev_ms:.2f}ms\n"
            f"  Throughput: {self.throughput_per_sec:.2f}/sec"
        )


def benchmark(func: Callable, iterations: int = 100, warmup: int = 5) -> BenchmarkResult:
    """
    Run a benchmark on a function.
    
    Args:
        func: Function to benchmark (takes no arguments)
        iterations: Number of iterations to run
        warmup: Number of warmup iterations
        
    Returns:
        BenchmarkResult with timing statistics
    """
    # Warmup
    for _ in range(warmup):
        func()
    
    # Force garbage collection
    gc.collect()
    
    # Actual benchmark
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        elapsed_ms = (time.perf_counter() - start) * 1000
        times.append(elapsed_ms)
    
    total = sum(times)
    avg = statistics.mean(times)
    std_dev = statistics.stdev(times) if len(times) > 1 else 0
    
    return BenchmarkResult(
        name=func.__name__ if hasattr(func, '__name__') else "benchmark",
        iterations=iterations,
        total_time_ms=total,
        avg_time_ms=avg,
        min_time_ms=min(times),
        max_time_ms=max(times),
        std_dev_ms=std_dev,
        throughput_per_sec=(iterations / total) * 1000 if total > 0 else 0,
    )


@contextmanager
def timer(name: str = "Operation"):
    """Context manager for timing operations."""
    start = time.perf_counter()
    yield
    elapsed = (time.perf_counter() - start) * 1000
    print(f"{name}: {elapsed:.2f}ms")


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def db_connection():
    """Create database connection for testing."""
    try:
        import psycopg2
        conn = psycopg2.connect(
            host="localhost",
            port=5433,
            database="compliance",
            user="compliance_user",
            password="compliance_dev_password_123"
        )
        yield conn
        conn.rollback()
        conn.close()
    except Exception as e:
        pytest.skip(f"Database not available: {e}")


@pytest.fixture
def sample_documents() -> Dict[str, str]:
    """Sample documents for testing."""
    base_docs = [
        "Concentration limits apply to sector exposure. The maximum allocation to any single sector should not exceed 30% of the portfolio NAV.",
        "Liquidity thresholds must be maintained at all times. A minimum of 10% of the portfolio should be held in liquid assets.",
        "SEC compliance requires filing Form PF quarterly. All position data must be accurately reported.",
        "Counterparty exposure limits are set at 5% of portfolio NAV. Derivatives positions count toward this limit.",
        "Risk limits are reviewed monthly by the investment committee. Breaches must be reported within 24 hours.",
    ]
    # Return dict of 100 documents
    return {f"doc_{i}": doc for i, doc in enumerate(base_docs * 20)}


@pytest.fixture
def bm25_index(sample_documents):
    """Create BM25 index with sample documents."""
    from rag.retriever import BM25Index
    
    index = BM25Index()
    index.build_index(sample_documents)
    return index


# =============================================================================
# RETRIEVAL PERFORMANCE TESTS
# =============================================================================

class TestRetrievalPerformance:
    """Test retrieval performance."""
    
    def test_bm25_index_build_performance(self, sample_documents):
        """BM25 index building should be fast."""
        from rag.retriever import BM25Index
        
        def build_index():
            index = BM25Index()
            index.build_index(sample_documents)
        
        result = benchmark(build_index, iterations=10, warmup=2)
        
        # Should build 100 docs in under 100ms on average
        assert result.avg_time_ms < 100, f"Index build too slow: {result.avg_time_ms:.2f}ms"
    
    def test_bm25_search_performance(self, bm25_index):
        """BM25 search should be fast."""
        queries = [
            "concentration limits sector",
            "liquidity thresholds portfolio",
            "SEC compliance Form PF",
            "counterparty exposure derivatives",
            "risk limits monthly review",
        ]
        
        def search_all():
            for query in queries:
                bm25_index.search(query, top_k=5)
        
        result = benchmark(search_all, iterations=50, warmup=5)
        
        # Should search 5 queries in under 50ms on average
        assert result.avg_time_ms < 50, f"Search too slow: {result.avg_time_ms:.2f}ms"
    
    def test_bm25_search_scaling(self, sample_documents):
        """Search time should scale reasonably with document count."""
        from rag.retriever import BM25Index
        
        times = []
        doc_list = list(sample_documents.values())
        for doc_count in [10, 50, 100]:
            docs = {f"doc_{i}": doc_list[i % len(doc_list)] for i in range(doc_count)}
            index = BM25Index()
            index.build_index(docs)
            
            start = time.perf_counter()
            for _ in range(10):
                index.search("concentration limits", top_k=5)
            elapsed = (time.perf_counter() - start) * 1000
            times.append((doc_count, elapsed))
        
        # Check that scaling is sub-linear (not O(n^2) or worse)
        ratio = times[2][1] / times[0][1]  # 100 docs vs 10 docs
        assert ratio < 20, f"Scaling too poor: {ratio:.2f}x slowdown for 10x docs"
    
    def test_confidence_calibrator_performance(self):
        """Confidence calibration should be fast with mock chunks."""
        from rag.retriever import ConfidenceCalibrator
        from dataclasses import dataclass
        
        @dataclass
        class MockChunk:
            content: str
            similarity: float = 0.8
        
        calibrator = ConfidenceCalibrator()
        mock_chunks = [
            MockChunk(content="Concentration limits are set at 30% threshold.", similarity=0.85),
            MockChunk(content="Sector exposure should not exceed limits.", similarity=0.75),
            MockChunk(content="Risk limits are reviewed monthly.", similarity=0.65),
        ]
        
        def calibrate_batch():
            for _ in range(5):
                calibrator.calculate("concentration limits", mock_chunks)  # type: ignore[arg-type]
        
        result = benchmark(calibrate_batch, iterations=100, warmup=10)
        
        # Should be fast
        assert result.avg_time_ms < 5, f"Calibration too slow: {result.avg_time_ms:.2f}ms"
    
    def test_query_rewriter_performance(self):
        """Query rewriting should be fast."""
        from rag.retriever import QueryRewriter
        
        rewriter = QueryRewriter()
        queries = [
            "concentration limits",
            "liquidity thresholds",
            "SEC compliance",
            "counterparty exposure",
            "risk limits",
        ]
        
        def rewrite_all():
            for query in queries:
                rewriter.rewrite(query)
        
        result = benchmark(rewrite_all, iterations=50, warmup=5)
        
        # Should rewrite 5 queries in under 10ms
        assert result.avg_time_ms < 10, f"Rewriting too slow: {result.avg_time_ms:.2f}ms"


# =============================================================================
# DATABASE PERFORMANCE TESTS
# =============================================================================

class TestDatabasePerformance:
    """Test database operation performance."""
    
    @pytest.mark.requires_db
    def test_simple_query_latency(self, db_connection):
        """Simple queries should be fast."""
        def simple_query():
            with db_connection.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
        
        result = benchmark(simple_query, iterations=100, warmup=10)
        
        # Simple query should be under 5ms on average
        assert result.avg_time_ms < 5, f"Simple query too slow: {result.avg_time_ms:.2f}ms"
    
    @pytest.mark.requires_db
    def test_chunk_count_query(self, db_connection):
        """Counting chunks should be fast."""
        def count_chunks():
            with db_connection.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM policy_chunks")
                cur.fetchone()
        
        result = benchmark(count_chunks, iterations=50, warmup=5)
        
        # Count query should be under 10ms on average
        assert result.avg_time_ms < 10, f"Count too slow: {result.avg_time_ms:.2f}ms"
    
    @pytest.mark.requires_db
    def test_fetch_chunks_performance(self, db_connection):
        """Fetching chunks should be fast."""
        def fetch_chunks():
            with db_connection.cursor() as cur:
                cur.execute("SELECT chunk_id, content FROM policy_chunks LIMIT 10")
                cur.fetchall()
        
        result = benchmark(fetch_chunks, iterations=50, warmup=5)
        
        # Fetching 10 chunks should be under 20ms
        assert result.avg_time_ms < 20, f"Fetch too slow: {result.avg_time_ms:.2f}ms"


# =============================================================================
# CONTROL EVALUATION PERFORMANCE TESTS
# =============================================================================

class TestControlEvaluationPerformance:
    """Test control evaluation performance."""
    
    def test_threshold_evaluation_performance(self):
        """Threshold evaluation should be very fast."""
        from control_runner.controls import ControlDefinition, ControlCategory, ThresholdOperator
        
        control = ControlDefinition(
            control_code="PERF_001",
            control_name="Performance Test",
            description="Performance test control",
            category=ControlCategory.CONCENTRATION,
            computation_sql="SELECT 0.25 AS calculated_value",
            threshold_value=0.30,
            threshold_operator=ThresholdOperator.GTE,
        )
        
        def evaluate_many():
            for value in [0.1, 0.2, 0.25, 0.30, 0.35, 0.40]:
                control.evaluate_threshold(value)
        
        result = benchmark(evaluate_many, iterations=1000, warmup=100)
        
        # Pure Python comparison should be < 0.1ms
        assert result.avg_time_ms < 0.1, f"Evaluation too slow: {result.avg_time_ms:.4f}ms"
    
    def test_control_creation_performance(self):
        """Control creation should be fast."""
        from control_runner.controls import ControlDefinition, ControlCategory, ThresholdOperator
        
        def create_controls():
            for i in range(10):
                ControlDefinition(
                    control_code=f"BATCH_{i}",
                    control_name=f"Batch Control {i}",
                    description=f"Batch test control {i}",
                    category=ControlCategory.CONCENTRATION,
                    computation_sql=f"SELECT {i * 0.1} AS calculated_value",
                    threshold_value=0.30,
                    threshold_operator=ThresholdOperator.GTE,
                )
        
        result = benchmark(create_controls, iterations=100, warmup=10)
        
        # Creating 10 controls should be < 5ms
        assert result.avg_time_ms < 5, f"Creation too slow: {result.avg_time_ms:.2f}ms"


# =============================================================================
# HASHING PERFORMANCE TESTS
# =============================================================================

class TestHashingPerformance:
    """Test hashing performance for evidence chain."""
    
    def test_sha256_hashing_performance(self):
        """SHA256 hashing should be fast."""
        data = json.dumps({
            "control_code": "CONC_001",
            "calculated_value": 0.25,
            "threshold": 0.30,
            "status": "pass",
            "timestamp": "2026-01-19T12:00:00Z",
        })
        
        def hash_data():
            hashlib.sha256(data.encode()).hexdigest()
        
        result = benchmark(hash_data, iterations=10000, warmup=100)
        
        # SHA256 should be < 0.01ms per hash
        assert result.avg_time_ms < 0.01, f"Hashing too slow: {result.avg_time_ms:.4f}ms"
    
    def test_large_data_hashing(self):
        """Hashing larger data should still be fast."""
        # Simulate large evidence payload
        large_data = json.dumps({
            "positions": [{"id": i, "value": i * 1000} for i in range(1000)],
            "controls": [{"id": i, "status": "pass"} for i in range(50)],
        })
        
        def hash_large():
            hashlib.sha256(large_data.encode()).hexdigest()
        
        result = benchmark(hash_large, iterations=1000, warmup=10)
        
        # Large data hash should be < 1ms
        assert result.avg_time_ms < 1, f"Large hash too slow: {result.avg_time_ms:.4f}ms"


# =============================================================================
# JSON SERIALIZATION PERFORMANCE TESTS
# =============================================================================

class TestSerializationPerformance:
    """Test JSON serialization performance."""
    
    def test_json_dumps_performance(self):
        """JSON serialization should be fast."""
        data = {
            "control_code": "CONC_001",
            "calculated_value": 0.25,
            "threshold": 0.30,
            "status": "pass",
            "positions": [{"id": i, "value": i * 100} for i in range(100)],
        }
        
        def serialize():
            json.dumps(data, sort_keys=True)
        
        result = benchmark(serialize, iterations=1000, warmup=100)
        
        # Serialization should be < 0.5ms
        assert result.avg_time_ms < 0.5, f"Serialization too slow: {result.avg_time_ms:.4f}ms"
    
    def test_json_loads_performance(self):
        """JSON deserialization should be fast."""
        data_str = json.dumps({
            "control_code": "CONC_001",
            "calculated_value": 0.25,
            "threshold": 0.30,
            "status": "pass",
            "positions": [{"id": i, "value": i * 100} for i in range(100)],
        })
        
        def deserialize():
            json.loads(data_str)
        
        result = benchmark(deserialize, iterations=1000, warmup=100)
        
        # Deserialization should be < 0.5ms
        assert result.avg_time_ms < 0.5, f"Deserialization too slow: {result.avg_time_ms:.4f}ms"


# =============================================================================
# SCALABILITY TESTS
# =============================================================================

class TestScalability:
    """Test system scalability."""
    
    def test_bm25_scales_with_documents(self):
        """BM25 should scale well with document count."""
        from rag.retriever import BM25Index
        
        base_doc = "Concentration limits apply to sector exposure and portfolio allocation."
        
        results = []
        for doc_count in [100, 500, 1000]:
            docs = {f"doc_{i}": f"{base_doc} Document {i}." for i in range(doc_count)}
            
            index = BM25Index()
            
            # Time index building
            start = time.perf_counter()
            index.build_index(docs)
            build_time = (time.perf_counter() - start) * 1000
            
            # Time search
            start = time.perf_counter()
            for _ in range(10):
                index.search("concentration limits", top_k=5)
            search_time = (time.perf_counter() - start) * 1000
            
            results.append({
                "docs": doc_count,
                "build_ms": build_time,
                "search_ms": search_time,
            })
        
        # Check that 10x docs doesn't cause 10x slowdown
        build_ratio = results[2]["build_ms"] / results[0]["build_ms"]
        search_ratio = results[2]["search_ms"] / results[0]["search_ms"]
        
        assert build_ratio < 15, f"Build scaling poor: {build_ratio:.2f}x"
        assert search_ratio < 15, f"Search scaling poor: {search_ratio:.2f}x"
    
    def test_control_evaluation_scales(self):
        """Control evaluation should scale linearly."""
        from control_runner.controls import ControlDefinition, ControlCategory, ThresholdOperator
        
        control = ControlDefinition(
            control_code="SCALE_001",
            control_name="Scaling Test",
            description="Tests scaling behavior",
            category=ControlCategory.CONCENTRATION,
            computation_sql="SELECT 0.25 AS calculated_value",
            threshold_value=0.30,
            threshold_operator=ThresholdOperator.GTE,
        )
        
        results = []
        for eval_count in [100, 1000, 10000]:
            start = time.perf_counter()
            for i in range(eval_count):
                control.evaluate_threshold(0.25 + (i % 10) * 0.01)
            elapsed = (time.perf_counter() - start) * 1000
            results.append({"count": eval_count, "time_ms": elapsed})
        
        # Should scale linearly
        ratio_100_to_10k = results[2]["time_ms"] / results[0]["time_ms"]
        expected_ratio = 100  # 10000 / 100
        
        # Allow 2x overhead for linear scaling
        assert ratio_100_to_10k < expected_ratio * 2, f"Scaling not linear: {ratio_100_to_10k:.2f}x"


# =============================================================================
# STRESS TESTS
# =============================================================================

class TestStressTests:
    """High-load stress tests."""
    
    def test_rapid_bm25_queries(self):
        """BM25 should handle rapid queries."""
        from rag.retriever import BM25Index
        
        docs = {
            f"doc_{i}": f"Document about compliance topic {i}. Contains various keywords."
            for i in range(100)
        }
        
        index = BM25Index()
        index.build_index(docs)
        
        queries = ["compliance", "topic", "keywords", "document", "various"]
        
        start = time.perf_counter()
        for _ in range(1000):
            for query in queries:
                index.search(query, top_k=5)
        elapsed = (time.perf_counter() - start) * 1000
        
        # 5000 queries should complete in under 5 seconds
        assert elapsed < 5000, f"Rapid queries too slow: {elapsed:.2f}ms"
    
    def test_memory_stability(self):
        """Memory should remain stable under load."""
        from rag.retriever import BM25Index
        import gc
        
        # Force garbage collection
        gc.collect()
        
        # Create and destroy many indexes
        for _ in range(10):
            docs = {f"doc_{i}": f"Test document {i}" for i in range(1000)}
            index = BM25Index()
            index.build_index(docs)
            
            for _ in range(100):
                index.search("test", top_k=5)
            
            del index
            gc.collect()
        
        # If we get here without OOM, test passes
        assert True
    
    def test_concurrent_control_creation(self):
        """Control creation should be thread-safe."""
        from control_runner.controls import ControlDefinition, ControlCategory, ThresholdOperator
        from concurrent.futures import ThreadPoolExecutor
        
        def create_control(i):
            return ControlDefinition(
                control_code=f"CONC_{i:04d}",
                control_name=f"Concurrent Control {i}",
                description=f"Thread-safe test {i}",
                category=ControlCategory.CONCENTRATION,
                computation_sql=f"SELECT {i} AS calculated_value",
                threshold_value=0.30,
                threshold_operator=ThresholdOperator.GTE,
            )
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(create_control, range(100)))
        
        assert len(results) == 100
        assert all(r is not None for r in results)


# =============================================================================
# PERFORMANCE REGRESSION TESTS
# =============================================================================

class TestPerformanceRegression:
    """Regression tests for performance."""
    
    def test_bm25_search_regression(self):
        """BM25 search should meet baseline performance."""
        from rag.retriever import BM25Index
        
        docs = {
            f"doc_{i}": doc for i, doc in enumerate([
                "Concentration limits for sector exposure",
                "Liquidity requirements and thresholds",
                "SEC compliance and Form PF reporting",
            ] * 50)  # 150 docs
        }
        
        index = BM25Index()
        index.build_index(docs)
        
        times = []
        for _ in range(100):
            start = time.perf_counter()
            index.search("concentration limits", top_k=5)
            times.append((time.perf_counter() - start) * 1000)
        
        avg_time = statistics.mean(times)
        p99_time = sorted(times)[98]
        
        # Baseline expectations
        assert avg_time < 5, f"Avg search time regressed: {avg_time:.2f}ms"
        assert p99_time < 20, f"P99 search time regressed: {p99_time:.2f}ms"
    
    def test_hash_performance_regression(self):
        """Hashing should meet baseline performance."""
        data = json.dumps({"test": "data", "values": list(range(100))})
        
        times = []
        for _ in range(1000):
            start = time.perf_counter()
            hashlib.sha256(data.encode()).hexdigest()
            times.append((time.perf_counter() - start) * 1000)
        
        avg_time = statistics.mean(times)
        
        assert avg_time < 0.1, f"Hash time regressed: {avg_time:.4f}ms"
