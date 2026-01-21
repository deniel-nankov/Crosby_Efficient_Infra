"""
Evaluation Test Suite - pytest tests for RAG quality

Run with:
    pytest tests/eval/test_retrieval.py -v
    pytest tests/eval/test_retrieval.py -k "concentration" -v
    pytest tests/eval/test_retrieval.py --tb=short
"""

import pytest
import time
from typing import List, Set
from . import COMPLIANCE_EVAL_DATASET, EvaluationResult


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture(scope="module")
def retriever():
    """Get RAG retriever for testing."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
    
    try:
        from rag import RAGRetriever, VectorStore, LocalEmbedder
        
        connection_params = {
            "host": "localhost",
            "port": 5432,
            "database": "compliance",
            "user": "compliance_user",
            "password": "compliance_dev_password_123",
        }
        
        vector_store = VectorStore(connection_params)
        embedder = LocalEmbedder()
        
        if not embedder.available:
            pytest.skip("Embedding service not available")
        
        return RAGRetriever(
            vector_store=vector_store,
            embedder=embedder,
            use_hybrid=True,
            use_reranking=False,  # Faster for testing
        )
    except Exception as e:
        pytest.skip(f"Could not initialize retriever: {e}")


@pytest.fixture(scope="module")
def eval_dataset():
    """Get evaluation dataset."""
    return COMPLIANCE_EVAL_DATASET


# =============================================================================
# RETRIEVAL TESTS
# =============================================================================

class TestRetrievalQuality:
    """Test retrieval quality against ground-truth dataset."""
    
    @pytest.mark.parametrize("test_case", COMPLIANCE_EVAL_DATASET, ids=lambda x: x["id"])
    def test_retrieval_hit(self, retriever, test_case):
        """Test that relevant documents are retrieved."""
        query = test_case["query"]
        expected_docs = set(test_case["relevant_documents"])
        
        # Retrieve
        start = time.time()
        result = retriever.retrieve_for_control(
            control_name=query,
            control_type=test_case["category"],
            status="pass",
            calculated_value=0.0,
            threshold=0.0,
            limit=5,
        )
        latency_ms = (time.time() - start) * 1000
        
        # Check for hits
        retrieved_docs = set(c.document_name for c in result.chunks)
        hits = retrieved_docs & expected_docs
        
        # At least one expected document should be retrieved
        assert len(hits) > 0, (
            f"Query '{query[:50]}...' retrieved {retrieved_docs} "
            f"but expected at least one of {expected_docs}"
        )
    
    @pytest.mark.parametrize("test_case", [
        tc for tc in COMPLIANCE_EVAL_DATASET if tc["difficulty"] == "easy"
    ], ids=lambda x: x["id"])
    def test_easy_queries_high_confidence(self, retriever, test_case):
        """Easy queries should have high retrieval confidence."""
        result = retriever.retrieve_for_control(
            control_name=test_case["query"],
            control_type=test_case["category"],
            status="pass",
            calculated_value=0.0,
            threshold=0.0,
            limit=3,
        )
        
        assert result.confidence >= 0.5, (
            f"Easy query '{test_case['id']}' had low confidence: {result.confidence:.2%}"
        )
    
    def test_retrieval_latency_p95(self, retriever, eval_dataset):
        """P95 latency should be under 2 seconds."""
        latencies = []
        
        for test_case in eval_dataset[:10]:  # Sample for speed
            start = time.time()
            retriever.retrieve_for_control(
                control_name=test_case["query"],
                control_type=test_case["category"],
                status="pass",
                calculated_value=0.0,
                threshold=0.0,
                limit=3,
            )
            latencies.append((time.time() - start) * 1000)
        
        latencies.sort()
        p95 = latencies[int(len(latencies) * 0.95)] if len(latencies) >= 20 else latencies[-1]
        
        assert p95 < 2000, f"P95 latency {p95:.0f}ms exceeds 2000ms threshold"


class TestRetrievalByCategory:
    """Test retrieval quality by category."""
    
    @pytest.mark.parametrize("category", ["concentration", "liquidity", "exposure", "exception"])
    def test_category_pass_rate(self, retriever, category):
        """Each category should have at least 70% retrieval hit rate."""
        category_tests = [tc for tc in COMPLIANCE_EVAL_DATASET if tc["category"] == category]
        
        if not category_tests:
            pytest.skip(f"No tests for category: {category}")
        
        hits = 0
        for test_case in category_tests:
            result = retriever.retrieve_for_control(
                control_name=test_case["query"],
                control_type=test_case["category"],
                status="pass",
                calculated_value=0.0,
                threshold=0.0,
                limit=5,
            )
            
            expected_docs = set(test_case["relevant_documents"])
            retrieved_docs = set(c.document_name for c in result.chunks)
            
            if expected_docs & retrieved_docs:
                hits += 1
        
        pass_rate = hits / len(category_tests)
        assert pass_rate >= 0.70, (
            f"Category '{category}' pass rate {pass_rate:.1%} below 70% threshold"
        )


# =============================================================================
# CONFIDENCE CALIBRATION TESTS
# =============================================================================

class TestConfidenceCalibration:
    """Test that confidence scores are well-calibrated."""
    
    def test_confidence_correlates_with_quality(self, retriever, eval_dataset):
        """High confidence should correlate with correct retrievals."""
        high_conf_correct = 0
        high_conf_total = 0
        low_conf_correct = 0
        low_conf_total = 0
        
        for test_case in eval_dataset:
            result = retriever.retrieve_for_control(
                control_name=test_case["query"],
                control_type=test_case["category"],
                status="pass",
                calculated_value=0.0,
                threshold=0.0,
                limit=3,
            )
            
            expected_docs = set(test_case["relevant_documents"])
            retrieved_docs = set(c.document_name for c in result.chunks)
            is_correct = bool(expected_docs & retrieved_docs)
            
            if result.confidence >= 0.7:
                high_conf_total += 1
                if is_correct:
                    high_conf_correct += 1
            else:
                low_conf_total += 1
                if is_correct:
                    low_conf_correct += 1
        
        if high_conf_total > 0 and low_conf_total > 0:
            high_conf_accuracy = high_conf_correct / high_conf_total
            low_conf_accuracy = low_conf_correct / low_conf_total
            
            # High confidence should have higher accuracy than low confidence
            assert high_conf_accuracy >= low_conf_accuracy, (
                f"Confidence miscalibrated: high conf accuracy {high_conf_accuracy:.1%} "
                f"< low conf accuracy {low_conf_accuracy:.1%}"
            )
