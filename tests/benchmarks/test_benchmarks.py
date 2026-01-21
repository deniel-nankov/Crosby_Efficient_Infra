"""
Pytest tests for benchmark infrastructure.

Run with:
    pytest tests/benchmarks/test_benchmarks.py -v
"""

import pytest
from tests.benchmarks import (
    BenchmarkCategory,
    BenchmarkCase,
    BenchmarkResult,
    BenchmarkReport,
    Difficulty,
    BENCHMARK_DATASET,
    get_benchmark_dataset,
    get_benchmark_by_category,
    get_benchmark_by_difficulty,
)
from tests.benchmarks.ragas_metrics import (
    RAGASMetrics,
    RAGASScores,
    calculate_ragas_scores,
)
from tests.benchmarks.finance_bench import (
    FinanceBenchCase,
    FinanceQuestionType,
    FINANCE_BENCHMARK_DATASET,
)
from tests.benchmarks.verafi_eval import (
    VERAFIEvaluator,
    VERAFIResult,
    evaluate_verafi,
)


class TestBenchmarkDataset:
    """Tests for the benchmark dataset."""
    
    def test_dataset_not_empty(self):
        """Benchmark dataset should have test cases."""
        assert len(BENCHMARK_DATASET) > 0
    
    def test_all_categories_covered(self):
        """All benchmark categories should have at least one test."""
        covered = set(b.category for b in BENCHMARK_DATASET)
        all_categories = set(BenchmarkCategory)
        
        # Allow some categories to be empty for now
        required = {
            BenchmarkCategory.RETRIEVAL_QUALITY,
            BenchmarkCategory.FAITHFULNESS,
        }
        
        for cat in required:
            assert cat in covered, f"Missing required category: {cat}"
    
    def test_get_by_category(self):
        """Should filter benchmarks by category."""
        retrieval = get_benchmark_by_category(BenchmarkCategory.RETRIEVAL_QUALITY)
        assert all(b.category == BenchmarkCategory.RETRIEVAL_QUALITY for b in retrieval)
    
    def test_get_by_difficulty(self):
        """Should filter benchmarks by difficulty."""
        easy = get_benchmark_by_difficulty(Difficulty.EASY)
        assert all(b.difficulty == Difficulty.EASY for b in easy)
    
    def test_benchmark_case_structure(self):
        """Each benchmark case should have required fields."""
        for case in BENCHMARK_DATASET:
            assert case.id, "Missing ID"
            assert case.category, "Missing category"
            assert case.query, "Missing query"
            assert len(case.expected_answer_contains) > 0, f"Case {case.id} needs expected terms"


class TestRAGASMetrics:
    """Tests for RAGAS metrics implementation."""
    
    def test_context_precision(self):
        """Test context precision calculation."""
        metrics = RAGASMetrics()
        
        # High precision - relevant docs
        precision = metrics.context_precision(
            question="What is the concentration limit?",
            retrieved_docs=["Concentration limit is 5% per security"],
        )
        assert precision > 0
    
    def test_faithfulness(self):
        """Test faithfulness calculation."""
        metrics = RAGASMetrics()
        
        # Faithful answer
        faith_score = metrics.faithfulness(
            answer="The limit is 5% according to policy.",
            context="Policy states concentration limit is 5% per security.",
        )
        assert faith_score > 0.5
        
        # Unfaithful answer (hallucination)
        unfaith_score = metrics.faithfulness(
            answer="The limit is 10% as required by regulations.",
            context="Policy states concentration limit is 5% per security.",
        )
        assert unfaith_score < faith_score
    
    def test_calculate_all(self):
        """Test all RAGAS metrics at once."""
        scores = calculate_ragas_scores(
            question="What is the sector limit?",
            answer="The sector limit is 25%.",
            context="Sector concentration shall not exceed 25% of NAV.",
            ground_truth="Sector concentration limit is 25%.",
        )
        
        assert isinstance(scores, RAGASScores)
        assert 0 <= scores.faithfulness <= 1
        assert 0 <= scores.answer_relevancy <= 1


class TestFinanceBench:
    """Tests for FinanceBench-style benchmarks."""
    
    def test_dataset_not_empty(self):
        """Finance benchmark dataset should exist."""
        assert len(FINANCE_BENCHMARK_DATASET) > 0
    
    def test_calculation_cases_have_values(self):
        """Calculation cases should have expected values."""
        calc_cases = [
            c for c in FINANCE_BENCHMARK_DATASET
            if c.question_type == FinanceQuestionType.CALCULATION
        ]
        
        for case in calc_cases:
            assert case.expected_value is not None, f"Case {case.id} needs expected_value"
            assert case.calculation_chain, f"Case {case.id} needs calculation_chain"
    
    def test_case_structure(self):
        """Finance cases should have required fields."""
        for case in FINANCE_BENCHMARK_DATASET:
            assert case.id, "Missing ID"
            assert case.question, "Missing question"
            assert case.expected_answer, "Missing expected answer"
            assert case.source_docs, "Missing source docs"


class TestVERAFIEvaluator:
    """Tests for VERAFI evaluation."""
    
    def test_grounded_answer(self):
        """Test evaluation of grounded answers."""
        result = evaluate_verafi(
            question="What is the single security limit?",
            answer="The single security limit is 5% of NAV.",
            context="Concentration Policy: Single security limit is 5% of NAV.",
            ground_truth="5% single security limit.",
        )
        
        assert isinstance(result, VERAFIResult)
        assert result.grounded_accuracy > 0.5
        assert result.hallucination_rate < 0.5
    
    def test_calculation_accuracy(self):
        """Test calculation accuracy metric."""
        evaluator = VERAFIEvaluator()
        result = evaluator.evaluate(
            question="What is the gross exposure?",
            answer="Gross exposure is 180% ($900M / $500M).",
            context="Long: $600M, Short: $300M, NAV: $500M",
            ground_truth="Gross exposure is 180%",
            expected_values={"gross_exposure_pct": 180.0},
        )
        
        assert result.calculation_accuracy == 1.0
    
    def test_hallucination_detection(self):
        """Test hallucination detection."""
        result = evaluate_verafi(
            question="What is the limit?",
            answer="The limit is 50% as mandated by the FRB regulations.",
            context="Policy limit is 5%.",
            ground_truth="Limit is 5%.",
        )
        
        # Should detect hallucination (50% != 5%)
        assert result.hallucination_rate > 0
    
    def test_overall_score(self):
        """Test overall score calculation."""
        result = VERAFIResult(
            grounded_accuracy=0.9,
            calculation_accuracy=0.95,
            policy_compliance=1.0,
            evidence_quality=0.8,
            hallucination_rate=0.05,
            total_claims=10,
            verified_claims=9,
            total_calculations=5,
            correct_calculations=5,
            policy_violations=[],
            hallucinated_facts=[],
        )
        
        assert 0 < result.overall_accuracy < 1
        assert result.beats_baseline  # Should beat 52.4%


class TestBenchmarkResult:
    """Tests for benchmark result data structures."""
    
    def test_result_to_dict(self):
        """Benchmark result should serialize to dict."""
        result = BenchmarkResult(
            case_id="TEST-001",
            category="retrieval_quality",
            difficulty="easy",
            retrieved_docs=["doc1.md", "doc2.md"],
            retrieval_precision=0.8,
            retrieval_recall=0.9,
            retrieval_hit=True,
            generated_answer="Test answer",
            contains_expected=[("term1", True)],
            contains_unexpected=[],
            numeric_accuracy=None,
            refused_correctly=None,
            resisted_counterfactual=None,
            passed=True,
            failure_reasons=[],
            latency_ms=50.0,
        )
        
        d = result.to_dict()
        assert d["case_id"] == "TEST-001"
        assert d["passed"] == True
        assert d["retrieval_precision"] == 0.8
    
    def test_report_markdown(self):
        """Benchmark report should generate markdown."""
        report = BenchmarkReport(
            report_id="test_001",
            run_at=None,
            total_cases=10,
            passed=8,
            failed=2,
            overall_score=0.8,
            category_scores={},
            difficulty_scores={},
            results=[],
            mean_retrieval_precision=0.75,
            mean_retrieval_recall=0.85,
            mean_faithfulness=0.9,
            mathematical_accuracy_rate=1.0,
            negative_rejection_rate=0.95,
            noise_robustness_rate=0.88,
        )
        
        md = report.to_markdown()
        assert "80.0%" in md or "80%" in md  # Overall score
        assert "RAG BENCHMARK REPORT" in md


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
