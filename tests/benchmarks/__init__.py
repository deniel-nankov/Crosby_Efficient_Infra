"""
RAG Benchmark Suite - Comprehensive Testing for Production RAG Systems

This module implements industry-standard RAG benchmarks adapted for
financial compliance systems:

1. **Retrieval Quality** - Hit rate, MRR, NDCG, precision, recall
2. **Noise Robustness** - Can system handle irrelevant retrieved docs?
3. **Negative Rejection** - Does system refuse to answer when it shouldn't?
4. **Counterfactual Robustness** - Does system resist contradictory info?
5. **Information Integration** - Can system combine multiple sources?
6. **Faithfulness** - Is output grounded in retrieved context?
7. **Mathematical Accuracy** - Are calculations correct? (VERAFI-inspired)
8. **Temporal Consistency** - Are date/time references correct?

Based on:
- RGB Benchmark (Chen et al. 2024)
- RAGAS Metrics (Es et al. 2023)
- VERAFI (Bayless et al. 2025)
- FinanceBench (Islam et al. 2023)

Usage:
    python -m tests.benchmarks.run_benchmarks
    pytest tests/benchmarks/ -v
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import json


class BenchmarkCategory(Enum):
    """Categories of RAG benchmarks."""
    RETRIEVAL_QUALITY = "retrieval_quality"
    NOISE_ROBUSTNESS = "noise_robustness"
    NEGATIVE_REJECTION = "negative_rejection"
    COUNTERFACTUAL_ROBUSTNESS = "counterfactual_robustness"
    INFORMATION_INTEGRATION = "information_integration"
    FAITHFULNESS = "faithfulness"
    MATHEMATICAL_ACCURACY = "mathematical_accuracy"
    TEMPORAL_CONSISTENCY = "temporal_consistency"


class Difficulty(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


@dataclass
class BenchmarkCase:
    """A single benchmark test case."""
    id: str
    category: BenchmarkCategory
    difficulty: Difficulty
    
    # Input
    query: str
    context_documents: List[str]  # Which docs should be relevant
    
    # For noise robustness - inject irrelevant docs
    noise_documents: List[str] = field(default_factory=list)
    
    # Expected output
    expected_answer_contains: List[str] = field(default_factory=list)
    expected_answer_not_contains: List[str] = field(default_factory=list)
    expected_numeric_values: Dict[str, float] = field(default_factory=dict)
    
    # For negative rejection - query that should NOT be answered
    should_refuse: bool = False
    refusal_reason: Optional[str] = None
    
    # For counterfactual - conflicting information
    counterfactual_claim: Optional[str] = None
    correct_answer: Optional[str] = None
    
    # For temporal
    temporal_references: List[Dict[str, str]] = field(default_factory=list)
    
    # Metadata
    source: str = "compliance"  # e.g., "financebench", "compliance", "custom"
    notes: Optional[str] = None


@dataclass
class BenchmarkResult:
    """Result of running a single benchmark case."""
    case_id: str
    category: str
    difficulty: str
    
    # Retrieval metrics
    retrieved_docs: List[str]
    retrieval_precision: float
    retrieval_recall: float
    retrieval_hit: bool
    
    # Generation metrics
    generated_answer: Optional[str]
    contains_expected: List[Tuple[str, bool]]  # (term, found)
    contains_unexpected: List[Tuple[str, bool]]  # (term, found - should be False)
    
    # Specialized metrics
    numeric_accuracy: Optional[Dict[str, Dict]] = None  # {name: {expected, actual, correct}}
    faithfulness_score: float = 0.0
    refused_correctly: Optional[bool] = None
    resisted_counterfactual: Optional[bool] = None
    temporal_correct: Optional[bool] = None
    
    # Overall
    passed: bool = False
    failure_reasons: List[str] = field(default_factory=list)
    latency_ms: float = 0.0


@dataclass
class BenchmarkReport:
    """Complete benchmark report across all categories."""
    report_id: str
    run_at: datetime
    
    # Overall
    total_cases: int
    passed: int
    failed: int
    overall_score: float
    
    # By category
    category_scores: Dict[str, Dict[str, float]]
    
    # By difficulty
    difficulty_scores: Dict[str, Dict[str, float]]
    
    # Detailed results
    results: List[BenchmarkResult]
    
    # Key metrics
    mean_retrieval_precision: float
    mean_retrieval_recall: float
    mean_faithfulness: float
    mathematical_accuracy_rate: float
    negative_rejection_rate: float
    noise_robustness_rate: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "report_id": self.report_id,
            "run_at": self.run_at.isoformat(),
            "total_cases": self.total_cases,
            "passed": self.passed,
            "failed": self.failed,
            "overall_score": round(self.overall_score, 4),
            "category_scores": self.category_scores,
            "difficulty_scores": self.difficulty_scores,
            "mean_retrieval_precision": round(self.mean_retrieval_precision, 4),
            "mean_retrieval_recall": round(self.mean_retrieval_recall, 4),
            "mean_faithfulness": round(self.mean_faithfulness, 4),
            "mathematical_accuracy_rate": round(self.mathematical_accuracy_rate, 4),
            "negative_rejection_rate": round(self.negative_rejection_rate, 4),
            "noise_robustness_rate": round(self.noise_robustness_rate, 4),
        }
    
    def to_markdown(self) -> str:
        """Generate markdown report."""
        lines = [
            "# RAG Benchmark Report",
            "",
            f"**Report ID:** {self.report_id}",
            f"**Run At:** {self.run_at.isoformat()}",
            "",
            "## Overall Score",
            "",
            f"# {self.overall_score:.1%}",
            "",
            f"Passed {self.passed}/{self.total_cases} test cases",
            "",
            "## Category Breakdown",
            "",
            "| Category | Score | Passed | Total |",
            "|----------|-------|--------|-------|",
        ]
        
        for cat, metrics in self.category_scores.items():
            lines.append(
                f"| {cat.replace('_', ' ').title()} | "
                f"{metrics['score']:.1%} | {int(metrics['passed'])} | {int(metrics['total'])} |"
            )
        
        lines.extend([
            "",
            "## Key Metrics",
            "",
            "| Metric | Score | Target | Status |",
            "|--------|-------|--------|--------|",
            f"| Retrieval Precision | {self.mean_retrieval_precision:.1%} | ≥70% | {'✅' if self.mean_retrieval_precision >= 0.7 else '❌'} |",
            f"| Retrieval Recall | {self.mean_retrieval_recall:.1%} | ≥80% | {'✅' if self.mean_retrieval_recall >= 0.8 else '❌'} |",
            f"| Faithfulness | {self.mean_faithfulness:.1%} | ≥90% | {'✅' if self.mean_faithfulness >= 0.9 else '❌'} |",
            f"| Mathematical Accuracy | {self.mathematical_accuracy_rate:.1%} | ≥95% | {'✅' if self.mathematical_accuracy_rate >= 0.95 else '❌'} |",
            f"| Negative Rejection | {self.negative_rejection_rate:.1%} | ≥90% | {'✅' if self.negative_rejection_rate >= 0.9 else '❌'} |",
            f"| Noise Robustness | {self.noise_robustness_rate:.1%} | ≥85% | {'✅' if self.noise_robustness_rate >= 0.85 else '❌'} |",
            "",
            "## Difficulty Breakdown",
            "",
            "| Difficulty | Score | Cases |",
            "|------------|-------|-------|",
        ])
        
        for diff, metrics in self.difficulty_scores.items():
            lines.append(f"| {diff.title()} | {metrics['score']:.1%} | {int(metrics['total'])} |")
        
        # Failures
        failures = [r for r in self.results if not r.passed]
        if failures:
            lines.extend([
                "",
                "## Failures",
                "",
            ])
            for f in failures[:10]:  # Show first 10
                lines.append(f"- **{f.case_id}** ({f.category}): {', '.join(f.failure_reasons[:2])}")
            if len(failures) > 10:
                lines.append(f"- ... and {len(failures) - 10} more failures")
        
        return "\n".join(lines)


# =============================================================================
# BENCHMARK DATASET
# =============================================================================

BENCHMARK_DATASET: List[BenchmarkCase] = [
    # =========================================================================
    # RETRIEVAL QUALITY
    # =========================================================================
    BenchmarkCase(
        id="RQ-001",
        category=BenchmarkCategory.RETRIEVAL_QUALITY,
        difficulty=Difficulty.EASY,
        query="What is the single issuer concentration limit?",
        context_documents=["concentration_limits.md"],
        expected_answer_contains=["10%", "single issuer"],
    ),
    BenchmarkCase(
        id="RQ-002",
        category=BenchmarkCategory.RETRIEVAL_QUALITY,
        difficulty=Difficulty.MEDIUM,
        query="What is the escalation process when a concentration breach occurs?",
        context_documents=["concentration_limits.md", "exception_management.md"],
        expected_answer_contains=["escalation", "compliance", "notification"],
    ),
    BenchmarkCase(
        id="RQ-003",
        category=BenchmarkCategory.RETRIEVAL_QUALITY,
        difficulty=Difficulty.HARD,
        query="How does the liquidity policy interact with SEC Form PF reporting requirements?",
        context_documents=["liquidity_policy.md", "sec_compliance.md"],
        expected_answer_contains=["liquidity", "Form PF", "SEC"],
    ),
    
    # =========================================================================
    # NOISE ROBUSTNESS - System should extract correct info despite noise
    # =========================================================================
    BenchmarkCase(
        id="NR-001",
        category=BenchmarkCategory.NOISE_ROBUSTNESS,
        difficulty=Difficulty.MEDIUM,
        query="What is the technology sector concentration limit?",
        context_documents=["concentration_limits.md"],
        noise_documents=["commodity_trading.md", "sec_compliance.md"],  # Irrelevant docs
        expected_answer_contains=["30%", "sector"],
        expected_answer_not_contains=["commodity", "CFTC"],  # Shouldn't mention noise
    ),
    BenchmarkCase(
        id="NR-002",
        category=BenchmarkCategory.NOISE_ROBUSTNESS,
        difficulty=Difficulty.HARD,
        query="What is the minimum liquidity requirement?",
        context_documents=["liquidity_policy.md"],
        noise_documents=["exposure_limits.md", "investment_guidelines.md"],
        expected_answer_contains=["liquidity", "minimum"],
        expected_answer_not_contains=["leverage", "gross exposure"],
    ),
    
    # =========================================================================
    # NEGATIVE REJECTION - System should refuse unanswerable questions
    # =========================================================================
    BenchmarkCase(
        id="NEG-001",
        category=BenchmarkCategory.NEGATIVE_REJECTION,
        difficulty=Difficulty.MEDIUM,
        query="What is the cryptocurrency trading policy?",  # Not in our policies
        context_documents=[],
        should_refuse=True,
        refusal_reason="No cryptocurrency policy exists in the document corpus",
        expected_answer_contains=["cannot", "no information", "not covered", "unable"],
    ),
    BenchmarkCase(
        id="NEG-002",
        category=BenchmarkCategory.NEGATIVE_REJECTION,
        difficulty=Difficulty.HARD,
        query="What was the fund's return in Q3 2024?",  # Historical performance not in policies
        context_documents=[],
        should_refuse=True,
        refusal_reason="Performance data not available in policy documents",
        expected_answer_contains=["cannot", "no data", "not available", "unable"],
    ),
    
    # =========================================================================
    # COUNTERFACTUAL ROBUSTNESS - System should resist wrong info in context
    # =========================================================================
    BenchmarkCase(
        id="CF-001",
        category=BenchmarkCategory.COUNTERFACTUAL_ROBUSTNESS,
        difficulty=Difficulty.HARD,
        query="Is the single issuer limit 25% as stated in the draft policy?",
        context_documents=["concentration_limits.md"],
        counterfactual_claim="Draft policy suggests 25% single issuer limit",
        correct_answer="The official policy states 10% single issuer limit",
        expected_answer_contains=["10%"],
        expected_answer_not_contains=["25%"],
    ),
    
    # =========================================================================
    # INFORMATION INTEGRATION - Combine info from multiple sources
    # =========================================================================
    BenchmarkCase(
        id="II-001",
        category=BenchmarkCategory.INFORMATION_INTEGRATION,
        difficulty=Difficulty.HARD,
        query="If we breach the sector concentration limit, what is the timeline for cure and who approves exceptions?",
        context_documents=["concentration_limits.md", "exception_management.md"],
        expected_answer_contains=["days", "cure", "approval", "CIO"],
    ),
    BenchmarkCase(
        id="II-002",
        category=BenchmarkCategory.INFORMATION_INTEGRATION,
        difficulty=Difficulty.HARD,
        query="What are the liquidity requirements and how do they affect Form PF reporting?",
        context_documents=["liquidity_policy.md", "sec_compliance.md"],
        expected_answer_contains=["liquidity", "bucket", "Form PF"],
    ),
    
    # =========================================================================
    # MATHEMATICAL ACCURACY (VERAFI-inspired)
    # =========================================================================
    BenchmarkCase(
        id="MATH-001",
        category=BenchmarkCategory.MATHEMATICAL_ACCURACY,
        difficulty=Difficulty.EASY,
        query="If NAV is $100M and single issuer limit is 10%, what is the maximum position size?",
        context_documents=["concentration_limits.md"],
        expected_answer_contains=["$10", "million", "10%"],
        expected_numeric_values={"max_position": 10_000_000},
    ),
    BenchmarkCase(
        id="MATH-002",
        category=BenchmarkCategory.MATHEMATICAL_ACCURACY,
        difficulty=Difficulty.MEDIUM,
        query="With $500M NAV and 30% sector limit, if current tech exposure is $120M, how much more can we add?",
        context_documents=["concentration_limits.md"],
        expected_answer_contains=["$30", "million"],
        expected_numeric_values={"remaining_capacity": 30_000_000},  # 500M * 30% - 120M = 30M
    ),
    BenchmarkCase(
        id="MATH-003",
        category=BenchmarkCategory.MATHEMATICAL_ACCURACY,
        difficulty=Difficulty.HARD,
        query="If gross exposure is 180% of NAV with $200M NAV, and the limit is 200%, what is the remaining capacity in dollars?",
        context_documents=["exposure_limits.md"],
        expected_answer_contains=["$40", "million"],
        expected_numeric_values={"remaining_capacity": 40_000_000},  # 200M * (200% - 180%) = 40M
    ),
    
    # =========================================================================
    # TEMPORAL CONSISTENCY
    # =========================================================================
    BenchmarkCase(
        id="TEMP-001",
        category=BenchmarkCategory.TEMPORAL_CONSISTENCY,
        difficulty=Difficulty.MEDIUM,
        query="What is the cure period for concentration breaches?",
        context_documents=["exception_management.md"],
        expected_answer_contains=["days"],
        temporal_references=[{"type": "duration", "expected": "days"}],
    ),
    BenchmarkCase(
        id="TEMP-002",
        category=BenchmarkCategory.TEMPORAL_CONSISTENCY,
        difficulty=Difficulty.HARD,
        query="How often must Form PF be filed and what is the deadline after quarter end?",
        context_documents=["sec_compliance.md"],
        expected_answer_contains=["quarterly", "days"],
        temporal_references=[
            {"type": "frequency", "expected": "quarterly"},
            {"type": "deadline", "expected": "days"},
        ],
    ),
    
    # =========================================================================
    # FAITHFULNESS - Output must be grounded in context
    # =========================================================================
    BenchmarkCase(
        id="FAITH-001",
        category=BenchmarkCategory.FAITHFULNESS,
        difficulty=Difficulty.MEDIUM,
        query="What are the sector concentration limits according to our policy?",
        context_documents=["concentration_limits.md"],
        expected_answer_contains=["sector", "%"],
        notes="Answer must cite specific percentages from policy, not fabricate",
    ),
    BenchmarkCase(
        id="FAITH-002",
        category=BenchmarkCategory.FAITHFULNESS,
        difficulty=Difficulty.HARD,
        query="Describe the complete exception approval workflow",
        context_documents=["exception_management.md"],
        expected_answer_contains=["approval", "documentation"],
        notes="Must describe actual workflow from policy, not generic process",
    ),
]


def get_benchmark_dataset() -> List[BenchmarkCase]:
    """Get the complete benchmark dataset."""
    return BENCHMARK_DATASET


def get_benchmark_by_category(category: BenchmarkCategory) -> List[BenchmarkCase]:
    """Get benchmarks for a specific category."""
    return [b for b in BENCHMARK_DATASET if b.category == category]


def get_benchmark_by_difficulty(difficulty: Difficulty) -> List[BenchmarkCase]:
    """Get benchmarks for a specific difficulty."""
    return [b for b in BENCHMARK_DATASET if b.difficulty == difficulty]


# ============================================================================
# EXPORTS - Import from submodules
# ============================================================================

# RAGAS metrics
from tests.benchmarks.ragas_metrics import (
    RAGASMetrics,
    RAGASScores,
    calculate_ragas_scores,
)

# FinanceBench-style cases
from tests.benchmarks.finance_bench import (
    FinanceBenchCase,
    FinanceQuestionType,
    FINANCE_BENCHMARK_DATASET,
    get_finance_benchmark_dataset,
    get_by_question_type,
    get_by_difficulty as get_finance_by_difficulty,
    get_by_tag,
)

# VERAFI evaluation
from tests.benchmarks.verafi_eval import (
    VERAFIEvaluator,
    VERAFIResult,
    VERAFIMetric,
    evaluate_verafi,
)

__all__ = [
    # Core types
    "BenchmarkCategory",
    "BenchmarkCase",
    "BenchmarkResult",
    "BenchmarkReport",
    "Difficulty",
    # Dataset access
    "BENCHMARK_DATASET",
    "get_benchmark_dataset",
    "get_benchmark_by_category",
    "get_benchmark_by_difficulty",
    # RAGAS
    "RAGASMetrics",
    "RAGASScores",
    "calculate_ragas_scores",
    # FinanceBench
    "FinanceBenchCase",
    "FinanceQuestionType",
    "FINANCE_BENCHMARK_DATASET",
    "get_finance_benchmark_dataset",
    "get_by_question_type",
    "get_finance_by_difficulty",
    "get_by_tag",
    # VERAFI
    "VERAFIEvaluator",
    "VERAFIResult",
    "VERAFIMetric",
    "evaluate_verafi",
]
