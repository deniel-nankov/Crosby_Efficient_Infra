"""
Evaluation Framework - Ground-Truth Test Suite for RAG Quality

This module provides structured evaluation:
1. Ground-truth Q&A pairs for retrieval testing
2. End-to-end generation quality assessment
3. Faithfulness scoring (is output grounded in context?)
4. Regression detection across code changes

Usage:
    pytest tests/eval/ -v                    # Run all evaluation tests
    pytest tests/eval/ -k "retrieval" -v     # Run retrieval tests only
    python -m tests.eval.run_evaluation      # Generate full evaluation report
"""

import json
import pytest
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict

# Ground-truth evaluation dataset
# Each entry contains a query and expected relevant documents/answers

COMPLIANCE_EVAL_DATASET = [
    # ==========================================================================
    # CONCENTRATION LIMITS
    # ==========================================================================
    {
        "id": "CONC-001",
        "category": "concentration",
        "query": "What is the maximum single issuer concentration limit?",
        "expected_answer_contains": ["10%", "single issuer", "NAV"],
        "relevant_documents": ["concentration_limits.md"],
        "relevant_sections": ["Single Issuer Limits"],
        "difficulty": "easy",
    },
    {
        "id": "CONC-002",
        "category": "concentration",
        "query": "What is the technology sector concentration threshold?",
        "expected_answer_contains": ["30%", "sector", "technology"],
        "relevant_documents": ["concentration_limits.md", "investment_guidelines.md"],
        "relevant_sections": ["Sector Concentration"],
        "difficulty": "easy",
    },
    {
        "id": "CONC-003",
        "category": "concentration",
        "query": "How is geographic concentration calculated and what are the limits?",
        "expected_answer_contains": ["country", "region", "exposure"],
        "relevant_documents": ["concentration_limits.md"],
        "relevant_sections": ["Geographic Concentration"],
        "difficulty": "medium",
    },
    {
        "id": "CONC-004",
        "category": "concentration",
        "query": "Can we exceed concentration limits with approval?",
        "expected_answer_contains": ["exception", "approval", "CIO"],
        "relevant_documents": ["concentration_limits.md", "exception_management.md"],
        "relevant_sections": ["Exceptions", "Approval Process"],
        "difficulty": "medium",
    },
    
    # ==========================================================================
    # LIQUIDITY
    # ==========================================================================
    {
        "id": "LIQ-001",
        "category": "liquidity",
        "query": "What is the minimum T+1 liquidity bucket requirement?",
        "expected_answer_contains": ["T+1", "liquidity", "minimum"],
        "relevant_documents": ["liquidity_policy.md"],
        "relevant_sections": ["Liquidity Buckets"],
        "difficulty": "easy",
    },
    {
        "id": "LIQ-002",
        "category": "liquidity",
        "query": "How are liquidity buckets defined and monitored?",
        "expected_answer_contains": ["T+1", "T+7", "T+30", "bucket"],
        "relevant_documents": ["liquidity_policy.md"],
        "relevant_sections": ["Bucket Definitions", "Monitoring"],
        "difficulty": "medium",
    },
    {
        "id": "LIQ-003",
        "category": "liquidity",
        "query": "What happens when liquidity falls below minimum thresholds?",
        "expected_answer_contains": ["breach", "escalation", "remediation"],
        "relevant_documents": ["liquidity_policy.md", "exception_management.md"],
        "relevant_sections": ["Breach Response"],
        "difficulty": "hard",
    },
    
    # ==========================================================================
    # EXPOSURE LIMITS
    # ==========================================================================
    {
        "id": "EXP-001",
        "category": "exposure",
        "query": "What is the maximum gross exposure limit?",
        "expected_answer_contains": ["gross", "exposure", "limit"],
        "relevant_documents": ["exposure_limits.md", "investment_guidelines.md"],
        "relevant_sections": ["Gross Exposure"],
        "difficulty": "easy",
    },
    {
        "id": "EXP-002",
        "category": "exposure",
        "query": "How is net exposure calculated?",
        "expected_answer_contains": ["long", "short", "net"],
        "relevant_documents": ["exposure_limits.md"],
        "relevant_sections": ["Net Exposure Calculation"],
        "difficulty": "medium",
    },
    {
        "id": "EXP-003",
        "category": "exposure",
        "query": "What are the leverage constraints for the fund?",
        "expected_answer_contains": ["leverage", "limit", "gross"],
        "relevant_documents": ["exposure_limits.md", "investment_guidelines.md"],
        "relevant_sections": ["Leverage Limits"],
        "difficulty": "medium",
    },
    
    # ==========================================================================
    # EXCEPTION MANAGEMENT
    # ==========================================================================
    {
        "id": "EXC-001",
        "category": "exception",
        "query": "What is the escalation process for a limit breach?",
        "expected_answer_contains": ["escalation", "notification", "compliance"],
        "relevant_documents": ["exception_management.md"],
        "relevant_sections": ["Escalation Procedures"],
        "difficulty": "easy",
    },
    {
        "id": "EXC-002",
        "category": "exception",
        "query": "How long do we have to cure a breach?",
        "expected_answer_contains": ["cure", "days", "timeline", "remediation"],
        "relevant_documents": ["exception_management.md"],
        "relevant_sections": ["Cure Periods"],
        "difficulty": "medium",
    },
    {
        "id": "EXC-003",
        "category": "exception",
        "query": "Who can approve limit exceptions and what documentation is required?",
        "expected_answer_contains": ["approval", "CIO", "documentation"],
        "relevant_documents": ["exception_management.md"],
        "relevant_sections": ["Approval Authority", "Documentation Requirements"],
        "difficulty": "hard",
    },
    
    # ==========================================================================
    # REGULATORY / SEC
    # ==========================================================================
    {
        "id": "REG-001",
        "category": "regulatory",
        "query": "What are the Form PF filing requirements?",
        "expected_answer_contains": ["Form PF", "SEC", "filing"],
        "relevant_documents": ["sec_compliance.md"],
        "relevant_sections": ["Form PF"],
        "difficulty": "medium",
    },
    {
        "id": "REG-002",
        "category": "regulatory",
        "query": "What disclosures are required for material compliance breaches?",
        "expected_answer_contains": ["disclosure", "material", "SEC"],
        "relevant_documents": ["sec_compliance.md", "exception_management.md"],
        "relevant_sections": ["Disclosure Requirements"],
        "difficulty": "hard",
    },
    
    # ==========================================================================
    # COMMODITY / DERIVATIVES
    # ==========================================================================
    {
        "id": "COM-001",
        "category": "commodity",
        "query": "What are the position limits for commodity derivatives?",
        "expected_answer_contains": ["commodity", "position", "limit"],
        "relevant_documents": ["commodity_trading.md"],
        "relevant_sections": ["Position Limits"],
        "difficulty": "medium",
    },
    {
        "id": "COM-002",
        "category": "commodity",
        "query": "What CFTC reporting is required for our commodity positions?",
        "expected_answer_contains": ["CFTC", "reporting", "commodity"],
        "relevant_documents": ["commodity_trading.md"],
        "relevant_sections": ["CFTC Reporting"],
        "difficulty": "hard",
    },
    
    # ==========================================================================
    # CROSS-CUTTING / COMPLEX
    # ==========================================================================
    {
        "id": "COMPLEX-001",
        "category": "complex",
        "query": "If technology sector is at 28% and we want to add more, what approvals are needed?",
        "expected_answer_contains": ["30%", "exception", "approval"],
        "relevant_documents": ["concentration_limits.md", "exception_management.md"],
        "relevant_sections": ["Sector Concentration", "Pre-Trade Approval"],
        "difficulty": "hard",
    },
    {
        "id": "COMPLEX-002",
        "category": "complex",
        "query": "What controls prevent concentration limit breaches at trade time?",
        "expected_answer_contains": ["pre-trade", "check", "limit"],
        "relevant_documents": ["concentration_limits.md", "investment_guidelines.md"],
        "relevant_sections": ["Pre-Trade Controls"],
        "difficulty": "hard",
    },
]


@dataclass
class EvaluationResult:
    """Result of evaluating a single query."""
    query_id: str
    query: str
    category: str
    difficulty: str
    
    # Retrieval metrics
    retrieved_documents: List[str]
    expected_documents: List[str]
    retrieval_hit: bool
    retrieval_precision: float
    retrieval_recall: float
    retrieval_latency_ms: float
    
    # Answer quality (if generation tested)
    generated_answer: Optional[str] = None
    expected_contains: List[str] = field(default_factory=list)
    answer_contains_hits: int = 0
    answer_faithfulness: float = 0.0
    
    # Overall
    passed: bool = False
    failure_reason: Optional[str] = None


@dataclass
class EvaluationReport:
    """Complete evaluation report."""
    report_id: str
    run_at: datetime
    total_queries: int
    
    # By result
    passed: int
    failed: int
    pass_rate: float
    
    # By category
    category_results: Dict[str, Dict[str, float]]
    
    # By difficulty
    difficulty_results: Dict[str, Dict[str, float]]
    
    # Detailed results
    results: List[EvaluationResult]
    
    # Aggregate metrics
    mean_retrieval_precision: float
    mean_retrieval_recall: float
    mean_retrieval_latency_ms: float
    mean_answer_faithfulness: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "report_id": self.report_id,
            "run_at": self.run_at.isoformat(),
            "total_queries": self.total_queries,
            "passed": self.passed,
            "failed": self.failed,
            "pass_rate": round(self.pass_rate, 4),
            "category_results": self.category_results,
            "difficulty_results": self.difficulty_results,
            "mean_retrieval_precision": round(self.mean_retrieval_precision, 4),
            "mean_retrieval_recall": round(self.mean_retrieval_recall, 4),
            "mean_retrieval_latency_ms": round(self.mean_retrieval_latency_ms, 2),
            "mean_answer_faithfulness": round(self.mean_answer_faithfulness, 4),
        }
    
    def to_markdown(self) -> str:
        """Generate markdown report."""
        lines = [
            f"# RAG Evaluation Report",
            f"",
            f"**Report ID:** {self.report_id}",
            f"**Run At:** {self.run_at.isoformat()}",
            f"",
            f"## Summary",
            f"",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Total Queries | {self.total_queries} |",
            f"| Passed | {self.passed} |",
            f"| Failed | {self.failed} |",
            f"| **Pass Rate** | **{self.pass_rate:.1%}** |",
            f"| Mean Precision | {self.mean_retrieval_precision:.2%} |",
            f"| Mean Recall | {self.mean_retrieval_recall:.2%} |",
            f"| Mean Latency | {self.mean_retrieval_latency_ms:.0f}ms |",
            f"",
            f"## Results by Category",
            f"",
            f"| Category | Pass Rate | Count |",
            f"|----------|-----------|-------|",
        ]
        
        for cat, metrics in self.category_results.items():
            lines.append(f"| {cat} | {metrics['pass_rate']:.1%} | {int(metrics['count'])} |")
        
        lines.extend([
            f"",
            f"## Results by Difficulty",
            f"",
            f"| Difficulty | Pass Rate | Count |",
            f"|------------|-----------|-------|",
        ])
        
        for diff, metrics in self.difficulty_results.items():
            lines.append(f"| {diff} | {metrics['pass_rate']:.1%} | {int(metrics['count'])} |")
        
        # Add failures
        failures = [r for r in self.results if not r.passed]
        if failures:
            lines.extend([
                f"",
                f"## Failures",
                f"",
            ])
            for f in failures:
                lines.append(f"- **{f.query_id}** ({f.category}/{f.difficulty}): {f.failure_reason}")
        
        return "\n".join(lines)


def get_eval_dataset() -> List[Dict[str, Any]]:
    """Get the evaluation dataset."""
    return COMPLIANCE_EVAL_DATASET


def save_eval_dataset(path: Path):
    """Save evaluation dataset to JSON file."""
    with open(path, 'w') as f:
        json.dump(COMPLIANCE_EVAL_DATASET, f, indent=2)
