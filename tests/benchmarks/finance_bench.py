"""
FinanceBench-Style Compliance Benchmarks

Financial domain-specific benchmarks modeled after:
- FinanceBench (https://arxiv.org/abs/2311.11944) - Financial QA benchmark
- VERAFI evaluation methodology - 94.7% accuracy target

These benchmarks test:
1. Financial calculations (leverage, returns, ratios)
2. Temporal reasoning (YoY changes, multi-period analysis)
3. Multi-document synthesis (combining 10-K + 10-Q + earnings)
4. Regulatory compliance (SEC rules, Form PF, ADV)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum


class FinanceQuestionType(Enum):
    """Types of financial questions."""
    CALCULATION = "calculation"
    TEMPORAL = "temporal"
    MULTI_DOC = "multi_document"
    REGULATORY = "regulatory"
    RISK = "risk"
    COMPLEX = "complex"


@dataclass(frozen=True)
class FinanceBenchCase:
    """A FinanceBench-style test case."""
    id: str
    question: str
    question_type: FinanceQuestionType
    
    # Source documents
    source_docs: List[str]
    source_text: str  # Actual text from documents
    
    # Expected answer
    expected_answer: str
    
    # For calculations
    calculation_chain: Optional[List[str]] = None  # Step-by-step calc
    expected_value: Optional[float] = None
    tolerance_pct: float = 0.05  # 5% tolerance
    
    # For temporal
    time_periods: Optional[List[str]] = None
    
    # Difficulty and tags
    difficulty: str = "medium"  # easy, medium, hard, very_hard
    tags: Tuple[str, ...] = field(default_factory=tuple)


# ============================================================================
# FINANCE BENCHMARK DATASET
# ============================================================================

FINANCE_BENCHMARK_DATASET: List[FinanceBenchCase] = [
    
    # === CALCULATION QUESTIONS ===
    
    FinanceBenchCase(
        id="FIN-CALC-001",
        question="What is the portfolio's gross exposure as a percentage of NAV?",
        question_type=FinanceQuestionType.CALCULATION,
        source_docs=["exposure_limits.md", "daily_positions"],
        source_text="""
        Portfolio NAV: $500,000,000
        Total Long Positions: $600,000,000
        Total Short Positions: $300,000,000
        
        From exposure_limits.md:
        Gross exposure = |Long| + |Short| / NAV
        Maximum gross exposure: 200% of NAV
        """,
        expected_answer="The gross exposure is 180% of NAV, calculated as ($600M + $300M) / $500M = 180%",
        calculation_chain=[
            "Gross = |Long| + |Short|",
            "Gross = $600M + $300M = $900M",
            "Gross % = $900M / $500M = 180%",
        ],
        expected_value=180.0,
        difficulty="easy",
        tags=("exposure", "calculation", "compliance"),
    ),
    
    FinanceBenchCase(
        id="FIN-CALC-002",
        question="Calculate the sector concentration for Technology holdings.",
        question_type=FinanceQuestionType.CALCULATION,
        source_docs=["concentration_limits.md", "daily_positions"],
        source_text="""
        Portfolio NAV: $500,000,000
        Technology Sector Holdings:
        - AAPL: $50,000,000
        - MSFT: $40,000,000
        - GOOGL: $30,000,000
        - NVDA: $25,000,000
        Total Technology: $145,000,000
        
        From concentration_limits.md:
        Sector concentration limit: 25% of NAV
        """,
        expected_answer="Technology sector concentration is 29% ($145M / $500M = 29%), which exceeds the 25% limit.",
        calculation_chain=[
            "Tech Total = $50M + $40M + $30M + $25M = $145M",
            "Concentration = $145M / $500M = 29%",
            "Limit = 25%, Breach = Yes",
        ],
        expected_value=29.0,
        difficulty="medium",
        tags=("concentration", "sector", "breach"),
    ),
    
    FinanceBenchCase(
        id="FIN-CALC-003",
        question="What is the portfolio's net leverage ratio?",
        question_type=FinanceQuestionType.CALCULATION,
        source_docs=["exposure_limits.md", "daily_positions"],
        source_text="""
        Portfolio NAV: $500,000,000
        Long Market Value: $600,000,000
        Short Market Value: $300,000,000
        Borrowed Cash: $200,000,000
        Cash on Hand: $50,000,000
        
        Net Leverage = (Long - Short + Borrowed - Cash) / NAV
        """,
        expected_answer="Net leverage is 150% calculated as ($600M - $300M + $200M - $50M) / $500M = 90%",
        calculation_chain=[
            "Net = Long - Short + Borrowed - Cash",
            "Net = $600M - $300M + $200M - $50M = $450M",
            "Leverage = $450M / $500M = 90%",
        ],
        expected_value=90.0,
        difficulty="hard",
        tags=("leverage", "exposure", "calculation"),
    ),
    
    # === TEMPORAL QUESTIONS ===
    
    FinanceBenchCase(
        id="FIN-TEMP-001",
        question="How has the portfolio's liquidity ratio changed over the past quarter?",
        question_type=FinanceQuestionType.TEMPORAL,
        source_docs=["liquidity_policy.md", "quarterly_liquidity_report"],
        source_text="""
        Liquidity Requirements:
        - Minimum liquid assets: 15% of NAV
        - Highly liquid assets: Cash + T-Bills + Top-50 equities
        
        Quarterly Liquidity:
        - January 2025: 18.2% ($91M / $500M)
        - February 2025: 16.5% ($82.5M / $500M)
        - March 2025: 14.8% ($74M / $500M) ⚠️ BELOW MINIMUM
        """,
        expected_answer="Liquidity ratio decreased from 18.2% in January to 14.8% in March, a 3.4 percentage point decline. March breached the 15% minimum requirement.",
        time_periods=["January 2025", "February 2025", "March 2025"],
        expected_value=-3.4,  # Change
        difficulty="medium",
        tags=("liquidity", "temporal", "trend", "breach"),
    ),
    
    FinanceBenchCase(
        id="FIN-TEMP-002",
        question="What was the year-over-year change in AUM?",
        question_type=FinanceQuestionType.TEMPORAL,
        source_docs=["fund_overview", "historical_nav"],
        source_text="""
        Fund Assets Under Management:
        - December 2023: $425,000,000
        - December 2024: $500,000,000
        
        Change: +$75M (+17.6%)
        """,
        expected_answer="AUM increased by $75 million, representing a 17.6% year-over-year growth from $425M to $500M.",
        time_periods=["December 2023", "December 2024"],
        calculation_chain=[
            "Change = $500M - $425M = $75M",
            "YoY % = ($75M / $425M) × 100 = 17.6%",
        ],
        expected_value=17.6,
        difficulty="easy",
        tags=("aum", "temporal", "growth"),
    ),
    
    # === MULTI-DOCUMENT QUESTIONS ===
    
    FinanceBenchCase(
        id="FIN-MULTI-001",
        question="Does the current position in AAPL violate both single security and sector concentration limits?",
        question_type=FinanceQuestionType.MULTI_DOC,
        source_docs=["concentration_limits.md", "daily_positions", "exception_management.md"],
        source_text="""
        From concentration_limits.md:
        - Single security limit: 5% of NAV
        - Sector concentration limit: 25% of NAV
        
        From daily_positions:
        - AAPL position: $50,000,000
        - Portfolio NAV: $500,000,000
        - Technology sector total: $145,000,000
        
        From exception_management.md:
        - Active exception EXC-2025-042: AAPL allowed up to 10% for 90 days
        - Approved by: Chief Risk Officer
        - Expires: 2025-04-15
        """,
        expected_answer="AAPL at 10% ($50M/$500M) would normally breach the 5% single security limit, but is covered by exception EXC-2025-042. However, Tech sector at 29% still breaches the 25% sector limit with no exception.",
        calculation_chain=[
            "AAPL % = $50M / $500M = 10%",
            "Single limit = 5%, Breach = Yes (but exception exists)",
            "Tech % = $145M / $500M = 29%",
            "Sector limit = 25%, Breach = Yes (no exception)",
        ],
        difficulty="hard",
        tags=("multi-document", "exception", "breach", "concentration"),
    ),
    
    FinanceBenchCase(
        id="FIN-MULTI-002",
        question="Summarize all compliance breaches and their escalation status.",
        question_type=FinanceQuestionType.MULTI_DOC,
        source_docs=["concentration_limits.md", "liquidity_policy.md", "exception_management.md", "control_results"],
        source_text="""
        Active Breaches:
        
        1. CONC_SINGLE_001 - AAPL at 10% vs 5% limit
           - Status: Exception approved (EXC-2025-042)
           - Escalation: None required
        
        2. CONC_SECTOR_001 - Technology at 29% vs 25% limit
           - Status: No exception
           - Escalation: Pending CRO review
        
        3. LIQ_MIN_001 - Liquidity at 14.8% vs 15% minimum
           - Status: Remediation in progress
           - Escalation: CFO notified
        """,
        expected_answer="Three breaches: (1) AAPL single security covered by exception, (2) Tech sector 29% vs 25% limit pending CRO review, (3) Liquidity 14.8% vs 15% minimum with CFO notified and remediation in progress.",
        difficulty="very_hard",
        tags=("multi-document", "synthesis", "escalation", "breach"),
    ),
    
    # === REGULATORY QUESTIONS ===
    
    FinanceBenchCase(
        id="FIN-REG-001",
        question="What are the SEC Form PF filing requirements for this fund?",
        question_type=FinanceQuestionType.REGULATORY,
        source_docs=["sec_compliance.md"],
        source_text="""
        SEC Form PF Requirements:
        
        Large Private Fund Adviser (>$1.5B regulatory AUM):
        - Filing frequency: Quarterly
        - Due: 60 days after quarter end
        
        Required disclosures:
        - Gross asset value
        - Net asset value
        - Borrowings and leverage
        - Counterparty exposures
        - Geographic exposure
        - Turnover rate
        
        Fund Status:
        - Regulatory AUM: $2.1B
        - Classification: Large Private Fund Adviser
        - Next filing due: May 30, 2025 (Q1)
        """,
        expected_answer="As a Large Private Fund Adviser with $2.1B regulatory AUM (above $1.5B threshold), quarterly Form PF filing is required within 60 days of quarter end. Next filing due May 30, 2025.",
        difficulty="medium",
        tags=("regulatory", "sec", "form_pf"),
    ),
    
    FinanceBenchCase(
        id="FIN-REG-002",
        question="Does the fund need to file Form 13F and what positions must be reported?",
        question_type=FinanceQuestionType.REGULATORY,
        source_docs=["sec_compliance.md", "daily_positions"],
        source_text="""
        Form 13F Requirements:
        - Threshold: $100M+ in 13(f) securities
        - Filing: Quarterly within 45 days
        
        13(f) Securities include:
        - Exchange-traded stocks
        - Equity options
        - Convertible bonds
        
        Fund Holdings Summary:
        - US Equities: $450,000,000 (13f reportable)
        - US Treasuries: $30,000,000 (not 13f)
        - Private Securities: $20,000,000 (not 13f)
        
        Status: Filing required - $450M exceeds $100M threshold
        """,
        expected_answer="Yes, 13F filing required as $450M in reportable securities exceeds the $100M threshold. US equities must be reported; Treasuries and private securities are exempt.",
        calculation_chain=[
            "Reportable 13(f) = $450M (US equities)",
            "Threshold = $100M",
            "$450M > $100M → Filing Required",
        ],
        expected_value=450.0,
        difficulty="medium",
        tags=("regulatory", "sec", "13f", "disclosure"),
    ),
    
    # === RISK QUESTIONS ===
    
    FinanceBenchCase(
        id="FIN-RISK-001",
        question="What is the portfolio's Value at Risk (VaR) and how does it compare to the limit?",
        question_type=FinanceQuestionType.RISK,
        source_docs=["risk_policy", "daily_risk_report"],
        source_text="""
        Risk Limits:
        - Daily VaR (95%): Max 2% of NAV
        - Daily VaR (99%): Max 3% of NAV
        
        Current Risk Metrics (as of 2025-01-17):
        - Portfolio NAV: $500,000,000
        - Daily VaR (95%): $8,500,000 (1.7% of NAV)
        - Daily VaR (99%): $12,000,000 (2.4% of NAV)
        
        Status: Within limits
        """,
        expected_answer="VaR(95%) is $8.5M (1.7% of NAV) vs 2% limit - compliant. VaR(99%) is $12M (2.4% of NAV) vs 3% limit - compliant. Both metrics within risk tolerance.",
        calculation_chain=[
            "VaR(95%) % = $8.5M / $500M = 1.7%",
            "1.7% < 2% limit → Compliant",
            "VaR(99%) % = $12M / $500M = 2.4%",
            "2.4% < 3% limit → Compliant",
        ],
        expected_value=1.7,
        difficulty="medium",
        tags=("risk", "var", "compliance"),
    ),
    
    # === COMPLEX QUESTIONS ===
    
    FinanceBenchCase(
        id="FIN-COMPLEX-001",
        question="If we sell $20M of AAPL and buy $20M of healthcare stocks, will all concentration limits be satisfied?",
        question_type=FinanceQuestionType.COMPLEX,
        source_docs=["concentration_limits.md", "daily_positions"],
        source_text="""
        Current State:
        - NAV: $500M
        - AAPL: $50M (10%)
        - Technology sector: $145M (29%)
        - Healthcare sector: $60M (12%)
        
        Limits:
        - Single security: 5%
        - Sector: 25%
        
        After proposed trade:
        - AAPL: $30M (6%) - still exceeds 5%
        - Technology: $125M (25%) - at limit
        - Healthcare: $80M (16%) - within limit
        """,
        expected_answer="No. After the trade: AAPL at 6% still exceeds 5% single security limit. Technology at 25% would be exactly at sector limit. Healthcare at 16% is compliant. Need to sell additional $5M AAPL to fully comply.",
        calculation_chain=[
            "AAPL new = $50M - $20M = $30M",
            "AAPL % = $30M / $500M = 6%",
            "6% > 5% → Still breaching",
            "Tech new = $145M - $20M = $125M",
            "Tech % = $125M / $500M = 25%",
            "25% = 25% → At limit (compliant)",
            "Healthcare new = $60M + $20M = $80M",
            "Healthcare % = $80M / $500M = 16%",
            "16% < 25% → Compliant",
        ],
        expected_value=6.0,
        difficulty="very_hard",
        tags=("complex", "scenario", "concentration", "remediation"),
    ),
]


def get_finance_benchmark_dataset() -> List[FinanceBenchCase]:
    """Get the full finance benchmark dataset."""
    return FINANCE_BENCHMARK_DATASET


def get_by_question_type(
    question_type: FinanceQuestionType,
) -> List[FinanceBenchCase]:
    """Filter benchmarks by question type."""
    return [b for b in FINANCE_BENCHMARK_DATASET if b.question_type == question_type]


def get_by_difficulty(difficulty: str) -> List[FinanceBenchCase]:
    """Filter benchmarks by difficulty."""
    return [b for b in FINANCE_BENCHMARK_DATASET if b.difficulty == difficulty]


def get_by_tag(tag: str) -> List[FinanceBenchCase]:
    """Filter benchmarks by tag."""
    return [b for b in FINANCE_BENCHMARK_DATASET if tag in b.tags]


# Quick test
if __name__ == "__main__":
    print(f"Finance Benchmark Dataset: {len(FINANCE_BENCHMARK_DATASET)} cases")
    print()
    
    for q_type in FinanceQuestionType:
        cases = get_by_question_type(q_type)
        print(f"  {q_type.value}: {len(cases)} cases")
    
    print()
    for diff in ["easy", "medium", "hard", "very_hard"]:
        cases = get_by_difficulty(diff)
        print(f"  {diff}: {len(cases)} cases")
