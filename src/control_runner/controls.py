"""
Compliance Control Definitions

This module defines all compliance controls that can be executed.
Each control is defined with:
- Unique identifier and code
- SQL query for calculation (deterministic)
- Threshold and comparison operator
- Regulatory reference

CRITICAL: All calculations happen in SQL, NOT in the LLM.
The LLM is only used for narrative generation AFTER controls complete.

SEC Examination Note: Control definitions are version-controlled.
Any changes trigger a new config hash in the control run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Optional, Dict, Any, Callable
import hashlib


class ControlCategory(Enum):
    """Categories of compliance controls."""
    CONCENTRATION = "concentration"
    LIQUIDITY = "liquidity"
    COUNTERPARTY = "counterparty"
    EXPOSURE = "exposure"
    LEVERAGE = "leverage"
    VALUATION = "valuation"
    RECONCILIATION = "reconciliation"
    REGULATORY = "regulatory"


class ThresholdOperator(Enum):
    """Comparison operators for threshold checks."""
    LT = "lt"      # Less than
    LTE = "lte"    # Less than or equal
    GT = "gt"      # Greater than
    GTE = "gte"    # Greater than or equal
    EQ = "eq"      # Equal
    NEQ = "neq"    # Not equal


class ControlFrequency(Enum):
    """Execution frequency for controls."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUAL = "annual"


class ControlResultStatus(Enum):
    """Possible results of a control execution."""
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    SKIP = "skip"
    ERROR = "error"


@dataclass(frozen=True)
class ControlDefinition:
    """
    Immutable definition of a compliance control.
    
    Each control must be:
    - Deterministic: Same inputs always produce same outputs
    - Self-contained: SQL query returns all needed data
    - Traceable: Includes regulatory reference
    """
    control_code: str
    control_name: str
    category: ControlCategory
    description: str
    
    # The SQL query that computes the metric
    # Must return a single numeric value in column 'calculated_value'
    computation_sql: str
    
    # Threshold for pass/fail
    threshold_value: float
    threshold_operator: ThresholdOperator
    
    # Optional warning threshold (triggers warning instead of fail)
    warning_threshold: Optional[float] = None
    warning_operator: Optional[ThresholdOperator] = None
    
    # Metadata
    frequency: ControlFrequency = ControlFrequency.DAILY
    regulatory_reference: Optional[str] = None
    policy_document_id: Optional[str] = None
    
    # For controls that need fund-specific execution
    fund_specific: bool = True
    
    # Effective dates
    effective_date: date = field(default_factory=date.today)
    expiration_date: Optional[date] = None
    
    @property
    def is_active(self) -> bool:
        """Check if control is currently active."""
        today = date.today()
        if self.expiration_date and today > self.expiration_date:
            return False
        return today >= self.effective_date
    
    @property
    def query_hash(self) -> str:
        """SHA-256 hash of the computation SQL for audit trail."""
        return hashlib.sha256(self.computation_sql.encode()).hexdigest()
    
    def evaluate_threshold(self, calculated_value: float) -> ControlResultStatus:
        """
        Evaluate calculated value against threshold.
        
        This is pure Python comparison - no LLM involvement.
        """
        # Check for warning first
        if self.warning_threshold is not None and self.warning_operator is not None:
            if self._compare(calculated_value, self.warning_threshold, self.warning_operator):
                # Check if it also breaches the fail threshold
                if self._compare(calculated_value, self.threshold_value, self.threshold_operator):
                    return ControlResultStatus.FAIL
                return ControlResultStatus.WARNING
        
        # Check fail threshold
        if self._compare(calculated_value, self.threshold_value, self.threshold_operator):
            return ControlResultStatus.FAIL
        
        return ControlResultStatus.PASS
    
    @staticmethod
    def _compare(value: float, threshold: float, operator: ThresholdOperator) -> bool:
        """Perform threshold comparison."""
        comparisons = {
            ThresholdOperator.LT: lambda v, t: v < t,
            ThresholdOperator.LTE: lambda v, t: v <= t,
            ThresholdOperator.GT: lambda v, t: v > t,
            ThresholdOperator.GTE: lambda v, t: v >= t,
            ThresholdOperator.EQ: lambda v, t: v == t,
            ThresholdOperator.NEQ: lambda v, t: v != t,
        }
        return comparisons[operator](value, threshold)
    
    def get_breach_amount(self, calculated_value: float) -> Optional[float]:
        """Calculate how much the value exceeds/falls short of threshold."""
        if self.threshold_operator in (ThresholdOperator.GT, ThresholdOperator.GTE):
            return calculated_value - self.threshold_value
        elif self.threshold_operator in (ThresholdOperator.LT, ThresholdOperator.LTE):
            return self.threshold_value - calculated_value
        return None


# =============================================================================
# STANDARD CONTROL DEFINITIONS
# =============================================================================
# These are the actual compliance controls used in production.
# Each SQL query must work against the Snowflake views defined in snowflake_views.sql


CONCENTRATION_CONTROLS = [
    ControlDefinition(
        control_code="CONC_001",
        control_name="Single Issuer Concentration Limit",
        category=ControlCategory.CONCENTRATION,
        description="No single issuer exposure shall exceed 10% of NAV",
        computation_sql="""
            SELECT 
                MAX(issuer_concentration_pct) AS calculated_value,
                issuer_name AS breach_entity,
                issuer_exposure_usd AS breach_exposure
            FROM compliance.v_issuer_concentration
            WHERE snapshot_id = :snapshot_id
              AND fund_id = :fund_id
            GROUP BY issuer_name, issuer_exposure_usd
            ORDER BY issuer_concentration_pct DESC
            LIMIT 1
        """,
        threshold_value=10.0,
        threshold_operator=ThresholdOperator.GT,
        warning_threshold=8.0,
        warning_operator=ThresholdOperator.GT,
        regulatory_reference="Investment Policy Section 4.2.1",
        policy_document_id="POL-CONC-001",
    ),
    
    ControlDefinition(
        control_code="CONC_002",
        control_name="Sector Concentration Limit",
        category=ControlCategory.CONCENTRATION,
        description="No single sector exposure shall exceed 25% of NAV",
        computation_sql="""
            SELECT 
                MAX(sector_concentration_pct) AS calculated_value,
                sector AS breach_entity
            FROM compliance.v_sector_concentration
            WHERE snapshot_id = :snapshot_id
              AND fund_id = :fund_id
            GROUP BY sector
            ORDER BY sector_concentration_pct DESC
            LIMIT 1
        """,
        threshold_value=25.0,
        threshold_operator=ThresholdOperator.GT,
        warning_threshold=20.0,
        warning_operator=ThresholdOperator.GT,
        regulatory_reference="Investment Policy Section 4.2.2",
        policy_document_id="POL-CONC-001",
    ),
    
    ControlDefinition(
        control_code="CONC_003",
        control_name="Country Concentration Limit",
        category=ControlCategory.CONCENTRATION,
        description="Non-US country exposure shall not exceed 15% of NAV (excluding developed markets)",
        computation_sql="""
            SELECT 
                MAX(country_concentration_pct) AS calculated_value,
                country AS breach_entity
            FROM compliance.v_country_concentration
            WHERE snapshot_id = :snapshot_id
              AND fund_id = :fund_id
              AND country != 'United States'
              AND region NOT IN ('Europe', 'Japan', 'Australia', 'Canada')
            GROUP BY country
            ORDER BY country_concentration_pct DESC
            LIMIT 1
        """,
        threshold_value=15.0,
        threshold_operator=ThresholdOperator.GT,
        warning_threshold=12.0,
        warning_operator=ThresholdOperator.GT,
        regulatory_reference="Investment Policy Section 4.2.3",
        policy_document_id="POL-CONC-001",
    ),
    
    ControlDefinition(
        control_code="CONC_004",
        control_name="Top 5 Issuers Concentration",
        category=ControlCategory.CONCENTRATION,
        description="Top 5 issuers combined shall not exceed 30% of NAV",
        computation_sql="""
            WITH top_issuers AS (
                SELECT 
                    issuer_concentration_pct
                FROM compliance.v_issuer_concentration
                WHERE snapshot_id = :snapshot_id
                  AND fund_id = :fund_id
                ORDER BY issuer_concentration_pct DESC
                LIMIT 5
            )
            SELECT SUM(issuer_concentration_pct) AS calculated_value
            FROM top_issuers
        """,
        threshold_value=30.0,
        threshold_operator=ThresholdOperator.GT,
        warning_threshold=25.0,
        warning_operator=ThresholdOperator.GT,
        regulatory_reference="Investment Policy Section 4.2.4",
    ),
]


LIQUIDITY_CONTROLS = [
    ControlDefinition(
        control_code="LIQ_001",
        control_name="Minimum Liquidity - 1 Day",
        category=ControlCategory.LIQUIDITY,
        description="At least 10% of NAV must be liquidatable within 1 business day",
        computation_sql="""
            SELECT 
                cumulative_long_pct AS calculated_value
            FROM compliance.v_cumulative_liquidity
            WHERE snapshot_id = :snapshot_id
              AND fund_id = :fund_id
              AND liquidity_bucket = '1d'
        """,
        threshold_value=10.0,
        threshold_operator=ThresholdOperator.LT,
        warning_threshold=15.0,
        warning_operator=ThresholdOperator.LT,
        regulatory_reference="Form PF Section 22",
        policy_document_id="POL-LIQ-001",
    ),
    
    ControlDefinition(
        control_code="LIQ_002",
        control_name="Minimum Liquidity - 7 Days",
        category=ControlCategory.LIQUIDITY,
        description="At least 25% of NAV must be liquidatable within 7 business days",
        computation_sql="""
            SELECT 
                cumulative_long_pct AS calculated_value
            FROM compliance.v_cumulative_liquidity
            WHERE snapshot_id = :snapshot_id
              AND fund_id = :fund_id
              AND liquidity_bucket = '2-7d'
        """,
        threshold_value=25.0,
        threshold_operator=ThresholdOperator.LT,
        warning_threshold=30.0,
        warning_operator=ThresholdOperator.LT,
        regulatory_reference="Form PF Section 22",
        policy_document_id="POL-LIQ-001",
    ),
    
    ControlDefinition(
        control_code="LIQ_003",
        control_name="Minimum Liquidity - 30 Days",
        category=ControlCategory.LIQUIDITY,
        description="At least 50% of NAV must be liquidatable within 30 days",
        computation_sql="""
            SELECT 
                cumulative_long_pct AS calculated_value
            FROM compliance.v_cumulative_liquidity
            WHERE snapshot_id = :snapshot_id
              AND fund_id = :fund_id
              AND liquidity_bucket = '8-30d'
        """,
        threshold_value=50.0,
        threshold_operator=ThresholdOperator.LT,
        warning_threshold=55.0,
        warning_operator=ThresholdOperator.LT,
        regulatory_reference="Form PF Section 22",
        policy_document_id="POL-LIQ-001",
    ),
    
    ControlDefinition(
        control_code="LIQ_004",
        control_name="Illiquid Assets Limit",
        category=ControlCategory.LIQUIDITY,
        description="Assets with >365 day liquidity horizon shall not exceed 15% of NAV",
        computation_sql="""
            SELECT 
                long_pct_nav AS calculated_value
            FROM compliance.v_liquidity_buckets
            WHERE snapshot_id = :snapshot_id
              AND fund_id = :fund_id
              AND liquidity_bucket = '>365d'
        """,
        threshold_value=15.0,
        threshold_operator=ThresholdOperator.GT,
        warning_threshold=12.0,
        warning_operator=ThresholdOperator.GT,
        regulatory_reference="Form PF Section 22",
        policy_document_id="POL-LIQ-001",
    ),
]


COUNTERPARTY_CONTROLS = [
    ControlDefinition(
        control_code="CP_001",
        control_name="Single Counterparty Exposure Limit",
        category=ControlCategory.COUNTERPARTY,
        description="Net exposure to any single counterparty shall not exceed 15% of NAV",
        computation_sql="""
            WITH fund_nav AS (
                SELECT fund_nav FROM compliance.v_leverage_metrics
                WHERE snapshot_id = :snapshot_id AND fund_id = :fund_id
            )
            SELECT 
                MAX(ce.net_exposure / fn.fund_nav * 100) AS calculated_value
            FROM compliance.v_counterparty_exposure ce
            CROSS JOIN fund_nav fn
            WHERE ce.snapshot_id = :snapshot_id
              AND ce.fund_id = :fund_id
        """,
        threshold_value=15.0,
        threshold_operator=ThresholdOperator.GT,
        warning_threshold=12.0,
        warning_operator=ThresholdOperator.GT,
        regulatory_reference="Investment Policy Section 5.1",
        policy_document_id="POL-CP-001",
    ),
    
    ControlDefinition(
        control_code="CP_002",
        control_name="Counterparty Limit Utilization",
        category=ControlCategory.COUNTERPARTY,
        description="Counterparty exposure shall not exceed 95% of internal limit",
        computation_sql="""
            SELECT 
                MAX(limit_utilization_pct) AS calculated_value,
                counterparty_name AS breach_entity
            FROM compliance.v_counterparty_exposure
            WHERE snapshot_id = :snapshot_id
              AND fund_id = :fund_id
              AND internal_limit > 0
            GROUP BY counterparty_name
            ORDER BY limit_utilization_pct DESC
            LIMIT 1
        """,
        threshold_value=95.0,
        threshold_operator=ThresholdOperator.GT,
        warning_threshold=80.0,
        warning_operator=ThresholdOperator.GT,
        regulatory_reference="Investment Policy Section 5.2",
        policy_document_id="POL-CP-001",
    ),
    
    ControlDefinition(
        control_code="CP_003",
        control_name="Uncollateralized Exposure Limit",
        category=ControlCategory.COUNTERPARTY,
        description="Uncollateralized counterparty exposure shall not exceed 5% of NAV",
        computation_sql="""
            WITH fund_nav AS (
                SELECT fund_nav FROM compliance.v_leverage_metrics
                WHERE snapshot_id = :snapshot_id AND fund_id = :fund_id
            )
            SELECT 
                MAX((ce.net_exposure - ce.collateral_posted) / fn.fund_nav * 100) AS calculated_value
            FROM compliance.v_counterparty_exposure ce
            CROSS JOIN fund_nav fn
            WHERE ce.snapshot_id = :snapshot_id
              AND ce.fund_id = :fund_id
              AND ce.net_exposure > ce.collateral_posted
        """,
        threshold_value=5.0,
        threshold_operator=ThresholdOperator.GT,
        warning_threshold=4.0,
        warning_operator=ThresholdOperator.GT,
        regulatory_reference="Form PF Section 29",
        policy_document_id="POL-CP-001",
    ),
]


LEVERAGE_CONTROLS = [
    ControlDefinition(
        control_code="LEV_001",
        control_name="Gross Leverage Limit",
        category=ControlCategory.LEVERAGE,
        description="Gross leverage shall not exceed 2.0x NAV",
        computation_sql="""
            SELECT 
                gross_leverage_ratio AS calculated_value
            FROM compliance.v_leverage_metrics
            WHERE snapshot_id = :snapshot_id
              AND fund_id = :fund_id
        """,
        threshold_value=2.0,
        threshold_operator=ThresholdOperator.GT,
        warning_threshold=1.8,
        warning_operator=ThresholdOperator.GT,
        regulatory_reference="Investment Policy Section 3.1",
        policy_document_id="POL-LEV-001",
    ),
    
    ControlDefinition(
        control_code="LEV_002",
        control_name="Net Leverage Limit",
        category=ControlCategory.LEVERAGE,
        description="Net leverage shall not exceed 1.5x NAV",
        computation_sql="""
            SELECT 
                ABS(net_leverage_ratio) AS calculated_value
            FROM compliance.v_leverage_metrics
            WHERE snapshot_id = :snapshot_id
              AND fund_id = :fund_id
        """,
        threshold_value=1.5,
        threshold_operator=ThresholdOperator.GT,
        warning_threshold=1.3,
        warning_operator=ThresholdOperator.GT,
        regulatory_reference="Investment Policy Section 3.2",
        policy_document_id="POL-LEV-001",
    ),
    
    ControlDefinition(
        control_code="LEV_003",
        control_name="Borrowing to NAV Limit",
        category=ControlCategory.LEVERAGE,
        description="Total borrowing shall not exceed 50% of NAV",
        computation_sql="""
            SELECT 
                borrowing_to_nav_ratio * 100 AS calculated_value
            FROM compliance.v_leverage_metrics
            WHERE snapshot_id = :snapshot_id
              AND fund_id = :fund_id
        """,
        threshold_value=50.0,
        threshold_operator=ThresholdOperator.GT,
        warning_threshold=40.0,
        warning_operator=ThresholdOperator.GT,
        regulatory_reference="Investment Policy Section 3.3",
        policy_document_id="POL-LEV-001",
    ),
]


VALUATION_CONTROLS = [
    ControlDefinition(
        control_code="VAL_001",
        control_name="Stale Price Threshold",
        category=ControlCategory.VALUATION,
        description="Positions with stale prices (>3 days old) shall not exceed 5% of NAV",
        computation_sql="""
            WITH fund_nav AS (
                SELECT SUM(market_value_usd) AS nav
                FROM compliance.v_positions
                WHERE snapshot_id = :snapshot_id AND fund_id = :fund_id
            ),
            stale AS (
                SELECT SUM(market_value_usd) AS stale_value
                FROM compliance.v_stale_prices
                WHERE snapshot_id = :snapshot_id 
                  AND fund_id = :fund_id
                  AND days_stale > 3
            )
            SELECT 
                COALESCE(stale.stale_value / NULLIF(fn.nav, 0) * 100, 0) AS calculated_value
            FROM fund_nav fn
            CROSS JOIN stale
        """,
        threshold_value=5.0,
        threshold_operator=ThresholdOperator.GT,
        warning_threshold=3.0,
        warning_operator=ThresholdOperator.GT,
        regulatory_reference="Valuation Policy Section 2.3",
        policy_document_id="POL-VAL-001",
    ),
    
    ControlDefinition(
        control_code="VAL_002",
        control_name="Missing Reference Data",
        category=ControlCategory.VALUATION,
        description="Positions missing critical reference data (CUSIP, ISIN, Issuer) shall be flagged",
        computation_sql="""
            SELECT 
                COUNT(*) AS calculated_value
            FROM compliance.v_missing_reference_data
            WHERE snapshot_id = :snapshot_id
        """,
        threshold_value=0,
        threshold_operator=ThresholdOperator.GT,
        warning_threshold=0,
        warning_operator=ThresholdOperator.GT,
        regulatory_reference="Data Quality Policy Section 3.1",
        policy_document_id="POL-DQ-001",
        fund_specific=False,  # Runs once for all funds
    ),
]


REGULATORY_CONTROLS = [
    ControlDefinition(
        control_code="REG_001",
        control_name="13F Filing Completeness",
        category=ControlCategory.REGULATORY,
        description="All 13F-reportable positions must have valid CUSIP identifiers",
        computation_sql="""
            SELECT 
                COUNT(*) AS calculated_value
            FROM compliance.v_13f_holdings
            WHERE snapshot_id = :snapshot_id
              AND fund_id = :fund_id
              AND (cusip IS NULL OR LENGTH(cusip) != 9)
        """,
        threshold_value=0,
        threshold_operator=ThresholdOperator.GT,
        regulatory_reference="SEC Rule 13F",
        policy_document_id="POL-REG-001",
    ),
    
    ControlDefinition(
        control_code="REG_002",
        control_name="Form PF Data Availability",
        category=ControlCategory.REGULATORY,
        description="All Form PF required data elements must be populated",
        computation_sql="""
            WITH required_data AS (
                SELECT 
                    CASE WHEN COUNT(*) > 0 THEN 1 ELSE 0 END AS has_positions,
                    CASE WHEN SUM(CASE WHEN liquidity_bucket IS NOT NULL THEN 1 ELSE 0 END) > 0 THEN 1 ELSE 0 END AS has_liquidity,
                    CASE WHEN SUM(CASE WHEN country IS NOT NULL THEN 1 ELSE 0 END) > 0 THEN 1 ELSE 0 END AS has_geography
                FROM compliance.v_positions p
                LEFT JOIN compliance.v_liquidity_buckets l ON p.fund_id = l.fund_id
                WHERE p.snapshot_id = :snapshot_id AND p.fund_id = :fund_id
            )
            SELECT 
                3 - (has_positions + has_liquidity + has_geography) AS calculated_value
            FROM required_data
        """,
        threshold_value=0,
        threshold_operator=ThresholdOperator.GT,
        regulatory_reference="Form PF Instructions",
        policy_document_id="POL-REG-002",
    ),
]


# =============================================================================
# CONTROL REGISTRY
# =============================================================================

def get_all_controls() -> list[ControlDefinition]:
    """Get all defined controls."""
    return (
        CONCENTRATION_CONTROLS +
        LIQUIDITY_CONTROLS +
        COUNTERPARTY_CONTROLS +
        LEVERAGE_CONTROLS +
        VALUATION_CONTROLS +
        REGULATORY_CONTROLS
    )


def get_active_controls() -> list[ControlDefinition]:
    """Get only currently active controls."""
    return [c for c in get_all_controls() if c.is_active]


def get_controls_by_category(category: ControlCategory) -> list[ControlDefinition]:
    """Get controls for a specific category."""
    return [c for c in get_active_controls() if c.category == category]


def get_control_by_code(code: str) -> Optional[ControlDefinition]:
    """Get a specific control by its code."""
    for control in get_all_controls():
        if control.control_code == code:
            return control
    return None


def get_controls_config_hash() -> str:
    """
    Generate hash of all control definitions for audit trail.
    
    This hash changes when any control is added, modified, or removed.
    """
    controls_str = "|".join([
        f"{c.control_code}:{c.query_hash}:{c.threshold_value}:{c.threshold_operator.value}"
        for c in sorted(get_all_controls(), key=lambda x: x.control_code)
    ])
    return hashlib.sha256(controls_str.encode()).hexdigest()
