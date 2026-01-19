"""
Data Quality Framework - Pristine Data Validation

This module provides comprehensive data validation for the compliance RAG system.
Every piece of data entering the system must pass quality gates.

Quality Dimensions:
1. COMPLETENESS - No missing required fields
2. ACCURACY - Values within expected ranges
3. CONSISTENCY - Cross-field validation
4. TIMELINESS - Data freshness checks
5. UNIQUENESS - No duplicates
6. VALIDITY - Proper data types and formats
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, date, timedelta
from decimal import Decimal, InvalidOperation
from typing import Optional, Dict, Any, List, Callable, Set, Tuple
from enum import Enum
import re

logger = logging.getLogger(__name__)


class QualityDimension(Enum):
    """Data quality dimensions following DAMA standards."""
    COMPLETENESS = "completeness"
    ACCURACY = "accuracy"
    CONSISTENCY = "consistency"
    TIMELINESS = "timeliness"
    UNIQUENESS = "uniqueness"
    VALIDITY = "validity"


class QualitySeverity(Enum):
    """Severity levels for quality issues."""
    CRITICAL = "critical"  # Blocks processing
    HIGH = "high"          # Requires review before processing
    MEDIUM = "medium"      # Logged, processing continues
    LOW = "low"            # Informational only


@dataclass
class QualityIssue:
    """Represents a single data quality issue."""
    issue_id: str
    dimension: QualityDimension
    severity: QualitySeverity
    field_name: str
    record_id: Optional[str]
    expected_value: Optional[str]
    actual_value: Optional[str]
    message: str
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "issue_id": self.issue_id,
            "dimension": self.dimension.value,
            "severity": self.severity.value,
            "field_name": self.field_name,
            "record_id": self.record_id,
            "expected_value": self.expected_value,
            "actual_value": self.actual_value,
            "message": self.message,
            "detected_at": self.detected_at.isoformat(),
        }


@dataclass
class QualityReport:
    """Comprehensive quality report for a dataset."""
    report_id: str
    dataset_name: str
    record_count: int
    validated_at: datetime
    
    # Quality scores by dimension (0-100)
    completeness_score: float = 100.0
    accuracy_score: float = 100.0
    consistency_score: float = 100.0
    timeliness_score: float = 100.0
    uniqueness_score: float = 100.0
    validity_score: float = 100.0
    
    # Issues found
    issues: List[QualityIssue] = field(default_factory=list)
    
    # Processing decision
    is_acceptable: bool = True
    rejection_reason: Optional[str] = None
    
    @property
    def overall_score(self) -> float:
        """Weighted average of all quality dimensions."""
        weights = {
            'completeness': 0.20,
            'accuracy': 0.25,
            'consistency': 0.15,
            'timeliness': 0.15,
            'uniqueness': 0.15,
            'validity': 0.10,
        }
        return (
            self.completeness_score * weights['completeness'] +
            self.accuracy_score * weights['accuracy'] +
            self.consistency_score * weights['consistency'] +
            self.timeliness_score * weights['timeliness'] +
            self.uniqueness_score * weights['uniqueness'] +
            self.validity_score * weights['validity']
        )
    
    @property
    def critical_issues(self) -> List[QualityIssue]:
        return [i for i in self.issues if i.severity == QualitySeverity.CRITICAL]
    
    @property
    def high_issues(self) -> List[QualityIssue]:
        return [i for i in self.issues if i.severity == QualitySeverity.HIGH]
    
    def to_summary(self) -> str:
        """Human-readable summary."""
        return f"""
Data Quality Report: {self.dataset_name}
========================================
Records Validated: {self.record_count:,}
Validated At: {self.validated_at.isoformat()}

Quality Scores:
  Completeness: {self.completeness_score:.1f}%
  Accuracy:     {self.accuracy_score:.1f}%
  Consistency:  {self.consistency_score:.1f}%
  Timeliness:   {self.timeliness_score:.1f}%
  Uniqueness:   {self.uniqueness_score:.1f}%
  Validity:     {self.validity_score:.1f}%
  ─────────────────────
  OVERALL:      {self.overall_score:.1f}%

Issues Found:
  Critical: {len(self.critical_issues)}
  High:     {len(self.high_issues)}
  Total:    {len(self.issues)}

Status: {'✅ ACCEPTABLE' if self.is_acceptable else '❌ REJECTED - ' + (self.rejection_reason or 'Quality threshold not met')}
"""


# =============================================================================
# VALIDATION RULES
# =============================================================================

@dataclass
class ValidationRule:
    """A single validation rule."""
    rule_id: str
    name: str
    dimension: QualityDimension
    severity: QualitySeverity
    description: str
    validator: Callable[[Any, Dict[str, Any]], Optional[QualityIssue]]


class PositionDataValidator:
    """
    Validates position data quality for compliance calculations.
    
    Critical for accurate concentration and exposure calculations.
    """
    
    def __init__(self):
        self.rules = self._build_rules()
    
    def _build_rules(self) -> List[ValidationRule]:
        """Build validation rules for position data."""
        return [
            # COMPLETENESS RULES
            ValidationRule(
                rule_id="POS_COMP_001",
                name="Required Fields Present",
                dimension=QualityDimension.COMPLETENESS,
                severity=QualitySeverity.CRITICAL,
                description="All required position fields must be present",
                validator=self._validate_required_fields,
            ),
            ValidationRule(
                rule_id="POS_COMP_002",
                name="Security Identifier Present",
                dimension=QualityDimension.COMPLETENESS,
                severity=QualitySeverity.CRITICAL,
                description="At least one security identifier (ISIN, CUSIP, SEDOL, ticker) required",
                validator=self._validate_security_identifier,
            ),
            
            # ACCURACY RULES
            ValidationRule(
                rule_id="POS_ACC_001",
                name="Market Value Calculation",
                dimension=QualityDimension.ACCURACY,
                severity=QualitySeverity.HIGH,
                description="Market value should equal quantity × price (within tolerance)",
                validator=self._validate_market_value_calculation,
            ),
            ValidationRule(
                rule_id="POS_ACC_002",
                name="Reasonable Market Value",
                dimension=QualityDimension.ACCURACY,
                severity=QualitySeverity.HIGH,
                description="Market value within reasonable bounds for position type",
                validator=self._validate_reasonable_market_value,
            ),
            ValidationRule(
                rule_id="POS_ACC_003",
                name="FX Rate Reasonableness",
                dimension=QualityDimension.ACCURACY,
                severity=QualitySeverity.MEDIUM,
                description="FX rate to USD within expected range",
                validator=self._validate_fx_rate,
            ),
            
            # CONSISTENCY RULES
            ValidationRule(
                rule_id="POS_CON_001",
                name="Long/Short Sign Consistency",
                dimension=QualityDimension.CONSISTENCY,
                severity=QualitySeverity.CRITICAL,
                description="Quantity sign must match long/short indicator",
                validator=self._validate_long_short_consistency,
            ),
            ValidationRule(
                rule_id="POS_CON_002",
                name="Currency Consistency",
                dimension=QualityDimension.CONSISTENCY,
                severity=QualitySeverity.HIGH,
                description="Local currency and FX rate must be consistent",
                validator=self._validate_currency_consistency,
            ),
            
            # TIMELINESS RULES
            ValidationRule(
                rule_id="POS_TIM_001",
                name="Price Freshness",
                dimension=QualityDimension.TIMELINESS,
                severity=QualitySeverity.HIGH,
                description="Price date should not be stale (>2 business days)",
                validator=self._validate_price_freshness,
            ),
            ValidationRule(
                rule_id="POS_TIM_002",
                name="Snapshot Currency",
                dimension=QualityDimension.TIMELINESS,
                severity=QualitySeverity.MEDIUM,
                description="Snapshot should be from current or previous business day",
                validator=self._validate_snapshot_currency,
            ),
            
            # VALIDITY RULES
            ValidationRule(
                rule_id="POS_VAL_001",
                name="Valid Asset Class",
                dimension=QualityDimension.VALIDITY,
                severity=QualitySeverity.MEDIUM,
                description="Asset class must be from approved list",
                validator=self._validate_asset_class,
            ),
            ValidationRule(
                rule_id="POS_VAL_002",
                name="Valid ISIN Format",
                dimension=QualityDimension.VALIDITY,
                severity=QualitySeverity.MEDIUM,
                description="ISIN must match standard format (12 chars, 2 letter prefix)",
                validator=self._validate_isin_format,
            ),
        ]
    
    def validate(self, positions: List[Dict[str, Any]], snapshot_date: date) -> QualityReport:
        """
        Validate a batch of position records.
        
        Args:
            positions: List of position dictionaries
            snapshot_date: Date of the snapshot being validated
            
        Returns:
            QualityReport with detailed findings
        """
        import uuid
        
        report = QualityReport(
            report_id=str(uuid.uuid4()),
            dataset_name="positions",
            record_count=len(positions),
            validated_at=datetime.now(timezone.utc),
        )
        
        if not positions:
            report.is_acceptable = False
            report.rejection_reason = "No positions to validate"
            return report
        
        # Track issues by dimension
        dimension_issues: Dict[QualityDimension, List[QualityIssue]] = {
            dim: [] for dim in QualityDimension
        }
        
        # Validate each position against all rules
        for position in positions:
            context = {"snapshot_date": snapshot_date, "all_positions": positions}
            
            for rule in self.rules:
                issue = rule.validator(position, context)
                if issue:
                    report.issues.append(issue)
                    dimension_issues[rule.dimension].append(issue)
        
        # Calculate dimension scores
        report.completeness_score = self._calculate_dimension_score(
            dimension_issues[QualityDimension.COMPLETENESS], len(positions)
        )
        report.accuracy_score = self._calculate_dimension_score(
            dimension_issues[QualityDimension.ACCURACY], len(positions)
        )
        report.consistency_score = self._calculate_dimension_score(
            dimension_issues[QualityDimension.CONSISTENCY], len(positions)
        )
        report.timeliness_score = self._calculate_dimension_score(
            dimension_issues[QualityDimension.TIMELINESS], len(positions)
        )
        report.uniqueness_score = self._calculate_uniqueness_score(positions)
        report.validity_score = self._calculate_dimension_score(
            dimension_issues[QualityDimension.VALIDITY], len(positions)
        )
        
        # Determine if dataset is acceptable
        # Critical issues always reject
        if report.critical_issues:
            report.is_acceptable = False
            report.rejection_reason = f"Critical issues found: {len(report.critical_issues)}"
        # Overall score must be above threshold
        elif report.overall_score < 95.0:
            report.is_acceptable = False
            report.rejection_reason = f"Overall quality score {report.overall_score:.1f}% below 95% threshold"
        # Too many high-severity issues
        elif len(report.high_issues) > len(positions) * 0.05:  # >5% with high issues
            report.is_acceptable = False
            report.rejection_reason = f"High severity issues affect >5% of records"
        
        return report
    
    def _calculate_dimension_score(self, issues: List[QualityIssue], total_records: int) -> float:
        """Calculate quality score for a dimension."""
        if total_records == 0:
            return 0.0
        
        # Weight by severity
        penalty = 0.0
        for issue in issues:
            if issue.severity == QualitySeverity.CRITICAL:
                penalty += 10.0  # Each critical = 10% penalty
            elif issue.severity == QualitySeverity.HIGH:
                penalty += 2.0
            elif issue.severity == QualitySeverity.MEDIUM:
                penalty += 0.5
            else:
                penalty += 0.1
        
        # Normalize by record count
        penalty_rate = (penalty / total_records) * 100
        return max(0.0, 100.0 - penalty_rate)
    
    def _calculate_uniqueness_score(self, positions: List[Dict[str, Any]]) -> float:
        """Check for duplicate positions."""
        if not positions:
            return 100.0
        
        # Create composite key for uniqueness
        seen_keys: Set[str] = set()
        duplicates = 0
        
        for pos in positions:
            key = f"{pos.get('fund_id')}|{pos.get('account_id')}|{pos.get('security_id')}"
            if key in seen_keys:
                duplicates += 1
            seen_keys.add(key)
        
        if duplicates > 0:
            return max(0.0, (1 - duplicates / len(positions)) * 100)
        return 100.0
    
    # ==========================================================================
    # VALIDATION FUNCTIONS
    # ==========================================================================
    
    def _validate_required_fields(self, position: Dict[str, Any], context: Dict) -> Optional[QualityIssue]:
        """Check all required fields are present and non-null."""
        required_fields = [
            'position_id', 'snapshot_id', 'fund_id', 'security_id',
            'quantity', 'market_value_usd'
        ]
        
        missing = [f for f in required_fields if not position.get(f)]
        
        if missing:
            return QualityIssue(
                issue_id=f"COMP_{position.get('position_id', 'UNKNOWN')}",
                dimension=QualityDimension.COMPLETENESS,
                severity=QualitySeverity.CRITICAL,
                field_name=", ".join(missing),
                record_id=position.get('position_id'),
                expected_value="Non-null value",
                actual_value="NULL or missing",
                message=f"Required fields missing: {missing}",
            )
        return None
    
    def _validate_security_identifier(self, position: Dict[str, Any], context: Dict) -> Optional[QualityIssue]:
        """At least one security identifier must be present."""
        identifiers = ['isin', 'cusip', 'sedol', 'ticker']
        has_identifier = any(position.get(i) for i in identifiers)
        
        if not has_identifier:
            return QualityIssue(
                issue_id=f"SECID_{position.get('position_id', 'UNKNOWN')}",
                dimension=QualityDimension.COMPLETENESS,
                severity=QualitySeverity.CRITICAL,
                field_name="security_identifiers",
                record_id=position.get('position_id'),
                expected_value="At least one of: ISIN, CUSIP, SEDOL, ticker",
                actual_value="None present",
                message="No security identifier found - cannot identify position",
            )
        return None
    
    def _validate_market_value_calculation(self, position: Dict[str, Any], context: Dict) -> Optional[QualityIssue]:
        """Market value should approximately equal quantity × price."""
        try:
            quantity = Decimal(str(position.get('quantity', 0)))
            price = Decimal(str(position.get('price', 0)))
            market_value = Decimal(str(position.get('market_value_usd', 0)))
            
            if price == 0:
                return None  # Can't validate without price
            
            expected_mv = abs(quantity * price)
            actual_mv = abs(market_value)
            
            if expected_mv == 0:
                return None
            
            # Allow 5% tolerance for FX, fees, etc.
            diff_pct = abs((actual_mv - expected_mv) / expected_mv) * 100
            
            if diff_pct > 5:
                return QualityIssue(
                    issue_id=f"MV_{position.get('position_id', 'UNKNOWN')}",
                    dimension=QualityDimension.ACCURACY,
                    severity=QualitySeverity.HIGH,
                    field_name="market_value_usd",
                    record_id=position.get('position_id'),
                    expected_value=f"~{float(expected_mv):.2f} (qty × price)",
                    actual_value=str(market_value),
                    message=f"Market value differs from qty×price by {diff_pct:.1f}%",
                )
        except (InvalidOperation, TypeError, ZeroDivisionError):
            pass
        return None
    
    def _validate_reasonable_market_value(self, position: Dict[str, Any], context: Dict) -> Optional[QualityIssue]:
        """Market value should be within reasonable bounds."""
        try:
            mv = float(position.get('market_value_usd', 0))
            
            # Single position > $1B is suspicious for most hedge funds
            if abs(mv) > 1_000_000_000:
                return QualityIssue(
                    issue_id=f"MVBIG_{position.get('position_id', 'UNKNOWN')}",
                    dimension=QualityDimension.ACCURACY,
                    severity=QualitySeverity.HIGH,
                    field_name="market_value_usd",
                    record_id=position.get('position_id'),
                    expected_value="< $1B for typical position",
                    actual_value=f"${mv:,.2f}",
                    message="Unusually large market value - verify data",
                )
        except (TypeError, ValueError):
            pass
        return None
    
    def _validate_fx_rate(self, position: Dict[str, Any], context: Dict) -> Optional[QualityIssue]:
        """FX rate should be within reasonable bounds."""
        try:
            fx_rate = float(position.get('fx_rate_to_usd', 1.0))
            currency = position.get('currency', 'USD')
            
            # FX rates outside 0.001 to 1000 are suspicious
            if fx_rate <= 0 or fx_rate > 1000:
                return QualityIssue(
                    issue_id=f"FX_{position.get('position_id', 'UNKNOWN')}",
                    dimension=QualityDimension.ACCURACY,
                    severity=QualitySeverity.MEDIUM,
                    field_name="fx_rate_to_usd",
                    record_id=position.get('position_id'),
                    expected_value="0.001 to 1000",
                    actual_value=str(fx_rate),
                    message=f"FX rate for {currency} outside expected range",
                )
        except (TypeError, ValueError):
            pass
        return None
    
    def _validate_long_short_consistency(self, position: Dict[str, Any], context: Dict) -> Optional[QualityIssue]:
        """Quantity sign must be consistent with position direction."""
        try:
            quantity = float(position.get('quantity', 0))
            quantity_long = float(position.get('quantity_long', 0))
            quantity_short = float(position.get('quantity_short', 0))
            
            # If both long and short quantities provided, they should match
            if quantity > 0 and quantity_short > 0 and quantity_long == 0:
                return QualityIssue(
                    issue_id=f"LS_{position.get('position_id', 'UNKNOWN')}",
                    dimension=QualityDimension.CONSISTENCY,
                    severity=QualitySeverity.CRITICAL,
                    field_name="quantity/quantity_long/quantity_short",
                    record_id=position.get('position_id'),
                    expected_value="Positive quantity with quantity_long > 0",
                    actual_value=f"qty={quantity}, long={quantity_long}, short={quantity_short}",
                    message="Long/short quantities inconsistent with position quantity",
                )
        except (TypeError, ValueError):
            pass
        return None
    
    def _validate_currency_consistency(self, position: Dict[str, Any], context: Dict) -> Optional[QualityIssue]:
        """Currency and FX rate should be consistent."""
        currency = position.get('currency', 'USD')
        fx_rate = position.get('fx_rate_to_usd', 1.0)
        
        # USD should have FX rate of 1.0
        if currency == 'USD' and fx_rate != 1.0:
            return QualityIssue(
                issue_id=f"CURR_{position.get('position_id', 'UNKNOWN')}",
                dimension=QualityDimension.CONSISTENCY,
                severity=QualitySeverity.HIGH,
                field_name="currency/fx_rate_to_usd",
                record_id=position.get('position_id'),
                expected_value="FX rate = 1.0 for USD",
                actual_value=f"currency={currency}, fx_rate={fx_rate}",
                message="USD position should have FX rate of 1.0",
            )
        return None
    
    def _validate_price_freshness(self, position: Dict[str, Any], context: Dict) -> Optional[QualityIssue]:
        """Price should not be stale (>2 business days old)."""
        try:
            snapshot_date = context.get('snapshot_date', date.today())
            price_date_str = position.get('price_date')
            
            if not price_date_str:
                return None
            
            if isinstance(price_date_str, date):
                price_date = price_date_str
            else:
                price_date = datetime.strptime(str(price_date_str)[:10], '%Y-%m-%d').date()
            
            days_stale = (snapshot_date - price_date).days
            
            if days_stale > 2:
                return QualityIssue(
                    issue_id=f"STALE_{position.get('position_id', 'UNKNOWN')}",
                    dimension=QualityDimension.TIMELINESS,
                    severity=QualitySeverity.HIGH,
                    field_name="price_date",
                    record_id=position.get('position_id'),
                    expected_value=f"Within 2 days of {snapshot_date}",
                    actual_value=str(price_date),
                    message=f"Price is {days_stale} days stale",
                )
        except (ValueError, TypeError):
            pass
        return None
    
    def _validate_snapshot_currency(self, position: Dict[str, Any], context: Dict) -> Optional[QualityIssue]:
        """Snapshot should be current."""
        try:
            snapshot_date = context.get('snapshot_date', date.today())
            today = date.today()
            
            days_old = (today - snapshot_date).days
            
            if days_old > 1:
                return QualityIssue(
                    issue_id=f"SNAPOLD_{position.get('position_id', 'UNKNOWN')}",
                    dimension=QualityDimension.TIMELINESS,
                    severity=QualitySeverity.MEDIUM,
                    field_name="snapshot_date",
                    record_id=position.get('position_id'),
                    expected_value=f"Today ({today}) or yesterday",
                    actual_value=str(snapshot_date),
                    message=f"Processing snapshot that is {days_old} days old",
                )
        except (ValueError, TypeError):
            pass
        return None
    
    def _validate_asset_class(self, position: Dict[str, Any], context: Dict) -> Optional[QualityIssue]:
        """Asset class must be from approved list."""
        valid_classes = {
            'equity', 'fixed_income', 'bond', 'derivative', 'option', 'future',
            'fx', 'commodity', 'cash', 'money_market', 'etf', 'mutual_fund',
            'private_equity', 'real_estate', 'other'
        }
        
        asset_class = (position.get('asset_class') or '').lower()
        
        if asset_class and asset_class not in valid_classes:
            return QualityIssue(
                issue_id=f"AC_{position.get('position_id', 'UNKNOWN')}",
                dimension=QualityDimension.VALIDITY,
                severity=QualitySeverity.MEDIUM,
                field_name="asset_class",
                record_id=position.get('position_id'),
                expected_value=f"One of: {', '.join(sorted(valid_classes))}",
                actual_value=asset_class,
                message=f"Unknown asset class: {asset_class}",
            )
        return None
    
    def _validate_isin_format(self, position: Dict[str, Any], context: Dict) -> Optional[QualityIssue]:
        """ISIN must match standard format."""
        isin = position.get('isin')
        
        if not isin:
            return None
        
        # ISIN: 2 letter country + 9 alphanumeric + 1 check digit = 12 chars
        isin_pattern = re.compile(r'^[A-Z]{2}[A-Z0-9]{9}[0-9]$')
        
        if not isin_pattern.match(isin):
            return QualityIssue(
                issue_id=f"ISIN_{position.get('position_id', 'UNKNOWN')}",
                dimension=QualityDimension.VALIDITY,
                severity=QualitySeverity.MEDIUM,
                field_name="isin",
                record_id=position.get('position_id'),
                expected_value="12 chars: 2 letter country + 9 alphanum + check digit",
                actual_value=isin,
                message=f"Invalid ISIN format: {isin}",
            )
        return None


# =============================================================================
# POLICY DOCUMENT VALIDATOR
# =============================================================================

class PolicyDocumentValidator:
    """Validates policy documents before ingestion into RAG."""
    
    def validate_policy(self, content: str, metadata: Dict[str, Any]) -> QualityReport:
        """Validate a policy document."""
        import uuid
        
        report = QualityReport(
            report_id=str(uuid.uuid4()),
            dataset_name=f"policy:{metadata.get('policy_id', 'unknown')}",
            record_count=1,
            validated_at=datetime.now(timezone.utc),
        )
        
        # Completeness checks
        required_metadata = ['policy_id', 'title', 'effective_date', 'version']
        missing = [f for f in required_metadata if not metadata.get(f)]
        
        if missing:
            report.issues.append(QualityIssue(
                issue_id=f"POL_META_{metadata.get('policy_id', 'UNK')}",
                dimension=QualityDimension.COMPLETENESS,
                severity=QualitySeverity.HIGH,
                field_name=", ".join(missing),
                record_id=metadata.get('policy_id'),
                expected_value="All metadata fields present",
                actual_value=f"Missing: {missing}",
                message=f"Policy metadata incomplete: {missing}",
            ))
            report.completeness_score = 50.0
        
        # Content validation
        if len(content) < 100:
            report.issues.append(QualityIssue(
                issue_id=f"POL_LEN_{metadata.get('policy_id', 'UNK')}",
                dimension=QualityDimension.COMPLETENESS,
                severity=QualitySeverity.HIGH,
                field_name="content",
                record_id=metadata.get('policy_id'),
                expected_value=">100 characters",
                actual_value=f"{len(content)} characters",
                message="Policy content too short",
            ))
        
        # Check for required sections
        required_sections = ['limit', 'threshold', 'monitoring', 'frequency']
        content_lower = content.lower()
        missing_sections = [s for s in required_sections if s not in content_lower]
        
        if missing_sections:
            report.issues.append(QualityIssue(
                issue_id=f"POL_SEC_{metadata.get('policy_id', 'UNK')}",
                dimension=QualityDimension.COMPLETENESS,
                severity=QualitySeverity.MEDIUM,
                field_name="content_sections",
                record_id=metadata.get('policy_id'),
                expected_value=f"Sections mentioning: {required_sections}",
                actual_value=f"Missing mentions of: {missing_sections}",
                message="Policy may be missing key compliance concepts",
            ))
        
        # Determine acceptability
        report.is_acceptable = not report.critical_issues and report.overall_score >= 80.0
        
        return report


# =============================================================================
# CONTROL DEFINITION VALIDATOR  
# =============================================================================

class ControlDefinitionValidator:
    """Validates control definitions before insertion."""
    
    VALID_CATEGORIES = {
        'concentration', 'exposure', 'liquidity', 'counterparty',
        'performance', 'regulatory', 'operational', 'risk'
    }
    
    VALID_OPERATORS = {'gt', 'gte', 'lt', 'lte', 'eq', 'ne', 'between'}
    
    VALID_FREQUENCIES = {'realtime', 'intraday', 'daily', 'weekly', 'monthly', 'quarterly'}
    
    def validate_control(self, control: Dict[str, Any]) -> QualityReport:
        """Validate a control definition."""
        import uuid
        
        report = QualityReport(
            report_id=str(uuid.uuid4()),
            dataset_name=f"control:{control.get('control_code', 'unknown')}",
            record_count=1,
            validated_at=datetime.now(timezone.utc),
        )
        
        # Required fields
        required = ['control_code', 'control_name', 'control_category', 
                    'threshold_operator', 'frequency', 'description']
        missing = [f for f in required if not control.get(f)]
        
        if missing:
            report.issues.append(QualityIssue(
                issue_id=f"CTRL_REQ_{control.get('control_code', 'UNK')}",
                dimension=QualityDimension.COMPLETENESS,
                severity=QualitySeverity.CRITICAL,
                field_name=", ".join(missing),
                record_id=control.get('control_code'),
                expected_value="All required fields present",
                actual_value=f"Missing: {missing}",
                message=f"Control definition incomplete",
            ))
        
        # Validate category
        category = control.get('control_category', '').lower()
        if category and category not in self.VALID_CATEGORIES:
            report.issues.append(QualityIssue(
                issue_id=f"CTRL_CAT_{control.get('control_code', 'UNK')}",
                dimension=QualityDimension.VALIDITY,
                severity=QualitySeverity.HIGH,
                field_name="control_category",
                record_id=control.get('control_code'),
                expected_value=f"One of: {self.VALID_CATEGORIES}",
                actual_value=category,
                message="Invalid control category",
            ))
        
        # Validate operator
        operator = control.get('threshold_operator', '').lower()
        if operator and operator not in self.VALID_OPERATORS:
            report.issues.append(QualityIssue(
                issue_id=f"CTRL_OP_{control.get('control_code', 'UNK')}",
                dimension=QualityDimension.VALIDITY,
                severity=QualitySeverity.CRITICAL,
                field_name="threshold_operator",
                record_id=control.get('control_code'),
                expected_value=f"One of: {self.VALID_OPERATORS}",
                actual_value=operator,
                message="Invalid threshold operator - control cannot execute",
            ))
        
        # Validate frequency
        frequency = control.get('frequency', '').lower()
        if frequency and frequency not in self.VALID_FREQUENCIES:
            report.issues.append(QualityIssue(
                issue_id=f"CTRL_FREQ_{control.get('control_code', 'UNK')}",
                dimension=QualityDimension.VALIDITY,
                severity=QualitySeverity.MEDIUM,
                field_name="frequency",
                record_id=control.get('control_code'),
                expected_value=f"One of: {self.VALID_FREQUENCIES}",
                actual_value=frequency,
                message="Non-standard frequency",
            ))
        
        # Threshold value sanity check
        threshold = control.get('threshold_value')
        if threshold is not None:
            try:
                tv = float(threshold)
                if tv < 0:
                    report.issues.append(QualityIssue(
                        issue_id=f"CTRL_THR_{control.get('control_code', 'UNK')}",
                        dimension=QualityDimension.ACCURACY,
                        severity=QualitySeverity.HIGH,
                        field_name="threshold_value",
                        record_id=control.get('control_code'),
                        expected_value=">= 0",
                        actual_value=str(threshold),
                        message="Negative threshold value",
                    ))
            except (ValueError, TypeError):
                report.issues.append(QualityIssue(
                    issue_id=f"CTRL_THR_TYPE_{control.get('control_code', 'UNK')}",
                    dimension=QualityDimension.VALIDITY,
                    severity=QualitySeverity.CRITICAL,
                    field_name="threshold_value",
                    record_id=control.get('control_code'),
                    expected_value="Numeric value",
                    actual_value=str(type(threshold)),
                    message="Threshold value must be numeric",
                ))
        
        # Calculate scores
        report.completeness_score = 100.0 if not missing else 50.0
        report.validity_score = self._calculate_dimension_score(
            [i for i in report.issues if i.dimension == QualityDimension.VALIDITY], 1
        )
        report.is_acceptable = not report.critical_issues
        
        return report
    
    def _calculate_dimension_score(self, issues: List[QualityIssue], total: int) -> float:
        if total == 0:
            return 100.0
        penalty = sum(10 if i.severity == QualitySeverity.CRITICAL else 
                      5 if i.severity == QualitySeverity.HIGH else 1 
                      for i in issues)
        return max(0.0, 100.0 - (penalty / total) * 10)
