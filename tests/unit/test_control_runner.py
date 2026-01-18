"""
Unit tests for Compliance RAG System - Control Runner Module

Tests deterministic execution of compliance controls.
These tests verify that the control definitions and execution
engine work correctly without any LLM involvement.

SEC Examination Note: Tests validate deterministic behavior.
"""

import pytest
from datetime import date, datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
import hashlib
import json
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.control_runner.controls import (
    ControlDefinition,
    ControlCategory,
    ControlFrequency,
    ThresholdOperator,
    ControlResultStatus,
    get_all_controls,
    get_active_controls,
    get_controls_by_category,
)


class TestControlCategory:
    """Tests for ControlCategory enum."""
    
    def test_all_categories_defined(self):
        """Test all required control categories are defined."""
        categories = list(ControlCategory)
        
        assert ControlCategory.CONCENTRATION in categories
        assert ControlCategory.LIQUIDITY in categories
        assert ControlCategory.COUNTERPARTY in categories
        assert ControlCategory.EXPOSURE in categories
        assert ControlCategory.LEVERAGE in categories
        assert ControlCategory.VALUATION in categories
        assert ControlCategory.RECONCILIATION in categories
        assert ControlCategory.REGULATORY in categories
    
    def test_category_values(self):
        """Test category enum values are strings."""
        for category in ControlCategory:
            assert isinstance(category.value, str)
            assert len(category.value) > 0


class TestThresholdOperator:
    """Tests for ThresholdOperator enum."""
    
    def test_all_operators_defined(self):
        """Test all comparison operators are defined."""
        operators = list(ThresholdOperator)
        
        assert ThresholdOperator.LT in operators   # Less than
        assert ThresholdOperator.LTE in operators  # Less than or equal
        assert ThresholdOperator.GT in operators   # Greater than
        assert ThresholdOperator.GTE in operators  # Greater than or equal
        assert ThresholdOperator.EQ in operators   # Equal
        assert ThresholdOperator.NEQ in operators  # Not equal
    
    def test_operator_values(self):
        """Test operator enum values."""
        assert ThresholdOperator.LT.value == "lt"
        assert ThresholdOperator.LTE.value == "lte"
        assert ThresholdOperator.GT.value == "gt"
        assert ThresholdOperator.GTE.value == "gte"
        assert ThresholdOperator.EQ.value == "eq"
        assert ThresholdOperator.NEQ.value == "neq"


class TestControlFrequency:
    """Tests for ControlFrequency enum."""
    
    def test_all_frequencies_defined(self):
        """Test all execution frequencies are defined."""
        frequencies = list(ControlFrequency)
        
        assert ControlFrequency.DAILY in frequencies
        assert ControlFrequency.WEEKLY in frequencies
        assert ControlFrequency.MONTHLY in frequencies
        assert ControlFrequency.QUARTERLY in frequencies
        assert ControlFrequency.ANNUAL in frequencies


class TestControlResultStatus:
    """Tests for ControlResultStatus enum."""
    
    def test_all_statuses_defined(self):
        """Test all result statuses are defined."""
        statuses = list(ControlResultStatus)
        
        assert ControlResultStatus.PASS in statuses
        assert ControlResultStatus.FAIL in statuses
        assert ControlResultStatus.WARNING in statuses
        assert ControlResultStatus.SKIP in statuses
        assert ControlResultStatus.ERROR in statuses
    
    def test_status_values(self):
        """Test status enum values."""
        assert ControlResultStatus.PASS.value == "pass"
        assert ControlResultStatus.FAIL.value == "fail"
        assert ControlResultStatus.WARNING.value == "warning"


class TestControlDefinition:
    """Tests for ControlDefinition dataclass."""
    
    def test_create_control_definition(self):
        """Test creating a control definition."""
        control = ControlDefinition(
            control_code="TEST_001",
            control_name="Test Control",
            category=ControlCategory.CONCENTRATION,
            description="A test control for unit testing",
            computation_sql="SELECT 0.5 AS calculated_value",
            threshold_value=1.0,
            threshold_operator=ThresholdOperator.GT,
        )
        
        assert control.control_code == "TEST_001"
        assert control.control_name == "Test Control"
        assert control.category == ControlCategory.CONCENTRATION
        assert control.threshold_value == 1.0
        assert control.threshold_operator == ThresholdOperator.GT
    
    def test_control_with_warning_threshold(self):
        """Test control with warning threshold."""
        control = ControlDefinition(
            control_code="TEST_002",
            control_name="Test Control with Warning",
            category=ControlCategory.LIQUIDITY,
            description="Control with warning threshold",
            computation_sql="SELECT 0.85 AS calculated_value",
            threshold_value=1.0,
            threshold_operator=ThresholdOperator.GT,
            warning_threshold=0.8,
            warning_operator=ThresholdOperator.GT,
        )
        
        assert control.warning_threshold == 0.8
        assert control.warning_operator == ThresholdOperator.GT
    
    def test_control_is_immutable(self):
        """Test that control definitions are immutable (frozen)."""
        control = ControlDefinition(
            control_code="TEST_003",
            control_name="Immutable Test",
            category=ControlCategory.EXPOSURE,
            description="Should not be modifiable",
            computation_sql="SELECT 1 AS calculated_value",
            threshold_value=10.0,
            threshold_operator=ThresholdOperator.GT,
        )
        
        with pytest.raises(Exception):  # FrozenInstanceError
            control.control_code = "MODIFIED"
    
    def test_query_hash_is_deterministic(self):
        """Test that SQL query hash is consistent."""
        sql = "SELECT MAX(exposure) AS calculated_value FROM positions"
        
        control1 = ControlDefinition(
            control_code="TEST_004",
            control_name="Hash Test 1",
            category=ControlCategory.EXPOSURE,
            description="Test",
            computation_sql=sql,
            threshold_value=1.0,
            threshold_operator=ThresholdOperator.GT,
        )
        
        control2 = ControlDefinition(
            control_code="TEST_005",
            control_name="Hash Test 2",
            category=ControlCategory.EXPOSURE,
            description="Test",
            computation_sql=sql,
            threshold_value=1.0,
            threshold_operator=ThresholdOperator.GT,
        )
        
        # Same SQL should produce same hash
        assert control1.query_hash == control2.query_hash


class TestThresholdEvaluation:
    """Tests for threshold evaluation logic."""
    
    @pytest.fixture
    def control_gt(self):
        """Control with greater-than threshold."""
        return ControlDefinition(
            control_code="THRESH_GT",
            control_name="GT Threshold Test",
            category=ControlCategory.CONCENTRATION,
            description="Fails when value > threshold",
            computation_sql="SELECT 1 AS calculated_value",
            threshold_value=10.0,
            threshold_operator=ThresholdOperator.GT,
            warning_threshold=8.0,
            warning_operator=ThresholdOperator.GT,
        )
    
    @pytest.fixture
    def control_lt(self):
        """Control with less-than threshold."""
        return ControlDefinition(
            control_code="THRESH_LT",
            control_name="LT Threshold Test",
            category=ControlCategory.LIQUIDITY,
            description="Fails when value < threshold",
            computation_sql="SELECT 1 AS calculated_value",
            threshold_value=0.10,
            threshold_operator=ThresholdOperator.LT,
        )
    
    def test_gt_threshold_pass(self, control_gt):
        """Test value below threshold passes (GT)."""
        result = control_gt.evaluate_threshold(5.0)
        assert result == ControlResultStatus.PASS
    
    def test_gt_threshold_warning(self, control_gt):
        """Test value above warning but below fail (GT)."""
        result = control_gt.evaluate_threshold(9.0)
        assert result == ControlResultStatus.WARNING
    
    def test_gt_threshold_fail(self, control_gt):
        """Test value above threshold fails (GT)."""
        result = control_gt.evaluate_threshold(15.0)
        assert result == ControlResultStatus.FAIL
    
    def test_gt_threshold_at_boundary(self, control_gt):
        """Test value exactly at threshold (GT)."""
        result = control_gt.evaluate_threshold(10.0)
        assert result == ControlResultStatus.WARNING  # Not > 10, but > 8
    
    def test_lt_threshold_pass(self, control_lt):
        """Test value above threshold passes (LT)."""
        result = control_lt.evaluate_threshold(0.15)
        assert result == ControlResultStatus.PASS
    
    def test_lt_threshold_fail(self, control_lt):
        """Test value below threshold fails (LT)."""
        result = control_lt.evaluate_threshold(0.05)
        assert result == ControlResultStatus.FAIL
    
    def test_breach_amount_calculation_gt(self, control_gt):
        """Test breach amount for GT threshold."""
        breach = control_gt.get_breach_amount(15.0)
        assert breach == 5.0  # 15 - 10 = 5
    
    def test_breach_amount_calculation_lt(self, control_lt):
        """Test breach amount for LT threshold."""
        breach = control_lt.get_breach_amount(0.05)
        assert breach == 0.05  # 0.10 - 0.05 = 0.05


class TestControlRegistry:
    """Tests for control registry functions."""
    
    def test_get_all_controls(self):
        """Test retrieving all registered controls."""
        controls = get_all_controls()
        
        assert isinstance(controls, list)
        assert len(controls) > 0
        
        for control in controls:
            assert isinstance(control, ControlDefinition)
    
    def test_get_active_controls(self):
        """Test retrieving only active controls."""
        active = get_active_controls()
        
        for control in active:
            assert control.is_active
    
    def test_get_controls_by_category(self):
        """Test filtering controls by category."""
        concentration_controls = get_controls_by_category(ControlCategory.CONCENTRATION)
        
        for control in concentration_controls:
            assert control.category == ControlCategory.CONCENTRATION
    
    def test_controls_have_required_fields(self):
        """Test all controls have required fields populated."""
        controls = get_all_controls()
        
        for control in controls:
            assert control.control_code is not None
            assert len(control.control_code) > 0
            assert control.control_name is not None
            assert control.computation_sql is not None
            assert control.threshold_value is not None
            assert control.threshold_operator is not None


class TestControlEffectiveDates:
    """Tests for control effective date logic."""
    
    def test_control_is_active_by_default(self):
        """Test that controls are active by default."""
        control = ControlDefinition(
            control_code="DATE_001",
            control_name="Active Control",
            category=ControlCategory.EXPOSURE,
            description="Should be active",
            computation_sql="SELECT 1 AS calculated_value",
            threshold_value=1.0,
            threshold_operator=ThresholdOperator.GT,
        )
        
        assert control.is_active
    
    def test_control_with_future_effective_date(self):
        """Test control with future effective date is inactive."""
        from datetime import timedelta
        
        future_date = date.today() + timedelta(days=30)
        
        control = ControlDefinition(
            control_code="DATE_002",
            control_name="Future Control",
            category=ControlCategory.EXPOSURE,
            description="Not yet effective",
            computation_sql="SELECT 1 AS calculated_value",
            threshold_value=1.0,
            threshold_operator=ThresholdOperator.GT,
            effective_date=future_date,
        )
        
        assert not control.is_active
    
    def test_control_with_past_expiration(self):
        """Test control with past expiration date is inactive."""
        from datetime import timedelta
        
        past_date = date.today() - timedelta(days=30)
        
        control = ControlDefinition(
            control_code="DATE_003",
            control_name="Expired Control",
            category=ControlCategory.EXPOSURE,
            description="Has expired",
            computation_sql="SELECT 1 AS calculated_value",
            threshold_value=1.0,
            threshold_operator=ThresholdOperator.GT,
            expiration_date=past_date,
        )
        
        assert not control.is_active


class TestDeterministicBehavior:
    """Tests ensuring deterministic control behavior."""
    
    def test_same_inputs_same_evaluation(self):
        """Test that same inputs produce same evaluation results."""
        control = ControlDefinition(
            control_code="DET_001",
            control_name="Deterministic Test",
            category=ControlCategory.CONCENTRATION,
            description="Must be deterministic",
            computation_sql="SELECT 0.95 AS calculated_value",
            threshold_value=1.0,
            threshold_operator=ThresholdOperator.GT,
            warning_threshold=0.8,
            warning_operator=ThresholdOperator.GT,
        )
        
        # Run evaluation multiple times
        results = [control.evaluate_threshold(0.95) for _ in range(100)]
        
        # All results should be identical
        assert all(r == ControlResultStatus.WARNING for r in results)
    
    def test_hash_consistency(self):
        """Test that hashing is consistent across runs."""
        data = {
            "control_code": "HASH_001",
            "calculated_value": 0.95,
            "threshold_value": 1.0,
            "run_date": "2024-01-15",
        }
        
        # Hash should be deterministic
        hash1 = hashlib.sha256(
            json.dumps(data, sort_keys=True).encode()
        ).hexdigest()
        
        hash2 = hashlib.sha256(
            json.dumps(data, sort_keys=True).encode()
        ).hexdigest()
        
        assert hash1 == hash2
    
    def test_control_comparison_operators(self):
        """Test all comparison operators work correctly."""
        test_cases = [
            (ThresholdOperator.GT, 10.0, 15.0, True),   # 15 > 10 = True
            (ThresholdOperator.GT, 10.0, 5.0, False),   # 5 > 10 = False
            (ThresholdOperator.GTE, 10.0, 10.0, True),  # 10 >= 10 = True
            (ThresholdOperator.LT, 10.0, 5.0, True),    # 5 < 10 = True
            (ThresholdOperator.LT, 10.0, 15.0, False),  # 15 < 10 = False
            (ThresholdOperator.LTE, 10.0, 10.0, True),  # 10 <= 10 = True
            (ThresholdOperator.EQ, 10.0, 10.0, True),   # 10 == 10 = True
            (ThresholdOperator.NEQ, 10.0, 5.0, True),   # 5 != 10 = True
        ]
        
        for operator, threshold, value, expected in test_cases:
            result = ControlDefinition._compare(value, threshold, operator)
            assert result == expected, f"Failed: {value} {operator.value} {threshold}"


class TestControlAuditTrail:
    """Tests for control audit trail features."""
    
    def test_query_hash_changes_with_sql(self):
        """Test that query hash changes when SQL changes."""
        control1 = ControlDefinition(
            control_code="AUDIT_001",
            control_name="Audit Test 1",
            category=ControlCategory.EXPOSURE,
            description="Test",
            computation_sql="SELECT MAX(exposure) AS calculated_value FROM positions",
            threshold_value=1.0,
            threshold_operator=ThresholdOperator.GT,
        )
        
        control2 = ControlDefinition(
            control_code="AUDIT_002",
            control_name="Audit Test 2",
            category=ControlCategory.EXPOSURE,
            description="Test",
            computation_sql="SELECT SUM(exposure) AS calculated_value FROM positions",
            threshold_value=1.0,
            threshold_operator=ThresholdOperator.GT,
        )
        
        # Different SQL should produce different hash
        assert control1.query_hash != control2.query_hash
    
    def test_control_includes_regulatory_reference(self):
        """Test controls can include regulatory references."""
        control = ControlDefinition(
            control_code="REG_001",
            control_name="Regulatory Control",
            category=ControlCategory.REGULATORY,
            description="With regulatory reference",
            computation_sql="SELECT 1 AS calculated_value",
            threshold_value=1.0,
            threshold_operator=ThresholdOperator.GT,
            regulatory_reference="Investment Policy Section 4.2.1",
            policy_document_id="POL-CONC-001",
        )
        
        assert control.regulatory_reference == "Investment Policy Section 4.2.1"
        assert control.policy_document_id == "POL-CONC-001"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
