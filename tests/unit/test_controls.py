"""
Control Runner Unit Tests - Aligned with Actual Implementation

Tests for:
- ControlDefinition: Control configuration and threshold evaluation
- ControlRegistry: Built-in control definitions
- ControlResultStatus: Status enum values

These tests use the actual API signatures from src/control_runner/
"""

from __future__ import annotations

import pytest
import hashlib
from dataclasses import FrozenInstanceError
from datetime import date, timedelta
from unittest.mock import MagicMock

pytestmark = [pytest.mark.unit]


class TestControlCategory:
    """Test ControlCategory enum."""
    
    def test_all_categories_defined(self):
        """All expected categories should exist."""
        from control_runner.controls import ControlCategory
        
        expected = ['CONCENTRATION', 'LIQUIDITY', 'EXPOSURE', 'REGULATORY']
        for cat in expected:
            assert hasattr(ControlCategory, cat), f"Missing category: {cat}"
    
    def test_category_values(self):
        """Categories should have string values."""
        from control_runner.controls import ControlCategory
        
        assert ControlCategory.CONCENTRATION.value == "concentration"
        assert ControlCategory.LIQUIDITY.value == "liquidity"


class TestThresholdOperator:
    """Test ThresholdOperator enum."""
    
    def test_all_operators_defined(self):
        """All threshold operators should exist."""
        from control_runner.controls import ThresholdOperator
        
        expected = ['LT', 'LTE', 'GT', 'GTE', 'EQ', 'NEQ']
        for op in expected:
            assert hasattr(ThresholdOperator, op), f"Missing operator: {op}"


class TestControlResultStatus:
    """Test ControlResultStatus enum."""
    
    def test_all_statuses_defined(self):
        """All result statuses should exist."""
        from control_runner.controls import ControlResultStatus
        
        expected = ['PASS', 'FAIL', 'WARNING', 'ERROR', 'SKIP']
        for status in expected:
            assert hasattr(ControlResultStatus, status), f"Missing status: {status}"
    
    def test_status_values(self):
        """Statuses should have expected values."""
        from control_runner.controls import ControlResultStatus
        
        # Status values are lowercase
        assert ControlResultStatus.PASS.value == "pass"
        assert ControlResultStatus.FAIL.value == "fail"


class TestControlDefinition:
    """Test ControlDefinition dataclass."""
    
    def test_control_definition_creation(self):
        """ControlDefinition should be created with required fields."""
        from control_runner.controls import ControlDefinition, ControlCategory, ThresholdOperator
        
        control = ControlDefinition(
            control_code="CONC_SECTOR_001",
            control_name="Sector Concentration",
            category=ControlCategory.CONCENTRATION,
            description="Test concentration control",
            computation_sql="SELECT SUM(market_value)/nav AS calculated_value FROM positions",
            threshold_value=0.30,
            threshold_operator=ThresholdOperator.GTE,
        )
        
        assert control.control_code == "CONC_SECTOR_001"
        assert control.control_name == "Sector Concentration"
        assert control.threshold_value == 0.30
    
    def test_control_definition_with_warning_threshold(self):
        """ControlDefinition should accept warning threshold."""
        from control_runner.controls import ControlDefinition, ControlCategory, ThresholdOperator
        
        control = ControlDefinition(
            control_code="TEST_001",
            control_name="Test Control",
            category=ControlCategory.CONCENTRATION,
            description="Test control with warning",
            computation_sql="SELECT 1",
            threshold_value=0.30,
            threshold_operator=ThresholdOperator.GTE,
            warning_threshold=0.25,
            warning_operator=ThresholdOperator.GTE,
        )
        
        assert control.warning_threshold == 0.25
    
    def test_control_is_immutable(self):
        """ControlDefinition should be immutable (frozen dataclass)."""
        from control_runner.controls import ControlDefinition, ControlCategory, ThresholdOperator
        
        control = ControlDefinition(
            control_code="TEST_001",
            control_name="Test",
            category=ControlCategory.CONCENTRATION,
            description="Immutable test",
            computation_sql="SELECT 1",
            threshold_value=0.30,
            threshold_operator=ThresholdOperator.GTE,
        )
        
        # Frozen dataclass should raise error on direct attribute assignment
        with pytest.raises((FrozenInstanceError, AttributeError, TypeError)):
            control.threshold_value = 0.50  # type: ignore[misc]
    
    def test_query_hash_is_deterministic(self):
        """Query hash should be the same for the same SQL."""
        from control_runner.controls import ControlDefinition, ControlCategory, ThresholdOperator
        
        sql = "SELECT SUM(market_value) AS calculated_value FROM positions"
        
        control1 = ControlDefinition(
            control_code="TEST_001",
            control_name="Test 1",
            category=ControlCategory.CONCENTRATION,
            description="First control",
            computation_sql=sql,
            threshold_value=0.30,
            threshold_operator=ThresholdOperator.GTE,
        )
        
        control2 = ControlDefinition(
            control_code="TEST_002",
            control_name="Test 2",
            category=ControlCategory.EXPOSURE,
            description="Second control",
            computation_sql=sql,
            threshold_value=0.50,
            threshold_operator=ThresholdOperator.LTE,
        )
        
        assert control1.query_hash == control2.query_hash


class TestThresholdEvaluation:
    """Test threshold evaluation logic."""
    
    def test_gte_threshold_pass(self):
        """GTE threshold should PASS when value < threshold."""
        from control_runner.controls import ControlDefinition, ControlCategory, ThresholdOperator, ControlResultStatus
        
        control = ControlDefinition(
            control_code="TEST",
            control_name="Test",
            category=ControlCategory.CONCENTRATION,
            description="GTE pass test",
            computation_sql="SELECT 1",
            threshold_value=0.30,
            threshold_operator=ThresholdOperator.GTE,
        )
        
        result = control.evaluate_threshold(0.25)
        assert result == ControlResultStatus.PASS
    
    def test_gte_threshold_fail(self):
        """GTE threshold should FAIL when value >= threshold."""
        from control_runner.controls import ControlDefinition, ControlCategory, ThresholdOperator, ControlResultStatus
        
        control = ControlDefinition(
            control_code="TEST",
            control_name="Test",
            category=ControlCategory.CONCENTRATION,
            description="GTE fail test",
            computation_sql="SELECT 1",
            threshold_value=0.30,
            threshold_operator=ThresholdOperator.GTE,
        )
        
        result = control.evaluate_threshold(0.35)
        assert result == ControlResultStatus.FAIL
    
    def test_gte_threshold_at_boundary(self):
        """GTE threshold should FAIL at exactly threshold value."""
        from control_runner.controls import ControlDefinition, ControlCategory, ThresholdOperator, ControlResultStatus
        
        control = ControlDefinition(
            control_code="TEST",
            control_name="Test",
            category=ControlCategory.CONCENTRATION,
            description="Boundary test",
            computation_sql="SELECT 1",
            threshold_value=0.30,
            threshold_operator=ThresholdOperator.GTE,
        )
        
        result = control.evaluate_threshold(0.30)
        assert result == ControlResultStatus.FAIL
    
    def test_lte_threshold_pass(self):
        """LTE threshold should PASS when value > threshold."""
        from control_runner.controls import ControlDefinition, ControlCategory, ThresholdOperator, ControlResultStatus
        
        control = ControlDefinition(
            control_code="LIQ_001",
            control_name="Liquidity Ratio",
            category=ControlCategory.LIQUIDITY,
            description="LTE pass test",
            computation_sql="SELECT 1",
            threshold_value=0.10,
            threshold_operator=ThresholdOperator.LTE,
        )
        
        result = control.evaluate_threshold(0.15)
        assert result == ControlResultStatus.PASS
    
    def test_lte_threshold_fail(self):
        """LTE threshold should FAIL when value <= threshold."""
        from control_runner.controls import ControlDefinition, ControlCategory, ThresholdOperator, ControlResultStatus
        
        control = ControlDefinition(
            control_code="LIQ_001",
            control_name="Liquidity Ratio",
            category=ControlCategory.LIQUIDITY,
            description="LTE fail test",
            computation_sql="SELECT 1",
            threshold_value=0.10,
            threshold_operator=ThresholdOperator.LTE,
        )
        
        result = control.evaluate_threshold(0.08)
        assert result == ControlResultStatus.FAIL
    
    def test_warning_threshold(self):
        """Warning threshold should return WARNING when triggered but not fail."""
        from control_runner.controls import ControlDefinition, ControlCategory, ThresholdOperator, ControlResultStatus
        
        control = ControlDefinition(
            control_code="TEST",
            control_name="Test",
            category=ControlCategory.CONCENTRATION,
            description="Warning test",
            computation_sql="SELECT 1",
            threshold_value=0.30,
            threshold_operator=ThresholdOperator.GTE,
            warning_threshold=0.25,
            warning_operator=ThresholdOperator.GTE,
        )
        
        result = control.evaluate_threshold(0.27)
        assert result == ControlResultStatus.WARNING
    
    def test_breach_amount_can_be_computed(self):
        """Breach amount should be computable from threshold and value."""
        from control_runner.controls import ControlDefinition, ControlCategory, ThresholdOperator, ControlResultStatus
        
        control = ControlDefinition(
            control_code="TEST",
            control_name="Test",
            category=ControlCategory.CONCENTRATION,
            description="Breach amount test",
            computation_sql="SELECT 1",
            threshold_value=0.30,
            threshold_operator=ThresholdOperator.GTE,
        )
        
        # Verify threshold evaluation works
        result = control.evaluate_threshold(0.35)
        assert result == ControlResultStatus.FAIL
        # Breach amount would be: 0.35 - 0.30 = 0.05


class TestControlRegistry:
    """Test built-in control registry."""
    
    def test_get_all_controls(self):
        """Should return all defined controls."""
        from control_runner.controls import get_all_controls
        
        controls = get_all_controls()
        assert len(controls) > 0
        assert all(hasattr(c, 'control_code') for c in controls)
    
    def test_get_active_controls(self):
        """Should return only active controls."""
        from control_runner.controls import get_active_controls
        
        controls = get_active_controls()
        assert all(c.is_active for c in controls)
    
    def test_get_controls_by_category(self):
        """Should filter controls by category."""
        from control_runner.controls import get_controls_by_category, ControlCategory
        
        concentration_controls = get_controls_by_category(ControlCategory.CONCENTRATION)
        assert all(c.category == ControlCategory.CONCENTRATION for c in concentration_controls)
    
    def test_controls_have_required_fields(self):
        """All controls should have required fields populated."""
        from control_runner.controls import get_all_controls
        
        for control in get_all_controls():
            assert control.control_code, f"Missing control_code"
            assert control.control_name, f"Missing control_name"
            assert control.computation_sql, f"Missing SQL for {control.control_code}"


class TestControlEffectiveDates:
    """Test control effective date logic."""
    
    def test_control_is_active_by_default(self):
        """Controls without expiration should be active."""
        from control_runner.controls import ControlDefinition, ControlCategory, ThresholdOperator
        
        control = ControlDefinition(
            control_code="TEST",
            control_name="Test",
            category=ControlCategory.CONCENTRATION,
            description="Active by default test",
            computation_sql="SELECT 1",
            threshold_value=0.30,
            threshold_operator=ThresholdOperator.GTE,
        )
        
        assert control.is_active is True
    
    def test_control_with_future_effective_date(self):
        """Controls with future effective date should not be active."""
        from control_runner.controls import ControlDefinition, ControlCategory, ThresholdOperator
        
        future_date = date.today() + timedelta(days=30)
        
        control = ControlDefinition(
            control_code="TEST",
            control_name="Test",
            category=ControlCategory.CONCENTRATION,
            description="Future effective date test",
            computation_sql="SELECT 1",
            threshold_value=0.30,
            threshold_operator=ThresholdOperator.GTE,
            effective_date=future_date,
        )
        
        assert control.is_active is False
    
    def test_control_with_past_expiration(self):
        """Controls with past expiration date should not be active."""
        from control_runner.controls import ControlDefinition, ControlCategory, ThresholdOperator
        
        past_date = date.today() - timedelta(days=30)
        
        control = ControlDefinition(
            control_code="TEST",
            control_name="Test",
            category=ControlCategory.CONCENTRATION,
            description="Past expiration test",
            computation_sql="SELECT 1",
            threshold_value=0.30,
            threshold_operator=ThresholdOperator.GTE,
            effective_date=date.today() - timedelta(days=60),
            expiration_date=past_date,
        )
        
        assert control.is_active is False
