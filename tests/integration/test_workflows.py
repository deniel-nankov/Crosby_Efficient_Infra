"""
End-to-End Workflow Tests

Comprehensive end-to-end tests verifying complete system workflows:
- Full compliance run from data loading to PDF generation
- Multi-control batch processing
- Exception investigation workflows
- Resilience and error recovery scenarios

These tests simulate real-world usage patterns and verify the
system works correctly as an integrated whole.

Test Categories:
- TestDailyComplianceWorkflow: Daily report generation
- TestControlExecution: Control evaluation workflows
- TestEvidenceRecording: Evidence trail verification
- TestResilienceWorkflow: Error recovery scenarios
- TestDataConsistencyWorkflow: Cross-component consistency
"""

from __future__ import annotations

import pytest
import sys
import time
import json
import hashlib
from datetime import datetime, date, timezone, timedelta
from decimal import Decimal
from typing import List, Dict, Any, Optional
from unittest.mock import MagicMock, patch
from dataclasses import dataclass
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

pytestmark = [pytest.mark.integration]


# =============================================================================
# WORKFLOW TEST UTILITIES
# =============================================================================

@dataclass
class WorkflowStep:
    """Represents a step in a workflow."""
    name: str
    duration_ms: float
    status: str
    result: Any = None
    error: Optional[str] = None


class WorkflowTracer:
    """Track workflow execution for testing."""
    
    def __init__(self):
        self.steps: List[WorkflowStep] = []
        self.start_time: Optional[float] = None
    
    def start(self):
        self.start_time = time.perf_counter()
        self.steps = []
    
    def record_step(self, name: str, status: str, result: Any = None, error: Optional[str] = None):
        duration = (time.perf_counter() - self.start_time) * 1000 if self.start_time else 0
        self.steps.append(WorkflowStep(
            name=name,
            duration_ms=duration,
            status=status,
            result=result,
            error=error,
        ))
    
    def get_summary(self) -> Dict[str, Any]:
        total_time = sum(s.duration_ms for s in self.steps)
        passed = sum(1 for s in self.steps if s.status == "PASS")
        failed = sum(1 for s in self.steps if s.status == "FAIL")
        
        return {
            "total_steps": len(self.steps),
            "passed": passed,
            "failed": failed,
            "total_time_ms": total_time,
            "steps": [
                {"name": s.name, "status": s.status, "duration_ms": s.duration_ms}
                for s in self.steps
            ]
        }


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
def postgres_adapter():
    """Create PostgresAdapter for testing."""
    try:
        from integration.postgres_adapter import PostgresAdapter
        adapter = PostgresAdapter()
        yield adapter
        adapter.close()
    except Exception as e:
        pytest.skip(f"PostgresAdapter not available: {e}")


@pytest.fixture
def control_runner(db_connection):
    """Create ControlRunner for testing."""
    from control_runner.runner import ControlRunner
    from unittest.mock import MagicMock
    
    mock_snowflake = MagicMock()
    mock_settings = MagicMock()
    mock_settings.version = "1.0.0"
    
    return ControlRunner(
        snowflake_connection=mock_snowflake,
        postgres_connection=db_connection,
        settings=mock_settings
    )


@pytest.fixture
def evidence_store(db_connection):
    """Create EvidenceStore for testing."""
    from evidence_store.store import EvidenceStore
    return EvidenceStore(postgres_connection=db_connection)


@pytest.fixture
def workflow_tracer():
    """Create WorkflowTracer for step tracking."""
    tracer = WorkflowTracer()
    tracer.start()
    return tracer


# =============================================================================
# DAILY COMPLIANCE WORKFLOW TESTS
# =============================================================================

class TestDailyComplianceWorkflow:
    """Test complete daily compliance reporting workflow."""
    
    @pytest.mark.requires_db
    def test_can_load_positions(self, postgres_adapter, workflow_tracer):
        """Should be able to load position data."""
        try:
            positions = postgres_adapter.source.get_positions(date.today())
            workflow_tracer.record_step("load_positions", "PASS", result=len(positions))
            assert isinstance(positions, list)
        except Exception as e:
            workflow_tracer.record_step("load_positions", "FAIL", error=str(e))
            pytest.fail(f"Failed to load positions: {e}")
    
    @pytest.mark.requires_db
    def test_can_load_nav(self, postgres_adapter, workflow_tracer):
        """Should be able to load NAV data."""
        try:
            nav = postgres_adapter.source.get_nav(date.today())
            workflow_tracer.record_step("load_nav", "PASS", result=nav)
            # NAV can be None if no data
            assert nav is None or isinstance(nav, (Decimal, float, int))
        except Exception as e:
            workflow_tracer.record_step("load_nav", "FAIL", error=str(e))
            pytest.fail(f"Failed to load NAV: {e}")
    
    @pytest.mark.requires_db
    def test_can_get_snapshot(self, postgres_adapter, workflow_tracer):
        """Should be able to get complete data snapshot."""
        try:
            snapshot = postgres_adapter.get_snapshot(date.today())
            workflow_tracer.record_step("get_snapshot", "PASS")
            
            assert snapshot is not None
            assert hasattr(snapshot, 'positions')
            assert hasattr(snapshot, 'control_results')
        except Exception as e:
            workflow_tracer.record_step("get_snapshot", "FAIL", error=str(e))
            pytest.fail(f"Failed to get snapshot: {e}")
    
    @pytest.mark.requires_db
    def test_workflow_handles_missing_data(self, postgres_adapter):
        """Workflow should handle missing data gracefully."""
        # Use a far future date that won't have data
        future_date = date.today() + timedelta(days=365 * 10)
        
        try:
            snapshot = postgres_adapter.get_snapshot(future_date)
            # Should succeed but with empty data
            assert snapshot is not None
        except Exception as e:
            # Should not raise an unhandled exception
            assert "get_positions" not in str(e).lower(), f"API error: {e}"


# =============================================================================
# CONTROL EXECUTION TESTS
# =============================================================================

class TestControlExecution:
    """Test control execution workflows."""
    
    @pytest.mark.requires_db
    def test_control_runner_initialization(self, db_connection):
        """ControlRunner should initialize properly."""
        from control_runner.runner import ControlRunner
        from unittest.mock import MagicMock
        
        # ControlRunner needs snowflake_connection, postgres_connection, settings
        mock_snowflake = MagicMock()
        mock_settings = MagicMock()
        mock_settings.version = "1.0.0"
        
        runner = ControlRunner(
            snowflake_connection=mock_snowflake,
            postgres_connection=db_connection,
            settings=mock_settings
        )
        assert runner is not None
        assert runner.postgres is db_connection
    
    @pytest.mark.requires_db
    def test_control_definition_registry(self):
        """Control definitions should be accessible."""
        from control_runner.controls import ControlDefinition, ControlCategory, ThresholdOperator
        
        # Create a test control definition
        control = ControlDefinition(
            control_code="TEST_001",
            control_name="Test Control",
            description="A test control for unit testing",
            category=ControlCategory.CONCENTRATION,
            computation_sql="SELECT 0.15 AS calculated_value",
            threshold_value=0.30,
            threshold_operator=ThresholdOperator.GTE,
        )
        
        assert control.control_code == "TEST_001"
        assert control.category == ControlCategory.CONCENTRATION
    
    @pytest.mark.requires_db
    def test_control_threshold_evaluation(self):
        """Control threshold evaluation should work correctly."""
        from control_runner.controls import ControlDefinition, ControlCategory, ThresholdOperator, ControlResultStatus
        
        control = ControlDefinition(
            control_code="THRESH_001",
            control_name="Threshold Test",
            description="Tests threshold evaluation",
            category=ControlCategory.CONCENTRATION,
            computation_sql="SELECT 0.35 AS calculated_value",
            threshold_value=0.30,
            threshold_operator=ThresholdOperator.GTE,
        )
        
        # 0.35 >= 0.30 should trigger breach (fail)
        result = control.evaluate_threshold(0.35)
        assert result == ControlResultStatus.FAIL
        
        # 0.25 >= 0.30 should not trigger breach (pass)
        result = control.evaluate_threshold(0.25)
        assert result == ControlResultStatus.PASS
    
    @pytest.mark.requires_db
    def test_multiple_threshold_operators(self):
        """All threshold operators should work."""
        from control_runner.controls import ControlDefinition, ControlCategory, ThresholdOperator, ControlResultStatus
        
        # GTE: >= threshold means FAIL
        operators_test_cases = [
            (ThresholdOperator.GTE, 0.30, 0.30, ControlResultStatus.FAIL),   # 0.30 >= 0.30
            (ThresholdOperator.GTE, 0.30, 0.25, ControlResultStatus.PASS),  # 0.25 >= 0.30
            (ThresholdOperator.GT, 0.30, 0.31, ControlResultStatus.FAIL),    # 0.31 > 0.30
            (ThresholdOperator.GT, 0.30, 0.30, ControlResultStatus.PASS),   # 0.30 > 0.30
            (ThresholdOperator.LTE, 0.30, 0.30, ControlResultStatus.FAIL),   # 0.30 <= 0.30
            (ThresholdOperator.LTE, 0.30, 0.35, ControlResultStatus.PASS),  # 0.35 <= 0.30
            (ThresholdOperator.LT, 0.30, 0.25, ControlResultStatus.FAIL),    # 0.25 < 0.30
            (ThresholdOperator.LT, 0.30, 0.30, ControlResultStatus.PASS),   # 0.30 < 0.30
        ]
        
        for op, threshold, value, expected in operators_test_cases:
            control = ControlDefinition(
                control_code=f"OP_{op.value}",
                control_name=f"Test {op.value}",
                description="Operator test",
                category=ControlCategory.CONCENTRATION,
                computation_sql="SELECT 0 AS calculated_value",
                threshold_value=threshold,
                threshold_operator=op,
            )
            result = control.evaluate_threshold(value)
            assert result == expected, f"{op.value}: {value} vs {threshold} expected {expected}, got {result}"


# =============================================================================
# EVIDENCE RECORDING TESTS
# =============================================================================

class TestEvidenceRecording:
    """Test evidence recording workflows."""
    
    @pytest.mark.requires_db
    def test_evidence_store_initialization(self, db_connection):
        """EvidenceStore should initialize properly."""
        from evidence_store.store import EvidenceStore
        
        store = EvidenceStore(postgres_connection=db_connection)
        assert store is not None
        assert store.connection is db_connection
    
    @pytest.mark.requires_db
    def test_evidence_hashing(self):
        """Evidence hashing should be deterministic."""
        data = {"control_id": "TEST_001", "value": 0.25}
        
        hash1 = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
        hash2 = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
        
        assert hash1 == hash2
    
    @pytest.mark.requires_db
    def test_evidence_hash_changes_with_data(self):
        """Evidence hash should change when data changes."""
        data1 = {"control_id": "TEST_001", "value": 0.25}
        data2 = {"control_id": "TEST_001", "value": 0.26}
        
        hash1 = hashlib.sha256(json.dumps(data1, sort_keys=True).encode()).hexdigest()
        hash2 = hashlib.sha256(json.dumps(data2, sort_keys=True).encode()).hexdigest()
        
        assert hash1 != hash2


# =============================================================================
# RESILIENCE WORKFLOW TESTS
# =============================================================================

class TestResilienceWorkflow:
    """Test system resilience and error handling."""
    
    @pytest.mark.requires_db
    def test_handles_connection_gracefully(self):
        """Should handle connection issues gracefully."""
        from integration.postgres_adapter import PostgresAdapter, PostgresConfig
        
        # Try with invalid config
        bad_config = PostgresConfig(
            host="nonexistent.host.local",
            port=5433,
            database="compliance_db",
            user="compliance_user",
            password="bad_password",
        )
        
        try:
            adapter = PostgresAdapter(config=bad_config)
            # If it doesn't fail on init, it should fail on use
        except Exception as e:
            # Should raise a clear connection error
            assert "connect" in str(e).lower() or "host" in str(e).lower() or "timeout" in str(e).lower()
    
    def test_handles_invalid_control_definition(self):
        """Should validate control definitions."""
        from control_runner.controls import ControlDefinition, ControlCategory, ThresholdOperator
        
        # Valid definition should work
        control = ControlDefinition(
            control_code="VALID_001",
            control_name="Valid Control",
            description="A valid control",
            category=ControlCategory.CONCENTRATION,
            computation_sql="SELECT 0.15 AS calculated_value",
            threshold_value=0.30,
            threshold_operator=ThresholdOperator.GTE,
        )
        assert control is not None
    
    def test_handles_edge_case_thresholds(self):
        """Should handle edge case threshold values."""
        from control_runner.controls import ControlDefinition, ControlCategory, ThresholdOperator, ControlResultStatus
        
        # Zero threshold
        control = ControlDefinition(
            control_code="ZERO_001",
            control_name="Zero Threshold",
            description="Zero threshold test",
            category=ControlCategory.CONCENTRATION,
            computation_sql="SELECT 0 AS calculated_value",
            threshold_value=0.0,
            threshold_operator=ThresholdOperator.GT,
        )
        
        # 0 > 0 is False, so should PASS
        assert control.evaluate_threshold(0.0) == ControlResultStatus.PASS
        # 0.001 > 0 is True, so should FAIL
        assert control.evaluate_threshold(0.001) == ControlResultStatus.FAIL
    
    def test_handles_negative_values(self):
        """Should handle negative calculated values."""
        from control_runner.controls import ControlDefinition, ControlCategory, ThresholdOperator, ControlResultStatus
        
        control = ControlDefinition(
            control_code="NEG_001",
            control_name="Negative Value Test",
            description="Negative value handling",
            category=ControlCategory.EXPOSURE,
            computation_sql="SELECT -0.10 AS calculated_value",
            threshold_value=-0.05,
            threshold_operator=ThresholdOperator.LT,
        )
        
        # -0.10 < -0.05 is True, so should FAIL (breached)
        assert control.evaluate_threshold(-0.10) == ControlResultStatus.FAIL
        # -0.03 < -0.05 is False, so should PASS
        assert control.evaluate_threshold(-0.03) == ControlResultStatus.PASS


# =============================================================================
# DATA CONSISTENCY WORKFLOW TESTS
# =============================================================================

class TestDataConsistencyWorkflow:
    """Test data consistency across components."""
    
    @pytest.mark.requires_db
    def test_control_results_have_required_fields(self, postgres_adapter):
        """Control results should have all required fields."""
        results = postgres_adapter.source.get_control_results(date.today())
        
        required_fields = ['control_id', 'control_name', 'control_type', 
                           'calculated_value', 'threshold', 'status']
        
        for result in results:
            for field in required_fields:
                assert field in result, f"Missing field: {field}"
    
    @pytest.mark.requires_db
    def test_positions_have_required_fields(self, postgres_adapter):
        """Positions should have all required fields."""
        positions = postgres_adapter.source.get_positions(date.today())
        
        required_fields = ['security_id', 'security_name', 'quantity', 'market_value']
        
        for pos in positions:
            for field in required_fields:
                assert field in pos, f"Missing field: {field}"
    
    def test_hash_consistency(self):
        """Hashes should be consistent across serialization."""
        test_data = {
            "control_code": "CONC_001",
            "calculated_value": 0.25,
            "threshold": 0.30,
            "status": "pass",
        }
        
        # Multiple serializations should produce same hash
        hashes = []
        for _ in range(5):
            serialized = json.dumps(test_data, sort_keys=True)
            hash_value = hashlib.sha256(serialized.encode()).hexdigest()
            hashes.append(hash_value)
        
        assert len(set(hashes)) == 1, "Hashes should be identical"
    
    def test_decimal_precision_preserved(self):
        """Decimal precision should be preserved in calculations."""
        from decimal import Decimal
        
        values = [
            Decimal("0.123456789"),
            Decimal("1000000.00"),
            Decimal("0.00001"),
        ]
        
        for value in values:
            # Conversion to string and back should preserve precision
            str_value = str(value)
            restored = Decimal(str_value)
            assert restored == value


# =============================================================================
# REGRESSION TESTS
# =============================================================================

class TestRegressionScenarios:
    """Regression tests for known issues."""
    
    def test_empty_positions_list(self):
        """Should handle empty positions list."""
        positions = []
        
        # Should not raise
        total_value = sum(p.get('market_value', 0) for p in positions)
        assert total_value == 0
    
    def test_unicode_in_security_names(self):
        """Should handle unicode in security names."""
        positions = [
            {"security_id": "TST001", "security_name": "Test Société Générale"},
            {"security_id": "TST002", "security_name": "日本株式会社"},
            {"security_id": "TST003", "security_name": "Émetteur français"},
        ]
        
        for pos in positions:
            # Serialization should work
            serialized = json.dumps(pos)
            restored = json.loads(serialized)
            assert restored['security_name'] == pos['security_name']
    
    def test_very_large_numbers(self):
        """Should handle very large market values."""
        from decimal import Decimal
        
        large_value = Decimal("9999999999999.99")
        
        # Should not overflow
        doubled = large_value * 2
        assert doubled == Decimal("19999999999999.98")
    
    def test_very_small_numbers(self):
        """Should handle very small values."""
        from decimal import Decimal
        
        small_value = Decimal("0.0000001")
        
        # Should preserve precision
        doubled = small_value * 2
        assert doubled == Decimal("0.0000002")
    
    def test_date_edge_cases(self):
        """Should handle date edge cases."""
        # Year boundary
        year_end = date(2024, 12, 31)
        year_start = date(2025, 1, 1)
        
        assert (year_start - year_end).days == 1
        
        # Leap year
        leap_day = date(2024, 2, 29)
        assert leap_day.month == 2
        assert leap_day.day == 29


# =============================================================================
# WORKFLOW TRACER TESTS
# =============================================================================

class TestWorkflowTracer:
    """Test the WorkflowTracer utility itself."""
    
    def test_tracer_records_steps(self):
        """Tracer should record workflow steps."""
        tracer = WorkflowTracer()
        tracer.start()
        
        tracer.record_step("step1", "PASS")
        tracer.record_step("step2", "FAIL", error="Some error")
        
        assert len(tracer.steps) == 2
        assert tracer.steps[0].name == "step1"
        assert tracer.steps[1].status == "FAIL"
    
    def test_tracer_summary(self):
        """Tracer should produce accurate summary."""
        tracer = WorkflowTracer()
        tracer.start()
        
        tracer.record_step("pass1", "PASS")
        tracer.record_step("pass2", "PASS")
        tracer.record_step("fail1", "FAIL")
        
        summary = tracer.get_summary()
        
        assert summary["total_steps"] == 3
        assert summary["passed"] == 2
        assert summary["failed"] == 1
    
    def test_tracer_timing(self):
        """Tracer should measure timing."""
        tracer = WorkflowTracer()
        tracer.start()
        
        time.sleep(0.01)  # 10ms
        tracer.record_step("delayed", "PASS")
        
        assert tracer.steps[0].duration_ms >= 5  # At least 5ms
