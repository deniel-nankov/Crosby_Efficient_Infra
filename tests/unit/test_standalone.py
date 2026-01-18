"""
Standalone tests for core system functionality.

These tests don't require the full source modules and verify
fundamental concepts like hashing, data structures, and determinism.
"""

import pytest
from datetime import date, datetime, timezone, timedelta
from decimal import Decimal
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List
import hashlib
import json


# ============================================================================
# LOCAL TEST MODELS (mimic production structures)
# ============================================================================

class ControlCategory(Enum):
    """Categories of compliance controls."""
    CONCENTRATION = "concentration"
    LIQUIDITY = "liquidity"
    COUNTERPARTY = "counterparty"
    EXPOSURE = "exposure"


class ThresholdOperator(Enum):
    """Comparison operators."""
    LT = "lt"
    LTE = "lte"
    GT = "gt"
    GTE = "gte"
    EQ = "eq"


class ControlResultStatus(Enum):
    """Control result statuses."""
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    SKIP = "skip"
    ERROR = "error"


@dataclass(frozen=True)
class ControlDefinition:
    """Test control definition."""
    control_code: str
    control_name: str
    category: ControlCategory
    threshold_value: float
    threshold_operator: ThresholdOperator
    warning_threshold: Optional[float] = None
    effective_date: date = field(default_factory=date.today)
    expiration_date: Optional[date] = None
    
    @property
    def is_active(self) -> bool:
        today = date.today()
        if self.expiration_date and today > self.expiration_date:
            return False
        return today >= self.effective_date
    
    def evaluate(self, value: float) -> ControlResultStatus:
        """Evaluate value against threshold."""
        # Check warning first
        if self.warning_threshold is not None:
            if self._compare(value, self.warning_threshold):
                if self._compare(value, self.threshold_value):
                    return ControlResultStatus.FAIL
                return ControlResultStatus.WARNING
        
        if self._compare(value, self.threshold_value):
            return ControlResultStatus.FAIL
        return ControlResultStatus.PASS
    
    def _compare(self, value: float, threshold: float) -> bool:
        """Compare value to threshold based on operator."""
        ops = {
            ThresholdOperator.LT: lambda v, t: v < t,
            ThresholdOperator.LTE: lambda v, t: v <= t,
            ThresholdOperator.GT: lambda v, t: v > t,
            ThresholdOperator.GTE: lambda v, t: v >= t,
            ThresholdOperator.EQ: lambda v, t: v == t,
        }
        return ops[self.threshold_operator](value, threshold)


@dataclass
class ControlResult:
    """Test control result."""
    result_id: str
    control_code: str
    calculated_value: float
    threshold_value: float
    status: ControlResultStatus
    run_date: date
    evidence_hash: str


@dataclass
class EvidencePackage:
    """Test evidence package."""
    package_id: str
    run_date: date
    results: List[ControlResult]
    
    @property
    def package_hash(self) -> str:
        """Generate deterministic package hash."""
        data = {
            "package_id": self.package_id,
            "run_date": self.run_date.isoformat(),
            "result_hashes": [r.evidence_hash for r in self.results],
        }
        return hashlib.sha256(
            json.dumps(data, sort_keys=True).encode()
        ).hexdigest()


# ============================================================================
# TEST CLASSES
# ============================================================================

class TestControlCategories:
    """Test control category enumeration."""
    
    def test_all_categories_exist(self):
        """Verify all required categories are defined."""
        assert ControlCategory.CONCENTRATION.value == "concentration"
        assert ControlCategory.LIQUIDITY.value == "liquidity"
        assert ControlCategory.COUNTERPARTY.value == "counterparty"
        assert ControlCategory.EXPOSURE.value == "exposure"
    
    def test_category_iteration(self):
        """Test we can iterate over categories."""
        categories = list(ControlCategory)
        assert len(categories) == 4


class TestThresholdOperators:
    """Test threshold comparison operators."""
    
    def test_operator_values(self):
        """Verify operator enum values."""
        assert ThresholdOperator.LT.value == "lt"
        assert ThresholdOperator.LTE.value == "lte"
        assert ThresholdOperator.GT.value == "gt"
        assert ThresholdOperator.GTE.value == "gte"
        assert ThresholdOperator.EQ.value == "eq"
    
    def test_operator_count(self):
        """Test operator enumeration count."""
        operators = list(ThresholdOperator)
        assert len(operators) == 5


class TestControlResultStatuses:
    """Test control result status enumeration."""
    
    def test_all_statuses_exist(self):
        """Verify all status values are defined."""
        assert ControlResultStatus.PASS.value == "pass"
        assert ControlResultStatus.FAIL.value == "fail"
        assert ControlResultStatus.WARNING.value == "warning"
        assert ControlResultStatus.SKIP.value == "skip"
        assert ControlResultStatus.ERROR.value == "error"


class TestControlDefinition:
    """Test control definition functionality."""
    
    @pytest.fixture
    def sample_control(self):
        """Create a sample control for testing."""
        return ControlDefinition(
            control_code="TEST_001",
            control_name="Test Concentration Limit",
            category=ControlCategory.CONCENTRATION,
            threshold_value=10.0,
            threshold_operator=ThresholdOperator.GT,
            warning_threshold=8.0,
        )
    
    def test_control_creation(self, sample_control):
        """Test control can be created."""
        assert sample_control.control_code == "TEST_001"
        assert sample_control.threshold_value == 10.0
    
    def test_control_is_frozen(self, sample_control):
        """Test control is immutable."""
        with pytest.raises(Exception):
            sample_control.control_code = "MODIFIED"
    
    def test_control_is_active_by_default(self, sample_control):
        """Test control is active by default."""
        assert sample_control.is_active is True
    
    def test_control_with_future_effective_date(self):
        """Test control with future effective date is inactive."""
        future = date.today() + timedelta(days=30)
        control = ControlDefinition(
            control_code="FUTURE_001",
            control_name="Future Control",
            category=ControlCategory.EXPOSURE,
            threshold_value=100.0,
            threshold_operator=ThresholdOperator.GT,
            effective_date=future,
        )
        assert control.is_active is False
    
    def test_control_with_past_expiration(self):
        """Test expired control is inactive."""
        past = date.today() - timedelta(days=30)
        control = ControlDefinition(
            control_code="EXPIRED_001",
            control_name="Expired Control",
            category=ControlCategory.EXPOSURE,
            threshold_value=100.0,
            threshold_operator=ThresholdOperator.GT,
            expiration_date=past,
        )
        assert control.is_active is False


class TestThresholdEvaluation:
    """Test threshold evaluation logic."""
    
    @pytest.fixture
    def gt_control(self):
        """Control with greater-than threshold."""
        return ControlDefinition(
            control_code="GT_001",
            control_name="GT Control",
            category=ControlCategory.CONCENTRATION,
            threshold_value=10.0,
            threshold_operator=ThresholdOperator.GT,
            warning_threshold=8.0,
        )
    
    @pytest.fixture
    def lt_control(self):
        """Control with less-than threshold."""
        return ControlDefinition(
            control_code="LT_001",
            control_name="LT Control",
            category=ControlCategory.LIQUIDITY,
            threshold_value=0.1,
            threshold_operator=ThresholdOperator.LT,
        )
    
    def test_gt_pass(self, gt_control):
        """Value below warning threshold passes."""
        result = gt_control.evaluate(5.0)
        assert result == ControlResultStatus.PASS
    
    def test_gt_warning(self, gt_control):
        """Value above warning but below fail triggers warning."""
        result = gt_control.evaluate(9.0)
        assert result == ControlResultStatus.WARNING
    
    def test_gt_fail(self, gt_control):
        """Value above fail threshold fails."""
        result = gt_control.evaluate(15.0)
        assert result == ControlResultStatus.FAIL
    
    def test_gt_at_warning_boundary(self, gt_control):
        """Value exactly at warning boundary."""
        result = gt_control.evaluate(8.0)
        # 8.0 is not > 8.0, so should pass
        assert result == ControlResultStatus.PASS
    
    def test_gt_at_fail_boundary(self, gt_control):
        """Value exactly at fail boundary."""
        result = gt_control.evaluate(10.0)
        # 10.0 is not > 10.0, but is > 8.0 (warning)
        assert result == ControlResultStatus.WARNING
    
    def test_lt_pass(self, lt_control):
        """Value above threshold passes (LT)."""
        result = lt_control.evaluate(0.15)
        assert result == ControlResultStatus.PASS
    
    def test_lt_fail(self, lt_control):
        """Value below threshold fails (LT)."""
        result = lt_control.evaluate(0.05)
        assert result == ControlResultStatus.FAIL


class TestDeterministicHashing:
    """Test that all hashing is deterministic."""
    
    def test_sha256_determinism(self):
        """SHA-256 produces same output for same input."""
        data = "test data for hashing"
        
        hash1 = hashlib.sha256(data.encode()).hexdigest()
        hash2 = hashlib.sha256(data.encode()).hexdigest()
        
        assert hash1 == hash2
    
    def test_json_hash_determinism(self):
        """JSON hashing is deterministic with sorted keys."""
        data = {
            "z_field": 3,
            "a_field": 1,
            "m_field": 2,
        }
        
        hash1 = hashlib.sha256(
            json.dumps(data, sort_keys=True).encode()
        ).hexdigest()
        
        hash2 = hashlib.sha256(
            json.dumps(data, sort_keys=True).encode()
        ).hexdigest()
        
        assert hash1 == hash2
    
    def test_hash_changes_with_data(self):
        """Hash changes when data changes."""
        data1 = {"value": 1}
        data2 = {"value": 2}
        
        hash1 = hashlib.sha256(
            json.dumps(data1, sort_keys=True).encode()
        ).hexdigest()
        hash2 = hashlib.sha256(
            json.dumps(data2, sort_keys=True).encode()
        ).hexdigest()
        
        assert hash1 != hash2
    
    def test_hash_format(self):
        """SHA-256 produces 64 hex characters."""
        hash_value = hashlib.sha256(b"test").hexdigest()
        
        assert len(hash_value) == 64
        assert all(c in '0123456789abcdef' for c in hash_value)


class TestControlResult:
    """Test control result data structure."""
    
    def test_create_result(self):
        """Test creating a control result."""
        result = ControlResult(
            result_id="RES-001",
            control_code="CONC_001",
            calculated_value=8.5,
            threshold_value=10.0,
            status=ControlResultStatus.PASS,
            run_date=date(2024, 1, 15),
            evidence_hash="abc123",
        )
        
        assert result.result_id == "RES-001"
        assert result.status == ControlResultStatus.PASS
    
    def test_result_evidence_hash(self):
        """Test result can store evidence hash."""
        evidence_data = {
            "control_code": "CONC_001",
            "value": 8.5,
            "date": "2024-01-15",
        }
        evidence_hash = hashlib.sha256(
            json.dumps(evidence_data, sort_keys=True).encode()
        ).hexdigest()
        
        result = ControlResult(
            result_id="RES-001",
            control_code="CONC_001",
            calculated_value=8.5,
            threshold_value=10.0,
            status=ControlResultStatus.PASS,
            run_date=date(2024, 1, 15),
            evidence_hash=evidence_hash,
        )
        
        assert len(result.evidence_hash) == 64


class TestEvidencePackage:
    """Test evidence packaging functionality."""
    
    def test_create_package(self):
        """Test creating an evidence package."""
        results = [
            ControlResult(
                result_id="RES-001",
                control_code="CONC_001",
                calculated_value=8.5,
                threshold_value=10.0,
                status=ControlResultStatus.PASS,
                run_date=date(2024, 1, 15),
                evidence_hash="hash1",
            ),
            ControlResult(
                result_id="RES-002",
                control_code="CONC_002",
                calculated_value=20.0,
                threshold_value=25.0,
                status=ControlResultStatus.PASS,
                run_date=date(2024, 1, 15),
                evidence_hash="hash2",
            ),
        ]
        
        package = EvidencePackage(
            package_id="PKG-001",
            run_date=date(2024, 1, 15),
            results=results,
        )
        
        assert package.package_id == "PKG-001"
        assert len(package.results) == 2
    
    def test_package_hash_is_deterministic(self):
        """Test package hash is deterministic."""
        results = [
            ControlResult(
                result_id="RES-001",
                control_code="CONC_001",
                calculated_value=8.5,
                threshold_value=10.0,
                status=ControlResultStatus.PASS,
                run_date=date(2024, 1, 15),
                evidence_hash="hash1",
            ),
        ]
        
        package1 = EvidencePackage(
            package_id="PKG-001",
            run_date=date(2024, 1, 15),
            results=results,
        )
        
        package2 = EvidencePackage(
            package_id="PKG-001",
            run_date=date(2024, 1, 15),
            results=results,
        )
        
        assert package1.package_hash == package2.package_hash


class TestAuditTrailIntegrity:
    """Test audit trail functionality."""
    
    def test_chain_of_evidence(self):
        """Test building a chain of evidence."""
        chain = []
        prev_hash = "genesis"
        
        for i in range(5):
            record = {
                "id": i,
                "data": f"record_{i}",
                "prev_hash": prev_hash,
            }
            current_hash = hashlib.sha256(
                json.dumps(record, sort_keys=True).encode()
            ).hexdigest()
            record["hash"] = current_hash
            chain.append(record)
            prev_hash = current_hash
        
        # Verify chain integrity
        for i, record in enumerate(chain):
            if i == 0:
                assert record["prev_hash"] == "genesis"
            else:
                assert record["prev_hash"] == chain[i-1]["hash"]
    
    def test_tamper_detection(self):
        """Test that tampering is detectable."""
        original = {"value": 100, "status": "pass"}
        original_hash = hashlib.sha256(
            json.dumps(original, sort_keys=True).encode()
        ).hexdigest()
        
        # Tamper
        tampered = {"value": 50, "status": "pass"}
        tampered_hash = hashlib.sha256(
            json.dumps(tampered, sort_keys=True).encode()
        ).hexdigest()
        
        assert original_hash != tampered_hash


class TestDeterministicEvaluation:
    """Test that evaluations are always deterministic."""
    
    def test_same_input_same_output(self):
        """Same inputs always produce same outputs."""
        control = ControlDefinition(
            control_code="DET_001",
            control_name="Deterministic Test",
            category=ControlCategory.CONCENTRATION,
            threshold_value=10.0,
            threshold_operator=ThresholdOperator.GT,
            warning_threshold=8.0,
        )
        
        # Evaluate 100 times
        results = [control.evaluate(9.5) for _ in range(100)]
        
        # All should be WARNING
        assert all(r == ControlResultStatus.WARNING for r in results)
    
    def test_boundary_conditions_deterministic(self):
        """Boundary conditions are deterministic."""
        control = ControlDefinition(
            control_code="BOUND_001",
            control_name="Boundary Test",
            category=ControlCategory.CONCENTRATION,
            threshold_value=10.0,
            threshold_operator=ThresholdOperator.GT,
        )
        
        # Exactly at boundary
        results = [control.evaluate(10.0) for _ in range(100)]
        assert all(r == ControlResultStatus.PASS for r in results)
        
        # Just above boundary
        results = [control.evaluate(10.0001) for _ in range(100)]
        assert all(r == ControlResultStatus.FAIL for r in results)


class TestComparisonOperators:
    """Test all comparison operators work correctly."""
    
    @pytest.mark.parametrize("operator,threshold,value,expected_breach", [
        (ThresholdOperator.GT, 10.0, 15.0, True),   # 15 > 10
        (ThresholdOperator.GT, 10.0, 5.0, False),   # 5 > 10
        (ThresholdOperator.GT, 10.0, 10.0, False),  # 10 > 10
        (ThresholdOperator.GTE, 10.0, 10.0, True),  # 10 >= 10
        (ThresholdOperator.GTE, 10.0, 9.0, False),  # 9 >= 10
        (ThresholdOperator.LT, 10.0, 5.0, True),    # 5 < 10
        (ThresholdOperator.LT, 10.0, 15.0, False),  # 15 < 10
        (ThresholdOperator.LT, 10.0, 10.0, False),  # 10 < 10
        (ThresholdOperator.LTE, 10.0, 10.0, True),  # 10 <= 10
        (ThresholdOperator.LTE, 10.0, 11.0, False), # 11 <= 10
        (ThresholdOperator.EQ, 10.0, 10.0, True),   # 10 == 10
        (ThresholdOperator.EQ, 10.0, 10.1, False),  # 10.1 == 10
    ])
    def test_operator_comparison(self, operator, threshold, value, expected_breach):
        """Test each operator comparison."""
        control = ControlDefinition(
            control_code="OP_TEST",
            control_name="Operator Test",
            category=ControlCategory.EXPOSURE,
            threshold_value=threshold,
            threshold_operator=operator,
        )
        
        result = control.evaluate(value)
        
        if expected_breach:
            assert result == ControlResultStatus.FAIL
        else:
            assert result == ControlResultStatus.PASS


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
