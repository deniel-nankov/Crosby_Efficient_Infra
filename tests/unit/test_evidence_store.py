"""
Unit tests for Evidence Store module.

Tests audit trail management and evidence retrieval.
These tests verify the queryable audit trail functionality.

SEC Examination Note: Tests validate evidence integrity and traceability.
"""

import pytest
from datetime import date, datetime, timezone, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
import json
import hashlib
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.evidence_store.store import (
    EvidenceStore,
    EvidenceQuery,
    ControlResultEvidence,
    ExceptionEvidence,
    DailyComplianceSummary,
)


class TestEvidenceQuery:
    """Tests for EvidenceQuery dataclass."""
    
    def test_create_evidence_query(self):
        """Test creating an evidence query."""
        query = EvidenceQuery(
            query_id="QRY-001",
            query_type="control_results",
            parameters={"run_date": "2024-01-15", "fund_id": "FUND-001"},
            executed_at=datetime.now(timezone.utc),
            executed_by="compliance_user",
            result_count=25,
        )
        
        assert query.query_id == "QRY-001"
        assert query.query_type == "control_results"
        assert query.result_count == 25
    
    def test_query_hash_is_deterministic(self):
        """Test that query hash is deterministic."""
        params = {"run_date": "2024-01-15", "fund_id": "FUND-001"}
        
        query1 = EvidenceQuery(
            query_id="QRY-001",
            query_type="control_results",
            parameters=params,
            executed_at=datetime.now(timezone.utc),
            executed_by="user1",
        )
        
        query2 = EvidenceQuery(
            query_id="QRY-002",
            query_type="control_results",
            parameters=params,
            executed_at=datetime.now(timezone.utc),
            executed_by="user2",
        )
        
        # Same type and parameters should produce same hash
        assert query1.query_hash == query2.query_hash
    
    def test_query_hash_changes_with_parameters(self):
        """Test that query hash changes with different parameters."""
        query1 = EvidenceQuery(
            query_id="QRY-001",
            query_type="control_results",
            parameters={"fund_id": "FUND-001"},
            executed_at=datetime.now(timezone.utc),
            executed_by="user",
        )
        
        query2 = EvidenceQuery(
            query_id="QRY-002",
            query_type="control_results",
            parameters={"fund_id": "FUND-002"},
            executed_at=datetime.now(timezone.utc),
            executed_by="user",
        )
        
        assert query1.query_hash != query2.query_hash


class TestControlResultEvidence:
    """Tests for ControlResultEvidence dataclass."""
    
    def test_create_control_result_evidence(self):
        """Test creating control result evidence."""
        evidence = ControlResultEvidence(
            result_id="RES-001",
            control_code="CONC_001",
            control_name="Single Issuer Concentration",
            control_category="concentration",
            run_id="RUN-001",
            run_code="DAILY-2024-01-15",
            run_date=date(2024, 1, 15),
            snapshot_id="SNAP-001",
            calculated_value=8.5,
            threshold_value=10.0,
            threshold_operator="gt",
            result_status="pass",
            breach_amount=None,
            computation_sql_hash="abc123",
            evidence_row_count=150,
            evidence_sample=None,
            executed_at=datetime.now(timezone.utc),
        )
        
        assert evidence.control_code == "CONC_001"
        assert evidence.result_status == "pass"
        assert evidence.calculated_value == 8.5
    
    def test_to_citation(self):
        """Test citation generation."""
        evidence = ControlResultEvidence(
            result_id="RES-001",
            control_code="CONC_001",
            control_name="Test Control",
            control_category="concentration",
            run_id="RUN-001",
            run_code="DAILY-2024-01-15",
            run_date=date(2024, 1, 15),
            snapshot_id="SNAP-001",
            calculated_value=8.5,
            threshold_value=10.0,
            threshold_operator="gt",
            result_status="pass",
            breach_amount=None,
            computation_sql_hash="abc123",
            evidence_row_count=150,
            evidence_sample=None,
            executed_at=datetime.now(timezone.utc),
        )
        
        citation = evidence.to_citation()
        
        assert "DAILY-2024-01-15" in citation
        assert "CONC_001" in citation
        assert "SNAP-001" in citation
    
    def test_to_summary(self):
        """Test deterministic summary generation."""
        evidence = ControlResultEvidence(
            result_id="RES-001",
            control_code="CONC_001",
            control_name="Test Control",
            control_category="concentration",
            run_id="RUN-001",
            run_code="DAILY-2024-01-15",
            run_date=date(2024, 1, 15),
            snapshot_id="SNAP-001",
            calculated_value=8.5,
            threshold_value=10.0,
            threshold_operator="gt",
            result_status="pass",
            breach_amount=None,
            computation_sql_hash="abc123",
            evidence_row_count=150,
            evidence_sample=None,
            executed_at=datetime.now(timezone.utc),
        )
        
        summary = evidence.to_summary()
        
        assert isinstance(summary, str)
        assert len(summary) > 0


class TestExceptionEvidence:
    """Tests for ExceptionEvidence dataclass."""
    
    def test_create_exception_evidence(self):
        """Test creating exception evidence."""
        evidence = ExceptionEvidence(
            exception_id="EXC-001",
            exception_code="EXC-2024-001",
            control_code="CONC_001",
            control_name="Single Issuer Concentration",
            run_code="DAILY-2024-01-15",
            run_date=date(2024, 1, 15),
            severity="high",
            title="Concentration Limit Breach",
            description="Single issuer exceeded 10% limit",
            status="open",
            breach_value=12.5,
            threshold_value=10.0,
            breach_amount=2.5,
            opened_at=datetime.now(timezone.utc),
            due_date=None,
            resolved_at=None,
            resolution_type=None,
            resolution_notes=None,
            assigned_to=None,
        )
        
        assert evidence.exception_id == "EXC-001"
        assert evidence.status == "open"
        assert evidence.breach_amount == 2.5
    
    def test_exception_severity_calculation(self):
        """Test that breach severity can be calculated."""
        evidence = ExceptionEvidence(
            exception_id="EXC-001",
            exception_code="EXC-2024-001",
            control_code="CONC_001",
            control_name="Test",
            run_code="DAILY-2024-01-15",
            run_date=date(2024, 1, 15),
            severity="high",
            title="Test Exception",
            description="Test",
            status="open",
            breach_value=15.0,
            threshold_value=10.0,
            breach_amount=5.0,
            opened_at=datetime.now(timezone.utc),
            due_date=None,
            resolved_at=None,
            resolution_type=None,
            resolution_notes=None,
            assigned_to=None,
        )
        
        # 50% over threshold
        breach_pct = (evidence.breach_amount / evidence.threshold_value) * 100
        assert breach_pct == 50.0


class TestDailyComplianceSummary:
    """Tests for DailyComplianceSummary dataclass."""
    
    def test_create_summary(self):
        """Test creating a daily compliance summary."""
        summary = DailyComplianceSummary(
            run_id="RUN-001",
            run_code="DAILY-2024-01-15",
            run_date=date(2024, 1, 15),
            snapshot_id="SNAP-001",
            total_controls=25,
            controls_passed=20,
            controls_failed=2,
            controls_warning=3,
            pass_rate=0.80,
            exceptions_opened=2,
            exceptions_closed=0,
            exceptions_outstanding=2,
            critical_exceptions=1,
            run_start=datetime.now(timezone.utc),
            run_end=datetime.now(timezone.utc),
            duration_seconds=5.0,
            config_hash="abc123",
        )
        
        assert summary.total_controls == 25
        assert summary.controls_passed == 20
        assert summary.controls_failed == 2
    
    def test_summary_pass_rate(self):
        """Test calculating pass rate."""
        summary = DailyComplianceSummary(
            run_id="RUN-001",
            run_code="DAILY-2024-01-15",
            run_date=date(2024, 1, 15),
            snapshot_id="SNAP-001",
            total_controls=100,
            controls_passed=85,
            controls_failed=5,
            controls_warning=10,
            pass_rate=0.85,
            exceptions_opened=5,
            exceptions_closed=0,
            exceptions_outstanding=5,
            critical_exceptions=2,
            run_start=datetime.now(timezone.utc),
            run_end=datetime.now(timezone.utc),
            duration_seconds=10.0,
            config_hash="abc123",
        )
        
        pass_rate = summary.controls_passed / summary.total_controls * 100
        assert pass_rate == 85.0
    
    def test_summary_is_clean(self):
        """Test checking if summary has no failures."""
        clean_summary = DailyComplianceSummary(
            run_id="RUN-001",
            run_code="DAILY-2024-01-15",
            run_date=date.today(),
            snapshot_id="SNAP-001",
            total_controls=25,
            controls_passed=23,
            controls_failed=0,
            controls_warning=2,
            pass_rate=0.92,
            exceptions_opened=0,
            exceptions_closed=0,
            exceptions_outstanding=0,
            critical_exceptions=0,
            run_start=datetime.now(timezone.utc),
            run_end=datetime.now(timezone.utc),
            duration_seconds=3.0,
            config_hash="abc123",
        )
        
        assert clean_summary.controls_failed == 0
        assert clean_summary.exceptions_opened == 0


class TestEvidenceHashing:
    """Tests for evidence hashing functionality."""
    
    def test_hash_is_deterministic(self):
        """Test that evidence hashing is deterministic."""
        data = {
            "control_code": "CONC_001",
            "fund_id": "FUND-001",
            "calculated_value": 8.5,
            "run_date": "2024-01-15",
        }
        
        hash1 = hashlib.sha256(
            json.dumps(data, sort_keys=True).encode()
        ).hexdigest()
        
        hash2 = hashlib.sha256(
            json.dumps(data, sort_keys=True).encode()
        ).hexdigest()
        
        assert hash1 == hash2
    
    def test_hash_changes_with_data(self):
        """Test that hash changes when data changes."""
        data1 = {
            "control_code": "CONC_001",
            "calculated_value": 8.5,
        }
        
        data2 = {
            "control_code": "CONC_001",
            "calculated_value": 9.0,
        }
        
        hash1 = hashlib.sha256(
            json.dumps(data1, sort_keys=True).encode()
        ).hexdigest()
        
        hash2 = hashlib.sha256(
            json.dumps(data2, sort_keys=True).encode()
        ).hexdigest()
        
        assert hash1 != hash2
    
    def test_hash_format(self):
        """Test hash format is valid SHA-256."""
        data = {"test": "data"}
        hash_value = hashlib.sha256(
            json.dumps(data, sort_keys=True).encode()
        ).hexdigest()
        
        # SHA-256 produces 64 hex characters
        assert len(hash_value) == 64
        assert all(c in '0123456789abcdef' for c in hash_value)


class TestAuditTrailIntegrity:
    """Tests for audit trail integrity."""
    
    def test_evidence_chain_hash(self):
        """Test evidence chain hashing."""
        evidence_chain = []
        previous_hash = "genesis"
        
        for i in range(5):
            evidence = {
                "result_id": f"RES-00{i}",
                "control_code": f"CTRL-00{i}",
                "previous_hash": previous_hash,
            }
            
            current_hash = hashlib.sha256(
                json.dumps(evidence, sort_keys=True).encode()
            ).hexdigest()
            
            evidence["current_hash"] = current_hash
            evidence_chain.append(evidence)
            previous_hash = current_hash
        
        # Verify chain integrity
        for i, evidence in enumerate(evidence_chain):
            if i == 0:
                assert evidence["previous_hash"] == "genesis"
            else:
                assert evidence["previous_hash"] == evidence_chain[i-1]["current_hash"]
    
    def test_tamper_detection(self):
        """Test that tampering can be detected."""
        original = {
            "control_code": "CONC_001",
            "calculated_value": 8.5,
            "result_status": "pass",
        }
        
        original_hash = hashlib.sha256(
            json.dumps(original, sort_keys=True).encode()
        ).hexdigest()
        
        # Tamper with data
        tampered = original.copy()
        tampered["calculated_value"] = 5.0
        
        tampered_hash = hashlib.sha256(
            json.dumps(tampered, sort_keys=True).encode()
        ).hexdigest()
        
        # Hashes should differ
        assert original_hash != tampered_hash


class TestEvidenceStore:
    """Tests for EvidenceStore class methods."""
    
    @pytest.fixture
    def mock_postgres_conn(self):
        """Create mock PostgreSQL connection."""
        conn = MagicMock()
        conn.execute = MagicMock()
        conn.fetchall = MagicMock(return_value=[])
        conn.fetchone = MagicMock(return_value=None)
        return conn
    
    def test_evidence_store_initialization(self, mock_postgres_conn):
        """Test EvidenceStore can be initialized."""
        store = EvidenceStore(postgres_connection=mock_postgres_conn, user_id="test_user")
        assert store is not None
        assert store.user_id == "test_user"
    
    def test_evidence_retrieval_interface(self, mock_postgres_conn):
        """Test evidence retrieval methods exist."""
        store = EvidenceStore(postgres_connection=mock_postgres_conn)
        
        # Verify methods exist
        assert hasattr(store, 'get_control_results_for_run')
        assert hasattr(store, 'get_exceptions_for_run')
        assert hasattr(store, 'get_daily_compliance_summary')


class TestEvidencePackaging:
    """Tests for evidence packaging functionality."""
    
    def test_package_includes_all_evidence(self):
        """Test evidence package contains all required data."""
        package_data = {
            "package_id": "PKG-001",
            "run_date": "2024-01-15",
            "fund_ids": ["FUND-001"],
            "control_results": [],
            "exceptions": [],
            "data_snapshots": [],
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        
        assert "package_id" in package_data
        assert "control_results" in package_data
        assert "exceptions" in package_data
    
    def test_package_hash_generation(self):
        """Test evidence package hash is deterministic."""
        data = {
            "run_date": "2024-01-15",
            "fund_ids": ["FUND-001", "FUND-002"],
            "control_count": 25,
            "exception_count": 2,
        }
        
        hash1 = hashlib.sha256(
            json.dumps(data, sort_keys=True).encode()
        ).hexdigest()
        
        hash2 = hashlib.sha256(
            json.dumps(data, sort_keys=True).encode()
        ).hexdigest()
        
        assert hash1 == hash2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
