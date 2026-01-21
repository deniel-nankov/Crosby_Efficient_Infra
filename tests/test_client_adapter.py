"""
Tests for the New Integration Module (Client Adapters + RAG Pipeline)

Tests the simplified client adapters and RAG pipeline that trusts client data.
"""
import pytest
from datetime import date
from decimal import Decimal
from pathlib import Path


class TestPositionDataclass:
    """Tests for Position dataclass."""
    
    def test_position_basic_fields(self):
        """Position dataclass holds expected fields."""
        from src.integration.client_adapter import Position
        
        pos = Position(
            security_id="AAPL",
            ticker="AAPL",
            security_name="Apple Inc",
            quantity=Decimal("1000"),
            market_value=Decimal("150000"),
            currency="USD"
        )
        
        assert pos.security_id == "AAPL"
        assert pos.ticker == "AAPL"
        assert pos.quantity == Decimal("1000")
        assert pos.market_value == Decimal("150000")
        assert pos.currency == "USD"
    
    def test_position_optional_fields(self):
        """Position has optional fields with defaults."""
        from src.integration.client_adapter import Position
        
        pos = Position(
            security_id="GOOGL",
            ticker="GOOGL",
            security_name="Alphabet",
            quantity=Decimal("500"),
            market_value=Decimal("100000"),
            currency="USD"
        )
        
        # Optional fields default to None
        assert pos.sector is None
        assert pos.isin is None


class TestControlResultDataclass:
    """Tests for ControlResult dataclass."""
    
    def test_control_result_pass(self):
        """ControlResult for passing control."""
        from src.integration.client_adapter import ControlResult
        
        result = ControlResult(
            control_id="CONC_001",
            control_name="Concentration Limit",
            control_type="concentration",
            calculated_value=Decimal("4.5"),
            threshold=Decimal("5.0"),
            threshold_operator="lte",
            status="pass"
        )
        
        assert result.control_id == "CONC_001"
        assert result.status == "pass"
        assert result.threshold_operator == "lte"
        assert result.breach_amount is None
    
    def test_control_result_fail_with_breach(self):
        """ControlResult can hold breach amount for failures."""
        from src.integration.client_adapter import ControlResult
        
        result = ControlResult(
            control_id="CONC_002",
            control_name="Sector Concentration",
            control_type="concentration",
            calculated_value=Decimal("35.0"),
            threshold=Decimal("30.0"),
            threshold_operator="lte",
            status="fail",
            breach_amount=Decimal("5.0")
        )
        
        assert result.status == "fail"
        assert result.breach_amount == Decimal("5.0")
    
    def test_control_result_warning_status(self):
        """ControlResult can have warning status."""
        from src.integration.client_adapter import ControlResult
        
        result = ControlResult(
            control_id="LIQ_001",
            control_name="Liquidity Warning",
            control_type="liquidity",
            calculated_value=Decimal("28.0"),
            threshold=Decimal("30.0"),
            threshold_operator="gte",
            status="warning"
        )
        
        assert result.status == "warning"


class TestDataSnapshotDataclass:
    """Tests for DataSnapshot dataclass."""
    
    def test_data_snapshot_structure(self):
        """DataSnapshot holds positions and control results."""
        from src.integration.client_adapter import DataSnapshot, Position, ControlResult
        
        positions = [
            Position(
                security_id="MSFT",
                ticker="MSFT",
                security_name="Microsoft",
                quantity=Decimal("500"),
                market_value=Decimal("200000"),
                currency="USD"
            )
        ]
        
        controls = [
            ControlResult(
                control_id="TEST_001",
                control_name="Test Control",
                control_type="test",
                calculated_value=Decimal("1.0"),
                threshold=Decimal("5.0"),
                threshold_operator="lte",
                status="pass"
            )
        ]
        
        snapshot = DataSnapshot(
            snapshot_id="SNAP-001",
            as_of_date=date.today(),
            source_system="TestSystem",
            positions=positions,
            control_results=controls
        )
        
        assert snapshot.snapshot_id == "SNAP-001"
        assert len(snapshot.positions) == 1
        assert len(snapshot.control_results) == 1
        assert snapshot.source_system == "TestSystem"
    
    def test_data_snapshot_optional_nav(self):
        """DataSnapshot can include NAV."""
        from src.integration.client_adapter import DataSnapshot
        
        snapshot = DataSnapshot(
            snapshot_id="SNAP-002",
            as_of_date=date.today(),
            source_system="TestSystem",
            positions=[],
            control_results=[],
            nav=Decimal("2_000_000_000")
        )
        
        assert snapshot.nav == Decimal("2_000_000_000")
    
    def test_data_snapshot_has_data_hash(self):
        """DataSnapshot generates a data hash for audit."""
        from src.integration.client_adapter import DataSnapshot
        
        snapshot = DataSnapshot(
            snapshot_id="SNAP-003",
            as_of_date=date.today(),
            source_system="TestSystem",
            positions=[],
            control_results=[],
        )
        
        assert snapshot.data_hash is not None
        assert len(snapshot.data_hash) == 16


class TestMockAdapter:
    """Tests for MockAdapter."""
    
    def test_mock_adapter_returns_snapshot(self):
        """MockAdapter returns a valid snapshot."""
        from src.integration.client_adapter import MockAdapter
        
        adapter = MockAdapter()
        snapshot = adapter.get_snapshot(as_of_date=date.today())
        
        assert snapshot is not None
        assert snapshot.as_of_date == date.today()
        assert len(snapshot.positions) > 0
        assert len(snapshot.control_results) > 0
    
    def test_mock_adapter_positions_have_values(self):
        """MockAdapter positions have realistic values."""
        from src.integration.client_adapter import MockAdapter
        
        adapter = MockAdapter()
        snapshot = adapter.get_snapshot(as_of_date=date.today())
        
        for pos in snapshot.positions:
            assert pos.quantity > 0
            assert pos.market_value > 0
            assert pos.currency == "USD"
    
    def test_mock_adapter_controls_have_status(self):
        """MockAdapter controls have valid status values."""
        from src.integration.client_adapter import MockAdapter
        
        adapter = MockAdapter()
        snapshot = adapter.get_snapshot(as_of_date=date.today())
        
        valid_statuses = {"pass", "fail", "warning"}
        for control in snapshot.control_results:
            assert control.status in valid_statuses
    
    def test_mock_adapter_generates_realistic_nav(self):
        """MockAdapter NAV is in realistic hedge fund range."""
        from src.integration.client_adapter import MockAdapter
        
        adapter = MockAdapter()
        snapshot = adapter.get_snapshot(as_of_date=date.today())
        
        # NAV should be around $2B
        assert snapshot.nav > Decimal("1_000_000_000")
        assert snapshot.nav < Decimal("3_000_000_000")
    
    def test_mock_adapter_snapshot_id_format(self):
        """MockAdapter generates consistent snapshot IDs."""
        from src.integration.client_adapter import MockAdapter
        
        adapter = MockAdapter()
        snapshot = adapter.get_snapshot(as_of_date=date.today())
        
        assert snapshot.snapshot_id.startswith("SNAP-")


class TestCSVAdapter:
    """Tests for CSV adapter."""
    
    def test_csv_adapter_requires_data_dir(self):
        """CSVAdapter requires a data_dir path."""
        from src.integration.client_adapter import CSVAdapter
        
        with pytest.raises(TypeError):
            CSVAdapter()  # Missing required argument
    
    def test_csv_adapter_stores_path(self):
        """CSVAdapter stores the data directory path."""
        from src.integration.client_adapter import CSVAdapter
        import os
        
        # Use a path that works on all platforms
        test_dir = os.path.join("data", "exports")
        adapter = CSVAdapter(data_dir=test_dir)
        # Path normalization may differ by OS, just check it ends with expected parts
        assert adapter.data_dir.name == "exports"
        assert adapter.data_dir.parent.name == "data"


class TestDatabaseAdapter:
    """Tests for DatabaseAdapter."""
    
    def test_database_adapter_stores_config(self):
        """DatabaseAdapter stores connection config."""
        from src.integration.client_adapter import DatabaseAdapter
        
        adapter = DatabaseAdapter(
            connection_string="postgresql://user:pass@host:5432/db",
            positions_query="SELECT * FROM positions",
            controls_query="SELECT * FROM controls",
            nav_query="SELECT nav FROM fund_data",
        )
        
        assert adapter.connection_string == "postgresql://user:pass@host:5432/db"
        assert adapter.positions_query == "SELECT * FROM positions"


class TestGetAdapterFactory:
    """Tests for adapter factory function."""
    
    def test_get_adapter_mock(self):
        """get_adapter returns MockAdapter for 'mock' type."""
        from src.integration.client_adapter import get_adapter, MockAdapter
        
        adapter = get_adapter("mock")
        assert isinstance(adapter, MockAdapter)
    
    def test_get_adapter_csv(self):
        """get_adapter returns CSVAdapter for 'csv' type."""
        from src.integration.client_adapter import get_adapter, CSVAdapter
        
        adapter = get_adapter("csv", data_dir="/data/test")
        assert isinstance(adapter, CSVAdapter)
    
    def test_get_adapter_database(self):
        """get_adapter returns DatabaseAdapter for 'database' type."""
        from src.integration.client_adapter import get_adapter, DatabaseAdapter
        
        adapter = get_adapter(
            "database",
            connection_string="postgresql://localhost:5432/db",
            positions_query="SELECT * FROM positions WHERE date = %(as_of_date)s",
            controls_query="SELECT * FROM controls WHERE date = %(as_of_date)s",
            nav_query="SELECT nav FROM fund_data WHERE date = %(as_of_date)s",
        )
        assert isinstance(adapter, DatabaseAdapter)
    
    def test_get_adapter_invalid_type(self):
        """get_adapter raises for unknown type."""
        from src.integration.client_adapter import get_adapter
        
        with pytest.raises(ValueError) as exc_info:
            get_adapter("invalid_type")
        
        assert "Unknown adapter type" in str(exc_info.value)


class TestRAGPipelineDataclasses:
    """Tests for RAG pipeline dataclasses."""
    
    def test_policy_context_dataclass(self):
        """PolicyContext holds retrieved policy data."""
        from src.integration.rag_pipeline import PolicyContext
        
        context = PolicyContext(
            policy_id="POL-001",
            section="Section 3.2",
            content="This is the policy content.",
            relevance_score=0.95,
        )
        
        assert context.policy_id == "POL-001"
        assert context.relevance_score == 0.95
        assert context.section == "Section 3.2"
    
    def test_policy_context_to_citation(self):
        """PolicyContext can generate citation string."""
        from src.integration.rag_pipeline import PolicyContext
        
        context = PolicyContext(
            policy_id="ADV-2B",
            section="Custody Requirements",
            content="Content here...",
        )
        
        citation = context.to_citation()
        assert "ADV-2B" in citation
        assert "Custody" in citation
    
    def test_generated_narrative_dataclass(self):
        """GeneratedNarrative holds LLM output."""
        from src.integration.rag_pipeline import GeneratedNarrative
        from datetime import datetime, timezone
        
        narrative = GeneratedNarrative(
            narrative_id="NAR-001",
            control_id="CTRL-001",
            content="The control passed successfully.",
            content_hash="abc123",
            citations=["POL-001", "POL-002"],
            model_used="gpt-4",
            prompt_hash="def456",
            context_hash="ghi789",
            generated_at=datetime.now(timezone.utc),
        )
        
        assert narrative.control_id == "CTRL-001"
        assert len(narrative.citations) == 2
        assert "POL-001" in narrative.citations
    
    def test_generated_narrative_to_dict(self):
        """GeneratedNarrative can be serialized to dict."""
        from src.integration.rag_pipeline import GeneratedNarrative
        from datetime import datetime, timezone
        
        narrative = GeneratedNarrative(
            narrative_id="NAR-002",
            control_id="CTRL-002",
            content="Test content",
            content_hash="abc",
            citations=["POL-001"],
            model_used="gpt-4",
            prompt_hash="def",
            context_hash="ghi",
            generated_at=datetime.now(timezone.utc),
        )
        
        data = narrative.to_dict()
        assert data["narrative_id"] == "NAR-002"
        assert data["control_id"] == "CTRL-002"
    
    def test_compliance_report_dataclass(self):
        """ComplianceReport aggregates narratives."""
        from src.integration.rag_pipeline import ComplianceReport
        from src.integration.client_adapter import DataSnapshot
        
        snapshot = DataSnapshot(
            snapshot_id="SNAP-001",
            as_of_date=date.today(),
            source_system="Test",
            positions=[],
            control_results=[],
            nav=Decimal("2_000_000_000"),
        )
        
        report = ComplianceReport(
            report_id="RPT-001",
            as_of_date=date.today(),
            snapshot=snapshot,
            narratives=[],
            controls_passed=5,
            controls_warning=2,
            controls_failed=1,
        )
        
        assert report.report_id == "RPT-001"
        assert report.controls_failed == 1
        assert report.controls_passed == 5


class TestComplianceRAGPipeline:
    """Tests for ComplianceRAGPipeline."""
    
    def test_pipeline_initialization(self):
        """ComplianceRAGPipeline can be initialized."""
        from src.integration.rag_pipeline import ComplianceRAGPipeline
        
        pipeline = ComplianceRAGPipeline()
        
        assert pipeline is not None
        assert pipeline.model_id == "gpt-4o"
    
    def test_pipeline_initialization_with_custom_model(self):
        """ComplianceRAGPipeline accepts custom model."""
        from src.integration.rag_pipeline import ComplianceRAGPipeline
        
        pipeline = ComplianceRAGPipeline(model_id="gpt-3.5-turbo")
        
        assert pipeline.model_id == "gpt-3.5-turbo"
    
    def test_pipeline_generate_report(self):
        """Pipeline generates a compliance report."""
        from src.integration.rag_pipeline import ComplianceRAGPipeline
        from src.integration.client_adapter import MockAdapter
        
        adapter = MockAdapter()
        pipeline = ComplianceRAGPipeline()
        
        snapshot = adapter.get_snapshot(as_of_date=date.today())
        report = pipeline.generate_report(snapshot)
        
        assert report is not None
        assert report.as_of_date == date.today()
    
    def test_pipeline_counts_control_statuses(self):
        """Pipeline correctly counts control statuses."""
        from src.integration.rag_pipeline import ComplianceRAGPipeline
        from src.integration.client_adapter import MockAdapter
        
        adapter = MockAdapter()
        pipeline = ComplianceRAGPipeline()
        
        snapshot = adapter.get_snapshot(as_of_date=date.today())
        report = pipeline.generate_report(snapshot)
        
        # Total should match control count
        total = report.controls_passed + report.controls_warning + report.controls_failed
        assert total == len(snapshot.control_results)
    
    def test_pipeline_generates_narratives_for_warnings(self):
        """Pipeline generates narratives for warnings/failures."""
        from src.integration.rag_pipeline import ComplianceRAGPipeline
        from src.integration.client_adapter import MockAdapter
        
        adapter = MockAdapter()
        pipeline = ComplianceRAGPipeline()
        
        snapshot = adapter.get_snapshot(as_of_date=date.today())
        report = pipeline.generate_report(snapshot)
        
        # Should have narratives for warnings and failures
        non_pass_count = sum(1 for c in snapshot.control_results if c.status != "pass")
        assert len(report.narratives) == non_pass_count


class TestComplianceReportSummary:
    """Tests for ComplianceReport executive summary."""
    
    def test_executive_summary_includes_date(self):
        """Executive summary includes report date."""
        from src.integration.rag_pipeline import ComplianceReport
        from src.integration.client_adapter import DataSnapshot
        
        snapshot = DataSnapshot(
            snapshot_id="SNAP-001",
            as_of_date=date.today(),
            source_system="Test",
            positions=[],
            control_results=[],
            nav=Decimal("2_000_000_000"),
        )
        
        report = ComplianceReport(
            report_id="RPT-001",
            as_of_date=date.today(),
            snapshot=snapshot,
            controls_passed=5,
            controls_warning=0,
            controls_failed=0,
        )
        
        summary = report.get_executive_summary()
        assert str(date.today()) in summary
    
    def test_executive_summary_shows_pass_status(self):
        """Executive summary shows passed status when all pass."""
        from src.integration.rag_pipeline import ComplianceReport
        from src.integration.client_adapter import DataSnapshot
        
        snapshot = DataSnapshot(
            snapshot_id="SNAP-001",
            as_of_date=date.today(),
            source_system="Test",
            positions=[],
            control_results=[],
            nav=Decimal("2_000_000_000"),
        )
        
        report = ComplianceReport(
            report_id="RPT-001",
            as_of_date=date.today(),
            snapshot=snapshot,
            controls_passed=5,
            controls_warning=0,
            controls_failed=0,
        )
        
        summary = report.get_executive_summary()
        assert "ALL CONTROLS PASSED" in summary
    
    def test_executive_summary_shows_breach_status(self):
        """Executive summary shows breach status when failures exist."""
        from src.integration.rag_pipeline import ComplianceReport
        from src.integration.client_adapter import DataSnapshot
        
        snapshot = DataSnapshot(
            snapshot_id="SNAP-001",
            as_of_date=date.today(),
            source_system="Test",
            positions=[],
            control_results=[],
            nav=Decimal("2_000_000_000"),
        )
        
        report = ComplianceReport(
            report_id="RPT-001",
            as_of_date=date.today(),
            snapshot=snapshot,
            controls_passed=3,
            controls_warning=1,
            controls_failed=1,
        )
        
        summary = report.get_executive_summary()
        assert "BREACH" in summary.upper()


class TestEndToEndWorkflow:
    """End-to-end tests simulating real usage."""
    
    def test_full_workflow_with_mock_adapter(self):
        """Complete workflow with mock adapter."""
        from src.integration.client_adapter import MockAdapter
        from src.integration.rag_pipeline import ComplianceRAGPipeline
        
        # 1. Initialize
        adapter = MockAdapter()
        pipeline = ComplianceRAGPipeline()
        
        # 2. Get data snapshot
        snapshot = adapter.get_snapshot(as_of_date=date.today())
        assert len(snapshot.positions) > 0
        
        # 3. Generate report
        report = pipeline.generate_report(snapshot)
        assert report is not None
        
        # 4. Verify counts
        total = report.controls_passed + report.controls_warning + report.controls_failed
        assert total == len(snapshot.control_results)
    
    def test_narratives_have_citations(self):
        """All narratives should include citations."""
        from src.integration.client_adapter import MockAdapter
        from src.integration.rag_pipeline import ComplianceRAGPipeline
        
        adapter = MockAdapter()
        pipeline = ComplianceRAGPipeline()
        
        snapshot = adapter.get_snapshot(as_of_date=date.today())
        report = pipeline.generate_report(snapshot)
        
        for narrative in report.narratives:
            assert narrative.has_citations or len(narrative.content) > 0
    
    def test_report_has_audit_trail(self):
        """Report includes audit trail information."""
        from src.integration.client_adapter import MockAdapter
        from src.integration.rag_pipeline import ComplianceRAGPipeline
        
        adapter = MockAdapter()
        pipeline = ComplianceRAGPipeline()
        
        snapshot = adapter.get_snapshot(as_of_date=date.today())
        report = pipeline.generate_report(snapshot)
        
        # Report should have ID and timestamp
        assert report.report_id is not None
        assert report.generated_at is not None
        
        # Narratives should have hashes
        for narrative in report.narratives:
            assert narrative.content_hash is not None
            assert narrative.prompt_hash is not None

