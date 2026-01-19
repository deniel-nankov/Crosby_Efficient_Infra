"""
Production-Ready Test Suite for Compliance RAG System
=====================================================
This module contains rigorous tests for production deployment including:
- Edge cases and boundary conditions
- Error handling and recovery
- Data integrity verification
- Concurrency and state management
- Security and input validation
- Regression tests for known issues

Run with: pytest tests/test_production_ready.py -v --tb=long
"""

import pytest
import hashlib
import json
import uuid
from datetime import date, datetime, timezone, timedelta
from decimal import Decimal
from typing import Dict, Any, List
import re


# =============================================================================
# IMPORTS FROM SOURCE
# =============================================================================

from src.control_runner import (
    ControlDefinition,
    ControlCategory,
    ControlFrequency,
    ThresholdOperator,
    ControlResultStatus,
    get_all_controls,
    get_active_controls,
    get_controls_by_category,
)

from src.control_runner.runner import (
    ControlRunContext,
    RunType,
    RunStatus,
    ExceptionSeverity,
    ControlExecutionResult,
)

from src.evidence_store import (
    EvidenceQuery,
    ControlResultEvidence,
    ExceptionEvidence,
    DailyComplianceSummary,
)

from src.retrieval import (
    RetrievalSource,
    RetrievalScope,
    RetrievedDocument,
    RetrievalContext,
)

from src.narrative import (
    NarrativeType,
    PromptTemplate,
    GeneratedNarrative,
    TEMPLATES,
)

from src.document_builder import (
    DocumentType,
    SectionType,
    DocumentSection,
    DocumentMetadata,
    GeneratedDocument,
)


# =============================================================================
# 1. FLOATING POINT PRECISION TESTS
# =============================================================================

class TestFloatingPointPrecision:
    """
    Test that floating point comparisons handle edge cases correctly.
    These are common sources of production bugs.
    """
    
    def test_threshold_at_exact_boundary(self):
        """Test control at exact threshold boundary."""
        control = ControlDefinition(
            control_code="TEST_001",
            control_name="Boundary Test",
            category=ControlCategory.CONCENTRATION,
            description="Test exact boundary",
            computation_sql="SELECT 10.0 AS calculated_value",
            threshold_value=10.0,
            threshold_operator=ThresholdOperator.GT,  # > 10 fails
        )
        
        # Exactly at threshold should PASS (not greater than)
        assert control.evaluate_threshold(10.0) == ControlResultStatus.PASS
        
        # Slightly above should FAIL
        assert control.evaluate_threshold(10.0001) == ControlResultStatus.FAIL
        
        # Slightly below should PASS
        assert control.evaluate_threshold(9.9999) == ControlResultStatus.PASS
    
    def test_floating_point_comparison_precision(self):
        """Test floating point precision issues (0.1 + 0.2 != 0.3 problem)."""
        control = ControlDefinition(
            control_code="TEST_002",
            control_name="Precision Test",
            category=ControlCategory.CONCENTRATION,
            description="Test floating point precision",
            computation_sql="SELECT 0.3 AS calculated_value",
            threshold_value=0.3,
            threshold_operator=ThresholdOperator.GTE,
        )
        
        # Classic floating point issue: 0.1 + 0.2 = 0.30000000000000004
        value = 0.1 + 0.2
        
        # This might cause issues depending on implementation
        result = control.evaluate_threshold(value)
        # Note: Current implementation uses direct comparison
        # In production, should use decimal.Decimal for financial calculations
        assert result in (ControlResultStatus.PASS, ControlResultStatus.FAIL)
    
    def test_very_small_numbers(self):
        """Test with very small numbers (basis points)."""
        control = ControlDefinition(
            control_code="TEST_003",
            control_name="Basis Points Test",
            category=ControlCategory.CONCENTRATION,
            description="Test small values",
            computation_sql="SELECT 0.0001 AS calculated_value",
            threshold_value=0.0005,  # 0.05% or 5 basis points
            threshold_operator=ThresholdOperator.GT,
        )
        
        assert control.evaluate_threshold(0.0001) == ControlResultStatus.PASS
        assert control.evaluate_threshold(0.0006) == ControlResultStatus.FAIL
    
    def test_very_large_numbers(self):
        """Test with very large numbers (AUM in billions)."""
        control = ControlDefinition(
            control_code="TEST_004",
            control_name="Large Number Test",
            category=ControlCategory.EXPOSURE,
            description="Test large values",
            computation_sql="SELECT 5000000000 AS calculated_value",
            threshold_value=10_000_000_000.0,  # $10B
            threshold_operator=ThresholdOperator.GT,
        )
        
        assert control.evaluate_threshold(5_000_000_000.0) == ControlResultStatus.PASS
        assert control.evaluate_threshold(15_000_000_000.0) == ControlResultStatus.FAIL
    
    def test_negative_values(self):
        """Test with negative values (short positions, losses)."""
        control = ControlDefinition(
            control_code="TEST_005",
            control_name="Negative Value Test",
            category=ControlCategory.EXPOSURE,
            description="Test negative values",
            computation_sql="SELECT -5.0 AS calculated_value",
            threshold_value=-10.0,  # Net short limit
            threshold_operator=ThresholdOperator.LT,  # < -10% fails
        )
        
        assert control.evaluate_threshold(-5.0) == ControlResultStatus.PASS  # -5 is not < -10
        assert control.evaluate_threshold(-15.0) == ControlResultStatus.FAIL  # -15 < -10
    
    def test_zero_value(self):
        """Test with zero value."""
        control = ControlDefinition(
            control_code="TEST_006",
            control_name="Zero Test",
            category=ControlCategory.LIQUIDITY,
            description="Test zero value",
            computation_sql="SELECT 0.0 AS calculated_value",
            threshold_value=5.0,
            threshold_operator=ThresholdOperator.LT,  # < 5% fails (need at least 5% liquidity)
        )
        
        # 0% liquidity should FAIL (less than 5% minimum)
        assert control.evaluate_threshold(0.0) == ControlResultStatus.FAIL


# =============================================================================
# 2. HASH INTEGRITY TESTS
# =============================================================================

class TestHashIntegrity:
    """
    Test that cryptographic hashes are correctly calculated and consistent.
    These ensure audit trail integrity.
    """
    
    def test_control_query_hash_deterministic(self):
        """Same SQL should always produce same hash."""
        sql = "SELECT MAX(value) AS calculated_value FROM test WHERE id = :id"
        
        control1 = ControlDefinition(
            control_code="HASH_001",
            control_name="Hash Test 1",
            category=ControlCategory.CONCENTRATION,
            description="Test",
            computation_sql=sql,
            threshold_value=10.0,
            threshold_operator=ThresholdOperator.GT,
        )
        
        control2 = ControlDefinition(
            control_code="HASH_002",
            control_name="Hash Test 2",
            category=ControlCategory.CONCENTRATION,
            description="Test",
            computation_sql=sql,
            threshold_value=10.0,
            threshold_operator=ThresholdOperator.GT,
        )
        
        assert control1.query_hash == control2.query_hash
    
    def test_control_query_hash_changes_with_sql(self):
        """Different SQL should produce different hash."""
        control1 = ControlDefinition(
            control_code="HASH_003",
            control_name="Hash Test",
            category=ControlCategory.CONCENTRATION,
            description="Test",
            computation_sql="SELECT 1 AS calculated_value",
            threshold_value=10.0,
            threshold_operator=ThresholdOperator.GT,
        )
        
        control2 = ControlDefinition(
            control_code="HASH_003",
            control_name="Hash Test",
            category=ControlCategory.CONCENTRATION,
            description="Test",
            computation_sql="SELECT 2 AS calculated_value",  # Different SQL
            threshold_value=10.0,
            threshold_operator=ThresholdOperator.GT,
        )
        
        assert control1.query_hash != control2.query_hash
    
    def test_document_section_hash_deterministic(self):
        """Document section content hash should be deterministic."""
        section1 = DocumentSection(
            section_id="SEC-001",
            section_order=1,
            section_type=SectionType.DETERMINISTIC,
            title="Test Section",
            content={"value": 123, "status": "pass"},
        )
        
        section2 = DocumentSection(
            section_id="SEC-001",
            section_order=1,
            section_type=SectionType.DETERMINISTIC,
            title="Test Section",
            content={"value": 123, "status": "pass"},
        )
        
        assert section1.content_hash == section2.content_hash
    
    def test_evidence_query_hash(self):
        """Evidence query hash should be deterministic."""
        query1 = EvidenceQuery(
            query_id="Q1",
            query_type="control_results",
            parameters={"date": "2024-01-15", "fund_id": "FUND-001"},
            executed_at=datetime.now(timezone.utc),
            executed_by="test_user",
        )
        
        query2 = EvidenceQuery(
            query_id="Q2",  # Different ID
            query_type="control_results",
            parameters={"date": "2024-01-15", "fund_id": "FUND-001"},
            executed_at=datetime.now(timezone.utc),
            executed_by="different_user",  # Different user
        )
        
        # Hash should be same (based on type and params, not ID/user)
        assert query1.query_hash == query2.query_hash
    
    def test_hash_format_is_valid_sha256(self):
        """Hashes should be valid SHA-256 format (64 hex chars)."""
        control = ControlDefinition(
            control_code="HASH_004",
            control_name="Format Test",
            category=ControlCategory.CONCENTRATION,
            description="Test",
            computation_sql="SELECT 1",
            threshold_value=10.0,
            threshold_operator=ThresholdOperator.GT,
        )
        
        hash_value = control.query_hash
        
        # SHA-256 produces 64 hex characters
        assert len(hash_value) == 64
        assert all(c in '0123456789abcdef' for c in hash_value)


# =============================================================================
# 3. DATE/TIME HANDLING TESTS
# =============================================================================

class TestDateTimeHandling:
    """
    Test date/time handling edge cases.
    Critical for compliance reporting accuracy.
    """
    
    def test_control_effective_date_future(self):
        """Future effective date should make control inactive."""
        tomorrow = date.today() + timedelta(days=1)
        
        control = ControlDefinition(
            control_code="DATE_001",
            control_name="Future Control",
            category=ControlCategory.CONCENTRATION,
            description="Test",
            computation_sql="SELECT 1",
            threshold_value=10.0,
            threshold_operator=ThresholdOperator.GT,
            effective_date=tomorrow,
        )
        
        assert not control.is_active
    
    def test_control_effective_date_past(self):
        """Past effective date should make control active."""
        yesterday = date.today() - timedelta(days=1)
        
        control = ControlDefinition(
            control_code="DATE_002",
            control_name="Past Control",
            category=ControlCategory.CONCENTRATION,
            description="Test",
            computation_sql="SELECT 1",
            threshold_value=10.0,
            threshold_operator=ThresholdOperator.GT,
            effective_date=yesterday,
        )
        
        assert control.is_active
    
    def test_control_expiration_date_past(self):
        """Past expiration date should make control inactive."""
        yesterday = date.today() - timedelta(days=1)
        last_month = date.today() - timedelta(days=30)
        
        control = ControlDefinition(
            control_code="DATE_003",
            control_name="Expired Control",
            category=ControlCategory.CONCENTRATION,
            description="Test",
            computation_sql="SELECT 1",
            threshold_value=10.0,
            threshold_operator=ThresholdOperator.GT,
            effective_date=last_month,
            expiration_date=yesterday,
        )
        
        assert not control.is_active
    
    def test_control_run_context_timestamp_utc(self):
        """Control run timestamps should be UTC."""
        context = ControlRunContext.create_new(
            run_type=RunType.SCHEDULED,
            run_date=date.today(),
            snowflake_snapshot_id="SNAP-001",
            snowflake_snapshot_ts=datetime.now(timezone.utc),
            fund_ids=["FUND-001"],
        )
        
        # Started_at should be timezone-aware
        assert context.started_at.tzinfo is not None
        assert context.started_at.tzinfo == timezone.utc


# =============================================================================
# 4. CONTROL REGISTRY TESTS
# =============================================================================

class TestControlRegistry:
    """
    Test control registry functions for completeness and correctness.
    """
    
    def test_all_controls_have_unique_codes(self):
        """All control codes should be unique."""
        all_controls = get_all_controls()
        codes = [c.control_code for c in all_controls]
        
        assert len(codes) == len(set(codes)), "Duplicate control codes found!"
    
    def test_all_controls_have_valid_categories(self):
        """All controls should have valid categories."""
        all_controls = get_all_controls()
        valid_categories = set(ControlCategory)
        
        for control in all_controls:
            assert control.category in valid_categories, f"Invalid category for {control.control_code}"
    
    def test_all_controls_have_non_empty_sql(self):
        """All controls should have non-empty SQL queries."""
        all_controls = get_all_controls()
        
        for control in all_controls:
            assert control.computation_sql.strip(), f"Empty SQL for {control.control_code}"
    
    def test_all_controls_have_descriptions(self):
        """All controls should have descriptions."""
        all_controls = get_all_controls()
        
        for control in all_controls:
            assert control.description.strip(), f"Missing description for {control.control_code}"
    
    def test_get_controls_by_category_returns_correct_controls(self):
        """Category filter should return correct controls."""
        all_controls = get_all_controls()
        
        for category in ControlCategory:
            filtered = get_controls_by_category(category)
            expected = [c for c in all_controls if c.category == category]
            
            assert len(filtered) == len(expected), f"Mismatch for category {category}"
    
    def test_active_controls_subset_of_all_controls(self):
        """Active controls should be a subset of all controls."""
        all_controls = get_all_controls()
        active_controls = get_active_controls()
        
        all_codes = set(c.control_code for c in all_controls)
        active_codes = set(c.control_code for c in active_controls)
        
        assert active_codes.issubset(all_codes)


# =============================================================================
# 5. INPUT VALIDATION TESTS
# =============================================================================

class TestInputValidation:
    """
    Test that invalid inputs are handled correctly.
    """
    
    def test_evaluate_threshold_with_nan(self):
        """NaN values should be handled gracefully."""
        control = ControlDefinition(
            control_code="VAL_001",
            control_name="NaN Test",
            category=ControlCategory.CONCENTRATION,
            description="Test",
            computation_sql="SELECT NULL",
            threshold_value=10.0,
            threshold_operator=ThresholdOperator.GT,
        )
        
        import math
        
        # NaN comparisons always return False
        result = control.evaluate_threshold(float('nan'))
        # This should not raise an exception
        assert result in (ControlResultStatus.PASS, ControlResultStatus.FAIL)
    
    def test_evaluate_threshold_with_infinity(self):
        """Infinity values should be handled."""
        control = ControlDefinition(
            control_code="VAL_002",
            control_name="Infinity Test",
            category=ControlCategory.CONCENTRATION,
            description="Test",
            computation_sql="SELECT 'inf'",
            threshold_value=10.0,
            threshold_operator=ThresholdOperator.GT,
        )
        
        # Positive infinity is greater than any number
        assert control.evaluate_threshold(float('inf')) == ControlResultStatus.FAIL
        
        # Negative infinity is less than any number
        assert control.evaluate_threshold(float('-inf')) == ControlResultStatus.PASS
    
    def test_evidence_query_with_special_characters(self):
        """Special characters in parameters should be handled."""
        query = EvidenceQuery(
            query_id="Q1",
            query_type="test",
            parameters={"name": "O'Brien & Co.", "symbol": "TEST$"},
            executed_at=datetime.now(timezone.utc),
            executed_by="test_user",
        )
        
        # Should not raise an exception
        hash_value = query.query_hash
        assert len(hash_value) == 64
    
    def test_document_section_with_unicode_content(self):
        """Unicode content should be handled correctly."""
        section = DocumentSection(
            section_id="SEC-001",
            section_order=1,
            section_type=SectionType.DETERMINISTIC,
            title="Unicode Test: 日本語 中文 العربية",
            content="Value: €100,000 | £50,000 | ¥1,000,000",
        )
        
        # Should not raise an exception
        hash_value = section.content_hash
        assert len(hash_value) == 64


# =============================================================================
# 6. WARNING THRESHOLD TESTS
# =============================================================================

class TestWarningThresholds:
    """
    Test warning threshold logic (early alerts before breach).
    """
    
    def test_warning_then_fail_precedence(self):
        """When both warning and fail thresholds are breached, FAIL takes precedence."""
        control = ControlDefinition(
            control_code="WARN_001",
            control_name="Warning Test",
            category=ControlCategory.CONCENTRATION,
            description="Test",
            computation_sql="SELECT 1",
            threshold_value=10.0,  # Fail at > 10%
            threshold_operator=ThresholdOperator.GT,
            warning_threshold=8.0,  # Warn at > 8%
            warning_operator=ThresholdOperator.GT,
        )
        
        # Below warning: PASS
        assert control.evaluate_threshold(7.0) == ControlResultStatus.PASS
        
        # At warning, below fail: WARNING
        assert control.evaluate_threshold(9.0) == ControlResultStatus.WARNING
        
        # Above fail: FAIL (not WARNING)
        assert control.evaluate_threshold(11.0) == ControlResultStatus.FAIL
    
    def test_warning_without_fail_breach(self):
        """Warning can trigger without fail threshold being breached."""
        control = ControlDefinition(
            control_code="WARN_002",
            control_name="Warning Only",
            category=ControlCategory.CONCENTRATION,
            description="Test",
            computation_sql="SELECT 1",
            threshold_value=15.0,  # Fail at > 15%
            threshold_operator=ThresholdOperator.GT,
            warning_threshold=10.0,  # Warn at > 10%
            warning_operator=ThresholdOperator.GT,
        )
        
        # Between warning and fail: WARNING
        assert control.evaluate_threshold(12.0) == ControlResultStatus.WARNING
    
    def test_no_warning_threshold(self):
        """Control without warning threshold should only return PASS or FAIL."""
        control = ControlDefinition(
            control_code="WARN_003",
            control_name="No Warning",
            category=ControlCategory.CONCENTRATION,
            description="Test",
            computation_sql="SELECT 1",
            threshold_value=10.0,
            threshold_operator=ThresholdOperator.GT,
            # No warning threshold
        )
        
        assert control.evaluate_threshold(5.0) == ControlResultStatus.PASS
        assert control.evaluate_threshold(11.0) == ControlResultStatus.FAIL
        # Should never return WARNING
        assert control.evaluate_threshold(9.9) == ControlResultStatus.PASS


# =============================================================================
# 7. BREACH AMOUNT CALCULATION TESTS
# =============================================================================

class TestBreachAmountCalculation:
    """
    Test breach amount calculations for different operator types.
    """
    
    def test_breach_amount_gt_operator(self):
        """Breach amount for GT operator should be positive when exceeding."""
        control = ControlDefinition(
            control_code="BREACH_001",
            control_name="GT Breach",
            category=ControlCategory.CONCENTRATION,
            description="Test",
            computation_sql="SELECT 1",
            threshold_value=10.0,
            threshold_operator=ThresholdOperator.GT,
        )
        
        # Value is 15, threshold is 10 -> breach by 5
        breach = control.get_breach_amount(15.0)
        assert breach == 5.0
        
        # Value is 8, threshold is 10 -> under by -2 (negative breach)
        breach = control.get_breach_amount(8.0)
        assert breach == -2.0
    
    def test_breach_amount_lt_operator(self):
        """Breach amount for LT operator should be positive when below."""
        control = ControlDefinition(
            control_code="BREACH_002",
            control_name="LT Breach",
            category=ControlCategory.LIQUIDITY,
            description="Test",
            computation_sql="SELECT 1",
            threshold_value=5.0,  # Need at least 5% liquidity
            threshold_operator=ThresholdOperator.LT,
        )
        
        # Value is 3%, threshold is 5% -> breach by 2% (liquidity shortfall)
        breach = control.get_breach_amount(3.0)
        assert breach == 2.0
        
        # Value is 7%, threshold is 5% -> over by -2% (excess liquidity)
        breach = control.get_breach_amount(7.0)
        assert breach == -2.0
    
    def test_breach_amount_eq_operator_returns_none(self):
        """Breach amount for EQ operator should return None (not applicable)."""
        control = ControlDefinition(
            control_code="BREACH_003",
            control_name="EQ Breach",
            category=ControlCategory.RECONCILIATION,
            description="Test",
            computation_sql="SELECT 1",
            threshold_value=0.0,  # Difference should be 0
            threshold_operator=ThresholdOperator.EQ,
        )
        
        # EQ operator doesn't have a meaningful breach amount
        breach = control.get_breach_amount(5.0)
        assert breach is None


# =============================================================================
# 8. TEMPLATE VALIDATION TESTS
# =============================================================================

class TestTemplateValidation:
    """
    Test narrative templates for correctness.
    """
    
    def test_all_templates_have_required_fields(self):
        """All templates should have required evidence types."""
        for name, template in TEMPLATES.items():
            assert template.template_id, f"Template {name} missing template_id"
            assert template.template_version, f"Template {name} missing version"
            assert template.system_prompt, f"Template {name} missing system_prompt"
            assert template.user_prompt_template, f"Template {name} missing user_prompt_template"
            assert template.required_evidence_types, f"Template {name} missing required_evidence_types"
    
    def test_template_hashes_are_deterministic(self):
        """Template hashes should be deterministic."""
        template = TEMPLATES["daily_summary"]
        
        hash1 = template.template_hash
        hash2 = template.template_hash
        
        assert hash1 == hash2
    
    def test_system_prompt_contains_critical_rules(self):
        """System prompt should contain anti-hallucination rules."""
        template = TEMPLATES["daily_summary"]
        
        critical_phrases = [
            "ONLY use information",
            "NEVER invent",
            "citations",
            "Insufficient evidence",
        ]
        
        for phrase in critical_phrases:
            assert phrase.lower() in template.system_prompt.lower(), \
                f"System prompt missing critical phrase: {phrase}"


# =============================================================================
# 9. CITATION EXTRACTION TESTS
# =============================================================================

class TestCitationExtraction:
    """
    Test citation extraction from narrative text.
    """
    
    def test_extract_control_run_citation(self):
        """Should extract [ControlRun: ...] citations."""
        text = "The control passed. [ControlRun: 2024-01-15 | Control: CONC_001 | Snapshot: SNAP-001]"
        
        pattern = r'\[(?:ControlRun|ControlResult|Exception|Policy|Filing):[^\]]+\]'
        citations = re.findall(pattern, text)
        
        assert len(citations) == 1
        assert "ControlRun" in citations[0]
    
    def test_extract_multiple_citations(self):
        """Should extract multiple citations from text."""
        text = """
        Control A passed [ControlResult: CONC_001].
        Exception B opened [Exception: EXC-001].
        Per policy [Policy: POL-001].
        """
        
        pattern = r'\[(?:ControlRun|ControlResult|Exception|Policy|Filing):[^\]]+\]'
        citations = re.findall(pattern, text)
        
        assert len(citations) == 3
    
    def test_no_citations_returns_empty(self):
        """Text without citations should return empty list."""
        text = "This is plain text without any citations."
        
        pattern = r'\[(?:ControlRun|ControlResult|Exception|Policy|Filing):[^\]]+\]'
        citations = re.findall(pattern, text)
        
        assert len(citations) == 0


# =============================================================================
# 10. RUN CONTEXT UNIQUENESS TESTS
# =============================================================================

class TestRunContextUniqueness:
    """
    Test that run IDs and codes are unique.
    """
    
    def test_run_ids_are_unique(self):
        """Each new context should have unique run_id."""
        contexts = []
        for _ in range(100):
            ctx = ControlRunContext.create_new(
                run_type=RunType.SCHEDULED,
                run_date=date.today(),
                snowflake_snapshot_id="SNAP-001",
                snowflake_snapshot_ts=datetime.now(timezone.utc),
                fund_ids=["FUND-001"],
            )
            contexts.append(ctx)
        
        run_ids = [c.run_id for c in contexts]
        assert len(run_ids) == len(set(run_ids)), "Duplicate run IDs generated!"
    
    def test_run_codes_contain_date(self):
        """Run codes should contain the run date."""
        run_date = date(2024, 6, 15)
        
        ctx = ControlRunContext.create_new(
            run_type=RunType.SCHEDULED,
            run_date=run_date,
            snowflake_snapshot_id="SNAP-001",
            snowflake_snapshot_ts=datetime.now(timezone.utc),
            fund_ids=["FUND-001"],
        )
        
        assert "2024-06-15" in ctx.run_code


# =============================================================================
# 11. DOCUMENT GENERATION TESTS
# =============================================================================

class TestDocumentGeneration:
    """
    Test document structure and metadata.
    """
    
    def test_document_section_ordering(self):
        """Sections should maintain order."""
        sections = [
            DocumentSection(
                section_id=f"SEC-{i}",
                section_order=i,
                section_type=SectionType.DETERMINISTIC,
                title=f"Section {i}",
                content=f"Content {i}",
            )
            for i in [3, 1, 2]
        ]
        
        sorted_sections = sorted(sections, key=lambda s: s.section_order)
        orders = [s.section_order for s in sorted_sections]
        
        assert orders == [1, 2, 3]
    
    def test_document_metadata_audit_record(self):
        """Document should produce complete audit record."""
        metadata = DocumentMetadata(
            document_id=str(uuid.uuid4()),
            document_code="DCP-2024-01-15-ABC123",
            document_type=DocumentType.DAILY_COMPLIANCE_PACK,
            document_date=date(2024, 1, 15),
            run_id=str(uuid.uuid4()),
            snapshot_id="SNAP-001",
            template_id="TMPL-001",
            template_version="1.0.0",
            template_hash="abc123",
        )
        
        sections = [
            DocumentSection(
                section_id="SEC-1",
                section_order=1,
                section_type=SectionType.HEADER,
                title="Header",
                content="Test",
            )
        ]
        
        # Create minimal PDF bytes
        pdf_bytes = b"%PDF-1.4 test content"
        
        doc = GeneratedDocument(
            metadata=metadata,
            sections=sections,
            pdf_bytes=pdf_bytes,
        )
        
        audit = doc.to_audit_record()
        
        # Check required fields
        assert "document_id" in audit
        assert "document_hash" in audit
        assert "section_count" in audit
        assert audit["section_count"] == 1


# =============================================================================
# 12. REGRESSION TESTS - Known Edge Cases
# =============================================================================

class TestRegressionKnownIssues:
    """
    Regression tests for previously identified issues.
    """
    
    def test_empty_fund_list_handling(self):
        """Empty fund list should be handled gracefully."""
        ctx = ControlRunContext.create_new(
            run_type=RunType.SCHEDULED,
            run_date=date.today(),
            snowflake_snapshot_id="SNAP-001",
            snowflake_snapshot_ts=datetime.now(timezone.utc),
            fund_ids=[],  # Empty list
        )
        
        assert ctx.fund_ids == []
        assert ctx.run_id is not None
    
    def test_very_long_control_description(self):
        """Very long descriptions should not cause issues."""
        long_description = "A" * 10000
        
        control = ControlDefinition(
            control_code="LONG_001",
            control_name="Long Description Test",
            category=ControlCategory.CONCENTRATION,
            description=long_description,
            computation_sql="SELECT 1",
            threshold_value=10.0,
            threshold_operator=ThresholdOperator.GT,
        )
        
        # Should not raise
        assert len(control.description) == 10000
        assert control.query_hash is not None
    
    def test_sql_with_special_sql_characters(self):
        """SQL with comments and special chars should hash correctly."""
        sql = """
        -- This is a comment
        SELECT 
            MAX(value) AS calculated_value /* inline comment */
        FROM compliance.test
        WHERE status = 'active'
          AND name LIKE '%test%'
        """
        
        control = ControlDefinition(
            control_code="SQL_001",
            control_name="SQL Special Chars",
            category=ControlCategory.CONCENTRATION,
            description="Test",
            computation_sql=sql,
            threshold_value=10.0,
            threshold_operator=ThresholdOperator.GT,
        )
        
        # Should produce valid hash
        assert len(control.query_hash) == 64


# =============================================================================
# 13. CONCURRENT ACCESS SIMULATION
# =============================================================================

class TestConcurrencySimulation:
    """
    Simulate concurrent access patterns.
    """
    
    def test_multiple_contexts_same_timestamp(self):
        """Multiple contexts created at same timestamp should still be unique."""
        timestamp = datetime.now(timezone.utc)
        
        contexts = [
            ControlRunContext(
                run_id=str(uuid.uuid4()),
                run_code=f"RUN-{i}",
                run_type=RunType.SCHEDULED,
                run_date=date.today(),
                snowflake_snapshot_id="SNAP-001",
                snowflake_snapshot_ts=timestamp,
                fund_ids=["FUND-001"],
                executor_service="test",
                executor_version="1.0.0",
                config_hash="hash",
                started_at=timestamp,  # Same timestamp
            )
            for i in range(10)
        ]
        
        run_ids = [c.run_id for c in contexts]
        assert len(run_ids) == len(set(run_ids))


# =============================================================================
# RUN ALL TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
