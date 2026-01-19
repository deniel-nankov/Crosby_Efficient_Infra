"""
Document Builder - PDF Generation with Locked Structure and Audit Trail

This module generates compliance documents (PDFs) with:
- Fixed, locked structure (no LLM control over layout)
- Deterministic tables from SQL results
- LLM-generated narratives with inline citations
- Complete audit appendices
- Document hashing for integrity verification

SEC Examination Note:
- All documents include generation metadata
- Document hash allows verification of integrity
- Section types (deterministic vs LLM) are clearly marked
- All evidence is traceable via citations
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, date
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any, List, BinaryIO
import uuid

logger = logging.getLogger(__name__)


class DocumentType(Enum):
    """Types of compliance documents."""
    DAILY_COMPLIANCE_PACK = "daily_compliance_pack"
    FORM_PF_WORKPAPER = "form_pf_workpaper"
    THIRTEENF_WORKPAPER = "13f_workpaper"
    ADV_WORKPAPER = "adv_workpaper"
    EXCEPTION_REPORT = "exception_report"
    QUARTERLY_REVIEW = "quarterly_review"


class SectionType(Enum):
    """Types of document sections for audit purposes."""
    HEADER = "header"
    DETERMINISTIC = "deterministic"  # SQL/Python generated
    LLM_NARRATIVE = "llm_narrative"  # LLM generated with citations
    TABLE = "table"
    APPENDIX = "appendix"
    SIGNATURE = "signature"


@dataclass
class DocumentSection:
    """
    A section of a compliance document.
    
    Each section tracks its source type for audit purposes.
    """
    section_id: str
    section_order: int
    section_type: SectionType
    title: str
    content: Any  # str, dict, or list depending on type
    
    # For LLM sections
    narrative_id: Optional[str] = None
    citations: List[str] = field(default_factory=list)
    
    # For table sections
    column_headers: Optional[List[str]] = None
    table_data: Optional[List[List[Any]]] = None
    
    # Evidence linkage
    evidence_ids: List[str] = field(default_factory=list)
    
    @property
    def content_hash(self) -> str:
        """Hash of section content."""
        if isinstance(self.content, str):
            return hashlib.sha256(self.content.encode()).hexdigest()
        else:
            return hashlib.sha256(json.dumps(self.content, default=str).encode()).hexdigest()


@dataclass
class DocumentMetadata:
    """
    Metadata for a generated document.
    
    This is included in the document and stored in the database.
    """
    document_id: str
    document_code: str
    document_type: DocumentType
    document_date: date
    
    # Source data
    run_id: Optional[str] = None
    snapshot_id: Optional[str] = None
    
    # Generation info
    template_id: str = ""
    template_version: str = ""
    template_hash: str = ""
    
    # LLM info (if narratives were generated)
    llm_model_id: Optional[str] = None
    llm_model_version: Optional[str] = None
    llm_tokens_used: int = 0
    
    # Output info
    page_count: int = 0
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    generated_by: str = "system"
    
    # Integrity
    document_hash: str = ""


@dataclass
class GeneratedDocument:
    """
    Complete generated document with audit trail.
    """
    metadata: DocumentMetadata
    sections: List[DocumentSection]
    pdf_bytes: bytes
    
    @property
    def document_hash(self) -> str:
        """SHA-256 hash of PDF content."""
        return hashlib.sha256(self.pdf_bytes).hexdigest()
    
    def save(self, path: Path) -> None:
        """Save PDF to file."""
        path.write_bytes(self.pdf_bytes)
        logger.info(f"Document saved: {path}")
    
    def to_audit_record(self) -> Dict[str, Any]:
        """Create audit record for database storage."""
        return {
            "document_id": self.metadata.document_id,
            "document_code": self.metadata.document_code,
            "document_type": self.metadata.document_type.value,
            "document_date": self.metadata.document_date.isoformat(),
            "run_id": self.metadata.run_id,
            "snapshot_id": self.metadata.snapshot_id,
            "template_id": self.metadata.template_id,
            "template_version": self.metadata.template_version,
            "template_hash": self.metadata.template_hash,
            "llm_model_id": self.metadata.llm_model_id,
            "llm_model_version": self.metadata.llm_model_version,
            "llm_tokens_used": self.metadata.llm_tokens_used,
            "page_count": self.metadata.page_count,
            "document_hash": self.document_hash,
            "generated_at": self.metadata.generated_at.isoformat(),
            "section_count": len(self.sections),
            "section_hashes": [s.content_hash for s in self.sections],
        }


class DocumentBuilder:
    """
    Main document builder for compliance PDFs.
    
    This class constructs documents with a fixed structure:
    1. Header with document info and generation metadata
    2. Executive summary (deterministic metrics)
    3. Detailed results tables (deterministic)
    4. Narrative sections (LLM with citations)
    5. Appendices (evidence details)
    6. Signature section
    """
    
    def __init__(self, settings: Any):
        self.settings = settings
        self.logger = logging.getLogger(f"{__name__}.DocumentBuilder")
    
    def build_daily_compliance_pack(
        self,
        run_date: date,
        run_summary: Dict[str, Any],
        control_results: List[Dict[str, Any]],
        exceptions: List[Dict[str, Any]],
        outstanding_exceptions: List[Dict[str, Any]],
        narrative: Optional[str] = None,
        narrative_metadata: Optional[Dict[str, Any]] = None,
    ) -> GeneratedDocument:
        """
        Build the Daily Compliance Pack PDF.
        
        Structure:
        1. Cover page with summary metrics
        2. Control results summary table
        3. Exception summary table
        4. Narrative summary (LLM)
        5. Detailed results by category
        6. Audit appendix
        """
        document_id = str(uuid.uuid4())
        document_code = f"DCP-{run_date.isoformat()}-{document_id[:8]}"
        
        self.logger.info(f"Building daily compliance pack: {document_code}")
        
        sections = []
        section_order = 0
        
        # Section 1: Header
        section_order += 1
        sections.append(DocumentSection(
            section_id=str(uuid.uuid4()),
            section_order=section_order,
            section_type=SectionType.HEADER,
            title="Daily Compliance Pack",
            content={
                "document_code": document_code,
                "date": run_date.isoformat(),
                "run_code": run_summary.get("run_code", "N/A"),
                "snapshot_id": run_summary.get("snowflake_snapshot_id", "N/A"),
                "generated_at": datetime.now(timezone.utc).isoformat(),
            },
        ))
        
        # Section 2: Executive Summary (Deterministic)
        section_order += 1
        pass_rate = run_summary.get("controls_passed", 0) / max(run_summary.get("total_controls", 1), 1) * 100
        sections.append(DocumentSection(
            section_id=str(uuid.uuid4()),
            section_order=section_order,
            section_type=SectionType.DETERMINISTIC,
            title="Executive Summary",
            content={
                "total_controls": run_summary.get("total_controls", 0),
                "controls_passed": run_summary.get("controls_passed", 0),
                "controls_failed": run_summary.get("controls_failed", 0),
                "controls_warning": run_summary.get("controls_warning", 0),
                "pass_rate": f"{pass_rate:.1f}%",
                "exceptions_opened": len(exceptions),
                "exceptions_outstanding": len(outstanding_exceptions),
                "critical_exceptions": len([e for e in outstanding_exceptions if e.get("severity") == "critical"]),
            },
            evidence_ids=[run_summary.get("run_id", "")],
        ))
        
        # Section 3: Control Results Table (Deterministic)
        section_order += 1
        failed_results = [r for r in control_results if r.get("result_status") == "fail"]
        sections.append(DocumentSection(
            section_id=str(uuid.uuid4()),
            section_order=section_order,
            section_type=SectionType.TABLE,
            title="Control Failures Summary",
            content="Table of control failures for the day",
            column_headers=["Control Code", "Control Name", "Category", "Calculated Value", "Threshold", "Breach Amount"],
            table_data=[
                [
                    r.get("control_code", ""),
                    r.get("control_name", ""),
                    r.get("control_category", ""),
                    f"{r.get('calculated_value', 0):.4f}",
                    f"{r.get('threshold_operator', '')} {r.get('threshold_value', '')}",
                    f"{r.get('breach_amount', 0):.4f}" if r.get('breach_amount') else "N/A",
                ]
                for r in failed_results
            ],
            evidence_ids=[r.get("result_id", "") for r in failed_results],
        ))
        
        # Section 4: Exceptions Table (Deterministic)
        section_order += 1
        sections.append(DocumentSection(
            section_id=str(uuid.uuid4()),
            section_order=section_order,
            section_type=SectionType.TABLE,
            title="New Exceptions",
            content="Exceptions opened today",
            column_headers=["Exception Code", "Control", "Severity", "Title", "Status", "Due Date"],
            table_data=[
                [
                    e.get("exception_code", ""),
                    e.get("control_code", ""),
                    e.get("severity", "").upper(),
                    e.get("title", "")[:50],
                    e.get("status", ""),
                    str(e.get("due_date", "N/A")),
                ]
                for e in exceptions
            ],
            evidence_ids=[e.get("exception_id", "") for e in exceptions],
        ))
        
        # Section 5: Outstanding Exceptions Table (Deterministic)
        section_order += 1
        sections.append(DocumentSection(
            section_id=str(uuid.uuid4()),
            section_order=section_order,
            section_type=SectionType.TABLE,
            title="Outstanding Exceptions",
            content="All currently open exceptions",
            column_headers=["Exception Code", "Control", "Severity", "Status", "Due Date", "Days Open"],
            table_data=[
                [
                    e.get("exception_code", ""),
                    e.get("control_code", ""),
                    e.get("severity", "").upper(),
                    e.get("status", ""),
                    str(e.get("due_date", "N/A")),
                    str(e.get("days_open", "N/A")),
                ]
                for e in outstanding_exceptions
            ],
            evidence_ids=[e.get("exception_id", "") for e in outstanding_exceptions],
        ))
        
        # Section 6: Narrative Summary (LLM if available)
        if narrative:
            section_order += 1
            citations = self._extract_citations(narrative)
            sections.append(DocumentSection(
                section_id=str(uuid.uuid4()),
                section_order=section_order,
                section_type=SectionType.LLM_NARRATIVE,
                title="Compliance Summary Narrative",
                content=narrative,
                narrative_id=narrative_metadata.get("narrative_id") if narrative_metadata else None,
                citations=citations,
            ))
        
        # Section 7: Detailed Results by Category (Deterministic)
        categories = {}
        for r in control_results:
            cat = r.get("control_category", "other")
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(r)
        
        for cat, results in sorted(categories.items()):
            section_order += 1
            sections.append(DocumentSection(
                section_id=str(uuid.uuid4()),
                section_order=section_order,
                section_type=SectionType.TABLE,
                title=f"Detailed Results: {cat.replace('_', ' ').title()}",
                content=f"All control results for {cat} category",
                column_headers=["Control", "Status", "Value", "Threshold"],
                table_data=[
                    [
                        r.get("control_code", ""),
                        r.get("result_status", "").upper(),
                        f"{r.get('calculated_value', 'N/A')}",
                        f"{r.get('threshold_operator', '')} {r.get('threshold_value', '')}",
                    ]
                    for r in results
                ],
                evidence_ids=[r.get("result_id", "") for r in results],
            ))
        
        # Section 8: Audit Appendix
        section_order += 1
        sections.append(DocumentSection(
            section_id=str(uuid.uuid4()),
            section_order=section_order,
            section_type=SectionType.APPENDIX,
            title="Audit Information",
            content={
                "run_id": run_summary.get("run_id", "N/A"),
                "run_code": run_summary.get("run_code", "N/A"),
                "snapshot_id": run_summary.get("snowflake_snapshot_id", "N/A"),
                "snapshot_timestamp": run_summary.get("snowflake_snapshot_ts", "N/A"),
                "config_hash": run_summary.get("config_hash", "N/A"),
                "executor_service": run_summary.get("executor_service", "N/A"),
                "executor_version": run_summary.get("executor_version", "N/A"),
                "run_start": str(run_summary.get("run_timestamp_start", "N/A")),
                "run_end": str(run_summary.get("run_timestamp_end", "N/A")),
                "total_evidence_ids": len(set(
                    eid for s in sections for eid in s.evidence_ids if eid
                )),
            },
        ))
        
        # Build PDF
        pdf_bytes = self._render_pdf(sections, DocumentType.DAILY_COMPLIANCE_PACK)
        
        # Create metadata
        metadata = DocumentMetadata(
            document_id=document_id,
            document_code=document_code,
            document_type=DocumentType.DAILY_COMPLIANCE_PACK,
            document_date=run_date,
            run_id=run_summary.get("run_id"),
            snapshot_id=run_summary.get("snowflake_snapshot_id"),
            template_id="DCP-TEMPLATE-001",
            template_version="1.0.0",
            template_hash=self._get_template_hash("DCP-TEMPLATE-001"),
            llm_model_id=narrative_metadata.get("model_id") if narrative_metadata else None,
            llm_model_version=narrative_metadata.get("model_version") if narrative_metadata else None,
            llm_tokens_used=narrative_metadata.get("tokens_used", 0) if narrative_metadata else 0,
        )
        
        document = GeneratedDocument(
            metadata=metadata,
            sections=sections,
            pdf_bytes=pdf_bytes,
        )
        
        # Update metadata with hash
        document.metadata.document_hash = document.document_hash
        
        self.logger.info(
            f"Daily compliance pack generated: {document_code}, "
            f"hash={document.document_hash[:16]}..."
        )
        
        return document
    
    def build_form_pf_workpaper(
        self,
        period_end: date,
        fund_id: str,
        metrics: Dict[str, Any],
        control_results: List[Dict[str, Any]],
        narrative: Optional[str] = None,
        narrative_metadata: Optional[Dict[str, Any]] = None,
    ) -> GeneratedDocument:
        """
        Build Form PF filing workpaper.
        
        Structure:
        1. Cover page
        2. Liquidity profile (Form PF Q22)
        3. Leverage metrics (Form PF Q26)
        4. Counterparty exposure (Form PF Q29)
        5. Geographic exposure (Form PF Q22)
        6. Control validation summary
        7. Methodology narrative
        8. Audit appendix
        """
        document_id = str(uuid.uuid4())
        document_code = f"PF-{period_end.isoformat()}-{fund_id}-{document_id[:8]}"
        
        self.logger.info(f"Building Form PF workpaper: {document_code}")
        
        sections = []
        section_order = 0
        
        # Section 1: Header
        section_order += 1
        sections.append(DocumentSection(
            section_id=str(uuid.uuid4()),
            section_order=section_order,
            section_type=SectionType.HEADER,
            title="Form PF Filing Workpaper",
            content={
                "document_code": document_code,
                "period_end": period_end.isoformat(),
                "fund_id": fund_id,
                "filing_type": "Form PF",
                "generated_at": datetime.now(timezone.utc).isoformat(),
            },
        ))
        
        # Section 2: Liquidity Profile (Q22) - SEC requires 7 specific time buckets
        section_order += 1
        liquidity_data = metrics.get("liquidity_buckets", [])
        # SEC Form PF Q22 required buckets: 1 day, 2-7 days, 8-30 days, 31-90 days, 91-180 days, 181-365 days, >365 days
        sections.append(DocumentSection(
            section_id=str(uuid.uuid4()),
            section_order=section_order,
            section_type=SectionType.TABLE,
            title="Question 22: Portfolio Liquidity Profile",
            content="Percentage of portfolio that can be liquidated within each time period (SEC required buckets)",
            column_headers=["Time Horizon", "Long Exposure ($)", "Long % NAV", "Short Exposure ($)", "Short % NAV"],
            table_data=[
                [
                    bucket.get("bucket", ""),
                    f"${bucket.get('long_value', 0):,.0f}",
                    f"{bucket.get('long_pct', 0):.1f}%",
                    f"${bucket.get('short_value', 0):,.0f}",
                    f"{bucket.get('short_pct', 0):.1f}%",
                ]
                for bucket in liquidity_data
            ],
        ))
        
        # Section 3: Leverage Metrics (Q26)
        section_order += 1
        leverage_data = metrics.get("leverage", {})
        sections.append(DocumentSection(
            section_id=str(uuid.uuid4()),
            section_order=section_order,
            section_type=SectionType.DETERMINISTIC,
            title="Question 26: Leverage",
            content={
                "gross_leverage_ratio": f"{leverage_data.get('gross_leverage', 0):.2f}x",
                "net_leverage_ratio": f"{leverage_data.get('net_leverage', 0):.2f}x",
                "borrowing_to_nav": f"{leverage_data.get('borrowing_to_nav', 0):.1f}%",
                "derivatives_notional": f"${leverage_data.get('derivatives_notional', 0):,.0f}",
                "gav_to_nav": f"{leverage_data.get('gav_to_nav', 0):.2f}x",
            },
        ))
        
        # Section 4: Counterparty Exposure (Q29)
        section_order += 1
        counterparty_data = metrics.get("counterparty_exposure", [])
        sections.append(DocumentSection(
            section_id=str(uuid.uuid4()),
            section_order=section_order,
            section_type=SectionType.TABLE,
            title="Question 29: Counterparty Credit Exposure",
            content="Top counterparty exposures",
            column_headers=["Counterparty", "Type", "Net Exposure ($)", "% NAV", "Credit Rating"],
            table_data=[
                [
                    cp.get("name", ""),
                    cp.get("type", ""),
                    f"${cp.get('net_exposure', 0):,.0f}",
                    f"{cp.get('pct_nav', 0):.1f}%",
                    cp.get("credit_rating", "N/A"),
                ]
                for cp in counterparty_data[:10]  # Top 10
            ],
        ))
        
        # Section 5: Geographic Exposure
        section_order += 1
        geo_data = metrics.get("geographic", [])
        sections.append(DocumentSection(
            section_id=str(uuid.uuid4()),
            section_order=section_order,
            section_type=SectionType.TABLE,
            title="Geographic Exposure",
            content="Exposure by country/region",
            column_headers=["Country", "Region", "Exposure ($)", "% NAV"],
            table_data=[
                [
                    geo.get("country", ""),
                    geo.get("region", ""),
                    f"${geo.get('exposure', 0):,.0f}",
                    f"{geo.get('pct_nav', 0):.1f}%",
                ]
                for geo in geo_data
            ],
        ))
        
        # Section 6: Control Validation
        section_order += 1
        pf_controls = [r for r in control_results if r.get("control_category") in ("liquidity", "leverage", "counterparty")]
        sections.append(DocumentSection(
            section_id=str(uuid.uuid4()),
            section_order=section_order,
            section_type=SectionType.TABLE,
            title="Control Validation Results",
            content="Relevant compliance controls for Form PF",
            column_headers=["Control", "Category", "Status", "Value", "Threshold"],
            table_data=[
                [
                    r.get("control_code", ""),
                    r.get("control_category", ""),
                    r.get("result_status", "").upper(),
                    f"{r.get('calculated_value', 'N/A')}",
                    f"{r.get('threshold_operator', '')} {r.get('threshold_value', '')}",
                ]
                for r in pf_controls
            ],
            evidence_ids=[r.get("result_id", "") for r in pf_controls],
        ))
        
        # Section 7: Methodology (LLM if available)
        if narrative:
            section_order += 1
            citations = self._extract_citations(narrative)
            sections.append(DocumentSection(
                section_id=str(uuid.uuid4()),
                section_order=section_order,
                section_type=SectionType.LLM_NARRATIVE,
                title="Methodology and Basis of Preparation",
                content=narrative,
                narrative_id=narrative_metadata.get("narrative_id") if narrative_metadata else None,
                citations=citations,
            ))
        
        # Section 8: Audit Appendix
        section_order += 1
        sections.append(DocumentSection(
            section_id=str(uuid.uuid4()),
            section_order=section_order,
            section_type=SectionType.APPENDIX,
            title="Audit Information",
            content={
                "period_end": period_end.isoformat(),
                "fund_id": fund_id,
                "data_snapshot": metrics.get("snapshot_id", "N/A"),
                "calculation_date": datetime.now(timezone.utc).isoformat(),
                "control_count": len(pf_controls),
                "controls_passed": len([r for r in pf_controls if r.get("result_status") == "pass"]),
            },
        ))
        
        # Build PDF
        pdf_bytes = self._render_pdf(sections, DocumentType.FORM_PF_WORKPAPER)
        
        # Create metadata
        metadata = DocumentMetadata(
            document_id=document_id,
            document_code=document_code,
            document_type=DocumentType.FORM_PF_WORKPAPER,
            document_date=period_end,
            snapshot_id=metrics.get("snapshot_id"),
            template_id="PF-TEMPLATE-001",
            template_version="1.0.0",
            template_hash=self._get_template_hash("PF-TEMPLATE-001"),
            llm_model_id=narrative_metadata.get("model_id") if narrative_metadata else None,
            llm_model_version=narrative_metadata.get("model_version") if narrative_metadata else None,
            llm_tokens_used=narrative_metadata.get("tokens_used", 0) if narrative_metadata else 0,
        )
        
        document = GeneratedDocument(
            metadata=metadata,
            sections=sections,
            pdf_bytes=pdf_bytes,
        )
        
        document.metadata.document_hash = document.document_hash
        
        return document
    
    def build_13f_workpaper(
        self,
        period_end: date,
        holdings: List[Dict[str, Any]],
        summary: Dict[str, Any],
    ) -> GeneratedDocument:
        """
        Build 13F filing workpaper.
        """
        document_id = str(uuid.uuid4())
        document_code = f"13F-{period_end.isoformat()}-{document_id[:8]}"
        
        sections = []
        section_order = 0
        
        # Header
        section_order += 1
        sections.append(DocumentSection(
            section_id=str(uuid.uuid4()),
            section_order=section_order,
            section_type=SectionType.HEADER,
            title="Form 13F Filing Workpaper",
            content={
                "document_code": document_code,
                "period_end": period_end.isoformat(),
                "filing_type": "Form 13F-HR",
                "generated_at": datetime.now(timezone.utc).isoformat(),
            },
        ))
        
        # Summary
        section_order += 1
        sections.append(DocumentSection(
            section_id=str(uuid.uuid4()),
            section_order=section_order,
            section_type=SectionType.DETERMINISTIC,
            title="Filing Summary",
            content={
                "total_positions": summary.get("total_positions", 0),
                "total_value": f"${summary.get('total_value', 0):,.0f}",
                "distinct_issuers": summary.get("distinct_issuers", 0),
                "13f_securities": summary.get("13f_count", 0),
            },
        ))
        
        # Holdings table - SEC Form 13F required columns
        section_order += 1
        sections.append(DocumentSection(
            section_id=str(uuid.uuid4()),
            section_order=section_order,
            section_type=SectionType.TABLE,
            title="13F Reportable Holdings",
            content="Securities reportable on Form 13F (values in thousands)",
            column_headers=["CUSIP", "Issuer", "Class", "Shares/PRN", "Value (x$1000)", "Discretion", "Voting Auth (Sole/Shared/None)"],
            table_data=[
                [
                    h.get("cusip", ""),
                    h.get("issuer_name", "")[:25],
                    h.get("security_class", "COM")[:10],
                    f"{h.get('shares', 0):,.0f}",
                    f"{h.get('value', 0) / 1000:,.0f}",  # SEC requires value in thousands
                    h.get("investment_discretion", "SOLE"),
                    f"{h.get('voting_sole', 0):,.0f}/{h.get('voting_shared', 0):,.0f}/{h.get('voting_none', 0):,.0f}",
                ]
                for h in holdings
            ],
        ))
        
        # Audit appendix
        section_order += 1
        sections.append(DocumentSection(
            section_id=str(uuid.uuid4()),
            section_order=section_order,
            section_type=SectionType.APPENDIX,
            title="Audit Information",
            content={
                "period_end": period_end.isoformat(),
                "data_snapshot": summary.get("snapshot_id", "N/A"),
                "position_count": len(holdings),
            },
        ))
        
        pdf_bytes = self._render_pdf(sections, DocumentType.THIRTEENF_WORKPAPER)
        
        metadata = DocumentMetadata(
            document_id=document_id,
            document_code=document_code,
            document_type=DocumentType.THIRTEENF_WORKPAPER,
            document_date=period_end,
            snapshot_id=summary.get("snapshot_id"),
            template_id="13F-TEMPLATE-001",
            template_version="1.0.0",
            template_hash=self._get_template_hash("13F-TEMPLATE-001"),
        )
        
        document = GeneratedDocument(
            metadata=metadata,
            sections=sections,
            pdf_bytes=pdf_bytes,
        )
        
        document.metadata.document_hash = document.document_hash
        
        return document
    
    def _render_pdf(
        self,
        sections: List[DocumentSection],
        doc_type: DocumentType,
    ) -> bytes:
        """
        Render sections to PDF bytes.
        
        Uses ReportLab for PDF generation.
        """
        try:
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import letter
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.platypus import (
                SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                PageBreak, HRFlowable
            )
            from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
        except ImportError:
            # If ReportLab not available, return placeholder
            self.logger.warning("ReportLab not available, returning placeholder PDF")
            return self._render_placeholder_pdf(sections, doc_type)
        
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72,
        )
        
        styles = getSampleStyleSheet()
        
        # Custom styles
        styles.add(ParagraphStyle(
            name='DocTitle',
            parent=styles['Heading1'],
            fontSize=18,
            alignment=TA_CENTER,
            spaceAfter=20,
        ))
        
        styles.add(ParagraphStyle(
            name='SectionTitle',
            parent=styles['Heading2'],
            fontSize=14,
            spaceBefore=15,
            spaceAfter=10,
        ))
        
        styles.add(ParagraphStyle(
            name='Citation',
            parent=styles['Normal'],
            fontSize=8,
            textColor=colors.grey,
            leftIndent=20,
        ))
        
        story = []
        
        for section in sections:
            if section.section_type == SectionType.HEADER:
                # Title
                story.append(Paragraph(section.title, styles['DocTitle']))
                
                # Header info
                if isinstance(section.content, dict):
                    for key, value in section.content.items():
                        story.append(Paragraph(
                            f"<b>{key.replace('_', ' ').title()}:</b> {value}",
                            styles['Normal']
                        ))
                
                story.append(HRFlowable(width="100%", thickness=1, color=colors.black))
                story.append(Spacer(1, 20))
                
            elif section.section_type == SectionType.DETERMINISTIC:
                story.append(Paragraph(section.title, styles['SectionTitle']))
                
                if isinstance(section.content, dict):
                    for key, value in section.content.items():
                        story.append(Paragraph(
                            f"<b>{key.replace('_', ' ').title()}:</b> {value}",
                            styles['Normal']
                        ))
                else:
                    story.append(Paragraph(str(section.content), styles['Normal']))
                
                story.append(Spacer(1, 10))
                
            elif section.section_type == SectionType.TABLE:
                story.append(Paragraph(section.title, styles['SectionTitle']))
                
                if section.column_headers and section.table_data:
                    table_data = [section.column_headers] + section.table_data
                    
                    t = Table(table_data, repeatRows=1)
                    t.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, 0), 9),
                        ('FONTSIZE', (0, 1), (-1, -1), 8),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
                        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
                    ]))
                    
                    story.append(t)
                
                story.append(Spacer(1, 10))
                
            elif section.section_type == SectionType.LLM_NARRATIVE:
                story.append(Paragraph(section.title, styles['SectionTitle']))
                story.append(Paragraph(
                    "<i>[LLM-Generated Content - See citations below]</i>",
                    styles['Citation']
                ))
                story.append(Spacer(1, 5))
                
                # Split content into paragraphs
                paragraphs = section.content.split('\n\n')
                for para in paragraphs:
                    if para.strip():
                        story.append(Paragraph(para.strip(), styles['Normal']))
                        story.append(Spacer(1, 5))
                
                # Add citations
                if section.citations:
                    story.append(Spacer(1, 5))
                    story.append(Paragraph("<b>Citations:</b>", styles['Citation']))
                    for citation in section.citations:
                        story.append(Paragraph(citation, styles['Citation']))
                
                story.append(Spacer(1, 10))
                
            elif section.section_type == SectionType.APPENDIX:
                story.append(PageBreak())
                story.append(Paragraph(section.title, styles['SectionTitle']))
                story.append(Paragraph(
                    "<i>This section contains audit and traceability information.</i>",
                    styles['Citation']
                ))
                story.append(Spacer(1, 10))
                
                if isinstance(section.content, dict):
                    for key, value in section.content.items():
                        story.append(Paragraph(
                            f"<b>{key.replace('_', ' ').title()}:</b> {value}",
                            styles['Normal']
                        ))
        
        # Build PDF
        doc.build(story)
        
        pdf_bytes = buffer.getvalue()
        buffer.close()
        
        return pdf_bytes
    
    def _render_placeholder_pdf(
        self,
        sections: List[DocumentSection],
        doc_type: DocumentType,
    ) -> bytes:
        """Render a simple placeholder when ReportLab is not available."""
        # Create a minimal valid PDF
        content = [
            "%PDF-1.4",
            "1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj",
            "2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj",
            "3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >> endobj",
            "4 0 obj << /Length 44 >> stream",
            "BT /F1 12 Tf 100 700 Td (Compliance Document) Tj ET",
            "endstream endobj",
            "xref",
            "0 5",
            "0000000000 65535 f",
            "0000000009 00000 n",
            "0000000058 00000 n",
            "0000000115 00000 n",
            "0000000206 00000 n",
            "trailer << /Size 5 /Root 1 0 R >>",
            "startxref",
            "300",
            "%%EOF",
        ]
        return "\n".join(content).encode('latin-1')
    
    def _extract_citations(self, text: str) -> List[str]:
        """Extract citation strings from text."""
        import re
        pattern = r'\[(?:ControlRun|ControlResult|Exception|Policy|Filing):[^\]]+\]'
        return re.findall(pattern, text)
    
    def _get_template_hash(self, template_id: str) -> str:
        """Get hash for a template ID."""
        # In production, this would load the actual template and hash it
        return hashlib.sha256(f"{template_id}:1.0.0".encode()).hexdigest()
