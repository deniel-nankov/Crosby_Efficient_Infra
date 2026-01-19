"""
Professional Finance PDF Generator - Clean & Condensed

Creates institutional-quality compliance documents:
- Times New Roman font throughout
- Clean, simple design
- Condensed to 1-2 pages maximum
- Professional but understated
"""

import io
import hashlib
from datetime import datetime, timezone, date
from typing import List, Dict, Any, Optional
from decimal import Decimal

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        HRFlowable, Flowable
    )
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


class ProfessionalCompliancePDF:
    """
    Generates clean, professional compliance PDFs.
    
    - Times New Roman font
    - Simple black/white design
    - 1-2 pages maximum
    - Condensed information
    """
    
    def __init__(
        self,
        fund_name: str = "Crosby Capital Management",
        fund_id: str = "CCM-001",
        confidentiality: str = "CONFIDENTIAL"
    ):
        self.fund_name = fund_name
        self.fund_id = fund_id
        self.confidentiality = confidentiality
        self._setup_styles()
    
    def _setup_styles(self):
        """Set up clean Times New Roman styles."""
        self.styles = getSampleStyleSheet()
        
        # Use Times-Roman (built-in PDF font, equivalent to Times New Roman)
        base_font = 'Times-Roman'
        bold_font = 'Times-Bold'
        italic_font = 'Times-Italic'
        
        # Document title
        self.styles.add(ParagraphStyle(
            name='DocTitle',
            fontName=bold_font,
            fontSize=16,
            alignment=TA_CENTER,
            spaceAfter=6,
            textColor=colors.black,
        ))
        
        # Subtitle
        self.styles.add(ParagraphStyle(
            name='DocSubtitle',
            fontName=base_font,
            fontSize=11,
            alignment=TA_CENTER,
            spaceAfter=12,
            textColor=colors.Color(0.3, 0.3, 0.3),
        ))
        
        # Section header
        self.styles.add(ParagraphStyle(
            name='SectionHead',
            fontName=bold_font,
            fontSize=11,
            spaceBefore=12,
            spaceAfter=6,
            textColor=colors.black,
        ))
        
        # Body text
        self.styles.add(ParagraphStyle(
            name='Body',
            fontName=base_font,
            fontSize=10,
            alignment=TA_JUSTIFY,
            spaceBefore=3,
            spaceAfter=3,
            leading=12,
            textColor=colors.black,
        ))
        
        # Small text for footer/metadata
        self.styles.add(ParagraphStyle(
            name='Small',
            fontName=base_font,
            fontSize=8,
            textColor=colors.Color(0.4, 0.4, 0.4),
        ))
    
    def _header_footer(self, canvas, doc):
        """Simple header and footer."""
        canvas.saveState()
        page_width, page_height = letter
        
        # Header line
        canvas.setStrokeColor(colors.black)
        canvas.setLineWidth(0.5)
        canvas.line(72, page_height - 50, page_width - 72, page_height - 50)
        
        # Fund name (left)
        canvas.setFont("Times-Bold", 10)
        canvas.drawString(72, page_height - 42, self.fund_name)
        
        # Confidentiality (right)
        canvas.setFont("Times-Roman", 9)
        canvas.drawRightString(page_width - 72, page_height - 42, self.confidentiality)
        
        # Footer line
        canvas.line(72, 45, page_width - 72, 45)
        
        # Page number (center)
        canvas.setFont("Times-Roman", 9)
        canvas.drawCentredString(page_width / 2, 32, f"Page {doc.page}")
        
        canvas.restoreState()
    
    def _simple_table(self, headers: List[str], data: List[List[Any]], 
                      col_widths: List[float] = None) -> Table:
        """Create a simple, clean table."""
        table_data = [headers] + data
        
        t = Table(table_data, colWidths=col_widths, repeatRows=1)
        t.setStyle(TableStyle([
            # Header
            ('FONTNAME', (0, 0), (-1, 0), 'Times-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
            ('TOPPADDING', (0, 0), (-1, 0), 6),
            ('LINEBELOW', (0, 0), (-1, 0), 1, colors.black),
            
            # Body
            ('FONTNAME', (0, 1), (-1, -1), 'Times-Roman'),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 4),
            ('TOPPADDING', (0, 1), (-1, -1), 4),
            ('LINEBELOW', (0, -1), (-1, -1), 0.5, colors.black),
            
            # Alignment
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ]))
        return t
    
    def generate_daily_compliance_report(
        self,
        report_date: date,
        nav: float,
        positions: List[Dict[str, Any]],
        control_results: List[Dict[str, Any]],
        narrative: str,
        snapshot_id: str = None,
        document_id: str = None,
    ) -> bytes:
        """
        Generate a clean, condensed Daily Compliance Report.
        
        Target: 1-2 pages maximum.
        """
        if not REPORTLAB_AVAILABLE:
            raise ImportError("ReportLab is required")
        
        buffer = io.BytesIO()
        
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=65,
            bottomMargin=55,
        )
        
        story = []
        
        # ===== TITLE =====
        story.append(Paragraph("Daily Compliance Report", self.styles['DocTitle']))
        story.append(Paragraph(
            f"{report_date.strftime('%B %d, %Y')}",
            self.styles['DocSubtitle']
        ))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.black))
        story.append(Spacer(1, 10))
        
        # ===== SUMMARY METRICS (condensed) =====
        total = len(control_results)
        passed = len([c for c in control_results if c.get('status', '').lower() in ('pass', 'passed')])
        warnings = len([c for c in control_results if c.get('status', '').lower() in ('warn', 'warning')])
        failed = len([c for c in control_results if c.get('status', '').lower() in ('fail', 'failed')])
        pass_rate = (passed / total * 100) if total > 0 else 0
        
        nav_str = f"${nav/1e9:.2f}B" if nav >= 1e9 else f"${nav/1e6:.0f}M"
        
        # Overall status
        if failed > 0:
            status_text = "BREACH"
        elif warnings > 0:
            status_text = "WARNING"
        else:
            status_text = "COMPLIANT"
        
        story.append(Paragraph("<b>Executive Summary</b>", self.styles['SectionHead']))
        
        # First line: key metrics
        summary_text = (
            f"Net Asset Value: <b>{nav_str}</b> | "
            f"Positions: <b>{len(positions)}</b> | "
            f"Controls Tested: <b>{total}</b> | "
            f"Status: <b>{status_text}</b>"
        )
        story.append(Paragraph(summary_text, self.styles['Body']))
        
        # Second line: control breakdown
        breakdown_text = (
            f"Results: <b>{passed}</b> passed, <b>{warnings}</b> warnings, <b>{failed}</b> breaches "
            f"({pass_rate:.0f}% pass rate)"
        )
        story.append(Paragraph(breakdown_text, self.styles['Body']))
        story.append(Spacer(1, 8))
        
        # ===== CONTROL RESULTS TABLE =====
        story.append(Paragraph("<b>Control Testing Results</b>", self.styles['SectionHead']))
        
        # Sort: failures first, then warnings, then passes
        def status_key(c):
            s = c.get('status', '').lower()
            if s in ('fail', 'failed'): return 0
            if s in ('warn', 'warning'): return 1
            return 2
        
        sorted_controls = sorted(control_results, key=status_key)
        
        control_data = []
        for c in sorted_controls:
            status = c.get('status', 'N/A').upper()
            if status in ('PASS', 'PASSED'):
                status_display = 'Pass'
            elif status in ('WARN', 'WARNING'):
                status_display = 'Warning'
            elif status in ('FAIL', 'FAILED'):
                status_display = 'FAIL'
            else:
                status_display = status
            
            val = c.get('current_value', c.get('calculated_value', 0))
            if isinstance(val, (int, float, Decimal)):
                val_str = f"{float(val):.1%}"
            else:
                val_str = str(val)
            
            # Include control category (type)
            control_type = c.get('control_type', '')
            if control_type:
                type_abbrev = control_type[:3].upper()  # LIQ, CON, EXP
            else:
                type_abbrev = '—'
            
            control_data.append([
                c.get('control_id', c.get('control_code', 'N/A')),
                type_abbrev,
                c.get('control_name', c.get('description', 'N/A'))[:30],
                val_str,
                str(c.get('threshold', c.get('threshold_value', 'N/A'))),
                status_display,
            ])
        
        story.append(self._simple_table(
            ["Control", "Type", "Description", "Actual", "Limit", "Status"],
            control_data,
            col_widths=[0.85*inch, 0.4*inch, 1.9*inch, 0.65*inch, 0.9*inch, 0.6*inch]
        ))
        story.append(Spacer(1, 10))
        
        # ===== COMMENTARY (condensed) =====
        story.append(Paragraph("<b>Commentary</b>", self.styles['SectionHead']))
        
        # Condense narrative - include breach amounts where relevant
        if warnings > 0 or failed > 0:
            issues = []
            for c in sorted_controls:
                s = c.get('status', '').lower()
                breach = c.get('breach_amount')
                if s in ('warn', 'warning'):
                    issues.append(f"• {c.get('control_name', 'Control')}: approaching limit")
                elif s in ('fail', 'failed'):
                    if breach:
                        issues.append(f"• {c.get('control_name', 'Control')}: exceeded by {breach:.2%}")
                    else:
                        issues.append(f"• {c.get('control_name', 'Control')}: LIMIT EXCEEDED")
            
            story.append(Paragraph(
                f"Compliance review identified {warnings} warning(s) and {failed} breach(es):",
                self.styles['Body']
            ))
            for issue in issues[:5]:  # Max 5 items
                story.append(Paragraph(issue, self.styles['Body']))
        else:
            story.append(Paragraph(
                "All controls passed within acceptable thresholds. No exceptions to report.",
                self.styles['Body']
            ))
        
        story.append(Spacer(1, 10))
        
        # ===== TOP POSITIONS (condensed - top 5 only) =====
        story.append(Paragraph("<b>Top Holdings</b>", self.styles['SectionHead']))
        
        def get_val(p):
            v = p.get('market_value', p.get('value', 0))
            if isinstance(v, str):
                v = float(v.replace('$', '').replace(',', ''))
            return abs(v) if v else 0
        
        top_positions = sorted(positions, key=get_val, reverse=True)[:5]
        
        pos_data = []
        for p in top_positions:
            val = p.get('market_value', p.get('value', 0))
            if isinstance(val, (int, float, Decimal)):
                val_str = f"${float(val):,.0f}"
                pct = (float(val) / nav * 100) if nav > 0 else 0
                pct_str = f"{pct:.1f}%"
            else:
                val_str = str(val)
                pct_str = "—"
            
            # Use ticker if available, else security_id
            sec_id = p.get('ticker') or p.get('security_id', 'N/A')
            sector = p.get('sector') or p.get('asset_class', '—')
            
            pos_data.append([
                sec_id[:10],
                p.get('security_name', p.get('name', 'N/A'))[:22],
                sector[:8] if sector else '—',
                val_str,
                pct_str,
            ])
        
        story.append(self._simple_table(
            ["Ticker", "Name", "Sector", "Value", "% NAV"],
            pos_data,
            col_widths=[0.7*inch, 1.9*inch, 0.7*inch, 1.3*inch, 0.7*inch]
        ))
        story.append(Spacer(1, 15))
        
        # ===== DOCUMENT CONTROL (condensed footer) =====
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.black))
        story.append(Spacer(1, 6))
        
        doc_hash = hashlib.sha256(str(datetime.now()).encode()).hexdigest()[:16]
        
        footer_text = (
            f"<b>Document Control:</b> "
            f"ID: {document_id or 'DCP-' + report_date.strftime('%Y%m%d')} | "
            f"Snapshot: {snapshot_id or 'LIVE'} | "
            f"Hash: {doc_hash} | "
            f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
        )
        story.append(Paragraph(footer_text, self.styles['Small']))
        
        # Build PDF
        doc.build(story, onFirstPage=self._header_footer, onLaterPages=self._header_footer)
        
        pdf_bytes = buffer.getvalue()
        buffer.close()
        
        return pdf_bytes


def generate_professional_pdf(
    report_date: date,
    nav: float,
    positions: List[Dict[str, Any]],
    control_results: List[Dict[str, Any]],
    narrative: str,
    fund_name: str = "Crosby Capital Management",
    output_path: str = None,
) -> bytes:
    """Generate a professional compliance PDF."""
    builder = ProfessionalCompliancePDF(fund_name=fund_name)
    pdf_bytes = builder.generate_daily_compliance_report(
        report_date=report_date,
        nav=nav,
        positions=positions,
        control_results=control_results,
        narrative=narrative,
    )
    
    if output_path:
        with open(output_path, 'wb') as f:
            f.write(pdf_bytes)
    
    return pdf_bytes
