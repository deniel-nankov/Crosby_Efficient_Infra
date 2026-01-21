"""
Institutional Hedge Fund Compliance PDF Generator

Generates comprehensive, audit-ready compliance documents for:
- Multi-billion dollar hedge funds
- SEC-registered investment advisers
- CFTC-registered CPO/CTAs
- Global macro / multi-strategy funds

Features:
- Executive Dashboard with risk gauges
- Limit utilization heat maps
- Historical trend comparisons
- Exception aging and escalation tracking
- Multi-fund/sleeve support
- Regulatory filing support (Form PF, ADV, 13F)
- Digital signature blocks
- Complete audit trail
"""
from __future__ import annotations

import io
import hashlib
from datetime import datetime, timezone, date, timedelta
from decimal import Decimal
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json

# ReportLab imports
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch, mm
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        HRFlowable, PageBreak, KeepTogether, ListFlowable, ListItem,
        Image, Flowable
    )
    from reportlab.graphics.shapes import Drawing, Rect, String, Line, Circle, Wedge
    from reportlab.graphics.charts.barcharts import VerticalBarChart
    from reportlab.graphics.charts.linecharts import HorizontalLineChart
    from reportlab.graphics.charts.piecharts import Pie
    from reportlab.graphics import renderPDF
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


class RiskLevel(Enum):
    """Risk levels for color coding."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class LimitStatus(Enum):
    """Limit utilization status."""
    COMPLIANT = "compliant"
    WARNING = "warning"  # >80% utilized
    BREACH = "breach"
    WAIVER = "waiver"


@dataclass
class ControlResult:
    """Structured control result for reporting."""
    control_id: str
    control_name: str
    category: str
    current_value: float
    threshold_value: float
    threshold_type: str  # 'max', 'min', 'range'
    status: LimitStatus
    utilization_pct: float  # How close to limit (0-100+)
    headroom: float  # Remaining capacity
    prior_day_value: Optional[float] = None
    avg_30d_value: Optional[float] = None
    breach_count_ytd: int = 0
    last_breach_date: Optional[date] = None
    evidence_hash: str = ""
    

@dataclass
class ExceptionRecord:
    """Exception with full tracking."""
    exception_id: str
    control_id: str
    control_name: str
    breach_date: date
    breach_value: float
    threshold_value: float
    severity_pct: float  # How much over/under
    status: str  # 'open', 'escalated', 'remediated', 'waived'
    escalation_level: int  # 0=none, 1=PM, 2=CIO, 3=CCO, 4=Board
    age_days: int
    assigned_to: str
    remediation_deadline: Optional[date] = None
    remediation_plan: str = ""
    commentary: str = ""
    evidence_hash: str = ""


@dataclass  
class PositionDetail:
    """Position with full attribution."""
    security_id: str
    security_name: str
    ticker: str
    asset_class: str
    sector: str
    country: str
    currency: str
    quantity: float
    price: float
    market_value: float
    pct_nav: float
    pct_gross: float
    side: str  # 'long', 'short'
    strategy: str = ""
    sleeve: str = ""
    days_held: int = 0
    unrealized_pnl: float = 0.0
    contribution_to_var: float = 0.0


@dataclass
class FundMetrics:
    """Comprehensive fund metrics."""
    fund_name: str
    fund_id: str
    report_date: date
    
    # NAV & AUM
    nav: float
    nav_change_1d: float
    nav_change_mtd: float
    nav_change_ytd: float
    aum: float
    
    # Exposure
    gross_exposure: float
    gross_limit: float
    net_exposure: float
    net_limit_long: float
    net_limit_short: float
    
    # Concentration
    top_10_concentration: float
    single_name_limit: float
    largest_position_pct: float
    
    # Liquidity
    t1_liquidity: float
    t1_requirement: float
    t5_liquidity: float
    t30_liquidity: float
    
    # Risk
    var_95_1d: float
    var_99_1d: float
    beta_to_spx: float
    sharpe_ratio_ytd: float
    
    # Leverage
    regulatory_leverage: float
    economic_leverage: float
    margin_utilization: float
    
    # Optional fields with defaults (must come last)
    sector_concentration: Dict[str, float] = field(default_factory=dict)


class LimitGauge(Flowable):
    """
    Visual gauge showing limit utilization.
    
    ┌────────────────────────────────────────┐
    │ ████████████████░░░░░░░░░░  82%       │
    │ Sector Concentration - Tech            │
    │ Current: 28.5%  |  Limit: 35%          │
    └────────────────────────────────────────┘
    """
    
    def __init__(
        self, 
        label: str,
        current: float,
        limit: float,
        width: float = 200,
        height: float = 50,
        show_values: bool = True
    ):
        Flowable.__init__(self)
        self.label = label
        self.current = current
        self.limit = limit
        self.width = width
        self.height = height
        self.show_values = show_values
        self.utilization = min(current / limit * 100 if limit > 0 else 0, 120)
    
    def draw(self):
        # Determine color based on utilization
        if self.utilization >= 100:
            fill_color = colors.Color(0.8, 0.2, 0.2)  # Red
        elif self.utilization >= 80:
            fill_color = colors.Color(0.9, 0.6, 0.1)  # Amber
        else:
            fill_color = colors.Color(0.2, 0.6, 0.3)  # Green
        
        # Background bar
        bar_height = 12
        bar_y = self.height - 20
        self.canv.setFillColor(colors.Color(0.9, 0.9, 0.9))
        self.canv.rect(0, bar_y, self.width - 40, bar_height, fill=1, stroke=0)
        
        # Filled portion
        fill_width = min((self.width - 40) * self.utilization / 100, self.width - 40)
        self.canv.setFillColor(fill_color)
        self.canv.rect(0, bar_y, fill_width, bar_height, fill=1, stroke=0)
        
        # 80% warning line
        warn_x = (self.width - 40) * 0.8
        self.canv.setStrokeColor(colors.Color(0.5, 0.5, 0.5))
        self.canv.setDash(2, 2)
        self.canv.line(warn_x, bar_y, warn_x, bar_y + bar_height)
        self.canv.setDash()
        
        # Percentage text
        self.canv.setFillColor(colors.black)
        self.canv.setFont("Times-Bold", 10)
        self.canv.drawString(self.width - 35, bar_y + 2, f"{self.utilization:.0f}%")
        
        # Label
        self.canv.setFont("Times-Bold", 9)
        self.canv.drawString(0, bar_y - 14, self.label)
        
        # Values
        if self.show_values:
            self.canv.setFont("Times-Roman", 8)
            self.canv.setFillColor(colors.Color(0.4, 0.4, 0.4))
            self.canv.drawString(0, bar_y - 26, f"Current: {self.current:.1f}%  |  Limit: {self.limit:.1f}%")


class InstitutionalCompliancePDF:
    """
    Generates comprehensive institutional-grade compliance documents.
    
    Document Types:
    1. Daily Compliance Pack - Full daily report for CCO
    2. Executive Summary - 1-page for CIO/PM
    3. Exception Report - Detailed breach analysis
    4. Risk Committee Pack - Monthly board report
    5. Form PF Workpaper - SEC regulatory filing support
    """
    
    def __init__(
        self,
        fund_name: str = "Institutional Fund LP",
        fund_id: str = "FUND-001",
        adviser_name: str = "Asset Management LLC",
        adviser_crd: str = "123456",
        confidentiality: str = "CONFIDENTIAL - PROPRIETARY"
    ):
        if not REPORTLAB_AVAILABLE:
            raise ImportError("reportlab required: pip install reportlab")
        
        self.fund_name = fund_name
        self.fund_id = fund_id
        self.adviser_name = adviser_name
        self.adviser_crd = adviser_crd
        self.confidentiality = confidentiality
        self._setup_styles()
    
    def _setup_styles(self):
        """Configure professional typography."""
        self.styles = getSampleStyleSheet()
        
        # Primary title
        self.styles.add(ParagraphStyle(
            name='Title1',
            fontName='Times-Bold',
            fontSize=18,
            alignment=TA_CENTER,
            spaceAfter=4,
            textColor=colors.Color(0.1, 0.1, 0.3),
        ))
        
        # Document subtitle
        self.styles.add(ParagraphStyle(
            name='Subtitle',
            fontName='Times-Roman',
            fontSize=12,
            alignment=TA_CENTER,
            spaceAfter=12,
            textColor=colors.Color(0.3, 0.3, 0.3),
        ))
        
        # Section headers
        self.styles.add(ParagraphStyle(
            name='Section',
            fontName='Times-Bold',
            fontSize=12,
            spaceBefore=16,
            spaceAfter=8,
            textColor=colors.Color(0.1, 0.1, 0.3),
            borderPadding=(0, 0, 4, 0),
        ))
        
        # Subsection headers
        self.styles.add(ParagraphStyle(
            name='Subsection',
            fontName='Times-Bold',
            fontSize=10,
            spaceBefore=10,
            spaceAfter=4,
            textColor=colors.black,
        ))
        
        # Body text
        self.styles.add(ParagraphStyle(
            name='Body',
            fontName='Times-Roman',
            fontSize=9,
            alignment=TA_JUSTIFY,
            spaceBefore=2,
            spaceAfter=2,
            leading=12,
        ))
        
        # Narrative (LLM-generated)
        self.styles.add(ParagraphStyle(
            name='Narrative',
            fontName='Times-Italic',
            fontSize=9,
            alignment=TA_JUSTIFY,
            spaceBefore=4,
            spaceAfter=4,
            leading=12,
            leftIndent=10,
            rightIndent=10,
            borderColor=colors.Color(0.8, 0.8, 0.8),
            borderWidth=0.5,
            borderPadding=6,
        ))
        
        # Alert/Warning
        self.styles.add(ParagraphStyle(
            name='Alert',
            fontName='Times-Bold',
            fontSize=9,
            textColor=colors.Color(0.7, 0.2, 0.1),
            spaceBefore=4,
            spaceAfter=4,
        ))
        
        # Success
        self.styles.add(ParagraphStyle(
            name='Success',
            fontName='Times-Bold',
            fontSize=9,
            textColor=colors.Color(0.1, 0.5, 0.2),
            spaceBefore=4,
            spaceAfter=4,
        ))
        
        # Metadata/footer
        self.styles.add(ParagraphStyle(
            name='Meta',
            fontName='Times-Roman',
            fontSize=7,
            textColor=colors.Color(0.5, 0.5, 0.5),
        ))
        
        # Table header
        self.styles.add(ParagraphStyle(
            name='TableHeader',
            fontName='Times-Bold',
            fontSize=8,
            alignment=TA_CENTER,
        ))
    
    def _header_footer(self, canvas, doc):
        """Professional header and footer with fund info."""
        canvas.saveState()
        width, height = letter
        
        # === HEADER ===
        # Top border
        canvas.setStrokeColor(colors.Color(0.1, 0.1, 0.3))
        canvas.setLineWidth(1.5)
        canvas.line(50, height - 40, width - 50, height - 40)
        
        # Fund name (left)
        canvas.setFont("Times-Bold", 11)
        canvas.setFillColor(colors.Color(0.1, 0.1, 0.3))
        canvas.drawString(50, height - 32, self.fund_name)
        
        # Confidentiality (right)
        canvas.setFont("Times-Bold", 8)
        canvas.setFillColor(colors.Color(0.6, 0.1, 0.1))
        canvas.drawRightString(width - 50, height - 32, self.confidentiality)
        
        # === FOOTER ===
        # Bottom border
        canvas.setStrokeColor(colors.Color(0.7, 0.7, 0.7))
        canvas.setLineWidth(0.5)
        canvas.line(50, 40, width - 50, 40)
        
        # Adviser info (left)
        canvas.setFont("Times-Roman", 7)
        canvas.setFillColor(colors.Color(0.5, 0.5, 0.5))
        canvas.drawString(50, 28, f"{self.adviser_name} | CRD# {self.adviser_crd}")
        
        # Page number (center)
        canvas.drawCentredString(width / 2, 28, f"Page {doc.page}")
        
        # Timestamp (right)
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        canvas.drawRightString(width - 50, 28, f"Generated: {timestamp}")
        
        canvas.restoreState()
    
    def _create_status_table(
        self,
        headers: List[str],
        data: List[List[Any]],
        col_widths: Optional[List[float]] = None,
        status_col: Optional[int] = None,
    ) -> Table:
        """Create a table with optional status color coding."""
        table_data = [headers] + data
        
        t = Table(table_data, colWidths=col_widths, repeatRows=1)
        
        style_commands = [
            # Header styling
            ('FONTNAME', (0, 0), (-1, 0), 'Times-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 8),
            ('BACKGROUND', (0, 0), (-1, 0), colors.Color(0.9, 0.9, 0.95)),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.Color(0.1, 0.1, 0.3)),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
            ('TOPPADDING', (0, 0), (-1, 0), 6),
            ('LINEBELOW', (0, 0), (-1, 0), 1, colors.Color(0.1, 0.1, 0.3)),
            
            # Body styling
            ('FONTNAME', (0, 1), (-1, -1), 'Times-Roman'),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 4),
            ('TOPPADDING', (0, 1), (-1, -1), 4),
            ('LINEBELOW', (0, 1), (-1, -1), 0.5, colors.Color(0.8, 0.8, 0.8)),
            
            # Alignment
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            
            # Alternating rows
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.Color(0.97, 0.97, 0.97)]),
        ]
        
        # Color code status column if specified
        if status_col is not None:
            for row_idx, row in enumerate(data, start=1):
                if row_idx <= len(data):
                    status = str(row[status_col]).upper() if status_col < len(row) else ""
                    if 'PASS' in status or 'COMPLIANT' in status or '✓' in status:
                        style_commands.append(
                            ('TEXTCOLOR', (status_col, row_idx), (status_col, row_idx), 
                             colors.Color(0.1, 0.5, 0.2))
                        )
                    elif 'WARN' in status or '⚠' in status:
                        style_commands.append(
                            ('TEXTCOLOR', (status_col, row_idx), (status_col, row_idx),
                             colors.Color(0.8, 0.5, 0.1))
                        )
                    elif 'FAIL' in status or 'BREACH' in status or '✗' in status:
                        style_commands.append(
                            ('TEXTCOLOR', (status_col, row_idx), (status_col, row_idx),
                             colors.Color(0.7, 0.1, 0.1))
                        )
        
        t.setStyle(TableStyle(style_commands))
        return t
    
    def _create_kpi_box(
        self,
        label: str,
        value: str,
        change: Optional[str] = None,
        status: str = "neutral"
    ) -> Table:
        """Create a KPI display box."""
        if status == "good":
            value_color = colors.Color(0.1, 0.5, 0.2)
        elif status == "bad":
            value_color = colors.Color(0.7, 0.1, 0.1)
        elif status == "warning":
            value_color = colors.Color(0.8, 0.5, 0.1)
        else:
            value_color = colors.black
        
        # Create mini table for KPI
        content = [[
            Paragraph(f"<font size='7' color='#666666'>{label}</font>", self.styles['Body']),
        ], [
            Paragraph(f"<font size='12'><b>{value}</b></font>", self.styles['Body']),
        ]]
        
        if change:
            change_color = '#228B22' if change.startswith('+') else '#B22222' if change.startswith('-') else '#666666'
            content.append([
                Paragraph(f"<font size='7' color='{change_color}'>{change}</font>", self.styles['Body']),
            ])
        
        t = Table(content, colWidths=[90])
        t.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOX', (0, 0), (-1, -1), 0.5, colors.Color(0.8, 0.8, 0.8)),
            ('BACKGROUND', (0, 0), (-1, -1), colors.Color(0.98, 0.98, 0.98)),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        return t
    
    def generate_daily_compliance_pack(
        self,
        metrics: FundMetrics,
        control_results: List[ControlResult],
        exceptions: List[ExceptionRecord],
        positions: List[PositionDetail],
        narratives: Dict[str, str],
        document_hash: Optional[str] = None,
    ) -> bytes:
        """
        Generate comprehensive Daily Compliance Pack.
        
        Sections:
        1. Executive Dashboard
        2. Limit Utilization Summary
        3. Control Test Results (by category)
        4. Exception Details & Escalation
        5. Position Concentration Analysis
        6. Liquidity Profile
        7. Risk Metrics
        8. Evidence Appendix
        """
        buffer = io.BytesIO()
        
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=50,
            leftMargin=50,
            topMargin=55,
            bottomMargin=50,
        )
        
        story = []
        
        # ========== PAGE 1: EXECUTIVE DASHBOARD ==========
        story.append(Paragraph("Daily Compliance Pack", self.styles['Title1']))
        story.append(Paragraph(
            f"{metrics.report_date.strftime('%A, %B %d, %Y')}",
            self.styles['Subtitle']
        ))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.Color(0.1, 0.1, 0.3)))
        story.append(Spacer(1, 12))
        
        # Key metrics row
        story.append(Paragraph("Key Metrics", self.styles['Section']))
        
        nav_change = f"+{metrics.nav_change_1d:.2f}%" if metrics.nav_change_1d >= 0 else f"{metrics.nav_change_1d:.2f}%"
        nav_status = "good" if metrics.nav_change_1d >= 0 else "bad"
        
        passed = len([c for c in control_results if c.status == LimitStatus.COMPLIANT])
        warnings = len([c for c in control_results if c.status == LimitStatus.WARNING])
        breaches = len([c for c in control_results if c.status == LimitStatus.BREACH])
        total = len(control_results)
        
        kpi_data = [
            [
                self._create_kpi_box("NAV", f"${metrics.nav/1e9:.2f}B", nav_change, nav_status),
                self._create_kpi_box("Gross Exposure", f"{metrics.gross_exposure:.1f}%", 
                    f"Limit: {metrics.gross_limit:.0f}%", 
                    "warning" if metrics.gross_exposure > metrics.gross_limit * 0.8 else "neutral"),
                self._create_kpi_box("Net Exposure", f"{metrics.net_exposure:.1f}%",
                    f"Range: ±{metrics.net_limit_long:.0f}%", "neutral"),
                self._create_kpi_box("Controls Passed", f"{passed}/{total}",
                    f"{passed/total*100:.0f}%" if total > 0 else "N/A",
                    "good" if breaches == 0 else "bad" if breaches > 0 else "warning"),
                self._create_kpi_box("Open Exceptions", f"{len(exceptions)}",
                    f"{warnings} warnings" if warnings > 0 else "All clear",
                    "bad" if len(exceptions) > 0 else "good"),
            ]
        ]
        
        kpi_table = Table(kpi_data, colWidths=[100] * 5)
        kpi_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        story.append(kpi_table)
        story.append(Spacer(1, 16))
        
        # Overall status banner
        if breaches > 0:
            story.append(Paragraph(
                f"⚠ ATTENTION REQUIRED: {breaches} control breach(es) identified requiring immediate review",
                self.styles['Alert']
            ))
        elif warnings > 0:
            story.append(Paragraph(
                f"⚡ WARNING: {warnings} control(s) approaching thresholds - monitoring required",
                self.styles['Alert']
            ))
        else:
            story.append(Paragraph(
                "✓ ALL CONTROLS PASSED - Fund operating within all risk limits",
                self.styles['Success']
            ))
        
        story.append(Spacer(1, 12))
        
        # Executive narrative
        if 'executive_summary' in narratives:
            story.append(Paragraph("Executive Summary", self.styles['Subsection']))
            story.append(Paragraph(narratives['executive_summary'], self.styles['Narrative']))
        
        story.append(Spacer(1, 12))
        
        # ========== LIMIT UTILIZATION GAUGES ==========
        story.append(Paragraph("Limit Utilization Dashboard", self.styles['Section']))
        
        # Create gauges for key limits
        gauge_data = [
            ("Gross Exposure", metrics.gross_exposure, metrics.gross_limit),
            ("Net Exposure (Long)", max(metrics.net_exposure, 0), metrics.net_limit_long),
            ("Top 10 Concentration", metrics.top_10_concentration, 50.0),  # Assuming 50% limit
            ("T+1 Liquidity", metrics.t1_liquidity, metrics.t1_requirement),
        ]
        
        gauges = []
        for label, current, limit in gauge_data:
            gauges.append(LimitGauge(label, current, limit, width=120, height=45))
        
        # Arrange gauges in 2x2 grid
        gauge_table = Table([[gauges[0], gauges[1]], [gauges[2], gauges[3]]], 
                           colWidths=[250, 250])
        gauge_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(gauge_table)
        
        story.append(PageBreak())
        
        # ========== PAGE 2: CONTROL RESULTS ==========
        story.append(Paragraph("Control Test Results", self.styles['Section']))
        
        # Group controls by category
        categories = {}
        for cr in control_results:
            cat = cr.category or "Other"
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(cr)
        
        for category, controls in categories.items():
            story.append(Paragraph(category.replace('_', ' ').title(), self.styles['Subsection']))
            
            headers = ['Control', 'Current', 'Threshold', 'Util %', 'Prior Day', '30D Avg', 'Status']
            data = []
            for c in controls:
                status_icon = {
                    LimitStatus.COMPLIANT: "✓ Pass",
                    LimitStatus.WARNING: "⚠ Warning",
                    LimitStatus.BREACH: "✗ Breach",
                    LimitStatus.WAIVER: "◐ Waiver",
                }.get(c.status, "—")
                
                data.append([
                    c.control_name[:30],
                    f"{c.current_value:.2f}%",
                    f"{c.threshold_value:.2f}%",
                    f"{c.utilization_pct:.0f}%",
                    f"{c.prior_day_value:.2f}%" if c.prior_day_value else "—",
                    f"{c.avg_30d_value:.2f}%" if c.avg_30d_value else "—",
                    status_icon,
                ])
            
            table = self._create_status_table(
                headers, data, 
                col_widths=[120, 55, 55, 45, 55, 55, 60],
                status_col=6
            )
            story.append(table)
            story.append(Spacer(1, 12))
        
        # ========== EXCEPTION DETAILS ==========
        if exceptions:
            story.append(PageBreak())
            story.append(Paragraph("Exception Details & Escalation Status", self.styles['Section']))
            
            for exc in exceptions:
                # Exception header
                esc_levels = ['None', 'PM', 'CIO', 'CCO', 'Board']
                esc_text = esc_levels[min(exc.escalation_level, 4)]
                
                story.append(Paragraph(
                    f"<b>{exc.control_name}</b> | Exception ID: {exc.exception_id[:12]}",
                    self.styles['Subsection']
                ))
                
                # Exception metrics table
                exc_data = [
                    ['Breach Date', exc.breach_date.strftime('%Y-%m-%d'),
                     'Age', f"{exc.age_days} days"],
                    ['Breach Value', f"{exc.breach_value:.2f}%",
                     'Threshold', f"{exc.threshold_value:.2f}%"],
                    ['Severity', f"{exc.severity_pct:.1f}% over",
                     'Escalation', esc_text],
                    ['Status', exc.status.upper(),
                     'Assigned To', exc.assigned_to],
                ]
                
                exc_table = Table(exc_data, colWidths=[80, 100, 80, 100])
                exc_table.setStyle(TableStyle([
                    ('FONTNAME', (0, 0), (-1, -1), 'Times-Roman'),
                    ('FONTSIZE', (0, 0), (-1, -1), 8),
                    ('FONTNAME', (0, 0), (0, -1), 'Times-Bold'),
                    ('FONTNAME', (2, 0), (2, -1), 'Times-Bold'),
                    ('TOPPADDING', (0, 0), (-1, -1), 2),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
                ]))
                story.append(exc_table)
                
                # Commentary
                if exc.commentary:
                    story.append(Paragraph("<b>Analysis:</b>", self.styles['Body']))
                    story.append(Paragraph(exc.commentary, self.styles['Narrative']))
                
                # Remediation plan
                if exc.remediation_plan:
                    story.append(Paragraph("<b>Remediation Plan:</b>", self.styles['Body']))
                    story.append(Paragraph(exc.remediation_plan, self.styles['Body']))
                    if exc.remediation_deadline:
                        story.append(Paragraph(
                            f"<b>Deadline:</b> {exc.remediation_deadline.strftime('%Y-%m-%d')}",
                            self.styles['Body']
                        ))
                
                story.append(Spacer(1, 12))
                story.append(HRFlowable(width="100%", thickness=0.5, color=colors.Color(0.8, 0.8, 0.8)))
        
        # ========== POSITION CONCENTRATION ==========
        story.append(PageBreak())
        story.append(Paragraph("Position Concentration Analysis", self.styles['Section']))
        
        # Top 10 positions
        story.append(Paragraph("Top 10 Positions by Market Value", self.styles['Subsection']))
        
        sorted_positions = sorted(positions, key=lambda p: abs(p.market_value), reverse=True)[:10]
        
        headers = ['Security', 'Ticker', 'Sector', 'Side', 'Market Value', '% NAV', '% Gross']
        data = []
        for p in sorted_positions:
            data.append([
                p.security_name[:25],
                p.ticker,
                p.sector[:15],
                p.side.upper(),
                f"${p.market_value/1e6:.1f}M",
                f"{p.pct_nav:.2f}%",
                f"{p.pct_gross:.2f}%",
            ])
        
        table = self._create_status_table(headers, data, col_widths=[110, 45, 70, 40, 70, 50, 50])
        story.append(table)
        story.append(Spacer(1, 16))
        
        # Sector concentration
        story.append(Paragraph("Sector Concentration", self.styles['Subsection']))
        
        # Aggregate by sector
        sector_totals: Dict[str, float] = {}
        for p in positions:
            sector = p.sector or "Other"
            sector_totals[sector] = sector_totals.get(sector, 0) + p.pct_nav
        
        sorted_sectors = sorted(sector_totals.items(), key=lambda x: abs(x[1]), reverse=True)
        
        headers = ['Sector', 'Net Exposure', 'Limit', 'Utilization']
        data = []
        for sector, exposure in sorted_sectors[:8]:
            limit = 25.0  # Default sector limit
            util = abs(exposure) / limit * 100
            data.append([
                sector,
                f"{exposure:.1f}%",
                f"{limit:.0f}%",
                f"{util:.0f}%",
            ])
        
        table = self._create_status_table(headers, data, col_widths=[150, 80, 80, 80])
        story.append(table)
        
        # ========== SIGNATURE BLOCK ==========
        story.append(PageBreak())
        story.append(Paragraph("Document Certification", self.styles['Section']))
        
        # Document hash
        if document_hash:
            story.append(Paragraph(f"<b>Document Integrity Hash:</b> {document_hash}", self.styles['Meta']))
        story.append(Spacer(1, 20))
        
        # Signature table
        sig_data = [
            ['Prepared By:', '_' * 30, 'Date:', '_' * 15],
            ['', '', '', ''],
            ['Reviewed By (CCO):', '_' * 30, 'Date:', '_' * 15],
            ['', '', '', ''],
            ['Approved By:', '_' * 30, 'Date:', '_' * 15],
        ]
        
        sig_table = Table(sig_data, colWidths=[100, 180, 40, 100])
        sig_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Times-Roman'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('VALIGN', (0, 0), (-1, -1), 'BOTTOM'),
        ]))
        story.append(sig_table)
        
        story.append(Spacer(1, 30))
        
        # Certification text
        cert_text = """
        I certify that this Daily Compliance Pack accurately reflects the compliance status 
        of the Fund as of the report date. All control tests were executed against verified 
        source data, and all exceptions have been documented with appropriate escalation. 
        Narrative commentary was generated with AI assistance using only cited evidence.
        """
        story.append(Paragraph(cert_text.strip(), self.styles['Body']))
        
        story.append(Spacer(1, 20))
        
        # Audit trail note
        audit_text = """
        <b>SEC Examination Note:</b> Complete audit trail including all source data, 
        SQL queries, LLM prompts, and evidence hashes is maintained in the compliance 
        database and available upon request. Document ID and integrity hash allow 
        verification of this document's authenticity.
        """
        story.append(Paragraph(audit_text, self.styles['Meta']))
        
        # Build PDF
        doc.build(story, onFirstPage=self._header_footer, onLaterPages=self._header_footer)
        
        return buffer.getvalue()
    
    def generate_executive_summary(
        self,
        metrics: FundMetrics,
        control_results: List[ControlResult],
        exceptions: List[ExceptionRecord],
        narrative: str,
    ) -> bytes:
        """
        Generate 1-page Executive Summary for CIO/PM.
        
        Quick glance format showing only critical information.
        """
        buffer = io.BytesIO()
        
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=50,
            leftMargin=50,
            topMargin=55,
            bottomMargin=50,
        )
        
        story = []
        
        # Title
        story.append(Paragraph("Compliance Summary", self.styles['Title1']))
        story.append(Paragraph(metrics.report_date.strftime('%B %d, %Y'), self.styles['Subtitle']))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.Color(0.1, 0.1, 0.3)))
        story.append(Spacer(1, 12))
        
        # Status indicator
        breaches = len([c for c in control_results if c.status == LimitStatus.BREACH])
        warnings = len([c for c in control_results if c.status == LimitStatus.WARNING])
        
        if breaches > 0:
            story.append(Paragraph(f"⚠ {breaches} BREACH(ES) REQUIRING ACTION", self.styles['Alert']))
        elif warnings > 0:
            story.append(Paragraph(f"⚡ {warnings} WARNING(S) - MONITORING REQUIRED", self.styles['Alert']))
        else:
            story.append(Paragraph("✓ ALL CLEAR - OPERATING WITHIN LIMITS", self.styles['Success']))
        
        story.append(Spacer(1, 16))
        
        # Key numbers
        summary_data = [
            ['NAV', f"${metrics.nav/1e9:.2f}B", 'Gross', f"{metrics.gross_exposure:.0f}%"],
            ['Net', f"{metrics.net_exposure:+.0f}%", 'Top 10', f"{metrics.top_10_concentration:.0f}%"],
            ['T+1 Liq', f"{metrics.t1_liquidity:.0f}%", 'VaR 95', f"${metrics.var_95_1d/1e6:.1f}M"],
        ]
        
        summary_table = Table(summary_data, colWidths=[60, 100, 60, 100])
        summary_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Times-Roman'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('FONTNAME', (1, 0), (1, -1), 'Times-Bold'),
            ('FONTNAME', (3, 0), (3, -1), 'Times-Bold'),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        story.append(summary_table)
        story.append(Spacer(1, 16))
        
        # Narrative
        story.append(Paragraph("Summary", self.styles['Subsection']))
        story.append(Paragraph(narrative, self.styles['Body']))
        
        # Exceptions if any
        if exceptions:
            story.append(Spacer(1, 12))
            story.append(Paragraph("Open Exceptions", self.styles['Subsection']))
            
            for exc in exceptions[:3]:  # Top 3 only
                story.append(Paragraph(
                    f"• <b>{exc.control_name}</b>: {exc.breach_value:.1f}% vs {exc.threshold_value:.1f}% "
                    f"(Age: {exc.age_days}d, Escalation: Level {exc.escalation_level})",
                    self.styles['Body']
                ))
        
        # Build
        doc.build(story, onFirstPage=self._header_footer, onLaterPages=self._header_footer)
        
        return buffer.getvalue()


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def create_sample_fund_metrics(report_date: Optional[date] = None) -> FundMetrics:
    """Create sample fund metrics for testing."""
    return FundMetrics(
        fund_name="Global Macro Fund LP",
        fund_id="GMF-001",
        report_date=report_date or date.today(),
        nav=2_350_000_000,
        nav_change_1d=0.42,
        nav_change_mtd=1.85,
        nav_change_ytd=8.34,
        aum=2_500_000_000,
        gross_exposure=165.5,
        gross_limit=200.0,
        net_exposure=45.2,
        net_limit_long=100.0,
        net_limit_short=100.0,
        top_10_concentration=38.5,
        single_name_limit=10.0,
        largest_position_pct=7.2,
        sector_concentration={
            "Technology": 28.5,
            "Financials": 22.3,
            "Healthcare": 15.8,
            "Energy": 12.4,
            "Consumer": 10.2,
        },
        t1_liquidity=22.5,
        t1_requirement=15.0,
        t5_liquidity=45.0,
        t30_liquidity=78.0,
        var_95_1d=18_500_000,
        var_99_1d=28_200_000,
        beta_to_spx=0.65,
        sharpe_ratio_ytd=1.85,
        regulatory_leverage=2.8,
        economic_leverage=1.65,
        margin_utilization=42.5,
    )


def create_sample_control_results() -> List[ControlResult]:
    """Create sample control results for testing."""
    return [
        ControlResult(
            control_id="CONC_SECTOR_001",
            control_name="Sector Concentration - Technology",
            category="Concentration",
            current_value=28.5,
            threshold_value=35.0,
            threshold_type="max",
            status=LimitStatus.COMPLIANT,
            utilization_pct=81.4,
            headroom=6.5,
            prior_day_value=27.8,
            avg_30d_value=26.5,
        ),
        ControlResult(
            control_id="CONC_ISSUER_001",
            control_name="Single Issuer Limit",
            category="Concentration",
            current_value=7.2,
            threshold_value=10.0,
            threshold_type="max",
            status=LimitStatus.COMPLIANT,
            utilization_pct=72.0,
            headroom=2.8,
            prior_day_value=7.0,
            avg_30d_value=6.8,
        ),
        ControlResult(
            control_id="EXP_GROSS_001",
            control_name="Gross Exposure",
            category="Exposure",
            current_value=165.5,
            threshold_value=200.0,
            threshold_type="max",
            status=LimitStatus.WARNING,
            utilization_pct=82.75,
            headroom=34.5,
            prior_day_value=162.0,
            avg_30d_value=158.0,
        ),
        ControlResult(
            control_id="LIQ_T1_001",
            control_name="T+1 Liquidity",
            category="Liquidity",
            current_value=22.5,
            threshold_value=15.0,
            threshold_type="min",
            status=LimitStatus.COMPLIANT,
            utilization_pct=66.7,
            headroom=7.5,
            prior_day_value=21.8,
            avg_30d_value=20.5,
        ),
    ]


def create_sample_exceptions() -> List[ExceptionRecord]:
    """Create sample exceptions for testing."""
    return [
        ExceptionRecord(
            exception_id="EXC-2026-001",
            control_id="CONC_SECTOR_002",
            control_name="Sector Concentration - Energy",
            breach_date=date(2026, 1, 18),
            breach_value=27.5,
            threshold_value=25.0,
            severity_pct=10.0,
            status="open",
            escalation_level=2,
            age_days=2,
            assigned_to="Jane Smith, CCO",
            remediation_deadline=date(2026, 1, 25),
            remediation_plan="Reduce energy exposure by selling 200,000 shares of XOM and 150,000 shares of CVX.",
            commentary="Energy sector concentration exceeded the 25% limit due to recent oil price rally "
                       "and existing long positions in integrated majors. Position was within limits as of "
                       "prior day close. Recommend gradual reduction over 3-5 trading days to minimize market impact.",
        ),
    ]


def create_sample_positions() -> List[PositionDetail]:
    """Create sample positions for testing."""
    return [
        PositionDetail(
            security_id="AAPL",
            security_name="Apple Inc",
            ticker="AAPL",
            asset_class="Equity",
            sector="Technology",
            country="US",
            currency="USD",
            quantity=500000,
            price=185.50,
            market_value=92_750_000,
            pct_nav=3.95,
            pct_gross=2.39,
            side="long",
        ),
        PositionDetail(
            security_id="MSFT",
            security_name="Microsoft Corp",
            ticker="MSFT",
            asset_class="Equity",
            sector="Technology",
            country="US",
            currency="USD",
            quantity=400000,
            price=420.25,
            market_value=168_100_000,
            pct_nav=7.15,
            pct_gross=4.33,
            side="long",
        ),
        PositionDetail(
            security_id="XOM",
            security_name="Exxon Mobil Corp",
            ticker="XOM",
            asset_class="Equity",
            sector="Energy",
            country="US",
            currency="USD",
            quantity=800000,
            price=115.30,
            market_value=92_240_000,
            pct_nav=3.92,
            pct_gross=2.38,
            side="long",
        ),
    ]


if __name__ == "__main__":
    # Test PDF generation
    pdf_gen = InstitutionalCompliancePDF(
        fund_name="Global Macro Master Fund LP",
        fund_id="GMMF-001",
        adviser_name="Crosby Capital Management LLC",
        adviser_crd="987654",
    )
    
    metrics = create_sample_fund_metrics()
    controls = create_sample_control_results()
    exceptions = create_sample_exceptions()
    positions = create_sample_positions()
    
    narratives = {
        "executive_summary": (
            "The Fund operated within all primary risk limits during the reporting period, "
            "with gross exposure at 82.75% of the 200% limit and net exposure well within "
            "the ±100% band. One exception was identified in the Energy sector, where "
            "concentration reached 27.5% against a 25% limit due to the recent rally in "
            "crude oil prices. The exception has been escalated to the CIO and a remediation "
            "plan is in place targeting a return to compliance by January 25, 2026. "
            "Liquidity remains strong with T+1 liquidity at 22.5%, well above the 15% requirement."
        ),
    }
    
    # Generate daily pack
    pdf_bytes = pdf_gen.generate_daily_compliance_pack(
        metrics=metrics,
        control_results=controls,
        exceptions=exceptions,
        positions=positions,
        narratives=narratives,
        document_hash="a1b2c3d4e5f6...",
    )
    
    # Save to file
    output_path = "output/institutional_daily_pack.pdf"
    with open(output_path, "wb") as f:
        f.write(pdf_bytes)
    
    print(f"Generated: {output_path} ({len(pdf_bytes):,} bytes)")
