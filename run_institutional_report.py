#!/usr/bin/env python3
"""
Institutional Compliance Report Generator

Generates comprehensive, hedge-fund-grade compliance documentation:
- Full Daily Compliance Pack (multi-page PDF)
- Executive Summary (1-page for CIO/PM)
- Exception Report with escalation tracking
- Risk Committee Pack (monthly)

Usage:
    python run_institutional_report.py              # Generate all reports
    python run_institutional_report.py --pack       # Daily pack only
    python run_institutional_report.py --summary    # Executive summary only
    python run_institutional_report.py --from-db    # Use database data
"""

import sys
from pathlib import Path
from datetime import date, datetime, timezone, timedelta
from decimal import Decimal
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.document_builder.institutional_pdf import (
    InstitutionalCompliancePDF,
    FundMetrics,
    ControlResult,
    ExceptionRecord,
    PositionDetail,
    LimitStatus,
    create_sample_fund_metrics,
    create_sample_control_results,
    create_sample_exceptions,
    create_sample_positions,
)


def create_comprehensive_fund_metrics() -> FundMetrics:
    """Create realistic hedge fund metrics."""
    return FundMetrics(
        fund_name="Global Opportunities Master Fund LP",
        fund_id="GOMF-001",
        report_date=date.today(),
        
        # NAV & AUM
        nav=2_847_500_000,  # $2.85B
        nav_change_1d=0.34,
        nav_change_mtd=1.92,
        nav_change_ytd=7.85,
        aum=3_150_000_000,  # $3.15B including managed accounts
        
        # Exposure
        gross_exposure=172.5,
        gross_limit=200.0,
        net_exposure=38.4,
        net_limit_long=100.0,
        net_limit_short=100.0,
        
        # Concentration
        top_10_concentration=42.3,
        single_name_limit=10.0,
        largest_position_pct=6.8,
        sector_concentration={
            "Technology": 32.5,
            "Healthcare": 18.7,
            "Financials": 15.2,
            "Consumer Discretionary": 12.8,
            "Energy": 11.5,
            "Industrials": 9.3,
        },
        
        # Liquidity
        t1_liquidity=24.8,
        t1_requirement=15.0,
        t5_liquidity=52.3,
        t30_liquidity=81.5,
        
        # Risk
        var_95_1d=22_400_000,
        var_99_1d=34_800_000,
        beta_to_spx=0.72,
        sharpe_ratio_ytd=2.15,
        
        # Leverage
        regulatory_leverage=3.2,
        economic_leverage=1.72,
        margin_utilization=48.5,
    )


def create_comprehensive_control_results() -> list[ControlResult]:
    """Create comprehensive control results across all categories."""
    return [
        # Exposure Controls
        ControlResult(
            control_id="EXP_GROSS_001",
            control_name="Gross Exposure",
            category="Exposure",
            current_value=172.5,
            threshold_value=200.0,
            threshold_type="max",
            status=LimitStatus.WARNING,
            utilization_pct=86.25,
            headroom=27.5,
            prior_day_value=168.3,
            avg_30d_value=165.8,
            breach_count_ytd=0,
        ),
        ControlResult(
            control_id="EXP_NET_001",
            control_name="Net Exposure (Long)",
            category="Exposure",
            current_value=38.4,
            threshold_value=100.0,
            threshold_type="max",
            status=LimitStatus.COMPLIANT,
            utilization_pct=38.4,
            headroom=61.6,
            prior_day_value=42.1,
            avg_30d_value=45.2,
        ),
        ControlResult(
            control_id="EXP_NET_002",
            control_name="Net Exposure (Short)",
            category="Exposure",
            current_value=-15.2,
            threshold_value=-100.0,
            threshold_type="min",
            status=LimitStatus.COMPLIANT,
            utilization_pct=15.2,
            headroom=84.8,
            prior_day_value=-18.5,
            avg_30d_value=-20.3,
        ),
        
        # Concentration Controls
        ControlResult(
            control_id="CONC_SECTOR_001",
            control_name="Sector Concentration - Technology",
            category="Concentration",
            current_value=32.5,
            threshold_value=35.0,
            threshold_type="max",
            status=LimitStatus.WARNING,
            utilization_pct=92.86,
            headroom=2.5,
            prior_day_value=31.8,
            avg_30d_value=30.2,
            breach_count_ytd=1,
            last_breach_date=date(2026, 1, 8),
        ),
        ControlResult(
            control_id="CONC_SECTOR_002",
            control_name="Sector Concentration - Healthcare",
            category="Concentration",
            current_value=18.7,
            threshold_value=25.0,
            threshold_type="max",
            status=LimitStatus.COMPLIANT,
            utilization_pct=74.8,
            headroom=6.3,
            prior_day_value=18.2,
            avg_30d_value=17.5,
        ),
        ControlResult(
            control_id="CONC_ISSUER_001",
            control_name="Single Issuer Limit",
            category="Concentration",
            current_value=6.8,
            threshold_value=10.0,
            threshold_type="max",
            status=LimitStatus.COMPLIANT,
            utilization_pct=68.0,
            headroom=3.2,
            prior_day_value=6.5,
            avg_30d_value=6.2,
        ),
        ControlResult(
            control_id="CONC_TOP10_001",
            control_name="Top 10 Concentration",
            category="Concentration",
            current_value=42.3,
            threshold_value=50.0,
            threshold_type="max",
            status=LimitStatus.COMPLIANT,
            utilization_pct=84.6,
            headroom=7.7,
            prior_day_value=41.5,
            avg_30d_value=40.8,
        ),
        
        # Liquidity Controls
        ControlResult(
            control_id="LIQ_T1_001",
            control_name="T+1 Liquidity",
            category="Liquidity",
            current_value=24.8,
            threshold_value=15.0,
            threshold_type="min",
            status=LimitStatus.COMPLIANT,
            utilization_pct=60.48,
            headroom=9.8,
            prior_day_value=23.5,
            avg_30d_value=22.8,
        ),
        ControlResult(
            control_id="LIQ_ILLIQ_001",
            control_name="Illiquid Assets",
            category="Liquidity",
            current_value=8.5,
            threshold_value=15.0,
            threshold_type="max",
            status=LimitStatus.COMPLIANT,
            utilization_pct=56.67,
            headroom=6.5,
            prior_day_value=8.2,
            avg_30d_value=7.8,
        ),
        
        # Counterparty Controls
        ControlResult(
            control_id="CP_PB_001",
            control_name="Prime Broker Concentration",
            category="Counterparty",
            current_value=45.2,
            threshold_value=60.0,
            threshold_type="max",
            status=LimitStatus.COMPLIANT,
            utilization_pct=75.33,
            headroom=14.8,
            prior_day_value=44.8,
            avg_30d_value=43.5,
        ),
        ControlResult(
            control_id="CP_OTC_001",
            control_name="OTC Counterparty Limit - GS",
            category="Counterparty",
            current_value=85_000_000,
            threshold_value=150_000_000,
            threshold_type="max",
            status=LimitStatus.COMPLIANT,
            utilization_pct=56.67,
            headroom=65_000_000,
        ),
        
        # Leverage Controls
        ControlResult(
            control_id="LEV_REG_001",
            control_name="Regulatory Leverage (13H)",
            category="Leverage",
            current_value=3.2,
            threshold_value=5.0,
            threshold_type="max",
            status=LimitStatus.COMPLIANT,
            utilization_pct=64.0,
            headroom=1.8,
            prior_day_value=3.1,
            avg_30d_value=3.0,
        ),
        ControlResult(
            control_id="LEV_MARGIN_001",
            control_name="Margin Utilization",
            category="Leverage",
            current_value=48.5,
            threshold_value=75.0,
            threshold_type="max",
            status=LimitStatus.COMPLIANT,
            utilization_pct=64.67,
            headroom=26.5,
            prior_day_value=46.2,
            avg_30d_value=45.0,
        ),
        
        # Risk Controls
        ControlResult(
            control_id="RISK_VAR_001",
            control_name="VaR (95%, 1D)",
            category="Risk",
            current_value=22_400_000,
            threshold_value=35_000_000,
            threshold_type="max",
            status=LimitStatus.COMPLIANT,
            utilization_pct=64.0,
            headroom=12_600_000,
            prior_day_value=21_800_000,
            avg_30d_value=20_500_000,
        ),
    ]


def create_comprehensive_exceptions() -> list[ExceptionRecord]:
    """Create exception records with full details."""
    return [
        ExceptionRecord(
            exception_id="EXC-2026-0042",
            control_id="CONC_SECTOR_001",
            control_name="Sector Concentration - Technology",
            breach_date=date.today() - timedelta(days=3),
            breach_value=36.2,
            threshold_value=35.0,
            severity_pct=3.43,
            status="open",
            escalation_level=2,  # CIO
            age_days=3,
            assigned_to="John Smith, Portfolio Manager",
            remediation_deadline=date.today() + timedelta(days=4),
            remediation_plan=(
                "Reduce technology sector exposure through the following actions:\n"
                "1. Trim NVDA position by 50,000 shares (~$6.2M) - Target: T+1\n"
                "2. Reduce MSFT position by 25,000 shares (~$10.5M) - Target: T+2\n"
                "3. Exit META call spread position (~$3.5M delta) - Target: T+1\n"
                "Expected post-remediation exposure: 32.5% (within limit)"
            ),
            commentary=(
                "Technology sector concentration exceeded the 35% limit due to a combination of:\n"
                "(1) Strong price appreciation in semiconductor holdings (+4.2% this week)\n"
                "(2) Redemption-driven NAV reduction of $45M, increasing relative weights\n"
                "(3) New GOOG position initiated prior to sector limit calculation refresh\n\n"
                "The breach was identified within 30 minutes of threshold crossing via automated "
                "monitoring. No willful policy violation occurred. Position was within limits "
                "as of prior day close (34.8%)."
            ),
            evidence_hash="a7b9c2d4e6f8",
        ),
        ExceptionRecord(
            exception_id="EXC-2026-0038",
            control_id="EXP_GROSS_001",
            control_name="Gross Exposure Warning",
            breach_date=date.today() - timedelta(days=1),
            breach_value=172.5,
            threshold_value=160.0,  # Warning threshold (80% of 200%)
            severity_pct=7.81,
            status="monitoring",
            escalation_level=1,  # PM
            age_days=1,
            assigned_to="Risk Committee",
            remediation_deadline=None,  # Monitoring, not remediation
            remediation_plan="",
            commentary=(
                "Gross exposure crossed the 80% warning threshold (160%) and is currently at "
                "86.25% utilization of the 200% hard limit. This is within normal operating "
                "range but warrants monitoring. No immediate action required unless exposure "
                "approaches 90% utilization (180% gross)."
            ),
            evidence_hash="b8c0d3e5f7g9",
        ),
    ]


def create_comprehensive_positions() -> list[PositionDetail]:
    """Create realistic portfolio positions."""
    positions = [
        # Top Long Positions
        PositionDetail(
            security_id="NVDA", security_name="NVIDIA Corporation", ticker="NVDA",
            asset_class="Equity", sector="Technology", country="US", currency="USD",
            quantity=320000, price=892.50, market_value=285_600_000,
            pct_nav=10.03, pct_gross=5.82, side="long", strategy="Growth",
            days_held=45, unrealized_pnl=52_400_000, contribution_to_var=4_200_000,
        ),
        PositionDetail(
            security_id="MSFT", security_name="Microsoft Corporation", ticker="MSFT",
            asset_class="Equity", sector="Technology", country="US", currency="USD",
            quantity=420000, price=415.80, market_value=174_636_000,
            pct_nav=6.13, pct_gross=3.56, side="long", strategy="Quality",
            days_held=180, unrealized_pnl=28_500_000, contribution_to_var=2_100_000,
        ),
        PositionDetail(
            security_id="AMZN", security_name="Amazon.com Inc", ticker="AMZN",
            asset_class="Equity", sector="Consumer Discretionary", country="US", currency="USD",
            quantity=550000, price=185.20, market_value=101_860_000,
            pct_nav=3.58, pct_gross=2.08, side="long", strategy="Growth",
            days_held=90, unrealized_pnl=15_200_000, contribution_to_var=1_800_000,
        ),
        PositionDetail(
            security_id="LLY", security_name="Eli Lilly and Company", ticker="LLY",
            asset_class="Equity", sector="Healthcare", country="US", currency="USD",
            quantity=125000, price=782.40, market_value=97_800_000,
            pct_nav=3.43, pct_gross=1.99, side="long", strategy="Quality",
            days_held=120, unrealized_pnl=22_800_000, contribution_to_var=1_500_000,
        ),
        PositionDetail(
            security_id="JPM", security_name="JPMorgan Chase & Co", ticker="JPM",
            asset_class="Equity", sector="Financials", country="US", currency="USD",
            quantity=480000, price=198.50, market_value=95_280_000,
            pct_nav=3.35, pct_gross=1.94, side="long", strategy="Value",
            days_held=200, unrealized_pnl=18_400_000, contribution_to_var=1_400_000,
        ),
        PositionDetail(
            security_id="GOOGL", security_name="Alphabet Inc Class A", ticker="GOOGL",
            asset_class="Equity", sector="Technology", country="US", currency="USD",
            quantity=520000, price=175.30, market_value=91_156_000,
            pct_nav=3.20, pct_gross=1.86, side="long", strategy="Growth",
            days_held=60, unrealized_pnl=8_200_000, contribution_to_var=1_650_000,
        ),
        PositionDetail(
            security_id="XOM", security_name="Exxon Mobil Corporation", ticker="XOM",
            asset_class="Equity", sector="Energy", country="US", currency="USD",
            quantity=750000, price=118.40, market_value=88_800_000,
            pct_nav=3.12, pct_gross=1.81, side="long", strategy="Value",
            days_held=150, unrealized_pnl=12_500_000, contribution_to_var=1_100_000,
        ),
        PositionDetail(
            security_id="UNH", security_name="UnitedHealth Group Inc", ticker="UNH",
            asset_class="Equity", sector="Healthcare", country="US", currency="USD",
            quantity=145000, price=525.80, market_value=76_241_000,
            pct_nav=2.68, pct_gross=1.55, side="long", strategy="Quality",
            days_held=85, unrealized_pnl=9_800_000, contribution_to_var=980_000,
        ),
        
        # Short Positions
        PositionDetail(
            security_id="TSLA", security_name="Tesla Inc", ticker="TSLA",
            asset_class="Equity", sector="Consumer Discretionary", country="US", currency="USD",
            quantity=-180000, price=245.60, market_value=-44_208_000,
            pct_nav=-1.55, pct_gross=0.90, side="short", strategy="Pair Trade",
            days_held=30, unrealized_pnl=5_800_000, contribution_to_var=2_200_000,
        ),
        PositionDetail(
            security_id="RIVN", security_name="Rivian Automotive Inc", ticker="RIVN",
            asset_class="Equity", sector="Consumer Discretionary", country="US", currency="USD",
            quantity=-1200000, price=18.45, market_value=-22_140_000,
            pct_nav=-0.78, pct_gross=0.45, side="short", strategy="Fundamental Short",
            days_held=60, unrealized_pnl=8_400_000, contribution_to_var=1_100_000,
        ),
        PositionDetail(
            security_id="COIN", security_name="Coinbase Global Inc", ticker="COIN",
            asset_class="Equity", sector="Financials", country="US", currency="USD",
            quantity=-120000, price=168.20, market_value=-20_184_000,
            pct_nav=-0.71, pct_gross=0.41, side="short", strategy="Thematic Short",
            days_held=45, unrealized_pnl=3_200_000, contribution_to_var=1_450_000,
        ),
    ]
    
    return positions


def generate_reports(args):
    """Generate institutional-grade compliance reports."""
    
    print("=" * 70)
    print("INSTITUTIONAL COMPLIANCE REPORT GENERATOR")
    print("=" * 70)
    print()
    
    # Initialize PDF generator
    pdf_gen = InstitutionalCompliancePDF(
        fund_name="Global Opportunities Master Fund LP",
        fund_id="GOMF-001",
        adviser_name="Crosby Capital Management LLC",
        adviser_crd="987654",
        confidentiality="CONFIDENTIAL - PROPRIETARY",
    )
    
    # Create data
    print("Loading fund data...")
    metrics = create_comprehensive_fund_metrics()
    control_results = create_comprehensive_control_results()
    exceptions = create_comprehensive_exceptions()
    positions = create_comprehensive_positions()
    
    print(f"  Fund: {metrics.fund_name}")
    print(f"  NAV: ${metrics.nav/1e9:.2f}B")
    print(f"  Report Date: {metrics.report_date}")
    print(f"  Controls: {len(control_results)}")
    print(f"  Exceptions: {len(exceptions)}")
    print(f"  Positions: {len(positions)}")
    print()
    
    # Generate narratives (in production, these come from the LLM)
    narratives = {
        "executive_summary": (
            f"The Fund operated within primary risk limits during the reporting period, "
            f"with gross exposure at {metrics.gross_exposure:.1f}% against a 200% limit "
            f"({metrics.gross_exposure/200*100:.1f}% utilization). Net exposure stands at "
            f"{metrics.net_exposure:+.1f}%, reflecting a moderately long bias consistent with "
            f"the current market outlook.\n\n"
            f"One active exception exists in Technology sector concentration, which exceeded "
            f"the 35% limit due to semiconductor appreciation and NAV reduction from redemptions. "
            f"A remediation plan is in place targeting return to compliance within 4 trading days.\n\n"
            f"Liquidity remains robust with T+1 liquidity at {metrics.t1_liquidity:.1f}%, well above "
            f"the 15% requirement. VaR (95%, 1D) of ${metrics.var_95_1d/1e6:.1f}M represents "
            f"{metrics.var_95_1d/metrics.nav*100:.2f}% of NAV, within normal operating range.\n\n"
            f"No material compliance concerns requiring immediate escalation beyond existing "
            f"exception management processes."
        ),
    }
    
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Generate Daily Compliance Pack
    if args.pack or args.all:
        print("Generating Daily Compliance Pack...")
        
        pdf_bytes = pdf_gen.generate_daily_compliance_pack(
            metrics=metrics,
            control_results=control_results,
            exceptions=exceptions,
            positions=positions,
            narratives=narratives,
            document_hash="sha256:a7b9c2d4e6f8901234567890abcdef...",
        )
        
        pack_path = output_dir / f"institutional_daily_pack_{date.today()}.pdf"
        with open(pack_path, "wb") as f:
            f.write(pdf_bytes)
        
        print(f"  ✓ Generated: {pack_path}")
        print(f"    Size: {len(pdf_bytes):,} bytes")
    
    # Generate Executive Summary
    if args.summary or args.all:
        print("Generating Executive Summary...")
        
        pdf_bytes = pdf_gen.generate_executive_summary(
            metrics=metrics,
            control_results=control_results,
            exceptions=exceptions,
            narrative=narratives["executive_summary"],
        )
        
        summary_path = output_dir / f"executive_summary_{date.today()}.pdf"
        with open(summary_path, "wb") as f:
            f.write(pdf_bytes)
        
        print(f"  ✓ Generated: {summary_path}")
        print(f"    Size: {len(pdf_bytes):,} bytes")
    
    print()
    print("=" * 70)
    print("GENERATION COMPLETE")
    print("=" * 70)
    print()
    print("Output files:")
    for f in output_dir.glob(f"*_{date.today()}.pdf"):
        print(f"  • {f}")
    print()
    print("These reports include:")
    print("  • Executive Dashboard with key risk metrics")
    print("  • Limit utilization gauges (visual)")
    print("  • Control test results by category")
    print("  • Exception details with escalation tracking")
    print("  • Position concentration analysis")
    print("  • Signature blocks for CCO approval")
    print("  • Complete audit trail with evidence hashes")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Generate institutional-grade compliance reports"
    )
    parser.add_argument(
        "--pack", action="store_true",
        help="Generate Daily Compliance Pack only"
    )
    parser.add_argument(
        "--summary", action="store_true", 
        help="Generate Executive Summary only"
    )
    parser.add_argument(
        "--from-db", action="store_true",
        help="Load data from PostgreSQL database"
    )
    parser.add_argument(
        "--all", action="store_true", default=True,
        help="Generate all reports (default)"
    )
    
    args = parser.parse_args()
    
    # If specific report requested, disable --all
    if args.pack or args.summary:
        args.all = False
    
    generate_reports(args)


if __name__ == "__main__":
    main()
