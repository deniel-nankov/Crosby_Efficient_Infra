#!/usr/bin/env python3
"""
Load High-Quality Data for RAG Demo

This script loads the realistic position and control data,
then generates sample compliance narratives showing RAG in action.
"""

import sys
from datetime import date
from decimal import Decimal
from pathlib import Path
import csv
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))


def load_positions(csv_path: str) -> list:
    """Load positions from CSV file."""
    positions = []
    with open(csv_path, 'r') as f:
        # Skip comment lines (lines starting with #)
        lines = [line for line in f if line.strip() and not line.strip().startswith('#')]
        reader = csv.DictReader(lines)
        for row in reader:
            positions.append({
                'security_id': row['security_id'],
                'ticker': row['ticker'],
                'security_name': row['security_name'],
                'quantity': Decimal(row['quantity']),
                'market_value': Decimal(row['market_value']),
                'currency': row['currency'],
                'sector': row['sector'],
                'issuer': row['issuer'],
                'asset_class': row['asset_class'],
                'country': row.get('country', 'US'),
                'liquidity_days': int(row.get('liquidity_days', 1)),
            })
    return positions


def load_controls(csv_path: str) -> list:
    """Load control results from CSV file."""
    controls = []
    with open(csv_path, 'r') as f:
        # Skip comment lines (lines starting with #)
        lines = [line for line in f if line.strip() and not line.strip().startswith('#')]
        reader = csv.DictReader(lines)
        for row in reader:
            details = None
            if row.get('details'):
                try:
                    details = json.loads(row['details'])
                except:
                    details = row['details']
            
            controls.append({
                'control_id': row['control_id'],
                'control_name': row['control_name'],
                'control_type': row['control_type'],
                'calculated_value': Decimal(row['calculated_value']),
                'threshold': Decimal(row['threshold']),
                'threshold_operator': row['threshold_operator'],
                'status': row['status'],
                'breach_amount': Decimal(row['breach_amount']) if row.get('breach_amount') else None,
                'details': details,
            })
    return controls


def analyze_data(positions: list, controls: list):
    """Analyze loaded data and print summary."""
    print("\n" + "=" * 70)
    print("DATA ANALYSIS SUMMARY")
    print("=" * 70)
    
    # Position analysis
    long_positions = [p for p in positions if p['market_value'] > 0]
    short_positions = [p for p in positions if p['market_value'] < 0]
    
    total_long = sum(p['market_value'] for p in long_positions)
    total_short = abs(sum(p['market_value'] for p in short_positions))
    nav = Decimal('2000000000')  # $2B NAV
    
    print(f"\nPOSITIONS ({len(positions)} total)")
    print("-" * 40)
    print(f"  Long positions:    {len(long_positions):>5}    ${total_long:>15,.0f}")
    print(f"  Short positions:   {len(short_positions):>5}    ${total_short:>15,.0f}")
    print(f"  Net exposure:             ${total_long - total_short:>15,.0f}")
    print(f"  NAV:                      ${nav:>15,.0f}")
    print(f"  Gross exposure:           {(total_long + total_short) / nav * 100:>14.1f}%")
    print(f"  Net exposure:             {(total_long - total_short) / nav * 100:>14.1f}%")
    
    # Sector breakdown
    print("\nSECTOR BREAKDOWN")
    print("-" * 40)
    sectors = {}
    for p in long_positions:
        if p['sector'] not in ['Index', 'Cash', 'Government']:
            sectors[p['sector']] = sectors.get(p['sector'], Decimal(0)) + p['market_value']
    
    sorted_sectors = sorted(sectors.items(), key=lambda x: x[1], reverse=True)
    for sector, value in sorted_sectors[:8]:
        pct = value / nav * 100
        print(f"  {sector:30} {pct:>6.1f}%  ${value:>12,.0f}")
    
    # Top positions
    print("\nTOP 10 POSITIONS")
    print("-" * 40)
    sorted_longs = sorted(long_positions, key=lambda x: x['market_value'], reverse=True)
    for p in sorted_longs[:10]:
        pct = p['market_value'] / nav * 100
        print(f"  {p['ticker']:6} {p['security_name'][:25]:25} {pct:>5.1f}%  ${p['market_value']:>12,.0f}")
    
    # Control results
    print("\nCONTROL RESULTS ({} total)".format(len(controls)))
    print("-" * 40)
    
    passes = [c for c in controls if c['status'] == 'pass']
    warnings = [c for c in controls if c['status'] == 'warning']
    fails = [c for c in controls if c['status'] == 'fail']
    
    print(f"  ✓ Passed:    {len(passes):>3}")
    print(f"  ⚠ Warnings:  {len(warnings):>3}")
    print(f"  ✗ Failed:    {len(fails):>3}")
    
    if fails:
        print("\nFAILED CONTROLS (Breaches)")
        print("-" * 40)
        for c in fails:
            print(f"  ✗ {c['control_name']}")
            print(f"      Value: {c['calculated_value']:.2f}%  Limit: {c['threshold']:.2f}%  Breach: {c['breach_amount']:.2f}%")
    
    if warnings:
        print("\nWARNING CONTROLS (Approaching Limits)")
        print("-" * 40)
        for c in warnings:
            print(f"  ⚠ {c['control_name']}")
            print(f"      Value: {c['calculated_value']:.2f}%  Limit: {c['threshold']:.2f}%")


def generate_sample_narratives(controls: list, policies_dir: Path):
    """Generate sample narratives showing what RAG would produce."""
    print("\n" + "=" * 70)
    print("SAMPLE RAG-GENERATED NARRATIVES")
    print("=" * 70)
    
    # Find breaches and warnings for narrative generation
    breaches = [c for c in controls if c['status'] == 'fail']
    warnings = [c for c in controls if c['status'] == 'warning']
    
    # Sample narrative for MSFT breach
    msft_breach = next((c for c in breaches if 'MSFT' in c['control_id']), None)
    if msft_breach:
        print("\n" + "-" * 70)
        print("NARRATIVE: Single Issuer Concentration Breach - Microsoft")
        print("-" * 70)
        print("""
FINDING:
The Fund's position in Microsoft Corporation (MSFT) exceeds the single issuer
concentration limit as of January 15, 2026.

CURRENT STATUS:
• Position Value: $218,400,000
• NAV: $2,000,000,000
• Concentration: 10.92%
• Limit: 10.00%
• Breach Amount: 0.92% (92 basis points)

POLICY REFERENCE:
Per the Concentration Limits Policy (POL-CONC-001), Section 3.1:
"Maximum Single Issuer Exposure: 10% of Net Asset Value (NAV)"

BREACH CLASSIFICATION:
Per Section 3.4 of the Concentration Limits Policy, this is classified as a
MINOR breach (10.0% - 10.5% range), requiring:
• Documentation within 24 hours
• Cure within 48 hours

RECOMMENDED ACTIONS:
1. Reduce MSFT position by approximately $18,400,000 (8,400 shares at current
   price) to bring concentration to 10.00%
2. Document root cause (likely passive breach due to MSFT price appreciation)
3. Consider implementing alert at 9.5% to prevent future breaches

ESCALATION:
Per Exception Management Policy (POL-EXC-001), minor breaches require PM
notification with CIO notification within 24 hours if not cured.
""")
    
    # Sample narrative for Net Exposure breach
    net_breach = next((c for c in breaches if c['control_id'] == 'EXP_NET_001'), None)
    if net_breach:
        print("\n" + "-" * 70)
        print("NARRATIVE: Net Exposure Limit Breach")
        print("-" * 70)
        print("""
FINDING:
The Fund's net exposure exceeds the maximum limit as of January 15, 2026.

CURRENT STATUS:
• Long Market Value: $2,689,625,000
• Short Market Value: $411,550,000
• Net Exposure: 113.90%
• Limit: 100.00%
• Breach Amount: 13.90%

POLICY REFERENCE:
Per the Exposure Limits Policy (POL-EXP-001), Section 4.1:
"Maximum Net Long: 100% of NAV"

BREACH CLASSIFICATION:
This is classified as a MAJOR breach (>10% over limit), requiring:
• Immediate CIO notification
• Same-day cure action
• CCO notification

ROOT CAUSE ANALYSIS:
• Current long positions total $2.69B against $2.0B NAV
• Short hedges of $411M are insufficient to bring net below 100%
• Requires either reduction of longs or increase of hedges

RECOMMENDED ACTIONS:
1. IMMEDIATE: Add index hedges (SPY, QQQ) totaling approximately $280M to
   bring net exposure to 100%
2. ALTERNATIVE: Reduce long positions by $280M across diversified holdings
3. Review hedge ratio policy with CIO

ESCALATION:
Per Exception Management Policy, major breaches require CIO cure within 24
hours with CCO notification. If not cured, escalates to CEO within 48 hours.
""")
    
    # Sample narrative for T+7 liquidity warning
    liq_warning = next((c for c in warnings if 'T7' in c['control_id']), None)
    if liq_warning:
        print("\n" + "-" * 70)
        print("NARRATIVE: T+7 Liquidity Warning")
        print("-" * 70)
        print("""
FINDING:
The Fund's T+7 liquidity is below the required minimum threshold.

CURRENT STATUS:
• T+7 Liquid Assets: $760,000,000
• NAV: $2,000,000,000
• T+7 Liquidity: 38.00%
• Minimum Required: 40.00%
• Shortfall: 2.00% ($40,000,000)

POLICY REFERENCE:
Per the Liquidity Risk Management Policy (POL-LIQ-001), Section 3.1:
"T+7 Liquidity Bucket Minimum: 40% of NAV"

BREACH CLASSIFICATION:
This is classified as a WARNING approaching breach (within 5% of minimum).
Current shortfall of 2% has crossed into minor breach territory.

ROOT CAUSE ANALYSIS:
• Several positions classified as T+7 or longer due to lower ADV
• International and emerging market positions contribute to lower liquidity
• Positions in ASML, NVO, BABA classified as T+3 or longer

RECOMMENDED ACTIONS:
1. Review liquidity classification for borderline positions
2. Consider reducing positions in less liquid names:
   - VALE (T+7): $18.85M
   - RIO (T+3): $21.38M
3. Build additional cash buffer from liquid position sales
4. Enhanced monitoring until T+7 liquidity exceeds 42%

REGULATORY CONTEXT:
Per SEC Rule 22e-4, funds must maintain liquidity consistent with redemption
terms. Our 45-day notice period provides substantial buffer, but maintaining
T+7 above 40% ensures adequate flexibility.
""")
    
    # Sample narrative for Tech sector warning
    tech_warning = next((c for c in warnings if 'TECH' in c['control_id']), None)
    if tech_warning:
        print("\n" + "-" * 70)
        print("NARRATIVE: Technology Sector Concentration Warning")
        print("-" * 70)
        print("""
FINDING:
Technology sector exposure is approaching the standard limit as of January
15, 2026.

CURRENT STATUS:
• Technology Sector Exposure: $578,700,000
• NAV: $2,000,000,000
• Concentration: 28.93%
• Standard Limit: 30.00%
• Exception Limit: 40.00% (with CIO approval)

POLICY REFERENCE:
Per the Concentration Limits Policy (POL-CONC-001), Section 4.2:
"The Technology sector is permitted up to 40% of NAV with prior CIO written
approval"

CURRENT TECHNOLOGY HOLDINGS:
• MSFT:  $218,400,000 (10.92%)
• AAPL:  $178,500,000 (8.93%)
• NVDA:  $156,800,000 (7.84%)
• AVGO:   $78,000,000 (3.90%)
• TSM:    $63,000,000 (3.15%)
• Others: Remaining technology positions

STATUS:
This is currently a WARNING, not a breach. The Technology sector has a
special exception allowing up to 40% with CIO approval.

RECOMMENDED ACTIONS:
1. Document current Technology thesis with PM
2. If exceeding 30% is intentional, obtain CIO written approval per policy
3. Implement enhanced monitoring at 28% threshold
4. Review Technology positions quarterly per policy requirement

NOTE: Short positions in QQQ (-$92.5M) partially offset Technology exposure
for risk purposes, though not for concentration calculation per policy.
""")


def main():
    print("=" * 70)
    print("COMPLIANCE RAG SYSTEM - HIGH-QUALITY DATA DEMO")
    print("=" * 70)
    print(f"As of: January 15, 2026")
    print(f"Fund: Master Fund LP")
    print(f"NAV: $2,000,000,000")
    
    # File paths
    data_dir = Path(__file__).parent / 'data'
    positions_file = data_dir / 'positions_20260115.csv'
    controls_file = data_dir / 'controls_20260115.csv'
    policies_dir = Path(__file__).parent / 'policies'
    
    # Check files exist
    if not positions_file.exists():
        print(f"\n✗ Position file not found: {positions_file}")
        return 1
    if not controls_file.exists():
        print(f"\n✗ Controls file not found: {controls_file}")
        return 1
    
    print(f"\n✓ Position file: {positions_file}")
    print(f"✓ Controls file: {controls_file}")
    
    # Load data
    print("\nLoading data...")
    positions = load_positions(str(positions_file))
    controls = load_controls(str(controls_file))
    
    print(f"✓ Loaded {len(positions)} positions")
    print(f"✓ Loaded {len(controls)} control results")
    
    # Analyze data
    analyze_data(positions, controls)
    
    # Generate sample narratives
    generate_sample_narratives(controls, policies_dir)
    
    print("\n" + "=" * 70)
    print("DATA QUALITY SUMMARY")
    print("=" * 70)
    print("""
This dataset demonstrates:

1. REALISTIC HEDGE FUND PORTFOLIO
   • 50+ long positions across all sectors
   • 5 short positions (index and sector hedges)
   • Cash and T-bills for liquidity
   • International exposure (EU, Asia, EM)

2. POLICY-ALIGNED CONTROL SCENARIOS
   • 3 BREACHES: MSFT concentration, Net exposure, Issuer limit
   • 2 WARNINGS: Tech sector approaching limit, T+7 liquidity
   • 30+ PASSES: Most controls within limits

3. COMPREHENSIVE POLICIES FOR RAG
   • Investment Guidelines (master policy)
   • Concentration Limits (with calculations)
   • Liquidity Policy (with buckets)
   • Exposure Limits (gross/net)
   • Exception Management (escalation procedures)
   • SEC Compliance (regulatory context)

4. RAG DEMONSTRATION VALUE
   • Breaches require policy citations
   • Calculations can be verified
   • Escalation paths are documented
   • Cure actions are specific
""")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
