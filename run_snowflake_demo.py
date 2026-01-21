#!/usr/bin/env python3
"""
=============================================================================
SNOWFLAKE INTEGRATION DEMO
=============================================================================

This script demonstrates how to use the Snowflake adapter with the 
Compliance RAG pipeline. It shows both mock mode (for testing) and 
production mode (with real Snowflake credentials).

Usage:
    # Mock mode (no credentials needed)
    python run_snowflake_demo.py --mock
    
    # Production mode (requires env vars)
    export SNOWFLAKE_ACCOUNT=xy12345.us-east-1
    export SNOWFLAKE_USER=compliance_svc
    export SNOWFLAKE_PASSWORD=your_password
    export SNOWFLAKE_WAREHOUSE=COMPLIANCE_WH
    export SNOWFLAKE_DATABASE=HEDGE_FUND_DATA
    export SNOWFLAKE_SCHEMA=COMPLIANCE
    python run_snowflake_demo.py

=============================================================================
"""

import argparse
import sys
from datetime import date
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from integration import (
    SnowflakeAdapter,
    SnowflakeConfig,
    SnowflakeViewConfig,
    get_snowflake_adapter,
)


def print_section(title: str):
    """Print a section header."""
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def run_demo(use_mock: bool = True):
    """Run the Snowflake integration demo."""
    
    print_section("SNOWFLAKE INTEGRATION DEMO")
    
    # =========================================================================
    # STEP 1: Configure Adapter
    # =========================================================================
    print_section("STEP 1: Configuring Snowflake Adapter")
    
    if use_mock:
        print("  Mode: MOCK (no Snowflake credentials needed)")
        print("  Use --no-mock flag with environment variables for production")
        adapter = get_snowflake_adapter(use_mock=True)
    else:
        print("  Mode: PRODUCTION (using environment variables)")
        try:
            config = SnowflakeConfig.from_env()
            print(f"  Account: {config.account}")
            print(f"  Database: {config.database}.{config.schema}")
            print(f"  Warehouse: {config.warehouse}")
            
            # Custom view configuration (modify for client's schema)
            view_config = SnowflakeViewConfig(
                positions_view="V_POSITIONS_CURRENT",
                controls_view="V_CONTROL_RESULTS",
                nav_view="V_NAV_DAILY",
            )
            
            adapter = SnowflakeAdapter(config, view_config)
            
        except Exception as e:
            print(f"\n  ERROR: {e}")
            print("\n  Set these environment variables:")
            print("    SNOWFLAKE_ACCOUNT")
            print("    SNOWFLAKE_USER")
            print("    SNOWFLAKE_PASSWORD")
            print("    SNOWFLAKE_WAREHOUSE")
            print("    SNOWFLAKE_DATABASE")
            print("    SNOWFLAKE_SCHEMA")
            return 1
    
    # =========================================================================
    # STEP 2: Get Data Snapshot
    # =========================================================================
    print_section("STEP 2: Fetching Data Snapshot")
    
    as_of_date = date.today()
    print(f"  As of Date: {as_of_date}")
    
    try:
        snapshot = adapter.get_snapshot(as_of_date)
        print(f"\n  Snapshot ID: {snapshot.snapshot_id}")
        print(f"  Source: {snapshot.source_system}")
        print(f"  Positions: {len(snapshot.positions)}")
        print(f"  Control Results: {len(snapshot.control_results)}")
        print(f"  NAV: ${snapshot.nav:,.0f}")
        print(f"  Data Hash: {snapshot.data_hash}")
        
    except Exception as e:
        print(f"\n  ERROR fetching snapshot: {e}")
        return 1
    
    # =========================================================================
    # STEP 3: Analyze Positions
    # =========================================================================
    print_section("STEP 3: Position Analysis")
    
    # Sort by market value
    sorted_positions = sorted(
        snapshot.positions, 
        key=lambda x: x.market_value, 
        reverse=True
    )
    
    print("\n  Top 10 Positions by Market Value:")
    print("  " + "-" * 60)
    for i, pos in enumerate(sorted_positions[:10], 1):
        pct = (pos.market_value / snapshot.nav * 100) if snapshot.nav else 0
        print(f"  {i:2}. {pos.ticker:8s} {pos.security_name[:25]:25s} ${pos.market_value:>15,.0f}  ({pct:.2f}%)")
    
    # Sector breakdown
    sectors = {}
    for pos in snapshot.positions:
        sector = pos.sector or "Unknown"
        sectors[sector] = sectors.get(sector, 0) + float(pos.market_value)
    
    print("\n  Sector Breakdown:")
    print("  " + "-" * 60)
    for sector, value in sorted(sectors.items(), key=lambda x: -x[1]):
        pct = (value / float(snapshot.nav) * 100) if snapshot.nav else 0
        print(f"  {sector:30s} ${value:>15,.0f}  ({pct:.1f}%)")
    
    # =========================================================================
    # STEP 4: Control Results Summary
    # =========================================================================
    print_section("STEP 4: Control Results Summary")
    
    # Count by status
    status_counts = {"pass": 0, "warning": 0, "fail": 0}
    for ctrl in snapshot.control_results:
        status_counts[ctrl.status] = status_counts.get(ctrl.status, 0) + 1
    
    print(f"\n  ✅ Passed:   {status_counts['pass']}")
    print(f"  ⚠️  Warnings: {status_counts['warning']}")
    print(f"  ❌ Failed:   {status_counts['fail']}")
    
    print("\n  All Control Results:")
    print("  " + "-" * 60)
    for ctrl in snapshot.control_results:
        status_icon = {"pass": "✅", "warning": "⚠️", "fail": "❌"}.get(ctrl.status, "?")
        print(f"  {status_icon} [{ctrl.control_id}] {ctrl.control_name}")
        print(f"      Value: {ctrl.calculated_value}  Threshold: {ctrl.threshold} ({ctrl.threshold_operator})")
    
    # =========================================================================
    # STEP 5: Integration with Pipeline
    # =========================================================================
    print_section("STEP 5: Ready for Pipeline Integration")
    
    print("""
  The snapshot is ready to be passed to the compliance pipeline:
  
  from src.orchestrator import ComplianceOrchestrator
  
  orchestrator = ComplianceOrchestrator()
  result = orchestrator.run_daily_compliance(
      snapshot=snapshot,
      generate_pdf=True,
  )
  
  Output:
    - PDF workpaper with narrative commentary
    - Audit trail with data hashes
    - Evidence store entries
  """)
    
    # =========================================================================
    # DONE
    # =========================================================================
    print_section("DEMO COMPLETE")
    
    print(f"""
  Successfully demonstrated Snowflake integration!
  
  Next Steps for Production:
  1. Deploy Snowflake views (schemas/snowflake_views.sql)
  2. Create service account with COMPLIANCE_READER role
  3. Set environment variables
  4. Run: python run_snowflake_demo.py --no-mock
  5. Integrate with daily_compliance_dag.py for automation
  """)
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Snowflake integration demo for Compliance RAG"
    )
    parser.add_argument(
        "--mock", 
        action="store_true",
        default=True,
        help="Use mock adapter (no credentials needed)"
    )
    parser.add_argument(
        "--no-mock", 
        action="store_true",
        help="Use real Snowflake connection (requires env vars)"
    )
    
    args = parser.parse_args()
    
    use_mock = not args.no_mock
    
    return run_demo(use_mock=use_mock)


if __name__ == "__main__":
    sys.exit(main())
