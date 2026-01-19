#!/usr/bin/env python3
"""
Data Integration Demo - Step-by-Step Walkthrough

This script demonstrates the complete data integration pipeline
with quality validation at each step.

Run with: python run_data_integration.py
"""

import os
import sys
from pathlib import Path
from datetime import date
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Load environment
load_dotenv()


def print_header(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def print_step(step_num: int, description: str):
    """Print a step indicator."""
    print(f"\n{'─' * 50}")
    print(f"  STEP {step_num}: {description}")
    print(f"{'─' * 50}\n")


def main():
    print_header("COMPLIANCE RAG - DATA INTEGRATION DEMO")
    print("This demo walks through the complete data integration pipeline")
    print("with quality validation at every step.\n")
    
    # ==========================================================================
    # STEP 1: Initialize Components
    # ==========================================================================
    print_step(1, "Initialize Data Quality Components")
    
    from src.data_quality import (
        PositionDataValidator,
        PolicyDocumentValidator,
        ControlDefinitionValidator,
        PolicyIngestionPipeline,
        PolicyStore,
        get_connector,
        DataIntegrationOrchestrator,
        get_sample_controls,
    )
    
    print("✓ Imported data quality validators")
    print("✓ Imported Snowflake connector")
    print("✓ Imported policy ingestion pipeline")
    print("✓ Imported orchestrator")
    
    # ==========================================================================
    # STEP 2: Connect to Data Sources
    # ==========================================================================
    print_step(2, "Connect to Data Sources")
    
    # PostgreSQL
    postgres_conn = None
    try:
        import psycopg2
        postgres_conn = psycopg2.connect(
            host=os.environ.get('POSTGRES_HOST', 'localhost'),
            port=int(os.environ.get('POSTGRES_PORT', 5432)),
            database=os.environ.get('POSTGRES_DB', 'compliance'),
            user=os.environ.get('POSTGRES_USER', 'compliance_user'),
            password=os.environ.get('POSTGRES_PASSWORD', 'compliance_pass'),
        )
        print("✓ Connected to PostgreSQL")
    except Exception as e:
        print(f"⚠ PostgreSQL not available: {e}")
        print("  (Continuing in demo mode)")
    
    # Snowflake (will use mock if not configured)
    snowflake = get_connector(use_mock=True)  # Use mock for demo
    print("✓ Snowflake connector initialized (mock mode)")
    
    # ==========================================================================
    # STEP 3: Extract Sample Position Data
    # ==========================================================================
    print_step(3, "Extract Position Data from Snowflake")
    
    snowflake.connect()
    
    # Get available snapshots
    snapshots = snowflake.get_available_snapshots(days_back=3)
    print(f"✓ Found {len(snapshots)} available snapshots:")
    for snap in snapshots[:3]:
        print(f"    - {snap.snapshot_id} ({snap.snapshot_date}): {snap.record_count} records")
    
    # Get positions from latest snapshot
    latest_snapshot = snapshots[0]
    positions, snapshot_meta = snowflake.get_positions_for_snapshot(latest_snapshot.snapshot_id)
    print(f"\n✓ Extracted {len(positions)} positions from {latest_snapshot.snapshot_id}")
    
    # Show sample position
    if positions:
        sample = positions[0]
        print(f"\n  Sample position:")
        print(f"    Security: {sample['ticker']} ({sample['isin']})")
        print(f"    Quantity: {sample['quantity']:,}")
        print(f"    Market Value: ${sample['market_value_usd']:,.2f}")
        print(f"    Sector: {sample['sector']}")
    
    # ==========================================================================
    # STEP 4: Validate Position Data Quality
    # ==========================================================================
    print_step(4, "Validate Position Data Quality")
    
    validator = PositionDataValidator()
    quality_report = validator.validate(positions, snapshot_meta.snapshot_date)
    
    print(quality_report.to_summary())
    
    if quality_report.is_acceptable:
        print("✅ DATA QUALITY GATE: PASSED")
    else:
        print("❌ DATA QUALITY GATE: FAILED")
        print(f"   Reason: {quality_report.rejection_reason}")
        print("\n   Critical Issues:")
        for issue in quality_report.critical_issues[:5]:
            print(f"   - {issue.field_name}: {issue.message}")
    
    # ==========================================================================
    # STEP 5: Ingest Policy Documents
    # ==========================================================================
    print_step(5, "Ingest Policy Documents")
    
    policy_pipeline = PolicyIngestionPipeline()
    policy_store = PolicyStore()  # In-memory for demo
    
    # Check if investment guidelines exist
    policy_path = Path(__file__).parent / 'policies' / 'investment_guidelines.md'
    
    if policy_path.exists():
        policy_doc = policy_pipeline.ingest_file(policy_path)
        policy_store.store_policy(policy_doc)
        
        print(f"✓ Ingested policy: {policy_doc.policy_id}")
        print(f"  Title: {policy_doc.title}")
        print(f"  Version: {policy_doc.version}")
        print(f"  Chunks created: {len(policy_doc.chunks)}")
        
        if policy_doc.quality_report:
            print(f"  Quality score: {policy_doc.quality_report.overall_score:.1f}%")
        
        # Show sample chunks
        print(f"\n  Sample sections:")
        for chunk in policy_doc.chunks[:5]:
            print(f"    - {chunk.section_path[:50]}...")
    else:
        print(f"⚠ Policy file not found: {policy_path}")
    
    # ==========================================================================
    # STEP 6: Validate Control Definitions
    # ==========================================================================
    print_step(6, "Validate Control Definitions")
    
    control_validator = ControlDefinitionValidator()
    sample_controls = get_sample_controls()
    
    print(f"Validating {len(sample_controls)} control definitions...\n")
    
    valid_count = 0
    for control in sample_controls:
        report = control_validator.validate_control(control)
        status = "✓" if report.is_acceptable else "✗"
        print(f"  {status} {control['control_code']}: {control['control_name']}")
        if report.is_acceptable:
            valid_count += 1
        else:
            for issue in report.critical_issues:
                print(f"      Issue: {issue.message}")
    
    print(f"\n  Summary: {valid_count}/{len(sample_controls)} controls valid")
    
    # ==========================================================================
    # STEP 7: Run Complete Integration (Optional)
    # ==========================================================================
    print_step(7, "Run Complete Integration Pipeline")
    
    orchestrator = DataIntegrationOrchestrator(
        postgres_conn=postgres_conn,
        use_mock=True,  # Use mock Snowflake
        quality_threshold=95.0,
    )
    
    print("Running daily sync with quality gates...\n")
    
    run = orchestrator.run_daily_sync(
        policy_dir=Path(__file__).parent / 'policies',
    )
    
    print(f"Integration Run: {run.run_id}")
    print(f"Status: {run.status}")
    print(f"Positions: {run.position_count}")
    print(f"Policies: {run.policy_count}")
    
    if run.position_quality:
        print(f"Position Quality: {run.position_quality.overall_score:.1f}%")
    
    if run.error_message:
        print(f"Error: {run.error_message}")
    
    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    print_header("INTEGRATION COMPLETE")
    
    print("""
Next Steps:
-----------
1. Configure Snowflake credentials in .env file
2. Deploy Snowflake views from schemas/snowflake_views.sql
3. Run with real data: python run_data_integration.py --live
4. Monitor quality reports in PostgreSQL

Quality Thresholds:
------------------
- Completeness: All required fields populated
- Accuracy: Market values match qty × price within 5%
- Timeliness: Prices not stale (>2 days)
- Uniqueness: No duplicate positions
- Overall: Must score ≥95% to pass quality gate

For production deployment, see docs/DATA_INTEGRATION_GUIDE.md
""")
    
    # Cleanup
    if postgres_conn:
        postgres_conn.close()
    snowflake.disconnect()


if __name__ == '__main__':
    main()
