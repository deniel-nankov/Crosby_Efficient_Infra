#!/usr/bin/env python3
"""
Quick Start Script - Load sample data and run compliance report

This replaces Snowflake with local PostgreSQL (FREE).
Run this after `docker-compose up -d` to get started.
"""

import sys
from datetime import date
from decimal import Decimal
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def main():
    print("=" * 60)
    print("Compliance RAG System - Quick Start")
    print("=" * 60)
    print()
    
    # Step 1: Check database connection
    print("Step 1: Connecting to PostgreSQL...")
    try:
        from integration.postgres_adapter import PostgresDataSource, PostgresConfig
        
        # Try to connect
        config = PostgresConfig(
            host="localhost",
            port=5432,
            database="compliance",
            user="compliance_user",
            password="compliance_dev_password_123",
        )
        source = PostgresDataSource(config)
        source.create_data_tables()
        print("   ✓ Connected to PostgreSQL")
        print("   ✓ Tables created")
    except Exception as e:
        print(f"   ✗ Database connection failed: {e}")
        print()
        print("   Make sure Docker is running:")
        print("   $ docker-compose up -d postgres")
        return 1
    
    # Step 2: Load sample data
    print()
    print("Step 2: Loading sample data...")
    today = date.today()
    try:
        source.load_sample_data(today)
        print(f"   ✓ Loaded positions for {today}")
        print(f"   ✓ Loaded control results for {today}")
        print(f"   ✓ Set NAV to $2,000,000,000")
    except Exception as e:
        print(f"   ✗ Failed to load sample data: {e}")
        return 1
    
    # Step 3: Verify data
    print()
    print("Step 3: Verifying data...")
    positions = source.get_positions(today)
    controls = source.get_control_results(today)
    nav = source.get_nav(today)
    
    print(f"   ✓ {len(positions)} positions loaded")
    print(f"   ✓ {len(controls)} control results loaded")
    print(f"   ✓ NAV: ${nav:,.2f}")
    
    # Step 4: Show sample position
    print()
    print("Step 4: Sample Position Data:")
    print("   " + "-" * 50)
    for pos in positions[:3]:
        print(f"   {pos['ticker']:6} | ${pos['market_value']:>12,.2f} | {pos['sector']}")
    print("   ...")
    
    # Step 5: Show control results
    print()
    print("Step 5: Control Results Summary:")
    print("   " + "-" * 50)
    for ctrl in controls:
        status_icon = "✓" if ctrl['status'] == 'pass' else "⚠" if ctrl['status'] == 'warning' else "✗"
        print(f"   {status_icon} {ctrl['control_name'][:35]:35} | {ctrl['calculated_value']:>6.1f}% vs {ctrl['threshold']:>5.1f}%")
    
    # Step 6: Generate narrative (mock)
    print()
    print("Step 6: Generating compliance narrative...")
    print("   " + "-" * 50)
    
    # Simple narrative generation without LLM
    breaches = [c for c in controls if c['status'] != 'pass']
    passes = [c for c in controls if c['status'] == 'pass']
    
    print()
    print("   COMPLIANCE REPORT")
    print(f"   As of: {today}")
    print(f"   NAV: ${nav:,.0f}")
    print()
    print(f"   Summary:")
    print(f"   - {len(passes)} controls PASSED")
    print(f"   - {len(breaches)} controls require attention")
    print()
    
    if breaches:
        print("   Items Requiring Attention:")
        for b in breaches:
            print(f"   • {b['control_name']}: {b['calculated_value']:.1f}% (limit: {b['threshold']:.1f}%)")
    
    print()
    print("=" * 60)
    print("✓ Quick start complete!")
    print()
    print("Next steps:")
    print("  1. Replace sample data with your real data (CSV files in data/)")
    print("  2. Add policy documents to policies/")
    print("  3. Start Ollama: docker-compose up -d ollama")
    print("  4. Run full pipeline: python run_demo.py")
    print("=" * 60)
    
    source.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
