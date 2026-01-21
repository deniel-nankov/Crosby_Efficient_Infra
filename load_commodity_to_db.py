#!/usr/bin/env python3
"""
Load real commodity data into PostgreSQL database.

This loads the commodity positions and controls generated from real market data
into the compliance database for the full pipeline test.
"""

import csv
import os
import sys
from datetime import date
from decimal import Decimal
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import psycopg2
from psycopg2.extras import execute_values

# Database config
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "5433")),
    "database": os.getenv("DB_NAME", "compliance"),
    "user": os.getenv("DB_USER", "compliance_user"),
    "password": os.getenv("DB_PASSWORD", "compliance_dev_password_123"),
}

DATA_DIR = Path(__file__).parent / "data"


def load_positions_from_csv(csv_path: Path):
    """Load positions from CSV file."""
    positions = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            positions.append({
                'security_id': row['security_id'],
                'ticker': row['ticker'],
                'security_name': row['security_name'],
                'quantity': int(float(row['quantity'])),
                'market_value': Decimal(row['market_value']),
                'currency': row['currency'],
                'sector': row['sector'],
                'issuer': row['issuer'],
                'asset_class': row.get('asset_class', 'equity'),
                'country': row.get('country', 'US'),
                'liquidity_days': int(row.get('liquidity_days', 1)),
            })
    return positions


def load_controls_from_csv(csv_path: Path):
    """Load controls from CSV file."""
    controls = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            controls.append({
                'control_id': row['control_id'],
                'control_name': row['control_name'],
                'control_type': row['control_type'],
                'calculated_value': Decimal(row['calculated_value']),
                'threshold': Decimal(row['threshold']),
                'threshold_operator': row['threshold_operator'],
                'status': row['status'],
                'details': row.get('details', '{}'),
            })
    return controls


def setup_database(conn):
    """Ensure tables exist."""
    cur = conn.cursor()
    
    # Check if tables exist
    cur.execute("""
        SELECT table_name FROM information_schema.tables 
        WHERE table_schema = 'public' 
        AND table_name IN ('fund_positions', 'fund_nav', 'fund_control_results')
    """)
    existing = {row[0] for row in cur.fetchall()}
    
    if 'fund_positions' not in existing:
        cur.execute("""
            CREATE TABLE fund_positions (
                id SERIAL PRIMARY KEY,
                as_of_date DATE NOT NULL,
                security_id VARCHAR(50) NOT NULL,
                ticker VARCHAR(20),
                security_name VARCHAR(255),
                quantity INTEGER,
                market_value DECIMAL(20, 2),
                currency VARCHAR(3) DEFAULT 'USD',
                sector VARCHAR(100),
                issuer VARCHAR(255),
                asset_class VARCHAR(50),
                country VARCHAR(50),
                liquidity_days INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        print("  Created fund_positions table")
    
    if 'fund_nav' not in existing:
        cur.execute("""
            CREATE TABLE fund_nav (
                id SERIAL PRIMARY KEY,
                as_of_date DATE NOT NULL,
                nav DECIMAL(20, 2) NOT NULL,
                currency VARCHAR(3) DEFAULT 'USD',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        print("  Created fund_nav table")
    
    if 'fund_control_results' not in existing:
        cur.execute("""
            CREATE TABLE fund_control_results (
                id SERIAL PRIMARY KEY,
                as_of_date DATE NOT NULL,
                control_id VARCHAR(50) NOT NULL,
                control_name VARCHAR(255),
                control_type VARCHAR(50),
                calculated_value DECIMAL(20, 6),
                threshold DECIMAL(20, 6),
                threshold_operator VARCHAR(10),
                status VARCHAR(20),
                details TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        print("  Created fund_control_results table")
    
    conn.commit()
    cur.close()


def clear_existing_data(conn, as_of_date: date):
    """Clear existing data for the date."""
    cur = conn.cursor()
    
    # Clear control results first (may have constraint issues)
    cur.execute("DELETE FROM fund_control_results WHERE as_of_date = %s", (as_of_date,))
    deleted_ctrl = cur.rowcount
    
    cur.execute("DELETE FROM fund_positions WHERE as_of_date = %s", (as_of_date,))
    deleted_pos = cur.rowcount
    
    cur.execute("DELETE FROM fund_nav WHERE as_of_date = %s", (as_of_date,))
    deleted_nav = cur.rowcount
    
    conn.commit()
    cur.close()
    
    if deleted_pos > 0 or deleted_nav > 0 or deleted_ctrl > 0:
        print(f"  Cleared existing data: {deleted_pos} positions, {deleted_nav} NAV, {deleted_ctrl} controls")


def insert_positions(conn, positions, as_of_date: date):
    """Insert positions into database."""
    cur = conn.cursor()
    
    # Use existing schema (no country or liquidity_days columns)
    data = [
        (
            as_of_date,
            p['security_id'],
            p['ticker'],
            p['security_name'],
            p['quantity'],
            p['market_value'],
            p['currency'],
            p['sector'],
            p['issuer'],
            p.get('asset_class', 'commodity'),
        )
        for p in positions
    ]
    
    execute_values(
        cur,
        """
        INSERT INTO fund_positions 
        (as_of_date, security_id, ticker, security_name, quantity, market_value, 
         currency, sector, issuer, asset_class)
        VALUES %s
        """,
        data
    )
    
    conn.commit()
    cur.close()
    print(f"  Inserted {len(positions)} positions")


def insert_nav(conn, nav: Decimal, as_of_date: date):
    """Insert NAV into database."""
    cur = conn.cursor()
    
    # Use existing schema (no currency column)
    cur.execute(
        "INSERT INTO fund_nav (as_of_date, nav) VALUES (%s, %s)",
        (as_of_date, nav)
    )
    
    conn.commit()
    cur.close()
    print(f"  Inserted NAV: ${nav:,.0f}")


def insert_controls(conn, controls, as_of_date: date):
    """Insert control results into database."""
    cur = conn.cursor()
    
    data = [
        (
            as_of_date,
            c['control_id'],
            c['control_name'],
            c['control_type'],
            c['calculated_value'],
            c['threshold'],
            c['threshold_operator'],
            c['status'],
            c.get('details', '{}'),
        )
        for c in controls
    ]
    
    execute_values(
        cur,
        """
        INSERT INTO fund_control_results 
        (as_of_date, control_id, control_name, control_type, calculated_value, 
         threshold, threshold_operator, status, details)
        VALUES %s
        """,
        data
    )
    
    conn.commit()
    cur.close()
    print(f"  Inserted {len(controls)} control results")


def main():
    print("=" * 70)
    print("LOADING REAL COMMODITY DATA INTO POSTGRESQL")
    print("=" * 70)
    
    as_of_date = date.today()
    nav = Decimal('2000000000')
    
    # Load data from CSV
    positions_file = DATA_DIR / "commodity_positions_20260117.csv"
    controls_file = DATA_DIR / "commodity_controls_20260117.csv"
    
    print(f"\nLoading data files...")
    print(f"  Positions: {positions_file}")
    print(f"  Controls: {controls_file}")
    
    positions = load_positions_from_csv(positions_file)
    controls = load_controls_from_csv(controls_file)
    
    print(f"\nLoaded {len(positions)} positions, {len(controls)} controls")
    
    # Connect to database
    print(f"\nConnecting to database: {DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}")
    
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        print("  Connected successfully!")
    except Exception as e:
        print(f"  ERROR: {e}")
        return 1
    
    # Setup and load
    print("\nSetting up tables...")
    setup_database(conn)
    
    print(f"\nLoading data for {as_of_date}...")
    clear_existing_data(conn, as_of_date)
    insert_nav(conn, nav, as_of_date)
    insert_positions(conn, positions, as_of_date)
    insert_controls(conn, controls, as_of_date)
    
    # Verify
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM fund_positions WHERE as_of_date = %s", (as_of_date,))
    pos_count = cur.fetchone()[0]
    
    cur.execute("SELECT nav FROM fund_nav WHERE as_of_date = %s", (as_of_date,))
    nav_result = cur.fetchone()
    
    cur.execute("SELECT COUNT(*) FROM fund_control_results WHERE as_of_date = %s", (as_of_date,))
    ctrl_count = cur.fetchone()[0]
    
    cur.close()
    conn.close()
    
    print("\n" + "=" * 70)
    print("VERIFICATION")
    print("=" * 70)
    print(f"  Positions in DB: {pos_count}")
    print(f"  NAV in DB:       ${nav_result[0]:,.0f}" if nav_result else "  NAV: NOT FOUND")
    print(f"  Controls in DB:  {ctrl_count}")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
