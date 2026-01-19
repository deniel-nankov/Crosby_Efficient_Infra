"""
PostgreSQL Adapter - Alternative to Snowflake

Uses the local PostgreSQL database (already in Docker stack) as the data source.
This is a FREE alternative to Snowflake that works identically.

Data can be loaded via:
1. CSV files (simplest)
2. Direct INSERT statements
3. Python scripts
4. dbt transformations
"""

from __future__ import annotations

import os
import logging
from datetime import date
from decimal import Decimal
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PostgresConfig:
    """PostgreSQL connection configuration."""
    host: str = "localhost"
    port: int = 5432
    database: str = "compliance"
    user: str = "compliance_user"
    password: str = "compliance_dev_password_123"
    
    @classmethod
    def from_env(cls) -> "PostgresConfig":
        """Create config from environment variables."""
        return cls(
            host=os.environ.get("POSTGRES_HOST", "localhost"),
            port=int(os.environ.get("POSTGRES_PORT", "5432")),
            database=os.environ.get("POSTGRES_DB", "compliance"),
            user=os.environ.get("POSTGRES_USER", "compliance_user"),
            password=os.environ.get("POSTGRES_PASSWORD", "compliance_dev_password_123"),
        )
    
    @property
    def connection_string(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


class PostgresDataSource:
    """
    PostgreSQL as data source (Snowflake alternative).
    
    Your data flows:
    1. Export from Bloomberg/Eze → CSV files
    2. Load CSVs into PostgreSQL tables
    3. RAG system reads from PostgreSQL
    
    Or use the load_sample_data() method for testing.
    """
    
    def __init__(self, config: Optional[PostgresConfig] = None):
        self.config = config or PostgresConfig.from_env()
        self._connection = None
    
    @property
    def connection(self):
        """Lazy connection initialization."""
        if self._connection is None:
            try:
                import psycopg2
                self._connection = psycopg2.connect(
                    host=self.config.host,
                    port=self.config.port,
                    database=self.config.database,
                    user=self.config.user,
                    password=self.config.password,
                )
                logger.info(f"Connected to PostgreSQL: {self.config.host}:{self.config.port}")
            except ImportError:
                raise ImportError("psycopg2 not installed. Run: pip install psycopg2-binary")
        return self._connection
    
    def create_data_tables(self):
        """Create the data tables (run once)."""
        cursor = self.connection.cursor()
        
        # Positions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS fund_positions (
                id SERIAL PRIMARY KEY,
                security_id VARCHAR(50) NOT NULL,
                ticker VARCHAR(20),
                security_name VARCHAR(500) NOT NULL,
                quantity DECIMAL(20,4) NOT NULL,
                market_value DECIMAL(20,2) NOT NULL,
                currency VARCHAR(3) DEFAULT 'USD',
                sector VARCHAR(100),
                issuer VARCHAR(200),
                asset_class VARCHAR(50),
                as_of_date DATE NOT NULL,
                created_at TIMESTAMP DEFAULT NOW(),
                UNIQUE(security_id, as_of_date)
            )
        """)
        
        # Control results table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS fund_control_results (
                id SERIAL PRIMARY KEY,
                control_id VARCHAR(100) NOT NULL,
                control_name VARCHAR(500) NOT NULL,
                control_type VARCHAR(50) NOT NULL,
                calculated_value DECIMAL(10,4) NOT NULL,
                threshold DECIMAL(10,4) NOT NULL,
                threshold_operator VARCHAR(10) NOT NULL,
                status VARCHAR(20) NOT NULL,
                breach_amount DECIMAL(10,4),
                details JSONB,
                as_of_date DATE NOT NULL,
                created_at TIMESTAMP DEFAULT NOW(),
                UNIQUE(control_id, as_of_date)
            )
        """)
        
        # NAV table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS fund_nav (
                id SERIAL PRIMARY KEY,
                nav DECIMAL(20,2) NOT NULL,
                as_of_date DATE NOT NULL UNIQUE,
                created_at TIMESTAMP DEFAULT NOW()
            )
        """)
        
        self.connection.commit()
        cursor.close()
        logger.info("Data tables created successfully")
    
    def load_positions_from_csv(self, csv_path: str, as_of_date: date):
        """
        Load positions from a CSV file.
        
        Expected CSV columns:
        security_id,ticker,security_name,quantity,market_value,currency,sector,issuer,asset_class
        """
        import csv
        
        cursor = self.connection.cursor()
        
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                cursor.execute("""
                    INSERT INTO fund_positions 
                    (security_id, ticker, security_name, quantity, market_value, 
                     currency, sector, issuer, asset_class, as_of_date)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (security_id, as_of_date) DO UPDATE SET
                        quantity = EXCLUDED.quantity,
                        market_value = EXCLUDED.market_value
                """, (
                    row['security_id'],
                    row.get('ticker'),
                    row['security_name'],
                    Decimal(row['quantity']),
                    Decimal(row['market_value']),
                    row.get('currency', 'USD'),
                    row.get('sector'),
                    row.get('issuer'),
                    row.get('asset_class'),
                    as_of_date,
                ))
        
        self.connection.commit()
        cursor.close()
        logger.info(f"Loaded positions from {csv_path}")
    
    def load_controls_from_csv(self, csv_path: str, as_of_date: date):
        """
        Load control results from a CSV file.
        
        Expected CSV columns:
        control_id,control_name,control_type,calculated_value,threshold,threshold_operator,status,breach_amount
        """
        import csv
        
        cursor = self.connection.cursor()
        
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                breach = row.get('breach_amount')
                cursor.execute("""
                    INSERT INTO fund_control_results 
                    (control_id, control_name, control_type, calculated_value, 
                     threshold, threshold_operator, status, breach_amount, as_of_date)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (control_id, as_of_date) DO UPDATE SET
                        calculated_value = EXCLUDED.calculated_value,
                        status = EXCLUDED.status,
                        breach_amount = EXCLUDED.breach_amount
                """, (
                    row['control_id'],
                    row['control_name'],
                    row['control_type'],
                    Decimal(row['calculated_value']),
                    Decimal(row['threshold']),
                    row['threshold_operator'],
                    row['status'],
                    Decimal(breach) if breach else None,
                    as_of_date,
                ))
        
        self.connection.commit()
        cursor.close()
        logger.info(f"Loaded controls from {csv_path}")
    
    def set_nav(self, nav: Decimal, as_of_date: date):
        """Set NAV for a date."""
        cursor = self.connection.cursor()
        cursor.execute("""
            INSERT INTO fund_nav (nav, as_of_date)
            VALUES (%s, %s)
            ON CONFLICT (as_of_date) DO UPDATE SET nav = EXCLUDED.nav
        """, (nav, as_of_date))
        self.connection.commit()
        cursor.close()
    
    def get_positions(self, as_of_date: date) -> List[Dict[str, Any]]:
        """Get positions for a date."""
        cursor = self.connection.cursor()
        cursor.execute("""
            SELECT security_id, ticker, security_name, quantity, market_value,
                   currency, sector, issuer, asset_class
            FROM fund_positions
            WHERE as_of_date = %s
            ORDER BY market_value DESC
        """, (as_of_date,))
        
        columns = ['security_id', 'ticker', 'security_name', 'quantity', 'market_value',
                   'currency', 'sector', 'issuer', 'asset_class']
        positions = [dict(zip(columns, row)) for row in cursor.fetchall()]
        cursor.close()
        return positions
    
    def get_control_results(self, as_of_date: date) -> List[Dict[str, Any]]:
        """Get control results for a date."""
        cursor = self.connection.cursor()
        cursor.execute("""
            SELECT control_id, control_name, control_type, calculated_value,
                   threshold, threshold_operator, status, breach_amount, details
            FROM fund_control_results
            WHERE as_of_date = %s
            ORDER BY 
                CASE status WHEN 'fail' THEN 1 WHEN 'warning' THEN 2 ELSE 3 END
        """, (as_of_date,))
        
        columns = ['control_id', 'control_name', 'control_type', 'calculated_value',
                   'threshold', 'threshold_operator', 'status', 'breach_amount', 'details']
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        cursor.close()
        return results
    
    def get_nav(self, as_of_date: date) -> Decimal:
        """Get NAV for a date."""
        cursor = self.connection.cursor()
        cursor.execute("SELECT nav FROM fund_nav WHERE as_of_date = %s", (as_of_date,))
        result = cursor.fetchone()
        cursor.close()
        
        if result:
            return Decimal(str(result[0]))
        raise ValueError(f"No NAV found for {as_of_date}")
    
    def load_sample_data(self, as_of_date: Optional[date] = None):
        """
        Load sample data for testing.
        Call this to populate the database with realistic test data.
        """
        if as_of_date is None:
            as_of_date = date.today()
        
        cursor = self.connection.cursor()
        
        # Sample positions
        positions = [
            ('SEC001', 'AAPL', 'Apple Inc', 100000, 15000000, 'USD', 'Technology', 'Apple Inc', 'equity'),
            ('SEC002', 'MSFT', 'Microsoft Corp', 80000, 12000000, 'USD', 'Technology', 'Microsoft Corp', 'equity'),
            ('SEC003', 'GOOGL', 'Alphabet Inc', 50000, 8500000, 'USD', 'Technology', 'Alphabet Inc', 'equity'),
            ('SEC004', 'JPM', 'JPMorgan Chase', 120000, 9500000, 'USD', 'Financials', 'JPMorgan Chase', 'equity'),
            ('SEC005', 'JNJ', 'Johnson & Johnson', 90000, 7200000, 'USD', 'Healthcare', 'Johnson & Johnson', 'equity'),
            ('SEC006', 'XOM', 'Exxon Mobil', 150000, 6800000, 'USD', 'Energy', 'Exxon Mobil', 'equity'),
            ('SEC007', 'PG', 'Procter & Gamble', 70000, 5500000, 'USD', 'Consumer Staples', 'Procter & Gamble', 'equity'),
            ('SEC008', 'NVDA', 'NVIDIA Corp', 25000, 11000000, 'USD', 'Technology', 'NVIDIA Corp', 'equity'),
            ('SEC009', 'V', 'Visa Inc', 60000, 8000000, 'USD', 'Financials', 'Visa Inc', 'equity'),
            ('SEC010', 'UNH', 'UnitedHealth', 40000, 7500000, 'USD', 'Healthcare', 'UnitedHealth Group', 'equity'),
        ]
        
        for pos in positions:
            cursor.execute("""
                INSERT INTO fund_positions 
                (security_id, ticker, security_name, quantity, market_value, 
                 currency, sector, issuer, asset_class, as_of_date)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (security_id, as_of_date) DO NOTHING
            """, (*pos, as_of_date))
        
        # Sample control results
        controls = [
            ('CONC_ISSUER_001', 'Single Issuer Concentration', 'concentration', 7.5, 10.0, 'lte', 'pass', None),
            ('CONC_SECTOR_001', 'Sector Concentration - Technology', 'concentration', 28.0, 30.0, 'lte', 'warning', None),
            ('EXP_GROSS_001', 'Gross Exposure', 'exposure', 145.0, 200.0, 'lte', 'pass', None),
            ('EXP_NET_001', 'Net Exposure', 'exposure', 72.0, 100.0, 'lte', 'pass', None),
            ('LIQ_T1_001', 'T+1 Liquidity', 'liquidity', 18.0, 10.0, 'gte', 'pass', None),
            ('LIQ_T7_001', 'T+7 Liquidity', 'liquidity', 35.0, 40.0, 'gte', 'warning', 5.0),
            ('CASH_MIN_001', 'Minimum Cash Buffer', 'liquidity', 3.2, 2.0, 'gte', 'pass', None),
        ]
        
        for ctrl in controls:
            cursor.execute("""
                INSERT INTO fund_control_results 
                (control_id, control_name, control_type, calculated_value, 
                 threshold, threshold_operator, status, breach_amount, as_of_date)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (control_id, as_of_date) DO NOTHING
            """, (*ctrl, as_of_date))
        
        # Sample NAV
        cursor.execute("""
            INSERT INTO fund_nav (nav, as_of_date)
            VALUES (%s, %s)
            ON CONFLICT (as_of_date) DO NOTHING
        """, (Decimal('2000000000'), as_of_date))
        
        self.connection.commit()
        cursor.close()
        logger.info(f"Sample data loaded for {as_of_date}")
    
    def close(self):
        """Close the connection."""
        if self._connection:
            self._connection.close()
            self._connection = None


class PostgresAdapter:
    """
    Adapter that wraps PostgresDataSource to match ClientSystemAdapter interface.
    Drop-in replacement for SnowflakeAdapter.
    """
    
    def __init__(self, config: Optional[PostgresConfig] = None):
        self.source = PostgresDataSource(config)
    
    def get_snapshot(self, as_of_date: date):
        """Get complete data snapshot from PostgreSQL."""
        from .client_adapter import Position, ControlResult, DataSnapshot
        
        positions_data = self.source.get_positions(as_of_date)
        controls_data = self.source.get_control_results(as_of_date)
        nav = self.source.get_nav(as_of_date)
        
        positions = [
            Position(
                security_id=p['security_id'],
                ticker=p.get('ticker'),
                security_name=p['security_name'],
                quantity=Decimal(str(p['quantity'])),
                market_value=Decimal(str(p['market_value'])),
                currency=p.get('currency', 'USD'),
                sector=p.get('sector'),
                issuer=p.get('issuer'),
                asset_class=p.get('asset_class'),
            )
            for p in positions_data
        ]
        
        control_results = [
            ControlResult(
                control_id=c['control_id'],
                control_name=c['control_name'],
                control_type=c['control_type'],
                calculated_value=Decimal(str(c['calculated_value'])),
                threshold=Decimal(str(c['threshold'])),
                threshold_operator=c['threshold_operator'],
                status=c['status'],
                breach_amount=Decimal(str(c['breach_amount'])) if c.get('breach_amount') else None,
                as_of_date=as_of_date,
                details=c.get('details'),
            )
            for c in controls_data
        ]
        
        return DataSnapshot(
            snapshot_id=f"PG-{as_of_date.strftime('%Y%m%d')}",
            as_of_date=as_of_date,
            source_system="PostgreSQL",
            positions=positions,
            control_results=control_results,
            nav=nav,
        )
    
    def close(self):
        self.source.close()


def get_postgres_adapter() -> PostgresAdapter:
    """Get a PostgreSQL adapter configured from environment."""
    return PostgresAdapter()
