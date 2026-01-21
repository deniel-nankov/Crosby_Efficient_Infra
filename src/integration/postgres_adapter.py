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
        
        # =================================================================
        # TRADE DATA TABLES - Full transaction lifecycle
        # =================================================================
        
        # Accounts table (for multi-account aggregation)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS accounts (
                account_id TEXT PRIMARY KEY,
                account_name TEXT NOT NULL,
                account_type TEXT NOT NULL,
                parent_account_id TEXT REFERENCES accounts(account_id),
                fund_id TEXT NOT NULL DEFAULT 'MASTER_FUND',
                currency TEXT DEFAULT 'USD',
                is_active BOOLEAN DEFAULT TRUE,
                custodian TEXT DEFAULT 'State Street',
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        
        # Orders table (OMS integration)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS orders (
                order_id TEXT PRIMARY KEY,
                account_id TEXT REFERENCES accounts(account_id),
                security_id TEXT NOT NULL,
                ticker TEXT NOT NULL,
                trade_type TEXT NOT NULL,
                quantity DECIMAL(20, 6) NOT NULL,
                limit_price DECIMAL(20, 6),
                order_time TIMESTAMPTZ NOT NULL,
                status TEXT NOT NULL,
                filled_quantity DECIMAL(20, 6) DEFAULT 0,
                average_fill_price DECIMAL(20, 6),
                pre_trade_check TEXT DEFAULT 'pending',
                compliance_notes TEXT,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        
        # Trades table (trade blotter)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                trade_id TEXT PRIMARY KEY,
                order_id TEXT REFERENCES orders(order_id),
                account_id TEXT REFERENCES accounts(account_id),
                security_id TEXT NOT NULL,
                ticker TEXT NOT NULL,
                trade_type TEXT NOT NULL,
                quantity DECIMAL(20, 6) NOT NULL,
                price DECIMAL(20, 6) NOT NULL,
                gross_amount DECIMAL(20, 2) NOT NULL,
                commission DECIMAL(20, 2) DEFAULT 0,
                fees DECIMAL(20, 2) DEFAULT 0,
                net_amount DECIMAL(20, 2) NOT NULL,
                exchange TEXT DEFAULT 'NASDAQ',
                execution_time TIMESTAMPTZ NOT NULL,
                broker TEXT DEFAULT 'Prime Broker',
                settlement_date DATE,
                settlement_status TEXT DEFAULT 'pending',
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        
        # Transaction ledger (complete audit trail)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS transactions (
                transaction_id TEXT PRIMARY KEY,
                transaction_type TEXT NOT NULL,
                account_id TEXT REFERENCES accounts(account_id),
                reference_id TEXT NOT NULL,
                reference_type TEXT NOT NULL,
                security_id TEXT,
                quantity_change DECIMAL(20, 6) DEFAULT 0,
                cash_change DECIMAL(20, 2) DEFAULT 0,
                effective_date DATE NOT NULL,
                posted_date DATE NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                notes TEXT
            )
        """)
        
        # Settlements table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS settlements (
                settlement_id TEXT PRIMARY KEY,
                trade_id TEXT REFERENCES trades(trade_id),
                settlement_date DATE NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                custodian TEXT DEFAULT 'State Street',
                broker_confirm_received BOOLEAN DEFAULT FALSE,
                custodian_matched BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        
        # Create indexes for trade tables
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_account ON trades(account_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_execution ON trades(execution_time)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_transactions_account ON transactions(account_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_transactions_date ON transactions(effective_date)")
        
        self.connection.commit()
        cursor.close()
        logger.info("Data tables created successfully (including trade data)")
    
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
        
        # Sample positions - larger values for realistic $2B fund
        # Technology sector is intentionally high to trigger concentration warning
        positions = [
            ('SEC001', 'AAPL', 'Apple Inc', 500000, 175000000, 'USD', 'Technology', 'Apple Inc', 'equity'),
            ('SEC002', 'MSFT', 'Microsoft Corp', 350000, 155000000, 'USD', 'Technology', 'Microsoft Corp', 'equity'),
            ('SEC003', 'NVDA', 'NVIDIA Corp', 200000, 180000000, 'USD', 'Technology', 'NVIDIA Corp', 'equity'),
            ('SEC004', 'GOOGL', 'Alphabet Inc', 150000, 50000000, 'USD', 'Technology', 'Alphabet Inc', 'equity'),
            ('SEC005', 'JPM', 'JPMorgan Chase', 800000, 150000000, 'USD', 'Financials', 'JPMorgan Chase', 'equity'),
            ('SEC006', 'BAC', 'Bank of America', 1200000, 45000000, 'USD', 'Financials', 'Bank of America', 'equity'),
            ('SEC007', 'JNJ', 'Johnson & Johnson', 400000, 72000000, 'USD', 'Healthcare', 'Johnson & Johnson', 'equity'),
            ('SEC008', 'UNH', 'UnitedHealth', 180000, 105000000, 'USD', 'Healthcare', 'UnitedHealth Group', 'equity'),
            ('SEC009', 'XOM', 'Exxon Mobil', 600000, 68000000, 'USD', 'Energy', 'Exxon Mobil', 'equity'),
            ('SEC010', 'CVX', 'Chevron Corp', 400000, 60000000, 'USD', 'Energy', 'Chevron Corp', 'equity'),
            ('SEC011', 'PG', 'Procter & Gamble', 350000, 55000000, 'USD', 'Consumer Staples', 'Procter & Gamble', 'equity'),
            ('SEC012', 'KO', 'Coca-Cola', 600000, 42000000, 'USD', 'Consumer Staples', 'Coca-Cola Co', 'equity'),
            ('SEC013', 'V', 'Visa Inc', 250000, 80000000, 'USD', 'Financials', 'Visa Inc', 'equity'),
            ('SEC014', 'MA', 'Mastercard', 150000, 75000000, 'USD', 'Financials', 'Mastercard Inc', 'equity'),
            ('SEC015', 'CASH', 'Cash & Equivalents', 1, 64000000, 'USD', 'Cash', 'N/A', 'cash'),
        ]
        
        for pos in positions:
            cursor.execute("""
                INSERT INTO fund_positions 
                (security_id, ticker, security_name, quantity, market_value, 
                 currency, sector, issuer, asset_class, as_of_date)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (security_id, as_of_date) DO UPDATE SET
                    market_value = EXCLUDED.market_value,
                    quantity = EXCLUDED.quantity
            """, (*pos, as_of_date))
        
        # Calculate actual sector concentration
        # Tech: 175+155+180+50 = 560M / 2000M = 28%
        # Sample control results
        controls = [
            ('CONC_ISSUER_001', 'Single Issuer Concentration', 'concentration', 9.0, 10.0, 'lte', 'warning', None),  # NVDA at 9%
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
                ON CONFLICT (control_id, as_of_date) DO UPDATE SET
                    calculated_value = EXCLUDED.calculated_value,
                    status = EXCLUDED.status
            """, (*ctrl, as_of_date))
        
        # Sample NAV
        cursor.execute("""
            INSERT INTO fund_nav (nav, as_of_date)
            VALUES (%s, %s)
            ON CONFLICT (as_of_date) DO NOTHING
        """, (Decimal('2000000000'), as_of_date))
        
        # Sample accounts and trades
        self._load_sample_trade_data(cursor, as_of_date)
        
        self.connection.commit()
        cursor.close()
        logger.info(f"Sample data loaded for {as_of_date}")
    
    def _load_sample_trade_data(self, cursor, as_of_date: date):
        """Load sample trade data for demonstration."""
        from datetime import timedelta
        import uuid
        
        # Sample accounts
        accounts = [
            ('MAIN-001', 'Master Fund - Main Account', 'fund', None, 'MASTER_FUND'),
            ('SLEEVE-TECH', 'Technology Sleeve', 'sleeve', 'MAIN-001', 'MASTER_FUND'),
            ('SLEEVE-FIN', 'Financials Sleeve', 'sleeve', 'MAIN-001', 'MASTER_FUND'),
        ]
        
        for acc in accounts:
            cursor.execute("""
                INSERT INTO accounts (account_id, account_name, account_type, parent_account_id, fund_id)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (account_id) DO NOTHING
            """, acc)
        
        # Sample trades
        trades = [
            ('TRD-001', None, 'SLEEVE-TECH', 'NVDA', 'NVDA', 'buy', 1000, 850.00, 850000, 50, 0, 850050, 'NASDAQ', as_of_date - timedelta(days=2), 'Goldman Sachs', as_of_date, 'settled'),
            ('TRD-002', None, 'SLEEVE-TECH', 'AAPL', 'AAPL', 'sell', 500, 175.00, 87500, 25, 0, 87475, 'NYSE', as_of_date - timedelta(days=1), 'Morgan Stanley', as_of_date, 'pending'),
            ('TRD-003', None, 'MAIN-001', 'MSFT', 'MSFT', 'buy', 2000, 415.00, 830000, 75, 0, 830075, 'NASDAQ', as_of_date, 'JPMorgan', as_of_date + timedelta(days=2), 'pending'),
        ]
        
        for trade in trades:
            cursor.execute("""
                INSERT INTO trades 
                (trade_id, order_id, account_id, security_id, ticker, trade_type,
                 quantity, price, gross_amount, commission, fees, net_amount,
                 exchange, execution_time, broker, settlement_date, settlement_status)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (trade_id) DO NOTHING
            """, trade)
            
            # Create transaction entry
            txn_id = str(uuid.uuid4())[:8]
            qty_change = trade[6] if trade[5] == 'buy' else -trade[6]
            cash_change = -trade[11] if trade[5] == 'buy' else trade[11]
            cursor.execute("""
                INSERT INTO transactions
                (transaction_id, transaction_type, account_id, reference_id, reference_type,
                 security_id, quantity_change, cash_change, effective_date, posted_date)
                VALUES (%s, 'trade', %s, %s, 'trade', %s, %s, %s, %s, %s)
                ON CONFLICT (transaction_id) DO NOTHING
            """, (txn_id, trade[2], trade[0], trade[3], qty_change, cash_change, as_of_date, as_of_date))
    
    # =========================================================================
    # TRADE DATA ACCESS METHODS
    # =========================================================================
    
    def get_trades(
        self,
        account_id: Optional[str] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get trades with optional filters."""
        cursor = self.connection.cursor()
        
        query = "SELECT * FROM trades WHERE 1=1"
        params = []
        
        if account_id:
            query += " AND account_id = %s"
            params.append(account_id)
        if start_date:
            query += " AND DATE(execution_time) >= %s"
            params.append(start_date)
        if end_date:
            query += " AND DATE(execution_time) <= %s"
            params.append(end_date)
        
        query += f" ORDER BY execution_time DESC LIMIT {limit}"
        
        cursor.execute(query, params)
        columns = [desc[0] for desc in cursor.description]
        trades = [dict(zip(columns, row)) for row in cursor.fetchall()]
        cursor.close()
        return trades
    
    def get_trade_summary(
        self,
        start_date: date,
        end_date: date,
    ) -> List[Dict[str, Any]]:
        """Aggregate trade summary by security."""
        cursor = self.connection.cursor()
        
        cursor.execute("""
            SELECT 
                security_id,
                ticker,
                SUM(CASE WHEN trade_type = 'buy' THEN quantity ELSE 0 END) as buy_qty,
                SUM(CASE WHEN trade_type = 'sell' THEN quantity ELSE 0 END) as sell_qty,
                SUM(CASE WHEN trade_type = 'buy' THEN gross_amount ELSE 0 END) as buy_amount,
                SUM(CASE WHEN trade_type = 'sell' THEN gross_amount ELSE 0 END) as sell_amount,
                COUNT(*) as trade_count
            FROM trades
            WHERE DATE(execution_time) BETWEEN %s AND %s
            GROUP BY security_id, ticker
            ORDER BY SUM(gross_amount) DESC
        """, (start_date, end_date))
        
        columns = [desc[0] for desc in cursor.description]
        summary = [dict(zip(columns, row)) for row in cursor.fetchall()]
        cursor.close()
        return summary
    
    def get_accounts(self, fund_id: str = "MASTER_FUND") -> List[Dict[str, Any]]:
        """Get all accounts for a fund."""
        cursor = self.connection.cursor()
        cursor.execute("""
            SELECT account_id, account_name, account_type, parent_account_id, custodian
            FROM accounts
            WHERE fund_id = %s AND is_active = TRUE
        """, (fund_id,))
        columns = [desc[0] for desc in cursor.description]
        accounts = [dict(zip(columns, row)) for row in cursor.fetchall()]
        cursor.close()
        return accounts
    
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
