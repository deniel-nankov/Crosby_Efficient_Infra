"""
Snowflake Data Warehouse Adapter for Hedge Fund Compliance

This adapter connects to Snowflake, the most common cloud data warehouse
used by institutional hedge funds. It reads position, NAV, and control
data from curated views that the fund's data engineering team maintains.

Typical Snowflake Architecture at Hedge Funds:
┌─────────────────────────────────────────────────────────────────────────┐
│  SOURCE SYSTEMS                                                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │ Bloomberg   │  │ Eze OMS    │  │ Geneva     │  │ Pricing     │    │
│  │ AIM/PORT    │  │            │  │            │  │ Vendors     │    │
│  └──────┬──────┘  └──────┬─────┘  └──────┬─────┘  └──────┬──────┘    │
│         │                │               │               │            │
│         └───────────────┬┴───────────────┴───────────────┘            │
│                         │                                              │
│                         ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │                    SNOWFLAKE                                     │  │
│  │  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐        │  │
│  │  │ RAW Layer     │→ │ CURATED Layer │→ │ ANALYTICS     │        │  │
│  │  │ (Landing)     │  │ (Clean)       │  │ (Aggregated)  │        │  │
│  │  └───────────────┘  └───────────────┘  └───────────────┘        │  │
│  │                              ↓                                   │  │
│  │                    ┌───────────────┐                             │  │
│  │                    │ COMPLIANCE    │  ← This adapter reads here  │  │
│  │                    │ VIEWS         │                             │  │
│  │                    └───────────────┘                             │  │
│  └─────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘

Environment Variables:
    SNOWFLAKE_ACCOUNT     - Snowflake account identifier (e.g., xy12345.us-east-1)
    SNOWFLAKE_USER        - Service account username
    SNOWFLAKE_PASSWORD    - Service account password (or use key-pair auth)
    SNOWFLAKE_WAREHOUSE   - Compute warehouse (e.g., COMPLIANCE_WH)
    SNOWFLAKE_DATABASE    - Database name (e.g., HEDGE_FUND_DATA)
    SNOWFLAKE_SCHEMA      - Schema name (e.g., COMPLIANCE)
    SNOWFLAKE_ROLE        - Role with read access (e.g., COMPLIANCE_READER)
    
    # For key-pair authentication (recommended for production):
    SNOWFLAKE_PRIVATE_KEY_PATH  - Path to private key file
    SNOWFLAKE_PRIVATE_KEY_PASSPHRASE - Passphrase for encrypted key

Usage:
    from src.integration.snowflake_adapter import SnowflakeAdapter, SnowflakeConfig
    
    config = SnowflakeConfig.from_env()
    adapter = SnowflakeAdapter(config)
    
    snapshot = adapter.get_snapshot(date.today())
    print(f"Loaded {len(snapshot.positions)} positions")
"""

from __future__ import annotations

import os
import logging
from abc import ABC
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
import json

from .client_adapter import (
    ClientSystemAdapter,
    Position,
    ControlResult,
    DataSnapshot,
)

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class SnowflakeConfig:
    """
    Configuration for Snowflake connection.
    
    Supports both password and key-pair authentication.
    Key-pair is recommended for production service accounts.
    """
    account: str
    user: str
    warehouse: str
    database: str
    schema: str
    role: Optional[str] = None
    
    # Authentication - use ONE of these methods:
    password: Optional[str] = None
    private_key_path: Optional[str] = None
    private_key_passphrase: Optional[str] = None
    
    # Connection options
    login_timeout: int = 60
    network_timeout: int = 120
    
    # Query options
    query_timeout: int = 300  # 5 minutes max per query
    
    @classmethod
    def from_env(cls) -> "SnowflakeConfig":
        """
        Load configuration from environment variables.
        
        Required:
            SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER, SNOWFLAKE_WAREHOUSE,
            SNOWFLAKE_DATABASE, SNOWFLAKE_SCHEMA
            
        Plus one of:
            SNOWFLAKE_PASSWORD
            SNOWFLAKE_PRIVATE_KEY_PATH (+ optional SNOWFLAKE_PRIVATE_KEY_PASSPHRASE)
        """
        account = os.environ.get("SNOWFLAKE_ACCOUNT")
        user = os.environ.get("SNOWFLAKE_USER")
        
        if not account or not user:
            raise ValueError(
                "Missing required environment variables: "
                "SNOWFLAKE_ACCOUNT and SNOWFLAKE_USER must be set"
            )
        
        return cls(
            account=account,
            user=user,
            password=os.environ.get("SNOWFLAKE_PASSWORD"),
            warehouse=os.environ.get("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH"),
            database=os.environ.get("SNOWFLAKE_DATABASE", "HEDGE_FUND_DATA"),
            schema=os.environ.get("SNOWFLAKE_SCHEMA", "COMPLIANCE"),
            role=os.environ.get("SNOWFLAKE_ROLE"),
            private_key_path=os.environ.get("SNOWFLAKE_PRIVATE_KEY_PATH"),
            private_key_passphrase=os.environ.get("SNOWFLAKE_PRIVATE_KEY_PASSPHRASE"),
        )
    
    def validate(self) -> None:
        """Validate configuration has required auth method."""
        if not self.password and not self.private_key_path:
            raise ValueError(
                "Must provide either SNOWFLAKE_PASSWORD or "
                "SNOWFLAKE_PRIVATE_KEY_PATH for authentication"
            )


@dataclass
class SnowflakeViewConfig:
    """
    Configuration for the views/tables to query.
    
    Customize these to match the fund's Snowflake schema.
    The default values assume a standard compliance data model.
    """
    # View/table names
    positions_view: str = "V_POSITIONS_CURRENT"
    controls_view: str = "V_CONTROL_RESULTS"
    nav_view: str = "V_NAV_DAILY"
    
    # Column mappings - map standard names to client's column names
    position_columns: Dict[str, str] = field(default_factory=lambda: {
        "security_id": "SECURITY_ID",
        "ticker": "TICKER",
        "security_name": "SECURITY_NAME",
        "quantity": "QUANTITY",
        "market_value": "MARKET_VALUE_USD",
        "currency": "CURRENCY",
        "sector": "GICS_SECTOR",
        "issuer": "ISSUER_NAME",
        "asset_class": "ASSET_CLASS",
        "isin": "ISIN",
        "cusip": "CUSIP",
        "price": "PRICE_LOCAL",
    })
    
    control_columns: Dict[str, str] = field(default_factory=lambda: {
        "control_id": "CONTROL_ID",
        "control_name": "CONTROL_NAME",
        "control_type": "CONTROL_TYPE",
        "calculated_value": "CALCULATED_VALUE",
        "threshold": "THRESHOLD_VALUE",
        "threshold_operator": "THRESHOLD_OPERATOR",
        "status": "STATUS",
        "breach_amount": "BREACH_AMOUNT",
        "details": "DETAILS_JSON",
    })
    
    nav_columns: Dict[str, str] = field(default_factory=lambda: {
        "nav": "NAV_USD",
        "as_of_date": "AS_OF_DATE",
    })
    
    # Date column name (typically same across views)
    date_column: str = "AS_OF_DATE"


# =============================================================================
# SNOWFLAKE ADAPTER
# =============================================================================

class SnowflakeAdapter(ClientSystemAdapter):
    """
    Adapter for reading hedge fund data from Snowflake.
    
    This adapter is READ-ONLY and designed to work with curated views
    that the fund's data engineering team maintains. We trust the data
    quality because it has already been validated upstream.
    
    Typical Usage:
        config = SnowflakeConfig.from_env()
        view_config = SnowflakeViewConfig(
            positions_view="COMPLIANCE.V_POSITIONS_EOD",
            # ... customize as needed
        )
        adapter = SnowflakeAdapter(config, view_config)
        
        snapshot = adapter.get_snapshot(date.today())
    """
    
    def __init__(
        self,
        config: SnowflakeConfig,
        view_config: Optional[SnowflakeViewConfig] = None,
    ):
        self.config = config
        self.view_config = view_config or SnowflakeViewConfig()
        self._connection = None
        self._cursor = None
        
        # Validate config on init
        self.config.validate()
    
    def connect(self) -> None:
        """
        Establish connection to Snowflake.
        
        Uses snowflake-connector-python which must be installed:
            pip install snowflake-connector-python
        """
        try:
            import snowflake.connector
        except ImportError:
            raise ImportError(
                "snowflake-connector-python is required. "
                "Install with: pip install snowflake-connector-python"
            )
        
        connect_params = {
            "account": self.config.account,
            "user": self.config.user,
            "warehouse": self.config.warehouse,
            "database": self.config.database,
            "schema": self.config.schema,
            "login_timeout": self.config.login_timeout,
            "network_timeout": self.config.network_timeout,
        }
        
        if self.config.role:
            connect_params["role"] = self.config.role
        
        # Authentication method
        if self.config.private_key_path:
            # Key-pair authentication (recommended for production)
            private_key = self._load_private_key()
            connect_params["private_key"] = private_key
        else:
            # Password authentication
            connect_params["password"] = self.config.password
        
        try:
            self._connection = snowflake.connector.connect(**connect_params)
            self._cursor = self._connection.cursor()
            
            # Set session parameters
            self._cursor.execute(
                f"ALTER SESSION SET QUERY_TAG = 'compliance_rag_adapter'"
            )
            self._cursor.execute(
                f"ALTER SESSION SET STATEMENT_TIMEOUT_IN_SECONDS = {self.config.query_timeout}"
            )
            
            logger.info(
                f"Connected to Snowflake: {self.config.account} "
                f"({self.config.database}.{self.config.schema})"
            )
            
        except Exception as e:
            logger.error(f"Failed to connect to Snowflake: {e}")
            raise
    
    def _load_private_key(self) -> bytes:
        """Load private key for key-pair authentication."""
        from cryptography.hazmat.backends import default_backend
        from cryptography.hazmat.primitives import serialization
        
        key_path = Path(self.config.private_key_path)
        if not key_path.exists():
            raise FileNotFoundError(f"Private key not found: {key_path}")
        
        with open(key_path, "rb") as key_file:
            private_key = serialization.load_pem_private_key(
                key_file.read(),
                password=self.config.private_key_passphrase.encode() 
                    if self.config.private_key_passphrase else None,
                backend=default_backend(),
            )
        
        return private_key.private_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    
    def close(self) -> None:
        """Close Snowflake connection."""
        if self._cursor:
            self._cursor.close()
            self._cursor = None
        if self._connection:
            self._connection.close()
            self._connection = None
            logger.info("Snowflake connection closed")
    
    def _ensure_connected(self) -> None:
        """Ensure we have an active connection."""
        if not self._connection or self._connection.is_closed():
            self.connect()
    
    def _execute_query(self, query: str, params: Optional[Dict] = None) -> List[Dict]:
        """
        Execute a query and return results as list of dicts.
        
        Handles Snowflake-specific result formatting.
        """
        self._ensure_connected()
        
        try:
            if params:
                self._cursor.execute(query, params)
            else:
                self._cursor.execute(query)
            
            columns = [desc[0] for desc in self._cursor.description]
            results = []
            
            for row in self._cursor.fetchall():
                results.append(dict(zip(columns, row)))
            
            logger.debug(f"Query returned {len(results)} rows")
            return results
            
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            logger.error(f"Query: {query[:200]}...")
            raise
    
    def get_positions(self, as_of_date: date) -> List[Position]:
        """
        Get positions from Snowflake as of a specific date.
        
        Queries the positions view and maps columns to Position objects.
        """
        cols = self.view_config.position_columns
        date_col = self.view_config.date_column
        view = self.view_config.positions_view
        
        # Build column selection with aliases
        select_cols = ", ".join([
            f"{sf_col} AS {std_name}"
            for std_name, sf_col in cols.items()
        ])
        
        query = f"""
        SELECT {select_cols}
        FROM {view}
        WHERE {date_col} = %(as_of_date)s
        ORDER BY {cols['market_value']} DESC
        """
        
        results = self._execute_query(query, {"as_of_date": as_of_date})
        
        positions = []
        for row in results:
            try:
                positions.append(Position(
                    security_id=str(row.get("security_id", "")),
                    ticker=row.get("ticker"),
                    security_name=str(row.get("security_name", "")),
                    quantity=Decimal(str(row.get("quantity", 0))),
                    market_value=Decimal(str(row.get("market_value", 0))),
                    currency=row.get("currency", "USD"),
                    sector=row.get("sector"),
                    issuer=row.get("issuer"),
                    asset_class=row.get("asset_class"),
                    isin=row.get("isin"),
                    cusip=row.get("cusip"),
                    price=Decimal(str(row["price"])) if row.get("price") else None,
                ))
            except Exception as e:
                logger.warning(f"Failed to parse position row: {e}")
                continue
        
        logger.info(f"Loaded {len(positions)} positions for {as_of_date}")
        return positions
    
    def get_control_results(self, as_of_date: date) -> List[ControlResult]:
        """
        Get control results from Snowflake as of a specific date.
        
        These are pre-calculated by the fund's existing compliance system.
        We trust these values and use them to generate narratives.
        """
        cols = self.view_config.control_columns
        date_col = self.view_config.date_column
        view = self.view_config.controls_view
        
        select_cols = ", ".join([
            f"{sf_col} AS {std_name}"
            for std_name, sf_col in cols.items()
        ])
        
        query = f"""
        SELECT {select_cols}
        FROM {view}
        WHERE {date_col} = %(as_of_date)s
        ORDER BY 
            CASE WHEN STATUS = 'fail' THEN 0
                 WHEN STATUS = 'warning' THEN 1
                 ELSE 2 END,
            {cols['control_id']}
        """
        
        results = self._execute_query(query, {"as_of_date": as_of_date})
        
        controls = []
        for row in results:
            try:
                # Parse details JSON if present
                details = None
                if row.get("details"):
                    try:
                        details = json.loads(row["details"]) if isinstance(row["details"], str) else row["details"]
                    except json.JSONDecodeError:
                        details = {"raw": row["details"]}
                
                controls.append(ControlResult(
                    control_id=str(row.get("control_id", "")),
                    control_name=str(row.get("control_name", "")),
                    control_type=str(row.get("control_type", "other")),
                    calculated_value=Decimal(str(row.get("calculated_value", 0))),
                    threshold=Decimal(str(row.get("threshold", 0))),
                    threshold_operator=str(row.get("threshold_operator", "lte")),
                    status=str(row.get("status", "pass")).lower(),
                    breach_amount=Decimal(str(row["breach_amount"])) if row.get("breach_amount") else None,
                    as_of_date=as_of_date,
                    details=details,
                ))
            except Exception as e:
                logger.warning(f"Failed to parse control row: {e}")
                continue
        
        logger.info(f"Loaded {len(controls)} control results for {as_of_date}")
        return controls
    
    def get_nav(self, as_of_date: date) -> Decimal:
        """
        Get NAV from Snowflake as of a specific date.
        
        Returns the fund's Net Asset Value for compliance calculations.
        """
        cols = self.view_config.nav_columns
        date_col = self.view_config.date_column
        view = self.view_config.nav_view
        
        query = f"""
        SELECT {cols['nav']} AS nav
        FROM {view}
        WHERE {date_col} = %(as_of_date)s
        LIMIT 1
        """
        
        results = self._execute_query(query, {"as_of_date": as_of_date})
        
        if not results:
            logger.warning(f"No NAV found for {as_of_date}")
            return Decimal("0")
        
        nav = Decimal(str(results[0]["nav"]))
        logger.info(f"NAV for {as_of_date}: ${nav:,.0f}")
        return nav
    
    def get_snapshot(self, as_of_date: date) -> DataSnapshot:
        """
        Get complete data snapshot from Snowflake.
        
        Combines positions, controls, and NAV into a single snapshot
        with full audit trail metadata.
        """
        snapshot = DataSnapshot(
            snapshot_id=f"SF-{as_of_date.strftime('%Y%m%d')}-{datetime.now().strftime('%H%M%S')}",
            as_of_date=as_of_date,
            source_system=f"Snowflake:{self.config.account}",
        )
        
        snapshot.positions = self.get_positions(as_of_date)
        snapshot.control_results = self.get_control_results(as_of_date)
        snapshot.nav = self.get_nav(as_of_date)
        
        logger.info(
            f"Snapshot {snapshot.snapshot_id}: "
            f"{len(snapshot.positions)} positions, "
            f"{len(snapshot.control_results)} controls, "
            f"NAV=${snapshot.nav:,.0f}"
        )
        
        return snapshot
    
    def test_connection(self) -> Dict[str, Any]:
        """
        Test Snowflake connection and return diagnostic info.
        
        Useful for validating configuration before running pipeline.
        """
        self._ensure_connected()
        
        diagnostics = {
            "connected": True,
            "account": self.config.account,
            "database": self.config.database,
            "schema": self.config.schema,
            "warehouse": self.config.warehouse,
        }
        
        try:
            # Check current session
            results = self._execute_query("SELECT CURRENT_USER(), CURRENT_ROLE(), CURRENT_WAREHOUSE()")
            if results:
                diagnostics["current_user"] = results[0].get("CURRENT_USER()")
                diagnostics["current_role"] = results[0].get("CURRENT_ROLE()")
                diagnostics["current_warehouse"] = results[0].get("CURRENT_WAREHOUSE()")
            
            # Check if views exist
            for view_name in [
                self.view_config.positions_view,
                self.view_config.controls_view,
                self.view_config.nav_view,
            ]:
                try:
                    self._execute_query(f"SELECT 1 FROM {view_name} LIMIT 1")
                    diagnostics[f"view_{view_name}"] = "exists"
                except Exception:
                    diagnostics[f"view_{view_name}"] = "NOT FOUND"
            
        except Exception as e:
            diagnostics["error"] = str(e)
        
        return diagnostics
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False


# =============================================================================
# MOCK SNOWFLAKE ADAPTER (for testing without credentials)
# =============================================================================

class MockSnowflakeAdapter(ClientSystemAdapter):
    """
    Mock Snowflake adapter for testing and demos.
    
    Returns realistic sample data without requiring Snowflake credentials.
    Use this for:
    - Local development
    - CI/CD pipelines
    - Demo environments
    """
    
    def __init__(self, nav: Decimal = Decimal("2000000000")):
        self._nav = nav
        logger.info("Using MockSnowflakeAdapter (no Snowflake credentials)")
    
    def get_positions(self, as_of_date: date) -> List[Position]:
        """Return mock positions."""
        import random
        random.seed(as_of_date.toordinal())  # Consistent per date
        
        securities = [
            ("AAPL", "Apple Inc", "Technology", "Apple Inc"),
            ("MSFT", "Microsoft Corp", "Technology", "Microsoft Corp"),
            ("GOOGL", "Alphabet Inc", "Technology", "Alphabet Inc"),
            ("AMZN", "Amazon.com Inc", "Consumer Discretionary", "Amazon.com"),
            ("JPM", "JPMorgan Chase", "Financials", "JPMorgan Chase"),
            ("JNJ", "Johnson & Johnson", "Healthcare", "Johnson & Johnson"),
            ("XOM", "Exxon Mobil", "Energy", "Exxon Mobil"),
            ("CVX", "Chevron Corp", "Energy", "Chevron Corp"),
            ("PG", "Procter & Gamble", "Consumer Staples", "Procter & Gamble"),
            ("NVDA", "NVIDIA Corp", "Technology", "NVIDIA Corp"),
            ("V", "Visa Inc", "Financials", "Visa Inc"),
            ("UNH", "UnitedHealth", "Healthcare", "UnitedHealth Group"),
            ("HD", "Home Depot", "Consumer Discretionary", "Home Depot"),
            ("MA", "Mastercard", "Financials", "Mastercard"),
            ("BAC", "Bank of America", "Financials", "Bank of America"),
        ]
        
        positions = []
        for i, (ticker, name, sector, issuer) in enumerate(securities):
            mv = Decimal(str(random.randint(80_000_000, 250_000_000)))
            price = Decimal(str(random.uniform(100, 600)))
            qty = (mv / price).quantize(Decimal('0.01'))
            
            positions.append(Position(
                security_id=f"SF-{i+1:04d}",
                ticker=ticker,
                security_name=name,
                quantity=qty,
                market_value=mv,
                currency="USD",
                sector=sector,
                issuer=issuer,
                asset_class="equity",
                price=price.quantize(Decimal('0.01')),
            ))
        
        return positions
    
    def get_control_results(self, as_of_date: date) -> List[ControlResult]:
        """Return mock control results."""
        return [
            ControlResult(
                control_id="CONC_ISSUER_001",
                control_name="Single Issuer Concentration",
                control_type="concentration",
                calculated_value=Decimal("7.8"),
                threshold=Decimal("10.0"),
                threshold_operator="lte",
                status="pass",
                as_of_date=as_of_date,
            ),
            ControlResult(
                control_id="CONC_SECTOR_001",
                control_name="Sector Concentration - Technology",
                control_type="concentration",
                calculated_value=Decimal("32.5"),
                threshold=Decimal("35.0"),
                threshold_operator="lte",
                status="warning",
                as_of_date=as_of_date,
                details={"sector": "Technology", "warning_at": "30%"},
            ),
            ControlResult(
                control_id="EXP_GROSS_001",
                control_name="Gross Exposure",
                control_type="exposure",
                calculated_value=Decimal("125.0"),
                threshold=Decimal("200.0"),
                threshold_operator="lte",
                status="pass",
                as_of_date=as_of_date,
            ),
            ControlResult(
                control_id="LIQ_T1_001",
                control_name="T+1 Liquidity",
                control_type="liquidity",
                calculated_value=Decimal("22.0"),
                threshold=Decimal("10.0"),
                threshold_operator="gte",
                status="pass",
                as_of_date=as_of_date,
            ),
        ]
    
    def get_nav(self, as_of_date: date) -> Decimal:
        return self._nav


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def get_snowflake_adapter(
    use_mock: bool = False,
    config: Optional[SnowflakeConfig] = None,
    view_config: Optional[SnowflakeViewConfig] = None,
) -> Union[SnowflakeAdapter, MockSnowflakeAdapter]:
    """
    Factory function to get Snowflake adapter.
    
    Args:
        use_mock: Force mock adapter (for testing)
        config: Snowflake configuration (loads from env if not provided)
        view_config: View/column configuration
        
    Returns:
        SnowflakeAdapter or MockSnowflakeAdapter
    
    Example:
        # Production - uses environment variables
        adapter = get_snowflake_adapter()
        
        # Testing
        adapter = get_snowflake_adapter(use_mock=True)
        
        # Custom config
        config = SnowflakeConfig(
            account="my_account",
            user="my_user",
            password="my_password",
            ...
        )
        adapter = get_snowflake_adapter(config=config)
    """
    if use_mock:
        return MockSnowflakeAdapter()
    
    # Try to load config from environment
    try:
        if config is None:
            config = SnowflakeConfig.from_env()
        return SnowflakeAdapter(config, view_config)
    except (ValueError, KeyError) as e:
        logger.warning(f"Snowflake config not available ({e}), using mock adapter")
        return MockSnowflakeAdapter()


# =============================================================================
# SAMPLE SNOWFLAKE VIEWS (SQL to create in client's Snowflake)
# =============================================================================

SAMPLE_SNOWFLAKE_VIEWS_SQL = """
-- =============================================================================
-- COMPLIANCE VIEWS FOR RAG SYSTEM
-- Execute these in your Snowflake environment
-- =============================================================================

-- Create compliance schema if not exists
CREATE SCHEMA IF NOT EXISTS COMPLIANCE;
USE SCHEMA COMPLIANCE;

-- =============================================================================
-- POSITIONS VIEW
-- Aggregates position data from your source systems
-- =============================================================================
CREATE OR REPLACE VIEW V_POSITIONS_CURRENT AS
SELECT
    p.SECURITY_ID,
    s.TICKER,
    s.SECURITY_NAME,
    p.QUANTITY,
    p.QUANTITY * pr.PRICE_USD AS MARKET_VALUE_USD,
    s.CURRENCY,
    s.GICS_SECTOR,
    s.ISSUER_NAME,
    s.ASSET_CLASS,
    s.ISIN,
    s.CUSIP,
    pr.PRICE_USD AS PRICE_LOCAL,
    p.AS_OF_DATE
FROM RAW.POSITIONS p
JOIN RAW.SECURITIES s ON p.SECURITY_ID = s.SECURITY_ID
JOIN RAW.PRICES pr ON p.SECURITY_ID = pr.SECURITY_ID 
    AND p.AS_OF_DATE = pr.PRICE_DATE
WHERE p.AS_OF_DATE >= DATEADD(day, -30, CURRENT_DATE());

-- =============================================================================
-- CONTROL RESULTS VIEW
-- Pre-calculated compliance control results
-- =============================================================================
CREATE OR REPLACE VIEW V_CONTROL_RESULTS AS
SELECT
    CONTROL_ID,
    CONTROL_NAME,
    CONTROL_TYPE,
    CALCULATED_VALUE,
    THRESHOLD_VALUE,
    THRESHOLD_OPERATOR,
    CASE 
        WHEN BREACH_FLAG = TRUE THEN 'fail'
        WHEN WARNING_FLAG = TRUE THEN 'warning'
        ELSE 'pass'
    END AS STATUS,
    CASE WHEN BREACH_FLAG THEN CALCULATED_VALUE - THRESHOLD_VALUE END AS BREACH_AMOUNT,
    DETAILS_JSON,
    AS_OF_DATE
FROM CURATED.CONTROL_CALCULATIONS
WHERE AS_OF_DATE >= DATEADD(day, -30, CURRENT_DATE());

-- =============================================================================
-- NAV VIEW
-- Daily Net Asset Value
-- =============================================================================
CREATE OR REPLACE VIEW V_NAV_DAILY AS
SELECT
    NAV_USD,
    AS_OF_DATE
FROM CURATED.FUND_NAV
WHERE AS_OF_DATE >= DATEADD(day, -30, CURRENT_DATE());

-- =============================================================================
-- Grant read access to compliance service account
-- =============================================================================
GRANT USAGE ON SCHEMA COMPLIANCE TO ROLE COMPLIANCE_READER;
GRANT SELECT ON ALL VIEWS IN SCHEMA COMPLIANCE TO ROLE COMPLIANCE_READER;
"""

if __name__ == "__main__":
    # Quick test of the adapter
    print("Testing Snowflake Adapter...")
    print("-" * 60)
    
    adapter = get_snowflake_adapter(use_mock=True)
    snapshot = adapter.get_snapshot(date.today())
    
    print(f"Snapshot ID: {snapshot.snapshot_id}")
    print(f"Positions: {len(snapshot.positions)}")
    print(f"Controls: {len(snapshot.control_results)}")
    print(f"NAV: ${snapshot.nav:,.0f}")
    print("-" * 60)
    print("Top 5 positions:")
    for p in sorted(snapshot.positions, key=lambda x: x.market_value, reverse=True)[:5]:
        print(f"  {p.ticker}: ${p.market_value:,.0f}")
