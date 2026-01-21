"""
ClickHouse Analytics Adapter for Hedge Fund Compliance

Institutional-grade time-series analytics engine for compliance data.
ClickHouse excels at sub-second queries across billions of rows,
making it ideal for historical compliance trend analysis.

Use Cases:
┌─────────────────────────────────────────────────────────────────────────────┐
│  CLICKHOUSE ANALYTICS USE CASES                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. HISTORICAL TREND ANALYSIS                                                │
│     • Control values over 5+ years                                           │
│     • Breach frequency by control type                                       │
│     • Sector concentration trends                                            │
│                                                                              │
│  2. REAL-TIME DASHBOARDS                                                     │
│     • Sub-second aggregations                                                │
│     • Live control monitoring                                                │
│     • Risk exposure heatmaps                                                 │
│                                                                              │
│  3. REGULATORY REPORTING                                                     │
│     • Form PF historical data                                                │
│     • SEC examination support                                                │
│     • AIFMD Annex IV analytics                                               │
│                                                                              │
│  4. AUDIT & FORENSICS                                                        │
│     • Point-in-time reconstruction                                           │
│     • Change detection                                                       │
│     • Anomaly identification                                                 │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

Architecture:
    
    PostgreSQL (Operational)          ClickHouse (Analytics)
    ━━━━━━━━━━━━━━━━━━━━━━           ━━━━━━━━━━━━━━━━━━━━━
    • Current day data                • Historical data (years)
    • Vector embeddings               • Time-series aggregations
    • Audit trail                     • Trend analysis
    • ACID transactions               • Sub-second queries
    
    Daily ETL: PostgreSQL → ClickHouse (end of day)

Environment Variables:
    CLICKHOUSE_HOST      - ClickHouse server hostname
    CLICKHOUSE_PORT      - Native protocol port (default: 9000)
    CLICKHOUSE_HTTP_PORT - HTTP port (default: 8123)
    CLICKHOUSE_USER      - Username
    CLICKHOUSE_PASSWORD  - Password
    CLICKHOUSE_DATABASE  - Database name (default: compliance)
    CLICKHOUSE_SECURE    - Use TLS (default: true for production)
    
    # For ClickHouse Cloud:
    CLICKHOUSE_CLOUD_URL - Full connection URL

Installation:
    pip install clickhouse-driver clickhouse-connect

Usage:
    from src.integration.clickhouse_adapter import ClickHouseAnalytics, ClickHouseConfig
    
    config = ClickHouseConfig.from_env()
    analytics = ClickHouseAnalytics(config)
    
    # Get control trends
    trends = analytics.get_control_trends(
        control_id="CONC_SECTOR_001",
        start_date=date(2024, 1, 1),
        end_date=date.today(),
    )
    
    # Get breach statistics
    stats = analytics.get_breach_statistics(
        start_date=date(2024, 1, 1),
        group_by="month",
    )
"""

from __future__ import annotations

import os
import logging
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime, timezone, timedelta
from decimal import Decimal
from enum import Enum
from typing import Optional, Dict, Any, List, Union, Tuple
import json

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

class ClickHouseDriver(Enum):
    """Available ClickHouse driver options."""
    NATIVE = "native"       # clickhouse-driver (binary protocol, fastest)
    HTTP = "http"           # clickhouse-connect (HTTP protocol)
    AUTO = "auto"           # Try native first, fall back to HTTP


@dataclass
class ClickHouseConfig:
    """
    Configuration for ClickHouse connection.
    
    Supports both self-hosted and ClickHouse Cloud deployments.
    For production, always use TLS (secure=True).
    """
    host: str
    port: int = 9000                    # Native protocol port
    http_port: int = 8123               # HTTP protocol port
    user: str = "default"
    password: str = ""
    database: str = "compliance"
    secure: bool = True                 # TLS encryption
    verify: bool = True                 # Verify TLS certificates
    
    # Connection pool settings
    pool_size: int = 10
    connect_timeout: float = 10.0
    send_receive_timeout: float = 300.0  # 5 min for large queries
    
    # Query settings
    max_execution_time: int = 300       # Max query time in seconds
    max_memory_usage: int = 10_000_000_000  # 10GB max memory per query
    
    # Driver selection
    driver: ClickHouseDriver = ClickHouseDriver.AUTO
    
    # ClickHouse Cloud specific
    cloud_url: Optional[str] = None     # Full connection URL for cloud
    
    @classmethod
    def from_env(cls) -> "ClickHouseConfig":
        """
        Load configuration from environment variables.
        
        Required:
            CLICKHOUSE_HOST
            
        Optional:
            CLICKHOUSE_PORT, CLICKHOUSE_USER, CLICKHOUSE_PASSWORD,
            CLICKHOUSE_DATABASE, CLICKHOUSE_SECURE, CLICKHOUSE_CLOUD_URL
        """
        host = os.environ.get("CLICKHOUSE_HOST")
        cloud_url = os.environ.get("CLICKHOUSE_CLOUD_URL")
        
        if not host and not cloud_url:
            raise ValueError(
                "Missing required environment variable: "
                "CLICKHOUSE_HOST or CLICKHOUSE_CLOUD_URL must be set"
            )
        
        return cls(
            host=host or "",
            port=int(os.environ.get("CLICKHOUSE_PORT", "9000")),
            http_port=int(os.environ.get("CLICKHOUSE_HTTP_PORT", "8123")),
            user=os.environ.get("CLICKHOUSE_USER", "default"),
            password=os.environ.get("CLICKHOUSE_PASSWORD", ""),
            database=os.environ.get("CLICKHOUSE_DATABASE", "compliance"),
            secure=os.environ.get("CLICKHOUSE_SECURE", "true").lower() == "true",
            cloud_url=cloud_url,
        )
    
    @classmethod
    def for_local_dev(cls) -> "ClickHouseConfig":
        """Configuration for local development (Docker)."""
        return cls(
            host="localhost",
            port=9000,
            http_port=8123,
            user="default",
            password="",
            database="compliance",
            secure=False,
            verify=False,
        )


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class ControlTrendPoint:
    """Single point in a control trend time series."""
    as_of_date: date
    calculated_value: Decimal
    threshold: Decimal
    status: str
    headroom_pct: float  # Distance from threshold as percentage
    
    @property
    def is_breach(self) -> bool:
        return self.status == "fail"
    
    @property
    def is_warning(self) -> bool:
        return self.status == "warning"


@dataclass
class ControlTrend:
    """Time series of control values with statistics."""
    control_id: str
    control_name: str
    control_type: str
    points: List[ControlTrendPoint]
    
    # Computed statistics
    min_value: Decimal = Decimal("0")
    max_value: Decimal = Decimal("0")
    avg_value: Decimal = Decimal("0")
    std_dev: float = 0.0
    breach_count: int = 0
    warning_count: int = 0
    
    def __post_init__(self):
        if self.points:
            values = [float(p.calculated_value) for p in self.points]
            self.min_value = Decimal(str(min(values)))
            self.max_value = Decimal(str(max(values)))
            self.avg_value = Decimal(str(sum(values) / len(values)))
            
            if len(values) > 1:
                mean = sum(values) / len(values)
                variance = sum((x - mean) ** 2 for x in values) / len(values)
                self.std_dev = variance ** 0.5
            
            self.breach_count = sum(1 for p in self.points if p.is_breach)
            self.warning_count = sum(1 for p in self.points if p.is_warning)


@dataclass
class BreachStatistics:
    """Aggregated breach statistics."""
    period: str                     # "2024-01", "2024-Q1", "2024", etc.
    total_controls: int
    total_breaches: int
    total_warnings: int
    breach_rate: float              # breaches / total as percentage
    warning_rate: float
    
    # Breakdown by type
    by_control_type: Dict[str, int] = field(default_factory=dict)
    by_severity: Dict[str, int] = field(default_factory=dict)
    
    # Top offenders
    top_breaching_controls: List[Tuple[str, int]] = field(default_factory=list)


@dataclass
class PositionSnapshot:
    """Historical position snapshot for point-in-time analysis."""
    as_of_date: date
    security_id: str
    ticker: str
    security_name: str
    market_value: Decimal
    quantity: Decimal
    sector: str
    weight_pct: float  # Percentage of NAV


@dataclass
class ConcentrationTrend:
    """Concentration metrics over time."""
    dimension: str          # "sector", "issuer", "asset_class"
    name: str               # e.g., "Technology", "Apple Inc"
    points: List[Tuple[date, float]]  # (date, weight_pct)
    
    max_concentration: float = 0.0
    avg_concentration: float = 0.0
    current_concentration: float = 0.0
    
    def __post_init__(self):
        if self.points:
            weights = [p[1] for p in self.points]
            self.max_concentration = max(weights)
            self.avg_concentration = sum(weights) / len(weights)
            self.current_concentration = self.points[-1][1]


# =============================================================================
# CLICKHOUSE ANALYTICS ENGINE
# =============================================================================

class ClickHouseAnalytics:
    """
    Institutional-grade analytics engine using ClickHouse.
    
    Provides sub-second queries for:
    - Historical control trends
    - Breach statistics and patterns
    - Position history and point-in-time
    - Concentration analysis
    - Regulatory reporting data
    
    Example:
        config = ClickHouseConfig.from_env()
        analytics = ClickHouseAnalytics(config)
        
        # 5-year control trend
        trend = analytics.get_control_trend(
            control_id="CONC_SECTOR_001",
            start_date=date(2021, 1, 1),
        )
        
        # Monthly breach stats
        stats = analytics.get_breach_statistics(
            start_date=date(2024, 1, 1),
            group_by="month",
        )
    """
    
    def __init__(self, config: ClickHouseConfig):
        self.config = config
        self._client = None
        self._driver_type = None
    
    def connect(self) -> None:
        """
        Establish connection to ClickHouse.
        
        Automatically selects best available driver.
        """
        if self.config.driver == ClickHouseDriver.AUTO:
            # Try native driver first (faster)
            try:
                self._connect_native()
                self._driver_type = ClickHouseDriver.NATIVE
                logger.info("Connected to ClickHouse using native driver")
                return
            except ImportError:
                pass
            
            # Fall back to HTTP driver
            try:
                self._connect_http()
                self._driver_type = ClickHouseDriver.HTTP
                logger.info("Connected to ClickHouse using HTTP driver")
                return
            except ImportError:
                pass
            
            raise ImportError(
                "No ClickHouse driver available. Install one of:\n"
                "  pip install clickhouse-driver    # Native protocol (recommended)\n"
                "  pip install clickhouse-connect   # HTTP protocol"
            )
        
        elif self.config.driver == ClickHouseDriver.NATIVE:
            self._connect_native()
            self._driver_type = ClickHouseDriver.NATIVE
        
        else:
            self._connect_http()
            self._driver_type = ClickHouseDriver.HTTP
    
    def _connect_native(self) -> None:
        """Connect using native binary protocol (fastest)."""
        from clickhouse_driver import Client
        
        self._client = Client(
            host=self.config.host,
            port=self.config.port,
            user=self.config.user,
            password=self.config.password,
            database=self.config.database,
            secure=self.config.secure,
            verify=self.config.verify,
            connect_timeout=self.config.connect_timeout,
            send_receive_timeout=self.config.send_receive_timeout,
            settings={
                "max_execution_time": self.config.max_execution_time,
                "max_memory_usage": self.config.max_memory_usage,
            },
        )
    
    def _connect_http(self) -> None:
        """Connect using HTTP protocol."""
        import clickhouse_connect
        
        if self.config.cloud_url:
            self._client = clickhouse_connect.get_client(
                dsn=self.config.cloud_url,
                send_receive_timeout=self.config.send_receive_timeout,
            )
        else:
            self._client = clickhouse_connect.get_client(
                host=self.config.host,
                port=self.config.http_port,
                username=self.config.user,
                password=self.config.password,
                database=self.config.database,
                secure=self.config.secure,
                verify=self.config.verify,
                send_receive_timeout=self.config.send_receive_timeout,
            )
    
    def close(self) -> None:
        """Close ClickHouse connection."""
        if self._client:
            if self._driver_type == ClickHouseDriver.NATIVE:
                self._client.disconnect()
            # HTTP client doesn't need explicit close
            self._client = None
            logger.info("ClickHouse connection closed")
    
    def _ensure_connected(self) -> None:
        """Ensure we have an active connection."""
        if not self._client:
            self.connect()
    
    def _execute(
        self, 
        query: str, 
        params: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        """
        Execute a query and return results as list of dicts.
        
        Handles driver-specific result formatting.
        """
        self._ensure_connected()
        
        try:
            if self._driver_type == ClickHouseDriver.NATIVE:
                result = self._client.execute(
                    query, 
                    params or {},
                    with_column_types=True,
                )
                rows, columns = result
                col_names = [c[0] for c in columns]
                return [dict(zip(col_names, row)) for row in rows]
            else:
                # HTTP driver
                result = self._client.query(query, parameters=params or {})
                return [dict(zip(result.column_names, row)) for row in result.result_rows]
                
        except Exception as e:
            logger.error(f"ClickHouse query failed: {e}")
            logger.debug(f"Query: {query[:500]}...")
            raise
    
    def _execute_command(self, query: str) -> None:
        """Execute a DDL/command query."""
        self._ensure_connected()
        
        if self._driver_type == ClickHouseDriver.NATIVE:
            self._client.execute(query)
        else:
            self._client.command(query)
    
    # =========================================================================
    # SCHEMA MANAGEMENT
    # =========================================================================
    
    def create_schema(self) -> None:
        """
        Create ClickHouse tables for compliance analytics.
        
        Uses MergeTree engine family optimized for time-series.
        """
        self._ensure_connected()
        
        # Control results history (main analytics table)
        self._execute_command("""
            CREATE TABLE IF NOT EXISTS control_results_history (
                as_of_date Date,
                control_id LowCardinality(String),
                control_name String,
                control_type LowCardinality(String),
                calculated_value Decimal(20, 6),
                threshold Decimal(20, 6),
                threshold_operator LowCardinality(String),
                status LowCardinality(String),
                breach_amount Nullable(Decimal(20, 6)),
                headroom_pct Float64,
                fund_id LowCardinality(String) DEFAULT 'MAIN',
                details String DEFAULT '{}',
                inserted_at DateTime64(3) DEFAULT now64(3),
                data_hash FixedString(32)
            )
            ENGINE = ReplacingMergeTree(inserted_at)
            PARTITION BY toYYYYMM(as_of_date)
            ORDER BY (fund_id, control_id, as_of_date)
            TTL as_of_date + INTERVAL 10 YEAR
            SETTINGS index_granularity = 8192
        """)
        
        # Positions history
        self._execute_command("""
            CREATE TABLE IF NOT EXISTS positions_history (
                as_of_date Date,
                security_id String,
                ticker LowCardinality(String),
                security_name String,
                quantity Decimal(20, 4),
                market_value Decimal(20, 2),
                price Decimal(20, 6),
                currency LowCardinality(String) DEFAULT 'USD',
                sector LowCardinality(String),
                issuer String,
                asset_class LowCardinality(String),
                country LowCardinality(String) DEFAULT 'US',
                weight_pct Float64,
                fund_id LowCardinality(String) DEFAULT 'MAIN',
                inserted_at DateTime64(3) DEFAULT now64(3)
            )
            ENGINE = ReplacingMergeTree(inserted_at)
            PARTITION BY toYYYYMM(as_of_date)
            ORDER BY (fund_id, as_of_date, security_id)
            TTL as_of_date + INTERVAL 10 YEAR
            SETTINGS index_granularity = 8192
        """)
        
        # NAV history
        self._execute_command("""
            CREATE TABLE IF NOT EXISTS nav_history (
                as_of_date Date,
                nav Decimal(20, 2),
                currency LowCardinality(String) DEFAULT 'USD',
                fund_id LowCardinality(String) DEFAULT 'MAIN',
                inserted_at DateTime64(3) DEFAULT now64(3)
            )
            ENGINE = ReplacingMergeTree(inserted_at)
            PARTITION BY toYYYYMM(as_of_date)
            ORDER BY (fund_id, as_of_date)
            SETTINGS index_granularity = 8192
        """)
        
        # Breach events (for fast breach analysis)
        self._execute_command("""
            CREATE TABLE IF NOT EXISTS breach_events (
                event_id UUID DEFAULT generateUUIDv4(),
                occurred_at DateTime64(3),
                as_of_date Date,
                control_id LowCardinality(String),
                control_name String,
                control_type LowCardinality(String),
                severity LowCardinality(String),  -- 'warning', 'breach', 'critical'
                calculated_value Decimal(20, 6),
                threshold Decimal(20, 6),
                breach_amount Decimal(20, 6),
                breach_pct Float64,
                fund_id LowCardinality(String) DEFAULT 'MAIN',
                resolved_at Nullable(DateTime64(3)),
                resolution_notes String DEFAULT ''
            )
            ENGINE = MergeTree()
            PARTITION BY toYYYYMM(as_of_date)
            ORDER BY (fund_id, occurred_at, control_id)
            SETTINGS index_granularity = 8192
        """)
        
        # Aggregated daily metrics (materialized view)
        self._execute_command("""
            CREATE MATERIALIZED VIEW IF NOT EXISTS daily_compliance_summary
            ENGINE = SummingMergeTree()
            PARTITION BY toYYYYMM(as_of_date)
            ORDER BY (fund_id, as_of_date)
            AS SELECT
                as_of_date,
                fund_id,
                count() AS total_controls,
                countIf(status = 'pass') AS passed_controls,
                countIf(status = 'warning') AS warning_controls,
                countIf(status = 'fail') AS failed_controls,
                avg(headroom_pct) AS avg_headroom_pct
            FROM control_results_history
            GROUP BY as_of_date, fund_id
        """)
        
        logger.info("ClickHouse schema created successfully")
    
    # =========================================================================
    # DATA INGESTION
    # =========================================================================
    
    def insert_control_results(
        self,
        results: List[Dict[str, Any]],
        fund_id: str = "MAIN",
    ) -> int:
        """
        Insert control results into ClickHouse.
        
        Args:
            results: List of control result dicts
            fund_id: Fund identifier for multi-fund support
            
        Returns:
            Number of rows inserted
        """
        self._ensure_connected()
        
        rows = []
        for r in results:
            # Calculate headroom percentage
            calc = float(r.get("calculated_value", 0))
            thresh = float(r.get("threshold", 0))
            op = r.get("threshold_operator", "lte")
            
            if thresh != 0:
                if op in ("lte", "lt"):
                    headroom = (thresh - calc) / thresh * 100
                else:  # gte, gt
                    headroom = (calc - thresh) / thresh * 100
            else:
                headroom = 0
            
            # Create data hash for deduplication
            hash_input = f"{r.get('control_id')}{r.get('as_of_date')}{calc}"
            data_hash = hashlib.md5(hash_input.encode()).hexdigest()
            
            rows.append({
                "as_of_date": r.get("as_of_date"),
                "control_id": r.get("control_id", ""),
                "control_name": r.get("control_name", ""),
                "control_type": r.get("control_type", "other"),
                "calculated_value": Decimal(str(calc)),
                "threshold": Decimal(str(thresh)),
                "threshold_operator": op,
                "status": r.get("status", "pass"),
                "breach_amount": Decimal(str(r["breach_amount"])) if r.get("breach_amount") else None,
                "headroom_pct": headroom,
                "fund_id": fund_id,
                "details": json.dumps(r.get("details", {})),
                "data_hash": data_hash,
            })
        
        if self._driver_type == ClickHouseDriver.NATIVE:
            self._client.execute(
                "INSERT INTO control_results_history VALUES",
                rows,
            )
        else:
            self._client.insert(
                "control_results_history",
                rows,
                column_names=list(rows[0].keys()) if rows else [],
            )
        
        logger.info(f"Inserted {len(rows)} control results into ClickHouse")
        return len(rows)
    
    def insert_positions(
        self,
        positions: List[Dict[str, Any]],
        nav: Decimal,
        as_of_date: date,
        fund_id: str = "MAIN",
    ) -> int:
        """
        Insert position snapshot into ClickHouse.
        
        Args:
            positions: List of position dicts
            nav: NAV for weight calculation
            as_of_date: Snapshot date
            fund_id: Fund identifier
            
        Returns:
            Number of rows inserted
        """
        self._ensure_connected()
        
        rows = []
        for p in positions:
            mv = float(p.get("market_value", 0))
            weight_pct = (mv / float(nav) * 100) if nav else 0
            
            rows.append({
                "as_of_date": as_of_date,
                "security_id": p.get("security_id", ""),
                "ticker": p.get("ticker", ""),
                "security_name": p.get("security_name", ""),
                "quantity": Decimal(str(p.get("quantity", 0))),
                "market_value": Decimal(str(mv)),
                "price": Decimal(str(p.get("price", 0))) if p.get("price") else Decimal("0"),
                "currency": p.get("currency", "USD"),
                "sector": p.get("sector", ""),
                "issuer": p.get("issuer", ""),
                "asset_class": p.get("asset_class", ""),
                "country": p.get("country", "US"),
                "weight_pct": weight_pct,
                "fund_id": fund_id,
            })
        
        # Also insert NAV
        nav_row = {
            "as_of_date": as_of_date,
            "nav": nav,
            "currency": "USD",
            "fund_id": fund_id,
        }
        
        if self._driver_type == ClickHouseDriver.NATIVE:
            self._client.execute("INSERT INTO positions_history VALUES", rows)
            self._client.execute("INSERT INTO nav_history VALUES", [nav_row])
        else:
            self._client.insert("positions_history", rows, column_names=list(rows[0].keys()) if rows else [])
            self._client.insert("nav_history", [nav_row], column_names=list(nav_row.keys()))
        
        logger.info(f"Inserted {len(rows)} positions + NAV into ClickHouse")
        return len(rows)
    
    # =========================================================================
    # CONTROL TREND ANALYSIS
    # =========================================================================
    
    def get_control_trend(
        self,
        control_id: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        fund_id: str = "MAIN",
    ) -> ControlTrend:
        """
        Get historical trend for a specific control.
        
        Args:
            control_id: Control identifier
            start_date: Start of period (default: 1 year ago)
            end_date: End of period (default: today)
            fund_id: Fund identifier
            
        Returns:
            ControlTrend with daily values and statistics
        """
        if not start_date:
            start_date = date.today() - timedelta(days=365)
        if not end_date:
            end_date = date.today()
        
        query = """
            SELECT 
                as_of_date,
                control_name,
                control_type,
                calculated_value,
                threshold,
                status,
                headroom_pct
            FROM control_results_history
            WHERE control_id = %(control_id)s
              AND fund_id = %(fund_id)s
              AND as_of_date BETWEEN %(start_date)s AND %(end_date)s
            ORDER BY as_of_date
        """
        
        results = self._execute(query, {
            "control_id": control_id,
            "fund_id": fund_id,
            "start_date": start_date,
            "end_date": end_date,
        })
        
        points = []
        control_name = ""
        control_type = ""
        
        for row in results:
            control_name = row.get("control_name", "")
            control_type = row.get("control_type", "")
            
            points.append(ControlTrendPoint(
                as_of_date=row["as_of_date"],
                calculated_value=Decimal(str(row["calculated_value"])),
                threshold=Decimal(str(row["threshold"])),
                status=row["status"],
                headroom_pct=float(row["headroom_pct"]),
            ))
        
        return ControlTrend(
            control_id=control_id,
            control_name=control_name,
            control_type=control_type,
            points=points,
        )
    
    def get_all_control_trends(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        fund_id: str = "MAIN",
    ) -> List[ControlTrend]:
        """
        Get trends for all controls in the period.
        
        Useful for generating comprehensive compliance reports.
        """
        if not start_date:
            start_date = date.today() - timedelta(days=90)
        if not end_date:
            end_date = date.today()
        
        # Get distinct controls
        controls_query = """
            SELECT DISTINCT control_id, control_name, control_type
            FROM control_results_history
            WHERE fund_id = %(fund_id)s
              AND as_of_date BETWEEN %(start_date)s AND %(end_date)s
            ORDER BY control_id
        """
        
        controls = self._execute(controls_query, {
            "fund_id": fund_id,
            "start_date": start_date,
            "end_date": end_date,
        })
        
        trends = []
        for ctrl in controls:
            trend = self.get_control_trend(
                control_id=ctrl["control_id"],
                start_date=start_date,
                end_date=end_date,
                fund_id=fund_id,
            )
            trends.append(trend)
        
        return trends
    
    # =========================================================================
    # BREACH STATISTICS
    # =========================================================================
    
    def get_breach_statistics(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        group_by: str = "month",  # "day", "week", "month", "quarter", "year"
        fund_id: str = "MAIN",
    ) -> List[BreachStatistics]:
        """
        Get aggregated breach statistics over time.
        
        Args:
            start_date: Start of period
            end_date: End of period
            group_by: Aggregation period
            fund_id: Fund identifier
            
        Returns:
            List of BreachStatistics per period
        """
        if not start_date:
            start_date = date.today() - timedelta(days=365)
        if not end_date:
            end_date = date.today()
        
        # Build date truncation based on grouping
        date_trunc_map = {
            "day": "toDate(as_of_date)",
            "week": "toStartOfWeek(as_of_date)",
            "month": "toStartOfMonth(as_of_date)",
            "quarter": "toStartOfQuarter(as_of_date)",
            "year": "toStartOfYear(as_of_date)",
        }
        date_format_map = {
            "day": "toString(period)",
            "week": "concat(toString(toYear(period)), '-W', toString(toWeek(period)))",
            "month": "formatDateTime(period, '%Y-%m')",
            "quarter": "concat(toString(toYear(period)), '-Q', toString(toQuarter(period)))",
            "year": "toString(toYear(period))",
        }
        
        date_trunc = date_trunc_map.get(group_by, date_trunc_map["month"])
        date_format = date_format_map.get(group_by, date_format_map["month"])
        
        query = f"""
            WITH period_stats AS (
                SELECT 
                    {date_trunc} AS period,
                    count() AS total_controls,
                    countIf(status = 'fail') AS total_breaches,
                    countIf(status = 'warning') AS total_warnings
                FROM control_results_history
                WHERE fund_id = %(fund_id)s
                  AND as_of_date BETWEEN %(start_date)s AND %(end_date)s
                GROUP BY period
            )
            SELECT 
                {date_format} AS period_label,
                total_controls,
                total_breaches,
                total_warnings,
                total_breaches * 100.0 / total_controls AS breach_rate,
                total_warnings * 100.0 / total_controls AS warning_rate
            FROM period_stats
            ORDER BY period
        """
        
        results = self._execute(query, {
            "fund_id": fund_id,
            "start_date": start_date,
            "end_date": end_date,
        })
        
        stats = []
        for row in results:
            stats.append(BreachStatistics(
                period=row["period_label"],
                total_controls=int(row["total_controls"]),
                total_breaches=int(row["total_breaches"]),
                total_warnings=int(row["total_warnings"]),
                breach_rate=float(row["breach_rate"]),
                warning_rate=float(row["warning_rate"]),
            ))
        
        return stats
    
    def get_top_breaching_controls(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        limit: int = 10,
        fund_id: str = "MAIN",
    ) -> List[Tuple[str, str, int]]:
        """
        Get controls with the most breaches.
        
        Returns:
            List of (control_id, control_name, breach_count)
        """
        if not start_date:
            start_date = date.today() - timedelta(days=365)
        if not end_date:
            end_date = date.today()
        
        query = """
            SELECT 
                control_id,
                any(control_name) AS control_name,
                countIf(status = 'fail') AS breach_count
            FROM control_results_history
            WHERE fund_id = %(fund_id)s
              AND as_of_date BETWEEN %(start_date)s AND %(end_date)s
            GROUP BY control_id
            HAVING breach_count > 0
            ORDER BY breach_count DESC
            LIMIT %(limit)s
        """
        
        results = self._execute(query, {
            "fund_id": fund_id,
            "start_date": start_date,
            "end_date": end_date,
            "limit": limit,
        })
        
        return [
            (row["control_id"], row["control_name"], int(row["breach_count"]))
            for row in results
        ]
    
    # =========================================================================
    # POSITION & CONCENTRATION ANALYSIS
    # =========================================================================
    
    def get_position_history(
        self,
        security_id: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        fund_id: str = "MAIN",
    ) -> List[PositionSnapshot]:
        """
        Get historical position data for a security.
        
        Useful for position sizing analysis and trade impact.
        """
        if not start_date:
            start_date = date.today() - timedelta(days=90)
        if not end_date:
            end_date = date.today()
        
        query = """
            SELECT 
                as_of_date,
                security_id,
                ticker,
                security_name,
                market_value,
                quantity,
                sector,
                weight_pct
            FROM positions_history
            WHERE security_id = %(security_id)s
              AND fund_id = %(fund_id)s
              AND as_of_date BETWEEN %(start_date)s AND %(end_date)s
            ORDER BY as_of_date
        """
        
        results = self._execute(query, {
            "security_id": security_id,
            "fund_id": fund_id,
            "start_date": start_date,
            "end_date": end_date,
        })
        
        return [
            PositionSnapshot(
                as_of_date=row["as_of_date"],
                security_id=row["security_id"],
                ticker=row["ticker"],
                security_name=row["security_name"],
                market_value=Decimal(str(row["market_value"])),
                quantity=Decimal(str(row["quantity"])),
                sector=row["sector"],
                weight_pct=float(row["weight_pct"]),
            )
            for row in results
        ]
    
    def get_concentration_trends(
        self,
        dimension: str = "sector",  # "sector", "issuer", "asset_class"
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        top_n: int = 10,
        fund_id: str = "MAIN",
    ) -> List[ConcentrationTrend]:
        """
        Get concentration trends by sector, issuer, or asset class.
        
        Args:
            dimension: Grouping dimension
            start_date: Start of period
            end_date: End of period
            top_n: Number of top concentrations to return
            fund_id: Fund identifier
            
        Returns:
            List of ConcentrationTrend objects
        """
        if not start_date:
            start_date = date.today() - timedelta(days=90)
        if not end_date:
            end_date = date.today()
        
        # Validate dimension
        valid_dimensions = {"sector", "issuer", "asset_class"}
        if dimension not in valid_dimensions:
            raise ValueError(f"Invalid dimension: {dimension}. Must be one of {valid_dimensions}")
        
        # First get top concentrations by average weight
        top_query = f"""
            SELECT 
                {dimension} AS name,
                avg(total_weight) AS avg_weight
            FROM (
                SELECT 
                    as_of_date,
                    {dimension},
                    sum(weight_pct) AS total_weight
                FROM positions_history
                WHERE fund_id = %(fund_id)s
                  AND as_of_date BETWEEN %(start_date)s AND %(end_date)s
                  AND {dimension} != ''
                GROUP BY as_of_date, {dimension}
            )
            GROUP BY {dimension}
            ORDER BY avg_weight DESC
            LIMIT %(top_n)s
        """
        
        top_names = self._execute(top_query, {
            "fund_id": fund_id,
            "start_date": start_date,
            "end_date": end_date,
            "top_n": top_n,
        })
        
        trends = []
        for name_row in top_names:
            name = name_row["name"]
            
            # Get time series for this concentration
            series_query = f"""
                SELECT 
                    as_of_date,
                    sum(weight_pct) AS total_weight
                FROM positions_history
                WHERE fund_id = %(fund_id)s
                  AND as_of_date BETWEEN %(start_date)s AND %(end_date)s
                  AND {dimension} = %(name)s
                GROUP BY as_of_date
                ORDER BY as_of_date
            """
            
            series = self._execute(series_query, {
                "fund_id": fund_id,
                "start_date": start_date,
                "end_date": end_date,
                "name": name,
            })
            
            points = [(row["as_of_date"], float(row["total_weight"])) for row in series]
            
            trends.append(ConcentrationTrend(
                dimension=dimension,
                name=name,
                points=points,
            ))
        
        return trends
    
    # =========================================================================
    # POINT-IN-TIME ANALYSIS
    # =========================================================================
    
    def get_portfolio_snapshot(
        self,
        as_of_date: date,
        fund_id: str = "MAIN",
    ) -> Dict[str, Any]:
        """
        Get complete portfolio state as of a specific date.
        
        Useful for regulatory examinations and historical analysis.
        
        Returns:
            Dict with positions, NAV, controls, and summary stats
        """
        # Get positions
        positions_query = """
            SELECT 
                security_id, ticker, security_name,
                quantity, market_value, price,
                sector, issuer, asset_class, weight_pct
            FROM positions_history
            WHERE as_of_date = %(as_of_date)s
              AND fund_id = %(fund_id)s
            ORDER BY market_value DESC
        """
        positions = self._execute(positions_query, {
            "as_of_date": as_of_date,
            "fund_id": fund_id,
        })
        
        # Get NAV
        nav_query = """
            SELECT nav
            FROM nav_history
            WHERE as_of_date = %(as_of_date)s
              AND fund_id = %(fund_id)s
            LIMIT 1
        """
        nav_result = self._execute(nav_query, {
            "as_of_date": as_of_date,
            "fund_id": fund_id,
        })
        nav = Decimal(str(nav_result[0]["nav"])) if nav_result else Decimal("0")
        
        # Get control results
        controls_query = """
            SELECT 
                control_id, control_name, control_type,
                calculated_value, threshold, threshold_operator,
                status, breach_amount, headroom_pct
            FROM control_results_history
            WHERE as_of_date = %(as_of_date)s
              AND fund_id = %(fund_id)s
            ORDER BY 
                CASE status WHEN 'fail' THEN 0 WHEN 'warning' THEN 1 ELSE 2 END,
                control_id
        """
        controls = self._execute(controls_query, {
            "as_of_date": as_of_date,
            "fund_id": fund_id,
        })
        
        # Calculate summary
        total_mv = sum(Decimal(str(p["market_value"])) for p in positions)
        sector_weights = {}
        for p in positions:
            sector = p.get("sector", "Unknown")
            sector_weights[sector] = sector_weights.get(sector, 0) + float(p["weight_pct"])
        
        return {
            "as_of_date": as_of_date,
            "fund_id": fund_id,
            "nav": nav,
            "total_market_value": total_mv,
            "position_count": len(positions),
            "positions": positions,
            "control_count": len(controls),
            "controls": controls,
            "passed_controls": sum(1 for c in controls if c["status"] == "pass"),
            "warning_controls": sum(1 for c in controls if c["status"] == "warning"),
            "failed_controls": sum(1 for c in controls if c["status"] == "fail"),
            "sector_weights": sector_weights,
        }
    
    # =========================================================================
    # REGULATORY REPORTING SUPPORT
    # =========================================================================
    
    def get_form_pf_data(
        self,
        reporting_period_start: date,
        reporting_period_end: date,
        fund_id: str = "MAIN",
    ) -> Dict[str, Any]:
        """
        Get data required for SEC Form PF filing.
        
        Includes:
        - Quarterly NAV highs/lows
        - Average control headroom
        - Breach summary
        - Concentration data
        """
        # NAV statistics
        nav_query = """
            SELECT 
                min(nav) AS min_nav,
                max(nav) AS max_nav,
                avg(nav) AS avg_nav,
                argMin(as_of_date, nav) AS min_nav_date,
                argMax(as_of_date, nav) AS max_nav_date
            FROM nav_history
            WHERE as_of_date BETWEEN %(start)s AND %(end)s
              AND fund_id = %(fund_id)s
        """
        nav_stats = self._execute(nav_query, {
            "start": reporting_period_start,
            "end": reporting_period_end,
            "fund_id": fund_id,
        })[0]
        
        # Exposure statistics
        exposure_query = """
            SELECT 
                control_id,
                avg(calculated_value) AS avg_value,
                max(calculated_value) AS max_value,
                min(headroom_pct) AS min_headroom
            FROM control_results_history
            WHERE as_of_date BETWEEN %(start)s AND %(end)s
              AND fund_id = %(fund_id)s
              AND control_type = 'exposure'
            GROUP BY control_id
        """
        exposure_stats = self._execute(exposure_query, {
            "start": reporting_period_start,
            "end": reporting_period_end,
            "fund_id": fund_id,
        })
        
        # Breach summary
        breach_query = """
            SELECT 
                countIf(status = 'fail') AS total_breaches,
                countIf(status = 'warning') AS total_warnings,
                count(DISTINCT as_of_date) AS trading_days,
                count(DISTINCT control_id) AS unique_controls
            FROM control_results_history
            WHERE as_of_date BETWEEN %(start)s AND %(end)s
              AND fund_id = %(fund_id)s
        """
        breach_stats = self._execute(breach_query, {
            "start": reporting_period_start,
            "end": reporting_period_end,
            "fund_id": fund_id,
        })[0]
        
        return {
            "reporting_period": {
                "start": reporting_period_start.isoformat(),
                "end": reporting_period_end.isoformat(),
            },
            "nav": {
                "minimum": float(nav_stats["min_nav"]) if nav_stats["min_nav"] else 0,
                "maximum": float(nav_stats["max_nav"]) if nav_stats["max_nav"] else 0,
                "average": float(nav_stats["avg_nav"]) if nav_stats["avg_nav"] else 0,
                "min_date": nav_stats["min_nav_date"].isoformat() if nav_stats["min_nav_date"] else None,
                "max_date": nav_stats["max_nav_date"].isoformat() if nav_stats["max_nav_date"] else None,
            },
            "exposure": [
                {
                    "control_id": e["control_id"],
                    "avg_value": float(e["avg_value"]),
                    "max_value": float(e["max_value"]),
                    "min_headroom_pct": float(e["min_headroom"]),
                }
                for e in exposure_stats
            ],
            "compliance": {
                "total_breaches": int(breach_stats["total_breaches"]),
                "total_warnings": int(breach_stats["total_warnings"]),
                "trading_days": int(breach_stats["trading_days"]),
                "unique_controls_monitored": int(breach_stats["unique_controls"]),
            },
        }
    
    # =========================================================================
    # UTILITIES
    # =========================================================================
    
    def test_connection(self) -> Dict[str, Any]:
        """Test ClickHouse connection and return diagnostics."""
        self._ensure_connected()
        
        diagnostics = {
            "connected": True,
            "driver": self._driver_type.value if self._driver_type else "unknown",
            "host": self.config.host,
            "database": self.config.database,
        }
        
        try:
            # Check version
            result = self._execute("SELECT version() AS version")
            diagnostics["version"] = result[0]["version"] if result else "unknown"
            
            # Check table sizes
            tables_query = """
                SELECT 
                    table,
                    formatReadableSize(sum(bytes_on_disk)) AS size,
                    sum(rows) AS rows
                FROM system.parts
                WHERE database = %(database)s
                  AND active = 1
                GROUP BY table
            """
            tables = self._execute(tables_query, {"database": self.config.database})
            diagnostics["tables"] = {
                t["table"]: {"size": t["size"], "rows": int(t["rows"])}
                for t in tables
            }
            
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
# MOCK ADAPTER FOR TESTING
# =============================================================================

class MockClickHouseAnalytics:
    """
    Mock ClickHouse analytics for testing without database.
    
    Generates realistic sample data for development and demos.
    """
    
    def __init__(self):
        logger.info("Using MockClickHouseAnalytics (no ClickHouse connection)")
        self._control_trends = self._generate_mock_trends()
    
    def _generate_mock_trends(self) -> Dict[str, ControlTrend]:
        """Generate realistic mock control trends."""
        import random
        
        controls = [
            ("CONC_SECTOR_001", "Sector Concentration - Technology", "concentration", 28.0, 35.0),
            ("CONC_ISSUER_001", "Single Issuer Concentration", "concentration", 7.5, 10.0),
            ("EXP_GROSS_001", "Gross Exposure", "exposure", 145.0, 200.0),
            ("EXP_NET_001", "Net Exposure", "exposure", 72.0, 100.0),
            ("LIQ_T1_001", "T+1 Liquidity", "liquidity", 18.0, 10.0),  # gte
        ]
        
        trends = {}
        today = date.today()
        
        for control_id, name, ctype, base_val, thresh in controls:
            points = []
            for i in range(365):
                d = today - timedelta(days=365-i)
                # Add some realistic variation
                val = base_val + random.uniform(-5, 5)
                
                # Occasional warning/breach
                if random.random() < 0.02:  # 2% breach rate
                    if ctype == "liquidity":
                        val = thresh - random.uniform(1, 5)  # Below minimum
                    else:
                        val = thresh + random.uniform(1, 10)  # Above maximum
                
                if ctype == "liquidity":
                    headroom = (val - thresh) / thresh * 100
                    status = "pass" if val >= thresh else ("warning" if val >= thresh * 0.9 else "fail")
                else:
                    headroom = (thresh - val) / thresh * 100
                    status = "pass" if val <= thresh else ("warning" if val <= thresh * 1.1 else "fail")
                
                points.append(ControlTrendPoint(
                    as_of_date=d,
                    calculated_value=Decimal(str(round(val, 2))),
                    threshold=Decimal(str(thresh)),
                    status=status,
                    headroom_pct=headroom,
                ))
            
            trends[control_id] = ControlTrend(
                control_id=control_id,
                control_name=name,
                control_type=ctype,
                points=points,
            )
        
        return trends
    
    def connect(self) -> None:
        pass
    
    def close(self) -> None:
        pass
    
    def create_schema(self) -> None:
        logger.info("Mock: Schema creation skipped")
    
    def get_control_trend(
        self,
        control_id: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        fund_id: str = "MAIN",
    ) -> ControlTrend:
        if control_id in self._control_trends:
            trend = self._control_trends[control_id]
            if start_date or end_date:
                filtered_points = [
                    p for p in trend.points
                    if (not start_date or p.as_of_date >= start_date)
                    and (not end_date or p.as_of_date <= end_date)
                ]
                return ControlTrend(
                    control_id=trend.control_id,
                    control_name=trend.control_name,
                    control_type=trend.control_type,
                    points=filtered_points,
                )
            return trend
        
        return ControlTrend(control_id=control_id, control_name="Unknown", control_type="other", points=[])
    
    def get_all_control_trends(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        fund_id: str = "MAIN",
    ) -> List[ControlTrend]:
        return [
            self.get_control_trend(cid, start_date, end_date, fund_id)
            for cid in self._control_trends
        ]
    
    def get_breach_statistics(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        group_by: str = "month",
        fund_id: str = "MAIN",
    ) -> List[BreachStatistics]:
        # Generate mock monthly stats
        stats = []
        today = date.today()
        
        for i in range(12):
            month_date = today - timedelta(days=30 * (12 - i))
            period = month_date.strftime("%Y-%m")
            
            import random
            random.seed(month_date.toordinal())
            
            total = 150 + random.randint(-10, 10)
            breaches = random.randint(0, 5)
            warnings = random.randint(2, 10)
            
            stats.append(BreachStatistics(
                period=period,
                total_controls=total,
                total_breaches=breaches,
                total_warnings=warnings,
                breach_rate=breaches / total * 100,
                warning_rate=warnings / total * 100,
            ))
        
        return stats
    
    def test_connection(self) -> Dict[str, Any]:
        return {
            "connected": True,
            "driver": "mock",
            "version": "mock-1.0",
            "tables": {},
        }
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def get_clickhouse_analytics(
    use_mock: bool = False,
    config: Optional[ClickHouseConfig] = None,
) -> Union[ClickHouseAnalytics, MockClickHouseAnalytics]:
    """
    Factory function to get ClickHouse analytics instance.
    
    Args:
        use_mock: Force mock adapter (for testing)
        config: ClickHouse configuration (loads from env if not provided)
        
    Returns:
        ClickHouseAnalytics or MockClickHouseAnalytics
    
    Example:
        # Production - uses environment variables
        analytics = get_clickhouse_analytics()
        
        # Testing
        analytics = get_clickhouse_analytics(use_mock=True)
    """
    if use_mock:
        return MockClickHouseAnalytics()
    
    try:
        if config is None:
            config = ClickHouseConfig.from_env()
        return ClickHouseAnalytics(config)
    except (ValueError, KeyError) as e:
        logger.warning(f"ClickHouse config not available ({e}), using mock analytics")
        return MockClickHouseAnalytics()


# =============================================================================
# CLI DEMO
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ClickHouse Analytics Demo")
    parser.add_argument("--mock", action="store_true", help="Use mock adapter")
    args = parser.parse_args()
    
    print("=" * 70)
    print("CLICKHOUSE ANALYTICS DEMO")
    print("=" * 70)
    
    analytics = get_clickhouse_analytics(use_mock=args.mock or True)
    
    # Test connection
    diag = analytics.test_connection()
    print(f"\nConnection: {diag}")
    
    # Get control trend
    print("\n" + "-" * 70)
    print("Control Trend: CONC_SECTOR_001 (last 30 days)")
    print("-" * 70)
    
    trend = analytics.get_control_trend(
        "CONC_SECTOR_001",
        start_date=date.today() - timedelta(days=30),
    )
    
    print(f"Control: {trend.control_name}")
    print(f"Type: {trend.control_type}")
    print(f"Points: {len(trend.points)}")
    print(f"Min: {trend.min_value}%, Max: {trend.max_value}%, Avg: {trend.avg_value:.2f}%")
    print(f"Breaches: {trend.breach_count}, Warnings: {trend.warning_count}")
    
    # Get breach statistics
    print("\n" + "-" * 70)
    print("Monthly Breach Statistics (last 12 months)")
    print("-" * 70)
    
    stats = analytics.get_breach_statistics(group_by="month")
    for s in stats[-6:]:  # Last 6 months
        print(f"  {s.period}: {s.total_breaches} breaches, {s.total_warnings} warnings ({s.breach_rate:.1f}%)")
    
    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
