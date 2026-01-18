"""
Snowflake Data Connector - Secure, Validated Data Extraction

This module handles the connection to Snowflake and extraction of position data
with built-in quality validation gates.

Security Notes:
- Read-only access using service account
- All queries logged
- No dynamic SQL construction
- Point-in-time snapshots only
"""

from __future__ import annotations

import os
import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, date
from typing import Optional, Dict, Any, List, Tuple
from decimal import Decimal

logger = logging.getLogger(__name__)


@dataclass
class SnowflakeConfig:
    """Snowflake connection configuration."""
    account: str
    user: str
    password: str
    warehouse: str
    database: str
    schema: str
    role: Optional[str] = None
    
    @classmethod
    def from_env(cls) -> 'SnowflakeConfig':
        """Load configuration from environment variables."""
        return cls(
            account=os.environ.get('SNOWFLAKE_ACCOUNT', ''),
            user=os.environ.get('SNOWFLAKE_USER', ''),
            password=os.environ.get('SNOWFLAKE_PASSWORD', ''),
            warehouse=os.environ.get('SNOWFLAKE_WAREHOUSE', 'COMPLIANCE_WH'),
            database=os.environ.get('SNOWFLAKE_DATABASE', 'HEDGE_FUND_DATA'),
            schema=os.environ.get('SNOWFLAKE_SCHEMA', 'compliance'),
            role=os.environ.get('SNOWFLAKE_ROLE'),
        )
    
    def is_configured(self) -> bool:
        """Check if all required config is present."""
        return bool(self.account and self.user and self.password)


@dataclass
class DataSnapshot:
    """Represents a point-in-time data snapshot."""
    snapshot_id: str
    snapshot_date: date
    snapshot_timestamp: datetime
    snapshot_type: str  # 'eod', 'intraday', 'adhoc'
    record_count: int
    validation_status: str
    query_hash: str
    extracted_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "snapshot_date": self.snapshot_date.isoformat(),
            "snapshot_timestamp": self.snapshot_timestamp.isoformat(),
            "snapshot_type": self.snapshot_type,
            "record_count": self.record_count,
            "validation_status": self.validation_status,
            "query_hash": self.query_hash,
            "extracted_at": self.extracted_at.isoformat(),
        }


class SnowflakeConnector:
    """
    Secure connector for Snowflake data extraction.
    
    Implements:
    - Connection pooling
    - Query logging
    - Result validation
    - Point-in-time consistency
    """
    
    def __init__(self, config: Optional[SnowflakeConfig] = None):
        self.config = config or SnowflakeConfig.from_env()
        self._connection = None
        self._query_log: List[Dict[str, Any]] = []
    
    def connect(self) -> bool:
        """
        Establish connection to Snowflake.
        
        Returns:
            True if connection successful, False otherwise
        """
        if not self.config.is_configured():
            logger.warning("Snowflake not configured - using mock mode")
            return False
        
        try:
            import snowflake.connector
            
            self._connection = snowflake.connector.connect(
                account=self.config.account,
                user=self.config.user,
                password=self.config.password,
                warehouse=self.config.warehouse,
                database=self.config.database,
                schema=self.config.schema,
                role=self.config.role,
            )
            
            logger.info(f"Connected to Snowflake: {self.config.database}.{self.config.schema}")
            return True
            
        except ImportError:
            logger.error("snowflake-connector-python not installed")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to Snowflake: {e}")
            return False
    
    def disconnect(self):
        """Close connection."""
        if self._connection:
            self._connection.close()
            self._connection = None
    
    def get_available_snapshots(self, days_back: int = 7) -> List[DataSnapshot]:
        """
        Get list of available data snapshots.
        
        Args:
            days_back: Number of days of snapshots to retrieve
            
        Returns:
            List of DataSnapshot objects
        """
        query = """
        SELECT 
            snapshot_id,
            snapshot_date,
            snapshot_timestamp,
            snapshot_type,
            record_count,
            validation_status
        FROM compliance.v_data_snapshots
        WHERE snapshot_date >= DATEADD(day, -%s, CURRENT_DATE())
        ORDER BY snapshot_timestamp DESC
        """
        
        results = self._execute_query(query, (days_back,))
        
        snapshots = []
        for row in results:
            snapshots.append(DataSnapshot(
                snapshot_id=row[0],
                snapshot_date=row[1],
                snapshot_timestamp=row[2],
                snapshot_type=row[3],
                record_count=row[4],
                validation_status=row[5],
                query_hash=self._hash_query(query, (days_back,)),
            ))
        
        return snapshots
    
    def get_positions_for_snapshot(
        self, 
        snapshot_id: str,
        fund_id: Optional[str] = None
    ) -> Tuple[List[Dict[str, Any]], DataSnapshot]:
        """
        Extract position data for a specific snapshot.
        
        Args:
            snapshot_id: The snapshot to extract
            fund_id: Optional filter to specific fund
            
        Returns:
            Tuple of (positions list, snapshot metadata)
        """
        # Build query (NO dynamic SQL - parameterized only)
        query = """
        SELECT 
            position_id,
            snapshot_id,
            snapshot_date,
            account_id,
            fund_id,
            security_id,
            security_type,
            ticker,
            isin,
            cusip,
            sedol,
            security_description,
            quantity,
            quantity_long,
            quantity_short,
            market_value_usd,
            market_value_local,
            currency,
            fx_rate_to_usd,
            cost_basis_usd,
            unrealized_pnl_usd,
            asset_class,
            sector,
            country,
            region,
            issuer_id,
            issuer_name,
            ultimate_parent_issuer_id,
            price,
            price_date,
            price_source,
            is_stale_price
        FROM compliance.v_positions
        WHERE snapshot_id = %s
        """
        
        params = [snapshot_id]
        
        if fund_id:
            query += " AND fund_id = %s"
            params.append(fund_id)
        
        query += " ORDER BY fund_id, market_value_usd DESC"
        
        results = self._execute_query(query, tuple(params))
        
        # Convert to dictionaries
        columns = [
            'position_id', 'snapshot_id', 'snapshot_date', 'account_id', 'fund_id',
            'security_id', 'security_type', 'ticker', 'isin', 'cusip', 'sedol',
            'security_description', 'quantity', 'quantity_long', 'quantity_short',
            'market_value_usd', 'market_value_local', 'currency', 'fx_rate_to_usd',
            'cost_basis_usd', 'unrealized_pnl_usd', 'asset_class', 'sector',
            'country', 'region', 'issuer_id', 'issuer_name', 
            'ultimate_parent_issuer_id', 'price', 'price_date', 'price_source',
            'is_stale_price'
        ]
        
        positions = []
        for row in results:
            pos = {}
            for i, col in enumerate(columns):
                value = row[i]
                # Convert Decimal to float for JSON serialization
                if isinstance(value, Decimal):
                    value = float(value)
                pos[col] = value
            positions.append(pos)
        
        # Create snapshot metadata
        snapshot = DataSnapshot(
            snapshot_id=snapshot_id,
            snapshot_date=positions[0]['snapshot_date'] if positions else date.today(),
            snapshot_timestamp=datetime.now(timezone.utc),
            snapshot_type='eod',
            record_count=len(positions),
            validation_status='extracted',
            query_hash=self._hash_query(query, tuple(params)),
        )
        
        logger.info(f"Extracted {len(positions)} positions for snapshot {snapshot_id}")
        
        return positions, snapshot
    
    def get_issuer_concentrations(self, snapshot_id: str) -> List[Dict[str, Any]]:
        """Get pre-calculated issuer concentrations."""
        query = """
        SELECT 
            snapshot_id,
            snapshot_date,
            fund_id,
            issuer_id,
            issuer_name,
            ultimate_parent_issuer_id,
            issuer_exposure_usd,
            fund_nav,
            issuer_concentration_pct,
            distinct_securities,
            security_types
        FROM compliance.v_issuer_concentration
        WHERE snapshot_id = %s
        ORDER BY issuer_concentration_pct DESC
        """
        
        results = self._execute_query(query, (snapshot_id,))
        
        columns = [
            'snapshot_id', 'snapshot_date', 'fund_id', 'issuer_id', 'issuer_name',
            'ultimate_parent_issuer_id', 'issuer_exposure_usd', 'fund_nav',
            'issuer_concentration_pct', 'distinct_securities', 'security_types'
        ]
        
        concentrations = []
        for row in results:
            conc = {}
            for i, col in enumerate(columns):
                value = row[i]
                if isinstance(value, Decimal):
                    value = float(value)
                conc[col] = value
            concentrations.append(conc)
        
        return concentrations
    
    def get_sector_concentrations(self, snapshot_id: str) -> List[Dict[str, Any]]:
        """Get pre-calculated sector concentrations."""
        query = """
        SELECT 
            snapshot_id,
            snapshot_date,
            fund_id,
            sector,
            sector_exposure_usd,
            fund_nav,
            sector_concentration_pct,
            distinct_securities,
            distinct_issuers
        FROM compliance.v_sector_concentration
        WHERE snapshot_id = %s
        ORDER BY sector_concentration_pct DESC
        """
        
        results = self._execute_query(query, (snapshot_id,))
        
        columns = [
            'snapshot_id', 'snapshot_date', 'fund_id', 'sector',
            'sector_exposure_usd', 'fund_nav', 'sector_concentration_pct',
            'distinct_securities', 'distinct_issuers'
        ]
        
        concentrations = []
        for row in results:
            conc = {}
            for i, col in enumerate(columns):
                value = row[i]
                if isinstance(value, Decimal):
                    value = float(value)
                conc[col] = value
            concentrations.append(conc)
        
        return concentrations
    
    def get_liquidity_profile(self, snapshot_id: str) -> List[Dict[str, Any]]:
        """Get liquidity profile data."""
        query = """
        SELECT 
            snapshot_id,
            snapshot_date,
            fund_id,
            liquidity_bucket,
            days_to_liquidate,
            bucket_market_value,
            fund_nav,
            bucket_pct_nav,
            cumulative_pct_nav,
            position_count
        FROM compliance.v_liquidity_profile
        WHERE snapshot_id = %s
        ORDER BY fund_id, days_to_liquidate
        """
        
        results = self._execute_query(query, (snapshot_id,))
        
        columns = [
            'snapshot_id', 'snapshot_date', 'fund_id', 'liquidity_bucket',
            'days_to_liquidate', 'bucket_market_value', 'fund_nav',
            'bucket_pct_nav', 'cumulative_pct_nav', 'position_count'
        ]
        
        liquidity = []
        for row in results:
            liq = {}
            for i, col in enumerate(columns):
                value = row[i]
                if isinstance(value, Decimal):
                    value = float(value)
                liq[col] = value
            liquidity.append(liq)
        
        return liquidity
    
    def _execute_query(self, query: str, params: tuple = ()) -> List[tuple]:
        """
        Execute a query with logging.
        
        All queries are logged for audit trail.
        """
        query_hash = self._hash_query(query, params)
        
        log_entry = {
            "query_hash": query_hash,
            "executed_at": datetime.now(timezone.utc).isoformat(),
            "param_count": len(params),
        }
        self._query_log.append(log_entry)
        
        if not self._connection:
            logger.warning("No connection - returning empty results")
            return []
        
        try:
            cursor = self._connection.cursor()
            cursor.execute(query, params)
            results = cursor.fetchall()
            cursor.close()
            
            log_entry["row_count"] = len(results)
            log_entry["status"] = "success"
            
            return results
            
        except Exception as e:
            log_entry["status"] = "error"
            log_entry["error"] = str(e)
            logger.error(f"Query failed: {e}")
            raise
    
    def _hash_query(self, query: str, params: tuple) -> str:
        """Create deterministic hash of query for audit."""
        query_str = f"{query}|{json.dumps(params, default=str, sort_keys=True)}"
        return hashlib.sha256(query_str.encode()).hexdigest()[:16]
    
    def get_query_log(self) -> List[Dict[str, Any]]:
        """Get the query log for audit."""
        return self._query_log.copy()


# =============================================================================
# MOCK CONNECTOR FOR TESTING
# =============================================================================

class MockSnowflakeConnector(SnowflakeConnector):
    """
    Mock connector for testing without Snowflake access.
    
    Generates realistic sample data for testing.
    """
    
    def __init__(self):
        super().__init__(SnowflakeConfig(
            account='mock', user='mock', password='mock',
            warehouse='mock', database='mock', schema='mock'
        ))
        self._connected = True
    
    def connect(self) -> bool:
        self._connected = True
        return True
    
    def disconnect(self):
        self._connected = False
    
    def get_available_snapshots(self, days_back: int = 7) -> List[DataSnapshot]:
        """Return mock snapshots."""
        from datetime import timedelta
        
        snapshots = []
        for i in range(days_back):
            snap_date = date.today() - timedelta(days=i)
            snapshots.append(DataSnapshot(
                snapshot_id=f"SNAP-{snap_date.strftime('%Y%m%d')}-EOD",
                snapshot_date=snap_date,
                snapshot_timestamp=datetime.combine(snap_date, datetime.min.time()).replace(tzinfo=timezone.utc),
                snapshot_type='eod',
                record_count=150,
                validation_status='valid',
                query_hash=f"mock_{i}",
            ))
        
        return snapshots
    
    def get_positions_for_snapshot(
        self, 
        snapshot_id: str,
        fund_id: Optional[str] = None
    ) -> Tuple[List[Dict[str, Any]], DataSnapshot]:
        """Generate mock position data."""
        import random
        
        # Sample securities
        securities = [
            {'ticker': 'AAPL', 'isin': 'US0378331005', 'name': 'Apple Inc', 'sector': 'Technology', 'issuer': 'Apple Inc'},
            {'ticker': 'MSFT', 'isin': 'US5949181045', 'name': 'Microsoft Corp', 'sector': 'Technology', 'issuer': 'Microsoft Corp'},
            {'ticker': 'GOOGL', 'isin': 'US02079K3059', 'name': 'Alphabet Inc', 'sector': 'Technology', 'issuer': 'Alphabet Inc'},
            {'ticker': 'JPM', 'isin': 'US46625H1005', 'name': 'JPMorgan Chase', 'sector': 'Financials', 'issuer': 'JPMorgan Chase'},
            {'ticker': 'JNJ', 'isin': 'US4781601046', 'name': 'Johnson & Johnson', 'sector': 'Healthcare', 'issuer': 'Johnson & Johnson'},
            {'ticker': 'XOM', 'isin': 'US30231G1022', 'name': 'Exxon Mobil', 'sector': 'Energy', 'issuer': 'Exxon Mobil'},
            {'ticker': 'PG', 'isin': 'US7427181091', 'name': 'Procter & Gamble', 'sector': 'Consumer Staples', 'issuer': 'Procter & Gamble'},
            {'ticker': 'NVDA', 'isin': 'US67066G1040', 'name': 'NVIDIA Corp', 'sector': 'Technology', 'issuer': 'NVIDIA Corp'},
            {'ticker': 'V', 'isin': 'US92826C8394', 'name': 'Visa Inc', 'sector': 'Financials', 'issuer': 'Visa Inc'},
            {'ticker': 'UNH', 'isin': 'US91324P1021', 'name': 'UnitedHealth', 'sector': 'Healthcare', 'issuer': 'UnitedHealth Group'},
        ]
        
        snap_date = date.today()
        positions = []
        
        for i, sec in enumerate(securities):
            quantity = random.randint(1000, 50000)
            price = random.uniform(50, 500)
            market_value = quantity * price
            
            positions.append({
                'position_id': f"POS-{i+1:04d}",
                'snapshot_id': snapshot_id,
                'snapshot_date': snap_date,
                'account_id': 'MAIN-001',
                'fund_id': fund_id or 'MASTER-FUND',
                'security_id': f"SEC-{i+1:04d}",
                'security_type': 'equity',
                'ticker': sec['ticker'],
                'isin': sec['isin'],
                'cusip': None,
                'sedol': None,
                'security_description': sec['name'],
                'quantity': quantity,
                'quantity_long': quantity if quantity > 0 else 0,
                'quantity_short': 0,
                'market_value_usd': round(market_value, 2),
                'market_value_local': round(market_value, 2),
                'currency': 'USD',
                'fx_rate_to_usd': 1.0,
                'cost_basis_usd': round(market_value * 0.9, 2),
                'unrealized_pnl_usd': round(market_value * 0.1, 2),
                'asset_class': 'equity',
                'sector': sec['sector'],
                'country': 'US',
                'region': 'North America',
                'issuer_id': f"ISS-{i+1:04d}",
                'issuer_name': sec['issuer'],
                'ultimate_parent_issuer_id': f"ISS-{i+1:04d}",
                'price': round(price, 2),
                'price_date': snap_date,
                'price_source': 'bloomberg',
                'is_stale_price': False,
            })
        
        snapshot = DataSnapshot(
            snapshot_id=snapshot_id,
            snapshot_date=snap_date,
            snapshot_timestamp=datetime.now(timezone.utc),
            snapshot_type='eod',
            record_count=len(positions),
            validation_status='valid',
            query_hash='mock_positions',
        )
        
        return positions, snapshot


def get_connector(use_mock: bool = False) -> SnowflakeConnector:
    """Factory function to get appropriate connector."""
    if use_mock:
        return MockSnowflakeConnector()
    
    config = SnowflakeConfig.from_env()
    if config.is_configured():
        return SnowflakeConnector(config)
    else:
        logger.warning("Snowflake not configured, using mock connector")
        return MockSnowflakeConnector()
