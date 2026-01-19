"""
Snowflake Connector for Compliance RAG System

Connects to Snowflake to pull:
- Position data
- Control results (pre-calculated by client's system)
- NAV and fund metadata

Configuration via environment variables:
- SNOWFLAKE_ACCOUNT
- SNOWFLAKE_USER
- SNOWFLAKE_PASSWORD
- SNOWFLAKE_WAREHOUSE
- SNOWFLAKE_DATABASE
- SNOWFLAKE_SCHEMA
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
    def from_env(cls) -> "SnowflakeConfig":
        """Create config from environment variables."""
        return cls(
            account=os.environ["SNOWFLAKE_ACCOUNT"],
            user=os.environ["SNOWFLAKE_USER"],
            password=os.environ["SNOWFLAKE_PASSWORD"],
            warehouse=os.environ.get("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH"),
            database=os.environ.get("SNOWFLAKE_DATABASE", "COMPLIANCE"),
            schema=os.environ.get("SNOWFLAKE_SCHEMA", "PUBLIC"),
            role=os.environ.get("SNOWFLAKE_ROLE"),
        )


class SnowflakeConnector:
    """
    Connector for Snowflake data warehouse.
    
    Expected Snowflake tables (client's data):
    
    POSITIONS:
        - SECURITY_ID VARCHAR
        - TICKER VARCHAR
        - SECURITY_NAME VARCHAR
        - QUANTITY NUMBER
        - MARKET_VALUE NUMBER
        - CURRENCY VARCHAR
        - SECTOR VARCHAR
        - AS_OF_DATE DATE
    
    CONTROL_RESULTS:
        - CONTROL_ID VARCHAR
        - CONTROL_NAME VARCHAR
        - CONTROL_TYPE VARCHAR
        - CALCULATED_VALUE NUMBER
        - THRESHOLD NUMBER
        - THRESHOLD_OPERATOR VARCHAR
        - STATUS VARCHAR (pass/warning/fail)
        - BREACH_AMOUNT NUMBER
        - AS_OF_DATE DATE
    
    FUND_NAV:
        - NAV NUMBER
        - AS_OF_DATE DATE
    """
    
    def __init__(self, config: Optional[SnowflakeConfig] = None):
        self.config = config or SnowflakeConfig.from_env()
        self._connection = None
    
    @property
    def connection(self):
        """Lazy connection initialization."""
        if self._connection is None:
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
                logger.info(f"Connected to Snowflake: {self.config.account}")
                
            except ImportError:
                raise ImportError(
                    "snowflake-connector-python not installed. "
                    "Run: pip install snowflake-connector-python"
                )
        
        return self._connection
    
    def get_positions(self, as_of_date: date) -> List[Dict[str, Any]]:
        """
        Get positions from Snowflake.
        
        Args:
            as_of_date: Date for position data
            
        Returns:
            List of position dictionaries
        """
        query = """
        SELECT 
            SECURITY_ID,
            TICKER,
            SECURITY_NAME,
            QUANTITY,
            MARKET_VALUE,
            CURRENCY,
            SECTOR,
            ISSUER,
            ASSET_CLASS
        FROM POSITIONS
        WHERE AS_OF_DATE = %(as_of_date)s
        ORDER BY MARKET_VALUE DESC
        """
        
        cursor = self.connection.cursor()
        cursor.execute(query, {"as_of_date": as_of_date})
        
        columns = [desc[0].lower() for desc in cursor.description]
        positions = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        cursor.close()
        logger.info(f"Retrieved {len(positions)} positions for {as_of_date}")
        
        return positions
    
    def get_control_results(self, as_of_date: date) -> List[Dict[str, Any]]:
        """
        Get control results from Snowflake.
        
        The client's system has already calculated these - we just retrieve them.
        
        Args:
            as_of_date: Date for control results
            
        Returns:
            List of control result dictionaries
        """
        query = """
        SELECT 
            CONTROL_ID,
            CONTROL_NAME,
            CONTROL_TYPE,
            CALCULATED_VALUE,
            THRESHOLD,
            THRESHOLD_OPERATOR,
            STATUS,
            BREACH_AMOUNT,
            DETAILS
        FROM CONTROL_RESULTS
        WHERE AS_OF_DATE = %(as_of_date)s
        ORDER BY 
            CASE STATUS 
                WHEN 'fail' THEN 1 
                WHEN 'warning' THEN 2 
                ELSE 3 
            END
        """
        
        cursor = self.connection.cursor()
        cursor.execute(query, {"as_of_date": as_of_date})
        
        columns = [desc[0].lower() for desc in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        cursor.close()
        logger.info(f"Retrieved {len(results)} control results for {as_of_date}")
        
        return results
    
    def get_nav(self, as_of_date: date) -> Decimal:
        """
        Get fund NAV from Snowflake.
        
        Args:
            as_of_date: Date for NAV
            
        Returns:
            NAV as Decimal
        """
        query = """
        SELECT NAV
        FROM FUND_NAV
        WHERE AS_OF_DATE = %(as_of_date)s
        """
        
        cursor = self.connection.cursor()
        cursor.execute(query, {"as_of_date": as_of_date})
        
        result = cursor.fetchone()
        cursor.close()
        
        if result:
            return Decimal(str(result[0]))
        else:
            raise ValueError(f"No NAV found for {as_of_date}")
    
    def close(self):
        """Close the connection."""
        if self._connection:
            self._connection.close()
            self._connection = None
            logger.info("Snowflake connection closed")


class SnowflakeAdapter:
    """
    Adapter that wraps SnowflakeConnector to match our ClientSystemAdapter interface.
    """
    
    def __init__(self, config: Optional[SnowflakeConfig] = None):
        self.connector = SnowflakeConnector(config)
    
    def get_snapshot(self, as_of_date: date):
        """Get complete data snapshot from Snowflake."""
        from .client_adapter import Position, ControlResult, DataSnapshot
        
        # Get data from Snowflake
        positions_data = self.connector.get_positions(as_of_date)
        controls_data = self.connector.get_control_results(as_of_date)
        nav = self.connector.get_nav(as_of_date)
        
        # Convert to our data classes
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
            snapshot_id=f"SNOW-{as_of_date.strftime('%Y%m%d')}",
            as_of_date=as_of_date,
            source_system="Snowflake",
            positions=positions,
            control_results=control_results,
            nav=nav,
        )
    
    def close(self):
        """Close the Snowflake connection."""
        self.connector.close()


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def get_snowflake_adapter() -> SnowflakeAdapter:
    """
    Get a Snowflake adapter configured from environment.
    
    Required environment variables:
        SNOWFLAKE_ACCOUNT
        SNOWFLAKE_USER
        SNOWFLAKE_PASSWORD
    
    Optional:
        SNOWFLAKE_WAREHOUSE (default: COMPUTE_WH)
        SNOWFLAKE_DATABASE (default: COMPLIANCE)
        SNOWFLAKE_SCHEMA (default: PUBLIC)
        SNOWFLAKE_ROLE
    """
    return SnowflakeAdapter()
