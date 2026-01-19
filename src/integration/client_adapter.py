"""
Client Data Adapter - Connect to Existing Hedge Fund Systems

This module provides adapters for common hedge fund systems.
We TRUST their data (it's already reconciled/audited) and just read it.

Supported Systems:
- Bloomberg AIM/PORT
- Eze EMS/OMS
- Advent Geneva/APX
- SS&C Eze
- Generic CSV/API

Design Philosophy:
- Read-only access to their systems
- No re-validation (their data is already clean)
- Simple logging for audit trail
- Focus on getting data INTO the RAG pipeline
"""

from __future__ import annotations

import hashlib
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone, date
from typing import Optional, Dict, Any, List, Protocol
from decimal import Decimal
import csv
from pathlib import Path

logger = logging.getLogger(__name__)


# =============================================================================
# DATA MODELS - Simple, Clean
# =============================================================================

@dataclass
class Position:
    """Position from client's system. Already validated by their PMS."""
    security_id: str
    ticker: Optional[str]
    security_name: str
    quantity: Decimal
    market_value: Decimal
    currency: str = "USD"
    sector: Optional[str] = None
    issuer: Optional[str] = None
    asset_class: Optional[str] = None
    
    # Optional fields their system may provide
    isin: Optional[str] = None
    cusip: Optional[str] = None
    price: Optional[Decimal] = None


@dataclass
class ControlResult:
    """Control result from client's existing compliance system."""
    control_id: str
    control_name: str
    control_type: str  # concentration, liquidity, exposure, etc.
    
    # The calculation (already done by their system)
    calculated_value: Decimal
    threshold: Decimal
    threshold_operator: str  # gt, gte, lt, lte
    
    # Result
    status: str  # pass, fail, warning
    breach_amount: Optional[Decimal] = None
    
    # Context
    as_of_date: date = field(default_factory=date.today)
    details: Optional[Dict[str, Any]] = None


@dataclass
class DataSnapshot:
    """A point-in-time snapshot of data from client system."""
    snapshot_id: str
    as_of_date: date
    source_system: str
    
    positions: List[Position] = field(default_factory=list)
    control_results: List[ControlResult] = field(default_factory=list)
    nav: Optional[Decimal] = None
    
    # Audit
    extracted_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def data_hash(self) -> str:
        """Hash for audit trail."""
        content = json.dumps({
            "snapshot_id": self.snapshot_id,
            "as_of_date": self.as_of_date.isoformat(),
            "position_count": len(self.positions),
            "control_count": len(self.control_results),
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]


# =============================================================================
# ADAPTER INTERFACE
# =============================================================================

class ClientSystemAdapter(ABC):
    """
    Abstract adapter for client hedge fund systems.
    
    Each fund will have a different system - we just need to read from it.
    """
    
    @abstractmethod
    def get_positions(self, as_of_date: date) -> List[Position]:
        """Get positions as of a specific date."""
        pass
    
    @abstractmethod
    def get_control_results(self, as_of_date: date) -> List[ControlResult]:
        """Get control results from their compliance system."""
        pass
    
    @abstractmethod
    def get_nav(self, as_of_date: date) -> Decimal:
        """Get NAV as of date."""
        pass
    
    def get_snapshot(self, as_of_date: date) -> DataSnapshot:
        """Get complete snapshot of all relevant data."""
        snapshot = DataSnapshot(
            snapshot_id=f"SNAP-{as_of_date.strftime('%Y%m%d')}",
            as_of_date=as_of_date,
            source_system=self.__class__.__name__,
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


# =============================================================================
# CONCRETE ADAPTERS
# =============================================================================

class CSVAdapter(ClientSystemAdapter):
    """
    Simple CSV adapter for testing or funds using Excel/CSV exports.
    
    Expected files:
    - positions_{date}.csv
    - controls_{date}.csv
    """
    
    def __init__(self, data_dir: Path, nav: Decimal = Decimal("1000000000")):
        self.data_dir = Path(data_dir)
        self._nav = nav
    
    def get_positions(self, as_of_date: date) -> List[Position]:
        """Load positions from CSV."""
        filename = self.data_dir / f"positions_{as_of_date.strftime('%Y%m%d')}.csv"
        
        if not filename.exists():
            logger.warning(f"Position file not found: {filename}")
            return []
        
        positions = []
        with open(filename, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                positions.append(Position(
                    security_id=row.get('security_id', ''),
                    ticker=row.get('ticker'),
                    security_name=row.get('security_name', row.get('name', '')),
                    quantity=Decimal(row.get('quantity', '0')),
                    market_value=Decimal(row.get('market_value', '0')),
                    currency=row.get('currency', 'USD'),
                    sector=row.get('sector'),
                    issuer=row.get('issuer'),
                    asset_class=row.get('asset_class'),
                ))
        
        return positions
    
    def get_control_results(self, as_of_date: date) -> List[ControlResult]:
        """Load control results from CSV."""
        filename = self.data_dir / f"controls_{as_of_date.strftime('%Y%m%d')}.csv"
        
        if not filename.exists():
            logger.warning(f"Controls file not found: {filename}")
            return []
        
        results = []
        with open(filename, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                results.append(ControlResult(
                    control_id=row.get('control_id', ''),
                    control_name=row.get('control_name', ''),
                    control_type=row.get('control_type', 'other'),
                    calculated_value=Decimal(row.get('calculated_value', '0')),
                    threshold=Decimal(row.get('threshold', '0')),
                    threshold_operator=row.get('threshold_operator', 'lte'),
                    status=row.get('status', 'pass'),
                    as_of_date=as_of_date,
                ))
        
        return results
    
    def get_nav(self, as_of_date: date) -> Decimal:
        return self._nav


class MockAdapter(ClientSystemAdapter):
    """
    Mock adapter for demos and testing.
    
    Generates realistic sample data.
    """
    
    def __init__(self, nav: Decimal = Decimal("2000000000")):  # $2B fund
        self._nav = nav
    
    def get_positions(self, as_of_date: date) -> List[Position]:
        """Generate mock positions."""
        import random
        
        securities = [
            ("AAPL", "Apple Inc", "Technology", "Apple Inc"),
            ("MSFT", "Microsoft Corp", "Technology", "Microsoft Corp"),
            ("GOOGL", "Alphabet Inc", "Technology", "Alphabet Inc"),
            ("JPM", "JPMorgan Chase", "Financials", "JPMorgan Chase"),
            ("JNJ", "Johnson & Johnson", "Healthcare", "Johnson & Johnson"),
            ("XOM", "Exxon Mobil", "Energy", "Exxon Mobil"),
            ("PG", "Procter & Gamble", "Consumer Staples", "Procter & Gamble"),
            ("NVDA", "NVIDIA Corp", "Technology", "NVIDIA Corp"),
            ("V", "Visa Inc", "Financials", "Visa Inc"),
            ("UNH", "UnitedHealth", "Healthcare", "UnitedHealth Group"),
            ("HD", "Home Depot", "Consumer Discretionary", "Home Depot"),
            ("MA", "Mastercard", "Financials", "Mastercard"),
            ("DIS", "Walt Disney", "Communication Services", "Walt Disney"),
            ("ADBE", "Adobe Inc", "Technology", "Adobe Inc"),
            ("CRM", "Salesforce", "Technology", "Salesforce"),
        ]
        
        positions = []
        for i, (ticker, name, sector, issuer) in enumerate(securities):
            # Random position size between $50M and $300M
            mv = Decimal(str(random.randint(50_000_000, 300_000_000)))
            price = Decimal(str(random.uniform(100, 800)))
            qty = mv / price
            
            positions.append(Position(
                security_id=f"SEC-{i+1:04d}",
                ticker=ticker,
                security_name=name,
                quantity=qty.quantize(Decimal('0.01')),
                market_value=mv,
                currency="USD",
                sector=sector,
                issuer=issuer,
                asset_class="equity",
                price=price.quantize(Decimal('0.01')),
            ))
        
        return positions
    
    def get_control_results(self, as_of_date: date) -> List[ControlResult]:
        """Generate mock control results (some passing, some warning, one breach)."""
        
        results = [
            ControlResult(
                control_id="CONC_ISSUER_001",
                control_name="Single Issuer Concentration",
                control_type="concentration",
                calculated_value=Decimal("8.5"),  # 8.5% - passing
                threshold=Decimal("10.0"),
                threshold_operator="lte",
                status="pass",
                as_of_date=as_of_date,
            ),
            ControlResult(
                control_id="CONC_SECTOR_001",
                control_name="Sector Concentration - Technology",
                control_type="concentration",
                calculated_value=Decimal("28.0"),  # 28% - warning at 25%
                threshold=Decimal("30.0"),
                threshold_operator="lte",
                status="warning",
                as_of_date=as_of_date,
                details={"sector": "Technology", "warning_threshold": "25%"},
            ),
            ControlResult(
                control_id="EXP_GROSS_001",
                control_name="Gross Exposure",
                control_type="exposure",
                calculated_value=Decimal("145.0"),  # 145% - passing
                threshold=Decimal("200.0"),
                threshold_operator="lte",
                status="pass",
                as_of_date=as_of_date,
            ),
            ControlResult(
                control_id="EXP_NET_001",
                control_name="Net Exposure",
                control_type="exposure",
                calculated_value=Decimal("72.0"),  # 72% - passing
                threshold=Decimal("100.0"),
                threshold_operator="lte",
                status="pass",
                as_of_date=as_of_date,
            ),
            ControlResult(
                control_id="LIQ_T1_001",
                control_name="T+1 Liquidity",
                control_type="liquidity",
                calculated_value=Decimal("18.0"),  # 18% - passing (min 10%)
                threshold=Decimal("10.0"),
                threshold_operator="gte",
                status="pass",
                as_of_date=as_of_date,
            ),
            ControlResult(
                control_id="LIQ_T7_001",
                control_name="T+7 Liquidity",
                control_type="liquidity",
                calculated_value=Decimal("35.0"),  # 35% - WARNING (min 40%)
                threshold=Decimal("40.0"),
                threshold_operator="gte",
                status="warning",
                breach_amount=Decimal("5.0"),
                as_of_date=as_of_date,
                details={"message": "5% below minimum liquidity threshold"},
            ),
            ControlResult(
                control_id="CASH_MIN_001",
                control_name="Minimum Cash Buffer",
                control_type="liquidity",
                calculated_value=Decimal("3.2"),  # 3.2% - passing (min 2%)
                threshold=Decimal("2.0"),
                threshold_operator="gte",
                status="pass",
                as_of_date=as_of_date,
            ),
        ]
        
        return results
    
    def get_nav(self, as_of_date: date) -> Decimal:
        return self._nav


class DatabaseAdapter(ClientSystemAdapter):
    """
    Adapter for direct database connection to client's system.
    
    Configure with client's read-only connection string.
    """
    
    def __init__(
        self, 
        connection_string: str,
        positions_query: str,
        controls_query: str,
        nav_query: str,
    ):
        self.connection_string = connection_string
        self.positions_query = positions_query
        self.controls_query = controls_query
        self.nav_query = nav_query
        self._connection = None
    
    def connect(self):
        """Establish database connection."""
        try:
            import psycopg2
            self._connection = psycopg2.connect(self.connection_string)
            logger.info("Connected to client database")
        except ImportError:
            logger.error("psycopg2 not installed")
            raise
    
    def get_positions(self, as_of_date: date) -> List[Position]:
        """Execute positions query."""
        if not self._connection:
            self.connect()
        
        cursor = self._connection.cursor()
        cursor.execute(self.positions_query, {"as_of_date": as_of_date})
        
        positions = []
        columns = [desc[0] for desc in cursor.description]
        
        for row in cursor.fetchall():
            data = dict(zip(columns, row))
            positions.append(Position(
                security_id=str(data.get('security_id', '')),
                ticker=data.get('ticker'),
                security_name=str(data.get('security_name', '')),
                quantity=Decimal(str(data.get('quantity', 0))),
                market_value=Decimal(str(data.get('market_value', 0))),
                currency=data.get('currency', 'USD'),
                sector=data.get('sector'),
                issuer=data.get('issuer'),
            ))
        
        cursor.close()
        return positions
    
    def get_control_results(self, as_of_date: date) -> List[ControlResult]:
        """Execute controls query."""
        if not self._connection:
            self.connect()
        
        cursor = self._connection.cursor()
        cursor.execute(self.controls_query, {"as_of_date": as_of_date})
        
        results = []
        columns = [desc[0] for desc in cursor.description]
        
        for row in cursor.fetchall():
            data = dict(zip(columns, row))
            results.append(ControlResult(
                control_id=str(data.get('control_id', '')),
                control_name=str(data.get('control_name', '')),
                control_type=str(data.get('control_type', 'other')),
                calculated_value=Decimal(str(data.get('calculated_value', 0))),
                threshold=Decimal(str(data.get('threshold', 0))),
                threshold_operator=str(data.get('threshold_operator', 'lte')),
                status=str(data.get('status', 'pass')),
                as_of_date=as_of_date,
            ))
        
        cursor.close()
        return results
    
    def get_nav(self, as_of_date: date) -> Decimal:
        """Execute NAV query."""
        if not self._connection:
            self.connect()
        
        cursor = self._connection.cursor()
        cursor.execute(self.nav_query, {"as_of_date": as_of_date})
        result = cursor.fetchone()
        cursor.close()
        
        return Decimal(str(result[0])) if result else Decimal("0")


# =============================================================================
# ADAPTER FACTORY
# =============================================================================

def get_adapter(
    adapter_type: str = "mock",
    **kwargs
) -> ClientSystemAdapter:
    """
    Factory function to get appropriate adapter.
    
    Args:
        adapter_type: "mock", "csv", "database"
        **kwargs: Adapter-specific configuration
    
    Returns:
        Configured adapter instance
    """
    if adapter_type == "mock":
        return MockAdapter(nav=kwargs.get("nav", Decimal("2000000000")))
    
    elif adapter_type == "csv":
        return CSVAdapter(
            data_dir=Path(kwargs.get("data_dir", "./data")),
            nav=kwargs.get("nav", Decimal("2000000000")),
        )
    
    elif adapter_type == "database":
        return DatabaseAdapter(
            connection_string=kwargs["connection_string"],
            positions_query=kwargs["positions_query"],
            controls_query=kwargs["controls_query"],
            nav_query=kwargs["nav_query"],
        )
    
    else:
        raise ValueError(f"Unknown adapter type: {adapter_type}")
