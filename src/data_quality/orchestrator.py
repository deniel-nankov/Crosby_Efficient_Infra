"""
Data Integration Orchestrator

Coordinates the complete data pipeline:
1. Extract from Snowflake
2. Validate quality
3. Load to PostgreSQL
4. Index for retrieval

This is the main entry point for daily data syncs.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, date
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import uuid

from .validators import (
    PositionDataValidator,
    ControlDefinitionValidator,
    QualityReport,
    QualitySeverity,
)
from .snowflake_connector import SnowflakeConnector, DataSnapshot, get_connector
from .policy_ingestion import PolicyIngestionPipeline, PolicyStore, PolicyDocument

logger = logging.getLogger(__name__)


@dataclass
class IntegrationRun:
    """Tracks a complete data integration run."""
    run_id: str
    run_type: str  # 'daily', 'adhoc', 'backfill'
    started_at: datetime
    completed_at: Optional[datetime] = None
    
    # Data extracted
    snapshot_id: Optional[str] = None
    position_count: int = 0
    policy_count: int = 0
    control_count: int = 0
    
    # Quality results
    position_quality: Optional[QualityReport] = None
    policies_quality: List[QualityReport] = field(default_factory=list)
    
    # Status
    status: str = "running"  # 'running', 'completed', 'failed', 'rejected'
    error_message: Optional[str] = None
    
    # Audit
    config_hash: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "run_type": self.run_type,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "snapshot_id": self.snapshot_id,
            "position_count": self.position_count,
            "policy_count": self.policy_count,
            "control_count": self.control_count,
            "position_quality_score": self.position_quality.overall_score if self.position_quality else None,
            "status": self.status,
            "error_message": self.error_message,
            "config_hash": self.config_hash,
        }
    
    def to_summary(self) -> str:
        """Human-readable summary."""
        duration = ""
        if self.completed_at:
            secs = (self.completed_at - self.started_at).total_seconds()
            duration = f" ({secs:.1f}s)"
        
        return f"""
Data Integration Run: {self.run_id}
=====================================
Type: {self.run_type}
Status: {self.status}{duration}
Started: {self.started_at.isoformat()}

Data Extracted:
  Snapshot: {self.snapshot_id or 'N/A'}
  Positions: {self.position_count:,}
  Policies: {self.policy_count}
  Controls: {self.control_count}

Quality:
  Position Quality: {f'{self.position_quality.overall_score:.1f}%' if self.position_quality else 'N/A'}

{f'Error: {self.error_message}' if self.error_message else ''}
"""


class DataIntegrationOrchestrator:
    """
    Main orchestrator for data integration.
    
    Coordinates:
    1. Snowflake data extraction
    2. Quality validation
    3. PostgreSQL loading
    4. Policy ingestion
    5. Control definitions sync
    
    All with quality gates and audit logging.
    """
    
    def __init__(
        self,
        postgres_conn=None,
        snowflake_connector: Optional[SnowflakeConnector] = None,
        quality_threshold: float = 95.0,
        use_mock: bool = False,
    ):
        self.postgres_conn = postgres_conn
        self.snowflake = snowflake_connector or get_connector(use_mock=use_mock)
        self.quality_threshold = quality_threshold
        
        # Initialize validators
        self.position_validator = PositionDataValidator()
        self.control_validator = ControlDefinitionValidator()
        self.policy_pipeline = PolicyIngestionPipeline()
        self.policy_store = PolicyStore(postgres_conn)
        
        # Run history
        self._runs: List[IntegrationRun] = []
    
    def run_daily_sync(
        self,
        snapshot_id: Optional[str] = None,
        fund_id: Optional[str] = None,
        policy_dir: Optional[Path] = None,
    ) -> IntegrationRun:
        """
        Execute a complete daily data sync.
        
        Steps:
        1. Get latest snapshot from Snowflake (or use provided)
        2. Extract and validate positions
        3. Load positions to evidence store
        4. Sync policy documents (if directory provided)
        5. Validate and report
        
        Args:
            snapshot_id: Specific snapshot to use (default: latest)
            fund_id: Optional filter to specific fund
            policy_dir: Directory containing policy Markdown files
            
        Returns:
            IntegrationRun with complete status and quality reports
        """
        run = IntegrationRun(
            run_id=str(uuid.uuid4()),
            run_type='daily',
            started_at=datetime.now(timezone.utc),
            config_hash=self._hash_config(),
        )
        self._runs.append(run)
        
        logger.info(f"Starting daily sync: {run.run_id}")
        
        try:
            # Step 1: Connect to Snowflake
            if not self.snowflake.connect():
                logger.warning("Running in mock mode - no Snowflake connection")
            
            # Step 2: Get snapshot
            if not snapshot_id:
                snapshots = self.snowflake.get_available_snapshots(days_back=1)
                if not snapshots:
                    raise ValueError("No valid snapshots available")
                snapshot_id = snapshots[0].snapshot_id
            
            run.snapshot_id = snapshot_id
            logger.info(f"Using snapshot: {snapshot_id}")
            
            # Step 3: Extract positions
            positions, snapshot_meta = self.snowflake.get_positions_for_snapshot(
                snapshot_id=snapshot_id,
                fund_id=fund_id,
            )
            run.position_count = len(positions)
            
            # Step 4: Validate positions (QUALITY GATE)
            logger.info(f"Validating {len(positions)} positions...")
            run.position_quality = self.position_validator.validate(
                positions=positions,
                snapshot_date=snapshot_meta.snapshot_date,
            )
            
            logger.info(f"Position quality score: {run.position_quality.overall_score:.1f}%")
            
            # Quality gate check
            if not run.position_quality.is_acceptable:
                run.status = 'rejected'
                run.error_message = f"Quality gate failed: {run.position_quality.rejection_reason}"
                logger.error(run.error_message)
                run.completed_at = datetime.now(timezone.utc)
                return run
            
            # Step 5: Load to PostgreSQL
            if self.postgres_conn:
                self._load_positions_to_postgres(positions, snapshot_meta)
            
            # Step 6: Sync policies if directory provided
            if policy_dir and policy_dir.exists():
                run.policy_count = self._sync_policies(policy_dir, run)
            
            # Success
            run.status = 'completed'
            run.completed_at = datetime.now(timezone.utc)
            
            logger.info(f"Daily sync completed: {run.run_id}")
            logger.info(run.to_summary())
            
            return run
            
        except Exception as e:
            run.status = 'failed'
            run.error_message = str(e)
            run.completed_at = datetime.now(timezone.utc)
            logger.exception(f"Daily sync failed: {e}")
            return run
        
        finally:
            self.snowflake.disconnect()
    
    def _load_positions_to_postgres(
        self, 
        positions: List[Dict[str, Any]], 
        snapshot: DataSnapshot
    ):
        """Load validated positions to PostgreSQL staging table."""
        cursor = self.postgres_conn.cursor()
        
        try:
            # Create staging record
            staging_id = str(uuid.uuid4())
            cursor.execute("""
                INSERT INTO position_staging (
                    staging_id, snapshot_id, snapshot_date, position_count,
                    loaded_at, status
                ) VALUES (%s, %s, %s, %s, %s, 'loaded')
            """, (
                staging_id,
                snapshot.snapshot_id,
                snapshot.snapshot_date,
                len(positions),
                datetime.now(timezone.utc),
            ))
            
            # Batch insert positions
            for pos in positions:
                cursor.execute("""
                    INSERT INTO positions_current (
                        staging_id, position_id, snapshot_id, fund_id,
                        security_id, ticker, isin, quantity, market_value_usd,
                        currency, asset_class, sector, issuer_id, issuer_name,
                        price, price_date, is_stale_price
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (snapshot_id, position_id) DO UPDATE SET
                        market_value_usd = EXCLUDED.market_value_usd,
                        price = EXCLUDED.price,
                        updated_at = NOW()
                """, (
                    staging_id,
                    pos.get('position_id'),
                    pos.get('snapshot_id'),
                    pos.get('fund_id'),
                    pos.get('security_id'),
                    pos.get('ticker'),
                    pos.get('isin'),
                    pos.get('quantity'),
                    pos.get('market_value_usd'),
                    pos.get('currency'),
                    pos.get('asset_class'),
                    pos.get('sector'),
                    pos.get('issuer_id'),
                    pos.get('issuer_name'),
                    pos.get('price'),
                    pos.get('price_date'),
                    pos.get('is_stale_price', False),
                ))
            
            self.postgres_conn.commit()
            logger.info(f"Loaded {len(positions)} positions to staging {staging_id}")
            
        except Exception as e:
            self.postgres_conn.rollback()
            logger.error(f"Failed to load positions: {e}")
            raise
        finally:
            cursor.close()
    
    def _sync_policies(self, policy_dir: Path, run: IntegrationRun) -> int:
        """Sync policy documents from directory."""
        policy_count = 0
        
        for policy_file in policy_dir.glob('*.md'):
            try:
                logger.info(f"Ingesting policy: {policy_file.name}")
                
                doc = self.policy_pipeline.ingest_file(policy_file)
                
                if doc.quality_report:
                    run.policies_quality.append(doc.quality_report)
                    
                    if not doc.quality_report.is_acceptable:
                        logger.warning(f"Policy {doc.policy_id} failed quality check")
                        continue
                
                self.policy_store.store_policy(doc)
                policy_count += 1
                
            except Exception as e:
                logger.error(f"Failed to ingest policy {policy_file}: {e}")
        
        return policy_count
    
    def seed_control_definitions(
        self, 
        controls: List[Dict[str, Any]]
    ) -> Tuple[int, List[QualityReport]]:
        """
        Seed control definitions to the database.
        
        Validates each control before insertion.
        
        Args:
            controls: List of control definition dictionaries
            
        Returns:
            Tuple of (count inserted, list of quality reports)
        """
        if not self.postgres_conn:
            logger.warning("No PostgreSQL connection - skipping control seeding")
            return 0, []
        
        inserted = 0
        reports = []
        
        cursor = self.postgres_conn.cursor()
        
        try:
            for control in controls:
                # Validate
                report = self.control_validator.validate_control(control)
                reports.append(report)
                
                if not report.is_acceptable:
                    logger.warning(
                        f"Control {control.get('control_code')} failed validation: "
                        f"{[i.message for i in report.critical_issues]}"
                    )
                    continue
                
                # Insert
                control_id = control.get('control_id', str(uuid.uuid4()))
                
                cursor.execute("""
                    INSERT INTO control_definitions (
                        control_id, control_code, control_name, control_category,
                        description, threshold_type, threshold_value, threshold_operator,
                        frequency, regulatory_reference, is_active, effective_date,
                        created_by, updated_by
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                    ON CONFLICT (control_code) DO UPDATE SET
                        control_name = EXCLUDED.control_name,
                        threshold_value = EXCLUDED.threshold_value,
                        updated_at = NOW(),
                        updated_by = EXCLUDED.updated_by
                """, (
                    control_id,
                    control.get('control_code'),
                    control.get('control_name'),
                    control.get('control_category'),
                    control.get('description'),
                    control.get('threshold_type', 'percentage'),
                    control.get('threshold_value'),
                    control.get('threshold_operator'),
                    control.get('frequency', 'daily'),
                    control.get('regulatory_reference'),
                    control.get('is_active', True),
                    control.get('effective_date', date.today()),
                    'data_integration',
                    'data_integration',
                ))
                
                inserted += 1
            
            self.postgres_conn.commit()
            logger.info(f"Seeded {inserted}/{len(controls)} control definitions")
            
            return inserted, reports
            
        except Exception as e:
            self.postgres_conn.rollback()
            logger.error(f"Failed to seed controls: {e}")
            raise
        finally:
            cursor.close()
    
    def _hash_config(self) -> str:
        """Hash current configuration for audit."""
        config = {
            "quality_threshold": self.quality_threshold,
            "snowflake_schema": self.snowflake.config.schema if hasattr(self.snowflake, 'config') else 'mock',
        }
        return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]
    
    def get_run_history(self) -> List[IntegrationRun]:
        """Get history of integration runs."""
        return self._runs.copy()


# =============================================================================
# SAMPLE CONTROL DEFINITIONS
# =============================================================================

SAMPLE_CONTROLS = [
    {
        "control_code": "CONC_ISSUER_001",
        "control_name": "Single Issuer Concentration",
        "control_category": "concentration",
        "description": "Maximum exposure to any single issuer as percentage of NAV",
        "threshold_type": "percentage",
        "threshold_value": 10.0,
        "threshold_operator": "lte",
        "frequency": "daily",
        "regulatory_reference": "Investment Guidelines Section 2.3",
    },
    {
        "control_code": "CONC_SECTOR_001",
        "control_name": "Sector Concentration",
        "control_category": "concentration",
        "description": "Maximum exposure to any single GICS sector as percentage of NAV",
        "threshold_type": "percentage",
        "threshold_value": 30.0,
        "threshold_operator": "lte",
        "frequency": "daily",
        "regulatory_reference": "Investment Guidelines Section 2.1",
    },
    {
        "control_code": "EXP_GROSS_001",
        "control_name": "Gross Exposure Limit",
        "control_category": "exposure",
        "description": "Maximum gross exposure (long + |short|) as percentage of NAV",
        "threshold_type": "percentage",
        "threshold_value": 200.0,
        "threshold_operator": "lte",
        "frequency": "daily",
        "regulatory_reference": "Investment Guidelines Section 1.1",
    },
    {
        "control_code": "EXP_NET_LONG_001",
        "control_name": "Net Long Exposure Limit",
        "control_category": "exposure",
        "description": "Maximum net long exposure as percentage of NAV",
        "threshold_type": "percentage",
        "threshold_value": 100.0,
        "threshold_operator": "lte",
        "frequency": "daily",
        "regulatory_reference": "Investment Guidelines Section 1.2",
    },
    {
        "control_code": "EXP_NET_SHORT_001",
        "control_name": "Net Short Exposure Limit",
        "control_category": "exposure",
        "description": "Maximum net short exposure as percentage of NAV",
        "threshold_type": "percentage",
        "threshold_value": 50.0,
        "threshold_operator": "lte",
        "frequency": "daily",
        "regulatory_reference": "Investment Guidelines Section 1.2",
    },
    {
        "control_code": "LIQ_T1_001",
        "control_name": "T+1 Liquidity Minimum",
        "control_category": "liquidity",
        "description": "Minimum percentage of NAV liquidatable within T+1",
        "threshold_type": "percentage",
        "threshold_value": 10.0,
        "threshold_operator": "gte",
        "frequency": "daily",
        "regulatory_reference": "Investment Guidelines Section 3.1",
    },
    {
        "control_code": "LIQ_T7_001",
        "control_name": "T+7 Liquidity Minimum",
        "control_category": "liquidity",
        "description": "Minimum percentage of NAV liquidatable within T+7",
        "threshold_type": "percentage",
        "threshold_value": 40.0,
        "threshold_operator": "gte",
        "frequency": "daily",
        "regulatory_reference": "Investment Guidelines Section 3.1",
    },
    {
        "control_code": "CASH_MIN_001",
        "control_name": "Minimum Cash Buffer",
        "control_category": "liquidity",
        "description": "Minimum cash as percentage of NAV",
        "threshold_type": "percentage",
        "threshold_value": 2.0,
        "threshold_operator": "gte",
        "frequency": "daily",
        "regulatory_reference": "Investment Guidelines Section 3.3",
    },
    {
        "control_code": "CPT_PRIME_001",
        "control_name": "Single Prime Broker Limit",
        "control_category": "counterparty",
        "description": "Maximum exposure to any single prime broker as percentage of NAV",
        "threshold_type": "percentage",
        "threshold_value": 50.0,
        "threshold_operator": "lte",
        "frequency": "daily",
        "regulatory_reference": "Investment Guidelines Section 4.1",
    },
    {
        "control_code": "POS_SINGLE_LONG_001",
        "control_name": "Single Position Long Limit",
        "control_category": "concentration",
        "description": "Maximum single long position as percentage of NAV",
        "threshold_type": "percentage",
        "threshold_value": 10.0,
        "threshold_operator": "lte",
        "frequency": "daily",
        "regulatory_reference": "Investment Guidelines Section 1.3",
    },
]


def get_sample_controls() -> List[Dict[str, Any]]:
    """Get sample control definitions for seeding."""
    return SAMPLE_CONTROLS.copy()
