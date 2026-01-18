"""
Control Runner - Deterministic Compliance Control Execution Engine

This module executes compliance controls against hedge fund data.
All calculations are performed in SQL/Python - no LLM involvement.

Key responsibilities:
1. Execute SQL queries against Snowflake snapshots
2. Evaluate results against thresholds
3. Record results to Postgres evidence store
4. Create exceptions for failures

SEC Examination Note: 
- All queries are logged with their hash
- All results are traceable to a specific data snapshot
- Execution is deterministic and reproducible
"""

from __future__ import annotations

import hashlib
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone, date
from typing import Optional, Dict, Any, List
from enum import Enum

from .controls import (
    ControlDefinition,
    ControlResultStatus,
    ControlCategory,
    get_active_controls,
    get_controls_config_hash,
)

logger = logging.getLogger(__name__)


class RunType(Enum):
    """Types of control runs."""
    SCHEDULED = "scheduled"
    AD_HOC = "ad-hoc"
    REMEDIATION = "remediation"
    BACKFILL = "backfill"


class RunStatus(Enum):
    """Status of a control run."""
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ExceptionSeverity(Enum):
    """Severity levels for exceptions."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class ControlRunContext:
    """
    Context for a control run execution.
    
    This object is immutable once created and provides
    all the information needed to execute and audit controls.
    """
    run_id: str
    run_code: str
    run_type: RunType
    run_date: date
    
    # Data lineage - CRITICAL for reproducibility
    snowflake_snapshot_id: str
    snowflake_snapshot_ts: datetime
    
    # Fund context (if fund-specific run)
    fund_ids: List[str]
    
    # Execution metadata
    executor_service: str
    executor_version: str
    config_hash: str
    
    # Timing
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    ended_at: Optional[datetime] = None
    
    @classmethod
    def create_new(
        cls,
        run_type: RunType,
        run_date: date,
        snowflake_snapshot_id: str,
        snowflake_snapshot_ts: datetime,
        fund_ids: List[str],
        executor_version: str = "1.0.0",
    ) -> "ControlRunContext":
        """Create a new control run context."""
        run_id = str(uuid.uuid4())
        run_code = f"{run_date.isoformat()}-{run_type.value.upper()}-{run_id[:8]}"
        
        return cls(
            run_id=run_id,
            run_code=run_code,
            run_type=run_type,
            run_date=run_date,
            snowflake_snapshot_id=snowflake_snapshot_id,
            snowflake_snapshot_ts=snowflake_snapshot_ts,
            fund_ids=fund_ids,
            executor_service="compliance-control-runner",
            executor_version=executor_version,
            config_hash=get_controls_config_hash(),
        )


@dataclass
class ControlExecutionResult:
    """
    Result of executing a single control.
    
    This is the output of deterministic calculation,
    before any LLM involvement.
    """
    result_id: str
    run_id: str
    control_code: str
    fund_id: Optional[str]
    
    # Deterministic outputs
    calculated_value: Optional[float]
    threshold_value: float
    threshold_operator: str
    result_status: ControlResultStatus
    
    # Breach details (if applicable)
    breach_amount: Optional[float] = None
    breach_percentage: Optional[float] = None
    breach_entity: Optional[str] = None  # e.g., issuer name, counterparty
    
    # Evidence
    evidence_query_hash: str = ""
    evidence_row_count: int = 0
    evidence_sample: Optional[Dict[str, Any]] = None
    
    # Computation details
    computation_sql: str = ""
    computation_duration_ms: int = 0
    
    # Error details (if status is ERROR)
    error_message: Optional[str] = None
    
    # Timestamp
    executed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_citation_string(self) -> str:
        """Generate citation string for narrative references."""
        return f"[ControlResult: {self.control_code} | Status: {self.result_status.value} | Value: {self.calculated_value}]"


@dataclass
class Exception:
    """
    Compliance exception raised from a control failure.
    
    Exceptions require human review and approval.
    """
    exception_id: str
    exception_code: str
    result_id: str
    run_id: str
    control_code: str
    fund_id: Optional[str]
    
    severity: ExceptionSeverity
    title: str
    description: str
    
    # Quantitative details
    breach_value: Optional[float] = None
    threshold_value: Optional[float] = None
    breach_amount: Optional[float] = None
    
    # Workflow
    status: str = "open"
    assigned_to: Optional[str] = None
    due_date: Optional[date] = None
    
    opened_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @classmethod
    def from_control_result(
        cls,
        result: ControlExecutionResult,
        control: ControlDefinition,
        run_context: ControlRunContext,
    ) -> "Exception":
        """Create an exception from a failed control result."""
        
        # Determine severity based on breach magnitude
        severity = cls._determine_severity(result, control)
        
        # Calculate due date based on severity
        due_days = {
            ExceptionSeverity.CRITICAL: 1,
            ExceptionSeverity.HIGH: 3,
            ExceptionSeverity.MEDIUM: 7,
            ExceptionSeverity.LOW: 14,
        }
        
        exception_id = str(uuid.uuid4())
        year = datetime.now().year
        
        # Generate deterministic exception code
        exception_code = f"EXC-{year}-{exception_id[:8].upper()}"
        
        # Build description with facts only (no LLM interpretation)
        description = (
            f"Control {control.control_code} ({control.control_name}) failed.\n"
            f"Calculated value: {result.calculated_value}\n"
            f"Threshold: {result.threshold_operator} {result.threshold_value}\n"
            f"Breach amount: {result.breach_amount}\n"
            f"Data snapshot: {run_context.snowflake_snapshot_id}\n"
        )
        if result.breach_entity:
            description += f"Breach entity: {result.breach_entity}\n"
        
        return cls(
            exception_id=exception_id,
            exception_code=exception_code,
            result_id=result.result_id,
            run_id=run_context.run_id,
            control_code=control.control_code,
            fund_id=result.fund_id,
            severity=severity,
            title=f"{control.control_name} - Limit Breach",
            description=description,
            breach_value=result.calculated_value,
            threshold_value=result.threshold_value,
            breach_amount=result.breach_amount,
            due_date=date.today() + __import__('datetime').timedelta(days=due_days[severity]),
        )
    
    @staticmethod
    def _determine_severity(
        result: ControlExecutionResult,
        control: ControlDefinition,
    ) -> ExceptionSeverity:
        """
        Determine severity based on breach magnitude.
        
        This is deterministic logic, not LLM-based.
        """
        if result.calculated_value is None:
            return ExceptionSeverity.HIGH
        
        # Calculate breach as percentage of threshold
        if result.breach_amount and control.threshold_value:
            breach_pct = abs(result.breach_amount / control.threshold_value * 100)
            
            if breach_pct > 50:
                return ExceptionSeverity.CRITICAL
            elif breach_pct > 25:
                return ExceptionSeverity.HIGH
            elif breach_pct > 10:
                return ExceptionSeverity.MEDIUM
            else:
                return ExceptionSeverity.LOW
        
        return ExceptionSeverity.MEDIUM


class ControlRunner:
    """
    Main control execution engine.
    
    This class orchestrates:
    1. Retrieving control definitions
    2. Executing SQL queries against Snowflake
    3. Evaluating results against thresholds
    4. Recording results to Postgres
    5. Creating exceptions for failures
    
    NO LLM CALLS happen in this class.
    """
    
    def __init__(
        self,
        snowflake_connection: Any,  # snowflake.connector.Connection
        postgres_connection: Any,   # psycopg2/asyncpg connection
        settings: Any,              # Settings object
    ):
        self.snowflake = snowflake_connection
        self.postgres = postgres_connection
        self.settings = settings
        self.logger = logging.getLogger(f"{__name__}.ControlRunner")
    
    def run_daily_controls(
        self,
        run_date: date,
        fund_ids: List[str],
        snapshot_id: Optional[str] = None,
    ) -> ControlRunContext:
        """
        Execute all daily controls for specified funds.
        
        Returns the run context with all results recorded to database.
        """
        self.logger.info(f"Starting daily control run for {run_date}")
        
        # Get or verify snapshot
        if snapshot_id is None:
            snapshot_id, snapshot_ts = self._get_latest_snapshot(run_date)
        else:
            snapshot_ts = self._verify_snapshot(snapshot_id)
        
        # Create run context
        context = ControlRunContext.create_new(
            run_type=RunType.SCHEDULED,
            run_date=run_date,
            snowflake_snapshot_id=snapshot_id,
            snowflake_snapshot_ts=snapshot_ts,
            fund_ids=fund_ids,
            executor_version=self.settings.version if hasattr(self.settings, 'version') else "1.0.0",
        )
        
        # Record run start
        self._record_run_start(context)
        
        try:
            # Get active controls
            controls = get_active_controls()
            self.logger.info(f"Executing {len(controls)} controls")
            
            results: List[ControlExecutionResult] = []
            exceptions: List[Exception] = []
            
            # Execute each control
            for control in controls:
                if control.fund_specific:
                    # Run for each fund
                    for fund_id in fund_ids:
                        result = self._execute_control(control, context, fund_id)
                        results.append(result)
                        
                        if result.result_status == ControlResultStatus.FAIL:
                            exc = Exception.from_control_result(result, control, context)
                            exceptions.append(exc)
                else:
                    # Run once (fund-agnostic)
                    result = self._execute_control(control, context, None)
                    results.append(result)
                    
                    if result.result_status == ControlResultStatus.FAIL:
                        exc = Exception.from_control_result(result, control, context)
                        exceptions.append(exc)
            
            # Record all results
            self._record_results(results)
            self._record_exceptions(exceptions)
            
            # Update run summary
            context = self._finalize_run(context, results)
            
            self.logger.info(
                f"Control run completed: {len(results)} results, "
                f"{len([r for r in results if r.result_status == ControlResultStatus.PASS])} passed, "
                f"{len([r for r in results if r.result_status == ControlResultStatus.FAIL])} failed, "
                f"{len(exceptions)} exceptions created"
            )
            
            return context
            
        except Exception as e:
            self.logger.error(f"Control run failed: {e}")
            self._record_run_failure(context, str(e))
            raise
    
    def _execute_control(
        self,
        control: ControlDefinition,
        context: ControlRunContext,
        fund_id: Optional[str],
    ) -> ControlExecutionResult:
        """
        Execute a single control.
        
        This method:
        1. Runs the SQL query
        2. Extracts the calculated value
        3. Compares against threshold
        4. Returns the result
        
        All operations are deterministic.
        """
        result_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            # Prepare query parameters
            params = {
                "snapshot_id": context.snowflake_snapshot_id,
                "fund_id": fund_id,
            }
            
            # Execute query
            query_result = self._execute_snowflake_query(
                control.computation_sql,
                params,
            )
            
            # Extract calculated value
            calculated_value = None
            evidence_sample = None
            row_count = 0
            breach_entity = None
            
            if query_result and len(query_result) > 0:
                row = query_result[0]
                if 'calculated_value' in row:
                    calculated_value = float(row['calculated_value']) if row['calculated_value'] is not None else None
                if 'breach_entity' in row:
                    breach_entity = row['breach_entity']
                
                row_count = len(query_result)
                evidence_sample = query_result[:10]  # First 10 rows as sample
            
            # Calculate duration
            duration_ms = int((time.time() - start_time) * 1000)
            
            # Evaluate threshold (deterministic comparison)
            if calculated_value is not None:
                result_status = control.evaluate_threshold(calculated_value)
                breach_amount = control.get_breach_amount(calculated_value)
                breach_percentage = None
                if breach_amount and control.threshold_value:
                    breach_percentage = (breach_amount / control.threshold_value) * 100
            else:
                result_status = ControlResultStatus.SKIP
                breach_amount = None
                breach_percentage = None
            
            return ControlExecutionResult(
                result_id=result_id,
                run_id=context.run_id,
                control_code=control.control_code,
                fund_id=fund_id,
                calculated_value=calculated_value,
                threshold_value=control.threshold_value,
                threshold_operator=control.threshold_operator.value,
                result_status=result_status,
                breach_amount=breach_amount,
                breach_percentage=breach_percentage,
                breach_entity=breach_entity,
                evidence_query_hash=control.query_hash,
                evidence_row_count=row_count,
                evidence_sample=evidence_sample,
                computation_sql=control.computation_sql,
                computation_duration_ms=duration_ms,
            )
            
        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            self.logger.error(f"Error executing control {control.control_code}: {e}")
            
            return ControlExecutionResult(
                result_id=result_id,
                run_id=context.run_id,
                control_code=control.control_code,
                fund_id=fund_id,
                calculated_value=None,
                threshold_value=control.threshold_value,
                threshold_operator=control.threshold_operator.value,
                result_status=ControlResultStatus.ERROR,
                error_message=str(e),
                evidence_query_hash=control.query_hash,
                computation_sql=control.computation_sql,
                computation_duration_ms=duration_ms,
            )
    
    def _execute_snowflake_query(
        self,
        sql: str,
        params: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Execute a query against Snowflake and return results.
        
        Uses parameterized queries to prevent SQL injection.
        """
        # This is a placeholder - actual implementation would use
        # snowflake.connector with proper parameter binding
        cursor = self.snowflake.cursor()
        try:
            # Convert named parameters to Snowflake format
            # Snowflake uses %(name)s for named parameters
            formatted_sql = sql
            for key, value in params.items():
                formatted_sql = formatted_sql.replace(f":{key}", f"%({key})s")
            
            cursor.execute(formatted_sql, params)
            columns = [desc[0].lower() for desc in cursor.description]
            rows = cursor.fetchall()
            
            return [dict(zip(columns, row)) for row in rows]
        finally:
            cursor.close()
    
    def _get_latest_snapshot(self, run_date: date) -> tuple[str, datetime]:
        """Get the latest valid snapshot for a given date."""
        query = """
            SELECT snapshot_id, snapshot_timestamp
            FROM compliance.v_data_snapshots
            WHERE snapshot_date = %(run_date)s
              AND validation_status = 'valid'
            ORDER BY snapshot_timestamp DESC
            LIMIT 1
        """
        result = self._execute_snowflake_query(query, {"run_date": run_date})
        
        if not result:
            raise ValueError(f"No valid snapshot found for date {run_date}")
        
        return result[0]['snapshot_id'], result[0]['snapshot_timestamp']
    
    def _verify_snapshot(self, snapshot_id: str) -> datetime:
        """Verify a snapshot exists and return its timestamp."""
        query = """
            SELECT snapshot_timestamp
            FROM compliance.v_data_snapshots
            WHERE snapshot_id = %(snapshot_id)s
              AND validation_status = 'valid'
        """
        result = self._execute_snowflake_query(query, {"snapshot_id": snapshot_id})
        
        if not result:
            raise ValueError(f"Snapshot {snapshot_id} not found or invalid")
        
        return result[0]['snapshot_timestamp']
    
    def _record_run_start(self, context: ControlRunContext) -> None:
        """Record the start of a control run to Postgres."""
        query = """
            INSERT INTO control_runs (
                run_id, run_code, run_type, run_date,
                run_timestamp_start, snowflake_snapshot_id, snowflake_snapshot_ts,
                executor_service, executor_version, config_hash, status
            ) VALUES (
                %(run_id)s, %(run_code)s, %(run_type)s, %(run_date)s,
                %(started_at)s, %(snapshot_id)s, %(snapshot_ts)s,
                %(executor)s, %(version)s, %(config_hash)s, 'running'
            )
        """
        self._execute_postgres_query(query, {
            "run_id": context.run_id,
            "run_code": context.run_code,
            "run_type": context.run_type.value,
            "run_date": context.run_date,
            "started_at": context.started_at,
            "snapshot_id": context.snowflake_snapshot_id,
            "snapshot_ts": context.snowflake_snapshot_ts,
            "executor": context.executor_service,
            "version": context.executor_version,
            "config_hash": context.config_hash,
        })
    
    def _record_results(self, results: List[ControlExecutionResult]) -> None:
        """Record control results to Postgres."""
        import json
        
        for result in results:
            query = """
                INSERT INTO control_results (
                    result_id, run_id, control_id,
                    calculated_value, threshold_value, threshold_operator,
                    result_status, breach_amount, breach_percentage,
                    evidence_query_hash, evidence_row_count, evidence_sample_json,
                    computation_sql, computation_duration_ms, created_at
                ) VALUES (
                    %(result_id)s, %(run_id)s,
                    (SELECT control_id FROM control_definitions WHERE control_code = %(control_code)s),
                    %(calculated_value)s, %(threshold_value)s, %(threshold_operator)s,
                    %(result_status)s, %(breach_amount)s, %(breach_percentage)s,
                    %(query_hash)s, %(row_count)s, %(evidence_sample)s,
                    %(computation_sql)s, %(duration_ms)s, %(executed_at)s
                )
            """
            self._execute_postgres_query(query, {
                "result_id": result.result_id,
                "run_id": result.run_id,
                "control_code": result.control_code,
                "calculated_value": result.calculated_value,
                "threshold_value": result.threshold_value,
                "threshold_operator": result.threshold_operator,
                "result_status": result.result_status.value,
                "breach_amount": result.breach_amount,
                "breach_percentage": result.breach_percentage,
                "query_hash": result.evidence_query_hash,
                "row_count": result.evidence_row_count,
                "evidence_sample": json.dumps(result.evidence_sample) if result.evidence_sample else None,
                "computation_sql": result.computation_sql,
                "duration_ms": result.computation_duration_ms,
                "executed_at": result.executed_at,
            })
    
    def _record_exceptions(self, exceptions: List[Exception]) -> None:
        """Record exceptions to Postgres."""
        for exc in exceptions:
            query = """
                INSERT INTO exceptions (
                    exception_id, exception_code, result_id, run_id, control_id,
                    severity, exception_type, title, description,
                    breach_value, threshold_value, breach_amount,
                    status, due_date, opened_at, last_updated_at, last_updated_by
                ) VALUES (
                    %(exception_id)s, %(exception_code)s, %(result_id)s, %(run_id)s,
                    (SELECT control_id FROM control_definitions WHERE control_code = %(control_code)s),
                    %(severity)s, 'breach', %(title)s, %(description)s,
                    %(breach_value)s, %(threshold_value)s, %(breach_amount)s,
                    'open', %(due_date)s, %(opened_at)s, %(opened_at)s, 'system'
                )
            """
            self._execute_postgres_query(query, {
                "exception_id": exc.exception_id,
                "exception_code": exc.exception_code,
                "result_id": exc.result_id,
                "run_id": exc.run_id,
                "control_code": exc.control_code,
                "severity": exc.severity.value,
                "title": exc.title,
                "description": exc.description,
                "breach_value": exc.breach_value,
                "threshold_value": exc.threshold_value,
                "breach_amount": exc.breach_amount,
                "due_date": exc.due_date,
                "opened_at": exc.opened_at,
            })
    
    def _finalize_run(
        self,
        context: ControlRunContext,
        results: List[ControlExecutionResult],
    ) -> ControlRunContext:
        """Update run record with final summary."""
        ended_at = datetime.now(timezone.utc)
        
        total = len(results)
        passed = len([r for r in results if r.result_status == ControlResultStatus.PASS])
        failed = len([r for r in results if r.result_status == ControlResultStatus.FAIL])
        warning = len([r for r in results if r.result_status == ControlResultStatus.WARNING])
        skipped = len([r for r in results if r.result_status == ControlResultStatus.SKIP])
        
        query = """
            UPDATE control_runs SET
                run_timestamp_end = %(ended_at)s,
                total_controls = %(total)s,
                controls_passed = %(passed)s,
                controls_failed = %(failed)s,
                controls_warning = %(warning)s,
                controls_skipped = %(skipped)s,
                status = 'completed'
            WHERE run_id = %(run_id)s
        """
        self._execute_postgres_query(query, {
            "run_id": context.run_id,
            "ended_at": ended_at,
            "total": total,
            "passed": passed,
            "failed": failed,
            "warning": warning,
            "skipped": skipped,
        })
        
        # Return updated context
        context.ended_at = ended_at
        return context
    
    def _record_run_failure(self, context: ControlRunContext, error: str) -> None:
        """Record run failure."""
        query = """
            UPDATE control_runs SET
                run_timestamp_end = %(ended_at)s,
                status = 'failed',
                error_message = %(error)s
            WHERE run_id = %(run_id)s
        """
        self._execute_postgres_query(query, {
            "run_id": context.run_id,
            "ended_at": datetime.now(timezone.utc),
            "error": error[:1000],  # Truncate long errors
        })
    
    def _execute_postgres_query(self, query: str, params: Dict[str, Any]) -> None:
        """Execute a query against Postgres."""
        # This is a placeholder - actual implementation would use
        # psycopg2 or asyncpg with proper parameter binding
        cursor = self.postgres.cursor()
        try:
            cursor.execute(query, params)
            self.postgres.commit()
        finally:
            cursor.close()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_run_summary(
    postgres_connection: Any,
    run_id: str,
) -> Dict[str, Any]:
    """
    Get a summary of a control run.
    
    This is used by the narrative generator and document builder.
    """
    query = """
        SELECT 
            cr.run_id,
            cr.run_code,
            cr.run_date,
            cr.snowflake_snapshot_id,
            cr.total_controls,
            cr.controls_passed,
            cr.controls_failed,
            cr.controls_warning,
            cr.status,
            cr.run_timestamp_start,
            cr.run_timestamp_end,
            COUNT(DISTINCT e.exception_id) AS exception_count
        FROM control_runs cr
        LEFT JOIN exceptions e ON cr.run_id = e.run_id
        WHERE cr.run_id = %(run_id)s
        GROUP BY cr.run_id, cr.run_code, cr.run_date, cr.snowflake_snapshot_id,
                 cr.total_controls, cr.controls_passed, cr.controls_failed,
                 cr.controls_warning, cr.status, cr.run_timestamp_start, cr.run_timestamp_end
    """
    cursor = postgres_connection.cursor()
    try:
        cursor.execute(query, {"run_id": run_id})
        row = cursor.fetchone()
        if row:
            columns = [desc[0] for desc in cursor.description]
            return dict(zip(columns, row))
        return {}
    finally:
        cursor.close()


def get_failed_controls(
    postgres_connection: Any,
    run_id: str,
) -> List[Dict[str, Any]]:
    """Get all failed control results for a run."""
    query = """
        SELECT 
            cr.result_id,
            cd.control_code,
            cd.control_name,
            cd.control_category,
            cr.calculated_value,
            cr.threshold_value,
            cr.threshold_operator,
            cr.breach_amount,
            cr.breach_percentage
        FROM control_results cr
        JOIN control_definitions cd ON cr.control_id = cd.control_id
        WHERE cr.run_id = %(run_id)s
          AND cr.result_status = 'fail'
        ORDER BY cd.control_code
    """
    cursor = postgres_connection.cursor()
    try:
        cursor.execute(query, {"run_id": run_id})
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in rows]
    finally:
        cursor.close()
