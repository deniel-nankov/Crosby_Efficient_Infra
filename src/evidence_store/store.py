"""
Evidence Store - Queryable Audit Trail for Compliance

This module provides the interface to the Postgres-based evidence store.
It handles:
- Structured storage of control results and exceptions
- Query interfaces for retrieving evidence by various dimensions
- Audit trail logging
- Evidence packaging for document generation

SEC Examination Note:
- All evidence queries are logged
- Evidence is immutable once created
- All data includes lineage to source snapshots
"""

from __future__ import annotations

import json
import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, date, timedelta
from typing import Optional, Dict, Any, List, Tuple
from enum import Enum
import uuid

logger = logging.getLogger(__name__)


@dataclass
class EvidenceQuery:
    """
    Represents a query for evidence.
    
    Queries are logged for audit purposes.
    """
    query_id: str
    query_type: str
    parameters: Dict[str, Any]
    executed_at: datetime
    executed_by: str
    result_count: int = 0
    
    @property
    def query_hash(self) -> str:
        """Hash of query for audit trail."""
        query_str = f"{self.query_type}|{json.dumps(self.parameters, sort_keys=True)}"
        return hashlib.sha256(query_str.encode()).hexdigest()


@dataclass
class ControlResultEvidence:
    """
    Evidence package for a single control result.
    
    This is the atomic unit of evidence for compliance reporting.
    """
    result_id: str
    control_code: str
    control_name: str
    control_category: str
    run_id: str
    run_code: str
    run_date: date
    snapshot_id: str
    
    # Result data
    calculated_value: Optional[float]
    threshold_value: float
    threshold_operator: str
    result_status: str
    breach_amount: Optional[float]
    
    # Computation evidence
    computation_sql_hash: str
    evidence_row_count: int
    evidence_sample: Optional[List[Dict[str, Any]]]
    
    # Timestamps
    executed_at: datetime
    
    def to_citation(self) -> str:
        """Generate a citation string for this evidence."""
        return (
            f"[ControlRun: {self.run_code} | "
            f"Control: {self.control_code} | "
            f"Snapshot: {self.snapshot_id}]"
        )
    
    def to_summary(self) -> str:
        """Generate a deterministic summary (no LLM)."""
        status_text = {
            'pass': 'passed',
            'fail': 'failed',
            'warning': 'triggered warning',
            'skip': 'was skipped',
            'error': 'encountered error',
        }
        
        return (
            f"Control {self.control_code} ({self.control_name}) {status_text.get(self.result_status, 'completed')}. "
            f"Calculated value: {self.calculated_value}, "
            f"Threshold: {self.threshold_operator} {self.threshold_value}. "
        )


@dataclass
class ExceptionEvidence:
    """
    Evidence package for an exception.
    """
    exception_id: str
    exception_code: str
    control_code: str
    control_name: str
    run_code: str
    run_date: date
    
    severity: str
    title: str
    description: str
    status: str
    
    breach_value: Optional[float]
    threshold_value: Optional[float]
    breach_amount: Optional[float]
    
    opened_at: datetime
    due_date: Optional[date]
    resolved_at: Optional[datetime]
    resolution_type: Optional[str]
    resolution_notes: Optional[str]
    
    # Workflow
    assigned_to: Optional[str]
    approvals: List[Dict[str, Any]] = field(default_factory=list)
    activities: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_citation(self) -> str:
        """Generate a citation string for this exception."""
        return f"[Exception: {self.exception_code} | Status: {self.status}]"


@dataclass
class DailyComplianceSummary:
    """
    Summary of daily compliance for a specific date.
    
    This is the primary evidence package for the daily compliance pack.
    """
    run_id: str
    run_code: str
    run_date: date
    snapshot_id: str
    
    # Summary counts
    total_controls: int
    controls_passed: int
    controls_failed: int
    controls_warning: int
    pass_rate: float
    
    # Exception counts
    exceptions_opened: int
    exceptions_closed: int
    exceptions_outstanding: int
    critical_exceptions: int
    
    # Execution metadata
    run_start: datetime
    run_end: datetime
    duration_seconds: float
    config_hash: str
    
    # Detailed results by category
    results_by_category: Dict[str, Dict[str, int]] = field(default_factory=dict)
    
    def to_citation(self) -> str:
        """Generate a citation string for this summary."""
        return f"[ControlRun: {self.run_code} | Date: {self.run_date} | Snapshot: {self.snapshot_id}]"


class EvidenceStore:
    """
    Main interface to the evidence store.
    
    Provides methods to:
    - Query control results
    - Query exceptions
    - Package evidence for documents
    - Log all evidence access
    """
    
    def __init__(self, postgres_connection: Any, user_id: str = "system"):
        self.connection = postgres_connection
        self.user_id = user_id
        self.logger = logging.getLogger(f"{__name__}.EvidenceStore")
    
    # =========================================================================
    # CONTROL RESULT QUERIES
    # =========================================================================
    
    def get_control_results_for_run(
        self,
        run_id: str,
        status_filter: Optional[List[str]] = None,
    ) -> List[ControlResultEvidence]:
        """
        Get all control results for a specific run.
        
        Args:
            run_id: The control run ID
            status_filter: Optional list of statuses to filter by
        
        Returns:
            List of ControlResultEvidence objects
        """
        query = """
            SELECT 
                cr.result_id,
                cd.control_code,
                cd.control_name,
                cd.control_category,
                r.run_id,
                r.run_code,
                r.run_date,
                r.snowflake_snapshot_id,
                cr.calculated_value,
                cr.threshold_value,
                cr.threshold_operator,
                cr.result_status,
                cr.breach_amount,
                cr.evidence_query_hash,
                cr.evidence_row_count,
                cr.evidence_sample_json,
                cr.created_at
            FROM control_results cr
            JOIN control_definitions cd ON cr.control_id = cd.control_id
            JOIN control_runs r ON cr.run_id = r.run_id
            WHERE r.run_id = %(run_id)s
        """
        
        params = {"run_id": run_id}
        
        if status_filter:
            query += " AND cr.result_status = ANY(%(statuses)s)"
            params["statuses"] = status_filter
        
        query += " ORDER BY cd.control_category, cd.control_code"
        
        # Log the query
        self._log_evidence_query("get_control_results_for_run", params)
        
        rows = self._execute_query(query, params)
        
        results = []
        for row in rows:
            evidence_sample = None
            if row['evidence_sample_json']:
                evidence_sample = json.loads(row['evidence_sample_json'])
            
            results.append(ControlResultEvidence(
                result_id=row['result_id'],
                control_code=row['control_code'],
                control_name=row['control_name'],
                control_category=row['control_category'],
                run_id=row['run_id'],
                run_code=row['run_code'],
                run_date=row['run_date'],
                snapshot_id=row['snowflake_snapshot_id'],
                calculated_value=row['calculated_value'],
                threshold_value=row['threshold_value'],
                threshold_operator=row['threshold_operator'],
                result_status=row['result_status'],
                breach_amount=row['breach_amount'],
                computation_sql_hash=row['evidence_query_hash'],
                evidence_row_count=row['evidence_row_count'],
                evidence_sample=evidence_sample,
                executed_at=row['created_at'],
            ))
        
        return results
    
    def get_failed_controls_for_run(self, run_id: str) -> List[ControlResultEvidence]:
        """Get only failed controls for a run."""
        return self.get_control_results_for_run(run_id, status_filter=['fail'])
    
    def get_control_result_by_id(self, result_id: str) -> Optional[ControlResultEvidence]:
        """Get a specific control result by ID."""
        results = self.get_control_results_for_run(
            run_id=self._get_run_id_for_result(result_id),
        )
        for result in results:
            if result.result_id == result_id:
                return result
        return None
    
    # =========================================================================
    # EXCEPTION QUERIES
    # =========================================================================
    
    def get_exceptions_for_run(self, run_id: str) -> List[ExceptionEvidence]:
        """Get all exceptions created for a specific run."""
        query = """
            SELECT 
                e.exception_id,
                e.exception_code,
                cd.control_code,
                cd.control_name,
                r.run_code,
                r.run_date,
                e.severity,
                e.title,
                e.description,
                e.status,
                e.breach_value,
                e.threshold_value,
                e.breach_amount,
                e.opened_at,
                e.due_date,
                e.resolved_at,
                e.resolution_type,
                e.resolution_notes,
                s.full_name as assigned_to
            FROM exceptions e
            JOIN control_definitions cd ON e.control_id = cd.control_id
            JOIN control_runs r ON e.run_id = r.run_id
            LEFT JOIN signatories s ON e.assigned_to = s.signatory_id
            WHERE e.run_id = %(run_id)s
            ORDER BY e.severity, e.opened_at
        """
        
        self._log_evidence_query("get_exceptions_for_run", {"run_id": run_id})
        
        rows = self._execute_query(query, {"run_id": run_id})
        
        return [self._row_to_exception_evidence(row) for row in rows]
    
    def get_open_exceptions(
        self,
        as_of_date: Optional[date] = None,
        severity_filter: Optional[List[str]] = None,
    ) -> List[ExceptionEvidence]:
        """Get all currently open exceptions."""
        as_of_date = as_of_date or date.today()
        
        query = """
            SELECT 
                e.exception_id,
                e.exception_code,
                cd.control_code,
                cd.control_name,
                r.run_code,
                r.run_date,
                e.severity,
                e.title,
                e.description,
                e.status,
                e.breach_value,
                e.threshold_value,
                e.breach_amount,
                e.opened_at,
                e.due_date,
                e.resolved_at,
                e.resolution_type,
                e.resolution_notes,
                s.full_name as assigned_to
            FROM exceptions e
            JOIN control_definitions cd ON e.control_id = cd.control_id
            JOIN control_runs r ON e.run_id = r.run_id
            LEFT JOIN signatories s ON e.assigned_to = s.signatory_id
            WHERE e.status NOT IN ('closed', 'approved')
              AND e.opened_at <= %(as_of_date)s
        """
        
        params = {"as_of_date": as_of_date}
        
        if severity_filter:
            query += " AND e.severity = ANY(%(severities)s)"
            params["severities"] = severity_filter
        
        query += " ORDER BY e.severity, e.due_date"
        
        self._log_evidence_query("get_open_exceptions", params)
        
        rows = self._execute_query(query, params)
        
        return [self._row_to_exception_evidence(row) for row in rows]
    
    def get_overdue_exceptions(self, as_of_date: Optional[date] = None) -> List[ExceptionEvidence]:
        """Get exceptions that are past their due date."""
        as_of_date = as_of_date or date.today()
        
        query = """
            SELECT 
                e.exception_id,
                e.exception_code,
                cd.control_code,
                cd.control_name,
                r.run_code,
                r.run_date,
                e.severity,
                e.title,
                e.description,
                e.status,
                e.breach_value,
                e.threshold_value,
                e.breach_amount,
                e.opened_at,
                e.due_date,
                e.resolved_at,
                e.resolution_type,
                e.resolution_notes,
                s.full_name as assigned_to
            FROM exceptions e
            JOIN control_definitions cd ON e.control_id = cd.control_id
            JOIN control_runs r ON e.run_id = r.run_id
            LEFT JOIN signatories s ON e.assigned_to = s.signatory_id
            WHERE e.status NOT IN ('closed', 'approved')
              AND e.due_date < %(as_of_date)s
            ORDER BY e.due_date, e.severity
        """
        
        self._log_evidence_query("get_overdue_exceptions", {"as_of_date": as_of_date})
        
        rows = self._execute_query(query, {"as_of_date": as_of_date})
        
        return [self._row_to_exception_evidence(row) for row in rows]
    
    def get_exception_with_activities(self, exception_id: str) -> Optional[ExceptionEvidence]:
        """Get a specific exception with its full activity history."""
        # Get base exception
        query = """
            SELECT 
                e.exception_id,
                e.exception_code,
                cd.control_code,
                cd.control_name,
                r.run_code,
                r.run_date,
                e.severity,
                e.title,
                e.description,
                e.status,
                e.breach_value,
                e.threshold_value,
                e.breach_amount,
                e.opened_at,
                e.due_date,
                e.resolved_at,
                e.resolution_type,
                e.resolution_notes,
                s.full_name as assigned_to
            FROM exceptions e
            JOIN control_definitions cd ON e.control_id = cd.control_id
            JOIN control_runs r ON e.run_id = r.run_id
            LEFT JOIN signatories s ON e.assigned_to = s.signatory_id
            WHERE e.exception_id = %(exception_id)s
        """
        
        rows = self._execute_query(query, {"exception_id": exception_id})
        if not rows:
            return None
        
        exception = self._row_to_exception_evidence(rows[0])
        
        # Get activities
        activity_query = """
            SELECT 
                ea.activity_id,
                ea.activity_type,
                ea.activity_description,
                ea.previous_value,
                ea.new_value,
                s.full_name as performed_by,
                ea.performed_at
            FROM exception_activities ea
            LEFT JOIN signatories s ON ea.performed_by = s.signatory_id
            WHERE ea.exception_id = %(exception_id)s
            ORDER BY ea.performed_at DESC
        """
        
        activities = self._execute_query(activity_query, {"exception_id": exception_id})
        exception.activities = activities
        
        # Get approvals
        approval_query = """
            SELECT 
                a.approval_id,
                a.approval_level,
                a.approval_status,
                a.approval_notes,
                s.full_name as approved_by,
                a.decided_at
            FROM approvals a
            JOIN signatories s ON a.approved_by = s.signatory_id
            WHERE a.reference_type = 'exception'
              AND a.reference_id = %(exception_id)s
            ORDER BY a.decided_at
        """
        
        approvals = self._execute_query(approval_query, {"exception_id": exception_id})
        exception.approvals = approvals
        
        return exception
    
    # =========================================================================
    # SUMMARY QUERIES
    # =========================================================================
    
    def get_daily_compliance_summary(self, run_date: date) -> Optional[DailyComplianceSummary]:
        """Get the compliance summary for a specific date."""
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
                ROUND(cr.controls_passed::NUMERIC / NULLIF(cr.total_controls, 0) * 100, 2) AS pass_rate,
                cr.run_timestamp_start,
                cr.run_timestamp_end,
                EXTRACT(EPOCH FROM (cr.run_timestamp_end - cr.run_timestamp_start)) AS duration_seconds,
                cr.config_hash
            FROM control_runs cr
            WHERE cr.run_date = %(run_date)s
              AND cr.status = 'completed'
            ORDER BY cr.run_timestamp_start DESC
            LIMIT 1
        """
        
        self._log_evidence_query("get_daily_compliance_summary", {"run_date": run_date})
        
        rows = self._execute_query(query, {"run_date": run_date})
        if not rows:
            return None
        
        row = rows[0]
        run_id = row['run_id']
        
        # Get exception counts
        exception_query = """
            SELECT 
                COUNT(*) FILTER (WHERE e.run_id = %(run_id)s) AS exceptions_opened,
                COUNT(*) FILTER (WHERE e.resolved_at::date = %(run_date)s) AS exceptions_closed,
                COUNT(*) FILTER (WHERE e.status NOT IN ('closed', 'approved')) AS exceptions_outstanding,
                COUNT(*) FILTER (WHERE e.severity = 'critical' AND e.status NOT IN ('closed', 'approved')) AS critical_exceptions
            FROM exceptions e
        """
        
        exc_rows = self._execute_query(exception_query, {"run_id": run_id, "run_date": run_date})
        exc_counts = exc_rows[0] if exc_rows else {}
        
        # Get results by category
        category_query = """
            SELECT 
                cd.control_category,
                cr.result_status,
                COUNT(*) as count
            FROM control_results cr
            JOIN control_definitions cd ON cr.control_id = cd.control_id
            WHERE cr.run_id = %(run_id)s
            GROUP BY cd.control_category, cr.result_status
        """
        
        category_rows = self._execute_query(category_query, {"run_id": run_id})
        results_by_category = {}
        for cat_row in category_rows:
            cat = cat_row['control_category']
            status = cat_row['result_status']
            if cat not in results_by_category:
                results_by_category[cat] = {}
            results_by_category[cat][status] = cat_row['count']
        
        return DailyComplianceSummary(
            run_id=row['run_id'],
            run_code=row['run_code'],
            run_date=row['run_date'],
            snapshot_id=row['snowflake_snapshot_id'],
            total_controls=row['total_controls'],
            controls_passed=row['controls_passed'],
            controls_failed=row['controls_failed'],
            controls_warning=row['controls_warning'],
            pass_rate=float(row['pass_rate'] or 0),
            exceptions_opened=exc_counts.get('exceptions_opened', 0),
            exceptions_closed=exc_counts.get('exceptions_closed', 0),
            exceptions_outstanding=exc_counts.get('exceptions_outstanding', 0),
            critical_exceptions=exc_counts.get('critical_exceptions', 0),
            run_start=row['run_timestamp_start'],
            run_end=row['run_timestamp_end'],
            duration_seconds=float(row['duration_seconds'] or 0),
            config_hash=row['config_hash'],
            results_by_category=results_by_category,
        )
    
    def get_compliance_trend(
        self,
        start_date: date,
        end_date: date,
    ) -> List[Dict[str, Any]]:
        """Get compliance pass rate trend over a date range."""
        query = """
            SELECT 
                cr.run_date,
                cr.total_controls,
                cr.controls_passed,
                cr.controls_failed,
                ROUND(cr.controls_passed::NUMERIC / NULLIF(cr.total_controls, 0) * 100, 2) AS pass_rate
            FROM control_runs cr
            WHERE cr.run_date BETWEEN %(start_date)s AND %(end_date)s
              AND cr.status = 'completed'
            ORDER BY cr.run_date
        """
        
        self._log_evidence_query("get_compliance_trend", {
            "start_date": start_date,
            "end_date": end_date,
        })
        
        return self._execute_query(query, {
            "start_date": start_date,
            "end_date": end_date,
        })
    
    # =========================================================================
    # APPROVAL QUERIES
    # =========================================================================
    
    def get_pending_approvals(self) -> List[Dict[str, Any]]:
        """Get all pending approvals."""
        query = """
            SELECT 
                a.approval_id,
                a.approval_type,
                a.reference_type,
                a.reference_id,
                a.approval_level,
                a.required_level,
                a.requested_at,
                CASE 
                    WHEN a.reference_type = 'exception' THEN e.exception_code
                    WHEN a.reference_type = 'document_generation' THEN d.document_code
                    ELSE 'Unknown'
                END as reference_code
            FROM approvals a
            LEFT JOIN exceptions e ON a.reference_type = 'exception' AND a.reference_id = e.exception_id
            LEFT JOIN document_generations d ON a.reference_type = 'document_generation' AND a.reference_id = d.document_id
            WHERE a.approval_status = 'pending'
            ORDER BY a.requested_at
        """
        
        return self._execute_query(query, {})
    
    # =========================================================================
    # EVIDENCE PACKAGING
    # =========================================================================
    
    def package_evidence_for_document(
        self,
        run_id: str,
        include_samples: bool = True,
    ) -> Dict[str, Any]:
        """
        Package all evidence needed for document generation.
        
        This creates a complete, self-contained evidence package
        that can be used by the document builder without additional queries.
        """
        package_id = str(uuid.uuid4())
        
        # Get run summary
        summary_query = """
            SELECT 
                cr.run_id,
                cr.run_code,
                cr.run_date,
                cr.snowflake_snapshot_id,
                cr.snowflake_snapshot_ts,
                cr.total_controls,
                cr.controls_passed,
                cr.controls_failed,
                cr.controls_warning,
                cr.run_timestamp_start,
                cr.run_timestamp_end,
                cr.config_hash,
                cr.executor_service,
                cr.executor_version
            FROM control_runs cr
            WHERE cr.run_id = %(run_id)s
        """
        
        run_rows = self._execute_query(summary_query, {"run_id": run_id})
        if not run_rows:
            raise ValueError(f"Run {run_id} not found")
        
        run_info = run_rows[0]
        
        # Get all results
        results = self.get_control_results_for_run(run_id)
        
        # Get exceptions
        exceptions = self.get_exceptions_for_run(run_id)
        
        # Get open exceptions as of run date
        open_exceptions = self.get_open_exceptions(as_of_date=run_info['run_date'])
        
        # Build package
        package = {
            "package_id": package_id,
            "package_created_at": datetime.now(timezone.utc).isoformat(),
            "package_created_by": self.user_id,
            "run": {
                "run_id": run_info['run_id'],
                "run_code": run_info['run_code'],
                "run_date": run_info['run_date'].isoformat(),
                "snapshot_id": run_info['snowflake_snapshot_id'],
                "snapshot_ts": run_info['snowflake_snapshot_ts'].isoformat() if run_info['snowflake_snapshot_ts'] else None,
                "config_hash": run_info['config_hash'],
                "executor": f"{run_info['executor_service']} v{run_info['executor_version']}",
            },
            "summary": {
                "total_controls": run_info['total_controls'],
                "passed": run_info['controls_passed'],
                "failed": run_info['controls_failed'],
                "warning": run_info['controls_warning'],
                "pass_rate": round(run_info['controls_passed'] / max(run_info['total_controls'], 1) * 100, 2),
            },
            "results": [
                {
                    "result_id": r.result_id,
                    "control_code": r.control_code,
                    "control_name": r.control_name,
                    "category": r.control_category,
                    "status": r.result_status,
                    "calculated_value": r.calculated_value,
                    "threshold": f"{r.threshold_operator} {r.threshold_value}",
                    "breach_amount": r.breach_amount,
                    "citation": r.to_citation(),
                }
                for r in results
            ],
            "exceptions_opened": [
                {
                    "exception_code": e.exception_code,
                    "control_code": e.control_code,
                    "severity": e.severity,
                    "title": e.title,
                    "status": e.status,
                    "due_date": e.due_date.isoformat() if e.due_date else None,
                    "citation": e.to_citation(),
                }
                for e in exceptions
            ],
            "exceptions_outstanding": [
                {
                    "exception_code": e.exception_code,
                    "control_code": e.control_code,
                    "severity": e.severity,
                    "title": e.title,
                    "status": e.status,
                    "due_date": e.due_date.isoformat() if e.due_date else None,
                    "days_open": (date.today() - e.opened_at.date()).days if e.opened_at else None,
                    "citation": e.to_citation(),
                }
                for e in open_exceptions
            ],
        }
        
        # Calculate package hash for integrity verification
        package["package_hash"] = hashlib.sha256(
            json.dumps(package, sort_keys=True, default=str).encode()
        ).hexdigest()
        
        self.logger.info(f"Created evidence package {package_id} for run {run_id}")
        
        return package
    
    # =========================================================================
    # INTERNAL METHODS
    # =========================================================================
    
    def _execute_query(self, query: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute a query and return results as list of dicts."""
        cursor = self.connection.cursor()
        try:
            cursor.execute(query, params)
            if cursor.description:
                columns = [desc[0] for desc in cursor.description]
                rows = cursor.fetchall()
                return [dict(zip(columns, row)) for row in rows]
            return []
        finally:
            cursor.close()
    
    def _log_evidence_query(self, query_type: str, params: Dict[str, Any]) -> None:
        """Log an evidence query for audit purposes."""
        query = EvidenceQuery(
            query_id=str(uuid.uuid4()),
            query_type=query_type,
            parameters=params,
            executed_at=datetime.now(timezone.utc),
            executed_by=self.user_id,
        )
        
        self.logger.info(
            f"Evidence query: type={query_type}, hash={query.query_hash[:16]}, "
            f"user={self.user_id}"
        )
        
        # In production, this would also write to audit_log table
    
    def _row_to_exception_evidence(self, row: Dict[str, Any]) -> ExceptionEvidence:
        """Convert a database row to ExceptionEvidence."""
        return ExceptionEvidence(
            exception_id=row['exception_id'],
            exception_code=row['exception_code'],
            control_code=row['control_code'],
            control_name=row['control_name'],
            run_code=row['run_code'],
            run_date=row['run_date'],
            severity=row['severity'],
            title=row['title'],
            description=row['description'],
            status=row['status'],
            breach_value=row.get('breach_value'),
            threshold_value=row.get('threshold_value'),
            breach_amount=row.get('breach_amount'),
            opened_at=row['opened_at'],
            due_date=row.get('due_date'),
            resolved_at=row.get('resolved_at'),
            resolution_type=row.get('resolution_type'),
            resolution_notes=row.get('resolution_notes'),
            assigned_to=row.get('assigned_to'),
        )
    
    def _get_run_id_for_result(self, result_id: str) -> str:
        """Get the run ID for a specific result."""
        query = "SELECT run_id FROM control_results WHERE result_id = %(result_id)s"
        rows = self._execute_query(query, {"result_id": result_id})
        if rows:
            return rows[0]['run_id']
        raise ValueError(f"Result {result_id} not found")
