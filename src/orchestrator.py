"""
Compliance RAG Orchestrator - Main Entry Point

This module provides the main orchestration layer that coordinates:
1. Control Runner (deterministic execution)
2. Evidence Store (audit trail)
3. Retrieval Layer (hybrid RAG)
4. Narrative Generator (LLM with citations)
5. Document Builder (PDF generation)

The orchestrator ensures all components work together while
maintaining the strict separation between deterministic computation
and LLM-assisted text generation.

SEC Examination Note:
- All operations are logged with full audit trail
- Each document generation is reproducible
- LLM is only used for narrative, never for calculations
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, List

from .config import get_settings, Settings
from .control_runner import ControlRunner, ControlRunContext, get_run_summary, get_failed_controls
from .evidence_store import EvidenceStore, DailyComplianceSummary
from .retrieval import HybridRetriever, RetrievalContext
from .narrative import NarrativeGenerator, GeneratedNarrative
from .document_builder import DocumentBuilder, GeneratedDocument, DocumentType

logger = logging.getLogger(__name__)


@dataclass
class ComplianceRunResult:
    """
    Complete result of a compliance run including document generation.
    """
    run_context: ControlRunContext
    summary: DailyComplianceSummary
    document: Optional[GeneratedDocument]
    narrative: Optional[GeneratedNarrative]
    
    # Metrics
    total_duration_seconds: float
    control_execution_seconds: float
    document_generation_seconds: float


class ComplianceOrchestrator:
    """
    Main orchestrator for the compliance RAG system.
    
    This class coordinates all system components to:
    1. Execute daily compliance controls
    2. Generate compliance documents
    3. Create SEC filing workpapers
    
    All operations maintain full audit trail and reproducibility.
    """
    
    def __init__(
        self,
        postgres_connection: Any,
        snowflake_connection: Any,
        llm_client: Optional[Any] = None,
        vector_store: Optional[Any] = None,
        settings: Optional[Settings] = None,
    ):
        self.settings = settings or get_settings()
        self.postgres = postgres_connection
        self.snowflake = snowflake_connection
        self.llm_client = llm_client
        
        # Initialize components
        self.control_runner = ControlRunner(
            snowflake_connection=snowflake_connection,
            postgres_connection=postgres_connection,
            settings=self.settings,
        )
        
        self.evidence_store = EvidenceStore(
            postgres_connection=postgres_connection,
            user_id="compliance_system",
        )
        
        self.retriever = HybridRetriever(
            postgres_connection=postgres_connection,
            vector_store=vector_store,
            settings=self.settings,
        )
        
        self.narrative_generator = NarrativeGenerator(
            llm_client=llm_client,
            settings=self.settings,
            postgres_connection=postgres_connection,
        ) if llm_client else None
        
        self.document_builder = DocumentBuilder(settings=self.settings)
        
        self.logger = logging.getLogger(f"{__name__}.ComplianceOrchestrator")
    
    def run_daily_compliance(
        self,
        run_date: date,
        fund_ids: List[str],
        generate_document: bool = True,
        generate_narrative: bool = True,
        output_path: Optional[Path] = None,
    ) -> ComplianceRunResult:
        """
        Execute complete daily compliance workflow.
        
        This is the main entry point for daily operations:
        1. Execute all active controls
        2. Record results and create exceptions
        3. Retrieve evidence for document
        4. Generate narrative (if LLM available)
        5. Build PDF document
        
        Args:
            run_date: Date for the compliance run
            fund_ids: List of fund IDs to run controls for
            generate_document: Whether to generate PDF
            generate_narrative: Whether to generate LLM narrative
            output_path: Where to save the PDF
        
        Returns:
            ComplianceRunResult with all outputs and metrics
        """
        import time
        start_time = time.time()
        
        self.logger.info(f"Starting daily compliance run for {run_date}")
        
        # Phase 1: Execute Controls (Deterministic)
        control_start = time.time()
        run_context = self.control_runner.run_daily_controls(
            run_date=run_date,
            fund_ids=fund_ids,
        )
        control_duration = time.time() - control_start
        
        self.logger.info(f"Controls executed in {control_duration:.2f}s")
        
        # Phase 2: Get Summary from Evidence Store
        summary = self.evidence_store.get_daily_compliance_summary(run_date)
        if not summary:
            raise ValueError(f"No compliance summary found for {run_date}")
        
        # Phase 3: Generate Document (if requested)
        document = None
        narrative = None
        doc_start = time.time()
        
        if generate_document:
            # Get detailed data for document
            control_results = self._get_all_control_results(run_context.run_id)
            exceptions = self._get_exceptions(run_context.run_id)
            outstanding = self._get_outstanding_exceptions(run_date)
            
            # Get run summary dict
            run_summary_dict = self._get_run_summary_dict(run_context.run_id)
            
            # Generate narrative if requested and LLM available
            narrative_text = None
            narrative_metadata = None
            
            if generate_narrative and self.narrative_generator:
                # Retrieve context for narrative
                context = self.retriever.retrieve_for_daily_pack(
                    run_id=run_context.run_id,
                    run_date=run_date,
                )
                
                # Generate narrative
                narrative = self.narrative_generator.generate_daily_summary(
                    context=context,
                    run_summary=run_summary_dict,
                )
                
                narrative_text = narrative.content
                narrative_metadata = {
                    "narrative_id": narrative.narrative_id,
                    "model_id": narrative.model_id,
                    "model_version": narrative.model_version,
                    "tokens_used": narrative.tokens_used,
                }
            
            # Build document
            document = self.document_builder.build_daily_compliance_pack(
                run_date=run_date,
                run_summary=run_summary_dict,
                control_results=control_results,
                exceptions=exceptions,
                outstanding_exceptions=outstanding,
                narrative=narrative_text,
                narrative_metadata=narrative_metadata,
            )
            
            # Save document if path provided
            if output_path:
                document.save(output_path)
            
            # Record document generation
            self._record_document_generation(document)
        
        doc_duration = time.time() - doc_start
        total_duration = time.time() - start_time
        
        self.logger.info(
            f"Daily compliance completed: "
            f"controls={summary.total_controls}, "
            f"passed={summary.controls_passed}, "
            f"failed={summary.controls_failed}, "
            f"duration={total_duration:.2f}s"
        )
        
        return ComplianceRunResult(
            run_context=run_context,
            summary=summary,
            document=document,
            narrative=narrative,
            total_duration_seconds=total_duration,
            control_execution_seconds=control_duration,
            document_generation_seconds=doc_duration,
        )
    
    def generate_form_pf_workpaper(
        self,
        period_end: date,
        fund_id: str,
        output_path: Optional[Path] = None,
    ) -> GeneratedDocument:
        """
        Generate Form PF filing workpaper.
        
        This retrieves metrics from Snowflake and generates
        a complete workpaper with supporting evidence.
        """
        self.logger.info(f"Generating Form PF workpaper for {fund_id}, period {period_end}")
        
        # Get metrics from Snowflake
        metrics = self._get_form_pf_metrics(period_end, fund_id)
        
        # Get relevant control results
        control_results = self._get_recent_control_results(
            categories=["liquidity", "leverage", "counterparty"],
            as_of_date=period_end,
        )
        
        # Generate methodology narrative if LLM available
        narrative = None
        narrative_metadata = None
        
        if self.narrative_generator:
            context = self.retriever.retrieve_for_filing_workpaper(
                filing_type="form_pf",
                period_end=period_end,
                fund_ids=[fund_id],
            )
            
            narrative_result = self.narrative_generator.generate_filing_workpaper(
                context=context,
                filing_type="form_pf",
                period=f"Quarter ending {period_end}",
                metrics=metrics,
            )
            
            narrative = narrative_result.content
            narrative_metadata = {
                "narrative_id": narrative_result.narrative_id,
                "model_id": narrative_result.model_id,
                "model_version": narrative_result.model_version,
                "tokens_used": narrative_result.tokens_used,
            }
        
        # Build document
        document = self.document_builder.build_form_pf_workpaper(
            period_end=period_end,
            fund_id=fund_id,
            metrics=metrics,
            control_results=control_results,
            narrative=narrative,
            narrative_metadata=narrative_metadata,
        )
        
        if output_path:
            document.save(output_path)
        
        self._record_document_generation(document)
        
        return document
    
    def generate_13f_workpaper(
        self,
        period_end: date,
        output_path: Optional[Path] = None,
    ) -> GeneratedDocument:
        """
        Generate 13F filing workpaper.
        """
        self.logger.info(f"Generating 13F workpaper for period {period_end}")
        
        # Get 13F holdings from Snowflake
        holdings, summary = self._get_13f_holdings(period_end)
        
        # Build document
        document = self.document_builder.build_13f_workpaper(
            period_end=period_end,
            holdings=holdings,
            summary=summary,
        )
        
        if output_path:
            document.save(output_path)
        
        self._record_document_generation(document)
        
        return document
    
    def regenerate_document(
        self,
        run_id: str,
        output_path: Optional[Path] = None,
    ) -> GeneratedDocument:
        """
        Regenerate a document from existing control run data.
        
        This demonstrates reproducibility - same inputs produce same outputs.
        """
        # Get run info
        run_summary = self._get_run_summary_dict(run_id)
        if not run_summary:
            raise ValueError(f"Run {run_id} not found")
        
        run_date = run_summary["run_date"]
        
        # Get all data from evidence store
        control_results = self._get_all_control_results(run_id)
        exceptions = self._get_exceptions(run_id)
        outstanding = self._get_outstanding_exceptions(run_date)
        
        # Regenerate (without LLM to ensure determinism)
        document = self.document_builder.build_daily_compliance_pack(
            run_date=run_date,
            run_summary=run_summary,
            control_results=control_results,
            exceptions=exceptions,
            outstanding_exceptions=outstanding,
            narrative=None,  # Skip LLM for deterministic regeneration
            narrative_metadata=None,
        )
        
        if output_path:
            document.save(output_path)
        
        return document
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    def _get_all_control_results(self, run_id: str) -> List[Dict[str, Any]]:
        """Get all control results for a run."""
        query = """
            SELECT 
                cr.result_id,
                cd.control_code,
                cd.control_name,
                cd.control_category,
                cr.calculated_value,
                cr.threshold_value,
                cr.threshold_operator,
                cr.result_status,
                cr.breach_amount,
                cr.breach_percentage
            FROM control_results cr
            JOIN control_definitions cd ON cr.control_id = cd.control_id
            WHERE cr.run_id = %(run_id)s
            ORDER BY cd.control_category, cd.control_code
        """
        return self._execute_postgres_query(query, {"run_id": run_id})
    
    def _get_exceptions(self, run_id: str) -> List[Dict[str, Any]]:
        """Get exceptions for a run."""
        query = """
            SELECT 
                e.exception_id,
                e.exception_code,
                cd.control_code,
                e.severity,
                e.title,
                e.status,
                e.due_date
            FROM exceptions e
            JOIN control_definitions cd ON e.control_id = cd.control_id
            WHERE e.run_id = %(run_id)s
            ORDER BY e.severity, e.opened_at
        """
        return self._execute_postgres_query(query, {"run_id": run_id})
    
    def _get_outstanding_exceptions(self, as_of_date: date) -> List[Dict[str, Any]]:
        """Get all open exceptions as of a date."""
        query = """
            SELECT 
                e.exception_id,
                e.exception_code,
                cd.control_code,
                e.severity,
                e.title,
                e.status,
                e.due_date,
                e.opened_at,
                CURRENT_DATE - e.opened_at::date AS days_open
            FROM exceptions e
            JOIN control_definitions cd ON e.control_id = cd.control_id
            WHERE e.status NOT IN ('closed', 'approved')
              AND e.opened_at <= %(as_of_date)s
            ORDER BY e.severity, e.due_date
        """
        return self._execute_postgres_query(query, {"as_of_date": as_of_date})
    
    def _get_run_summary_dict(self, run_id: str) -> Dict[str, Any]:
        """Get run summary as dictionary."""
        query = """
            SELECT 
                run_id,
                run_code,
                run_date,
                snowflake_snapshot_id,
                snowflake_snapshot_ts,
                total_controls,
                controls_passed,
                controls_failed,
                controls_warning,
                run_timestamp_start,
                run_timestamp_end,
                config_hash,
                executor_service,
                executor_version
            FROM control_runs
            WHERE run_id = %(run_id)s
        """
        rows = self._execute_postgres_query(query, {"run_id": run_id})
        return rows[0] if rows else {}
    
    def _get_recent_control_results(
        self,
        categories: List[str],
        as_of_date: date,
    ) -> List[Dict[str, Any]]:
        """Get recent control results by category."""
        query = """
            SELECT DISTINCT ON (cd.control_code)
                cr.result_id,
                cd.control_code,
                cd.control_name,
                cd.control_category,
                cr.calculated_value,
                cr.threshold_value,
                cr.threshold_operator,
                cr.result_status,
                r.run_date
            FROM control_results cr
            JOIN control_definitions cd ON cr.control_id = cd.control_id
            JOIN control_runs r ON cr.run_id = r.run_id
            WHERE cd.control_category = ANY(%(categories)s)
              AND r.run_date <= %(as_of_date)s
              AND r.status = 'completed'
            ORDER BY cd.control_code, r.run_date DESC
        """
        return self._execute_postgres_query(query, {
            "categories": categories,
            "as_of_date": as_of_date,
        })
    
    def _get_form_pf_metrics(self, period_end: date, fund_id: str) -> Dict[str, Any]:
        """Get Form PF metrics from Snowflake."""
        # Liquidity buckets
        liquidity_query = """
            SELECT 
                liquidity_bucket as bucket,
                long_market_value as long_value,
                long_pct_nav as long_pct,
                short_market_value as short_value,
                short_pct_nav as short_pct
            FROM compliance.v_liquidity_buckets
            WHERE snapshot_date = %(period_end)s
              AND fund_id = %(fund_id)s
            ORDER BY bucket_order
        """
        
        # Leverage
        leverage_query = """
            SELECT 
                gross_leverage_ratio as gross_leverage,
                net_leverage_ratio as net_leverage,
                borrowing_to_nav_ratio * 100 as borrowing_to_nav,
                derivatives_gross_notional as derivatives_notional,
                gav_to_nav_ratio as gav_to_nav
            FROM compliance.v_leverage_metrics
            WHERE snapshot_date = %(period_end)s
              AND fund_id = %(fund_id)s
        """
        
        # Counterparty
        cp_query = """
            SELECT 
                counterparty_name as name,
                counterparty_type as type,
                net_exposure,
                net_exposure / NULLIF(
                    (SELECT fund_nav FROM compliance.v_leverage_metrics 
                     WHERE snapshot_date = %(period_end)s AND fund_id = %(fund_id)s),
                0) * 100 as pct_nav,
                credit_rating
            FROM compliance.v_counterparty_exposure
            WHERE snapshot_date = %(period_end)s
              AND fund_id = %(fund_id)s
            ORDER BY net_exposure DESC
        """
        
        # Geographic
        geo_query = """
            SELECT 
                country,
                region,
                exposure_usd as exposure,
                pct_nav
            FROM compliance.v_form_pf_geographic
            WHERE snapshot_date = %(period_end)s
              AND fund_id = %(fund_id)s
            ORDER BY exposure_usd DESC
        """
        
        params = {"period_end": period_end, "fund_id": fund_id}
        
        return {
            "snapshot_id": f"SNOW_{period_end.isoformat().replace('-', '_')}",
            "liquidity_buckets": self._execute_snowflake_query(liquidity_query, params),
            "leverage": self._execute_snowflake_query(leverage_query, params)[0] if self._execute_snowflake_query(leverage_query, params) else {},
            "counterparty_exposure": self._execute_snowflake_query(cp_query, params),
            "geographic": self._execute_snowflake_query(geo_query, params),
        }
    
    def _get_13f_holdings(self, period_end: date) -> tuple:
        """Get 13F holdings from Snowflake."""
        query = """
            SELECT 
                cusip,
                issuer_name,
                security_type as security_class,
                shares_principal_amount as shares,
                value,
                investment_discretion
            FROM compliance.v_13f_holdings
            WHERE snapshot_date = %(period_end)s
              AND is_13f_security = TRUE
            ORDER BY value DESC
        """
        
        holdings = self._execute_snowflake_query(query, {"period_end": period_end})
        
        summary = {
            "total_positions": len(holdings),
            "total_value": sum(h.get("value", 0) for h in holdings),
            "distinct_issuers": len(set(h.get("issuer_name") for h in holdings)),
            "13f_count": len(holdings),
            "snapshot_id": f"SNOW_{period_end.isoformat().replace('-', '_')}",
        }
        
        return holdings, summary
    
    def _record_document_generation(self, document: GeneratedDocument) -> None:
        """Record document generation to database."""
        query = """
            INSERT INTO document_generations (
                document_id, document_code, document_type, document_date,
                run_id, source_snapshot_id,
                template_id, template_version, template_hash,
                llm_model_id, llm_model_version, llm_tokens_used,
                output_format, output_file_path, output_file_hash, output_file_size_bytes,
                status, generated_at
            ) VALUES (
                %(document_id)s, %(document_code)s, %(document_type)s, %(document_date)s,
                %(run_id)s, %(snapshot_id)s,
                %(template_id)s, %(template_version)s, %(template_hash)s,
                %(llm_model_id)s, %(llm_model_version)s, %(llm_tokens_used)s,
                'pdf', '', %(document_hash)s, %(file_size)s,
                'generated', %(generated_at)s
            )
        """
        
        self._execute_postgres_query(query, {
            "document_id": document.metadata.document_id,
            "document_code": document.metadata.document_code,
            "document_type": document.metadata.document_type.value,
            "document_date": document.metadata.document_date,
            "run_id": document.metadata.run_id,
            "snapshot_id": document.metadata.snapshot_id,
            "template_id": document.metadata.template_id,
            "template_version": document.metadata.template_version,
            "template_hash": document.metadata.template_hash,
            "llm_model_id": document.metadata.llm_model_id,
            "llm_model_version": document.metadata.llm_model_version,
            "llm_tokens_used": document.metadata.llm_tokens_used,
            "document_hash": document.document_hash,
            "file_size": len(document.pdf_bytes),
            "generated_at": document.metadata.generated_at,
        }, commit=True)
    
    def _execute_postgres_query(
        self,
        query: str,
        params: Dict[str, Any],
        commit: bool = False,
    ) -> List[Dict[str, Any]]:
        """Execute query against Postgres."""
        cursor = self.postgres.cursor()
        try:
            cursor.execute(query, params)
            if commit:
                self.postgres.commit()
                return []
            if cursor.description:
                columns = [desc[0] for desc in cursor.description]
                return [dict(zip(columns, row)) for row in cursor.fetchall()]
            return []
        finally:
            cursor.close()
    
    def _execute_snowflake_query(
        self,
        query: str,
        params: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Execute query against Snowflake."""
        cursor = self.snowflake.cursor()
        try:
            # Convert named parameters for Snowflake
            formatted_query = query
            for key, value in params.items():
                formatted_query = formatted_query.replace(f"%({key})s", f":{key}")
            
            cursor.execute(formatted_query, params)
            if cursor.description:
                columns = [desc[0].lower() for desc in cursor.description]
                return [dict(zip(columns, row)) for row in cursor.fetchall()]
            return []
        finally:
            cursor.close()
