"""
Retrieval Layer - Hybrid RAG for Compliance Evidence and Policies

This module provides the retrieval layer for the compliance RAG system.
It implements a three-tier retrieval strategy:

1. Structured SQL queries (FIRST) - for control results, exceptions, metrics
2. Lexical retrieval (SECOND) - for exact policy matches, regulatory text
3. Vector retrieval (THIRD) - for semantic similarity, finding related policies

SEC Examination Note:
- All retrievals are logged with query and result hashes
- Retrieved context is always permissioned and scoped
- No external data sources are queried
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, date
from enum import Enum
from typing import Optional, Dict, Any, List, Tuple
import uuid
import re

logger = logging.getLogger(__name__)


class RetrievalSource(Enum):
    """Types of retrieval sources."""
    STRUCTURED = "structured"  # SQL queries
    LEXICAL = "lexical"        # Full-text search
    VECTOR = "vector"          # Semantic similarity


class RetrievalScope(Enum):
    """Scopes for retrieval operations."""
    CONTROL_RESULTS = "control_results"
    EXCEPTIONS = "exceptions"
    POLICIES = "policies"
    PRIOR_FILINGS = "prior_filings"
    METRICS = "metrics"


@dataclass
class RetrievedDocument:
    """
    A document retrieved for context.
    
    All retrieved content must be traceable.
    """
    document_id: str
    source: RetrievalSource
    scope: RetrievalScope
    
    # Content
    content: str
    content_hash: str
    
    # Metadata
    title: Optional[str] = None
    section: Optional[str] = None
    page_number: Optional[int] = None
    
    # Relevance
    relevance_score: float = 1.0
    match_type: str = "exact"  # 'exact', 'partial', 'semantic'
    
    # Source reference
    source_id: Optional[str] = None  # e.g., policy_id, result_id
    source_path: Optional[str] = None
    effective_date: Optional[date] = None
    
    # Audit
    retrieved_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_citation(self) -> str:
        """Generate a citation string for this document."""
        if self.scope == RetrievalScope.POLICIES:
            return f"[Policy: {self.source_id} | Section: {self.section or 'N/A'}]"
        elif self.scope == RetrievalScope.CONTROL_RESULTS:
            return f"[ControlResult: {self.source_id}]"
        elif self.scope == RetrievalScope.EXCEPTIONS:
            return f"[Exception: {self.source_id}]"
        elif self.scope == RetrievalScope.PRIOR_FILINGS:
            return f"[Filing: {self.source_id} | Date: {self.effective_date}]"
        else:
            return f"[{self.scope.value}: {self.source_id}]"


@dataclass
class RetrievalContext:
    """
    Complete context package for LLM generation.
    
    This is what gets passed to the narrative generator.
    """
    context_id: str
    query: str
    query_hash: str
    
    # Retrieved documents by source
    structured_results: List[RetrievedDocument] = field(default_factory=list)
    lexical_results: List[RetrievedDocument] = field(default_factory=list)
    vector_results: List[RetrievedDocument] = field(default_factory=list)
    
    # Metadata
    total_documents: int = 0
    retrieval_duration_ms: int = 0
    retrieved_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def all_documents(self) -> List[RetrievedDocument]:
        """Get all retrieved documents in priority order."""
        return self.structured_results + self.lexical_results + self.vector_results
    
    @property
    def context_hash(self) -> str:
        """Hash of all retrieved content for audit trail."""
        content_str = "|".join([d.content_hash for d in self.all_documents])
        return hashlib.sha256(content_str.encode()).hexdigest()
    
    def to_prompt_context(self, max_tokens: int = 8000) -> str:
        """
        Format retrieved context for LLM prompt.
        
        This formats the context in a way that:
        - Clearly separates sources
        - Includes citations
        - Respects token limits
        """
        sections = []
        
        # Structured evidence (highest priority)
        if self.structured_results:
            section = "## EVIDENCE FROM CONTROL SYSTEM\n\n"
            for doc in self.structured_results:
                section += f"### {doc.title or 'Evidence'}\n"
                section += f"Citation: {doc.to_citation()}\n\n"
                section += doc.content + "\n\n"
            sections.append(section)
        
        # Policy excerpts
        if self.lexical_results:
            section = "## RELEVANT POLICY EXCERPTS\n\n"
            for doc in self.lexical_results:
                section += f"### {doc.title or 'Policy'}\n"
                section += f"Citation: {doc.to_citation()}\n\n"
                section += doc.content + "\n\n"
            sections.append(section)
        
        # Semantic matches (lowest priority)
        if self.vector_results:
            section = "## RELATED POLICY CONTEXT\n\n"
            for doc in self.vector_results[:3]:  # Limit semantic results
                section += f"### {doc.title or 'Related Content'}\n"
                section += f"Citation: {doc.to_citation()}\n"
                section += f"Relevance Score: {doc.relevance_score:.2f}\n\n"
                section += doc.content + "\n\n"
            sections.append(section)
        
        full_context = "\n".join(sections)
        
        # Truncate if needed (rough token estimation: 4 chars = 1 token)
        max_chars = max_tokens * 4
        if len(full_context) > max_chars:
            full_context = full_context[:max_chars] + "\n\n[Context truncated due to length]"
        
        return full_context


class HybridRetriever:
    """
    Main retrieval engine implementing hybrid search.
    
    Retrieval order:
    1. Structured queries (always first, most precise)
    2. Lexical search (for exact matches)
    3. Vector search (for semantic similarity)
    
    This ensures compliance evidence takes precedence over general policy text.
    """
    
    def __init__(
        self,
        postgres_connection: Any,
        vector_store: Optional[Any] = None,
        settings: Optional[Any] = None,
    ):
        self.postgres = postgres_connection
        self.vector_store = vector_store
        self.settings = settings
        self.logger = logging.getLogger(f"{__name__}.HybridRetriever")
    
    def retrieve_for_daily_pack(
        self,
        run_id: str,
        run_date: date,
    ) -> RetrievalContext:
        """
        Retrieve all context needed for daily compliance pack generation.
        
        This is a specialized retrieval that gathers:
        - Control run summary and results
        - Active exceptions
        - Relevant policy sections
        """
        import time
        start_time = time.time()
        
        query = f"Daily compliance pack for {run_date}"
        context_id = str(uuid.uuid4())
        
        self.logger.info(f"Retrieving context for daily pack: run_id={run_id}")
        
        # 1. Structured retrieval - control results and exceptions
        structured = []
        
        # Get run summary
        run_summary = self._get_run_summary(run_id)
        if run_summary:
            structured.append(RetrievedDocument(
                document_id=str(uuid.uuid4()),
                source=RetrievalSource.STRUCTURED,
                scope=RetrievalScope.CONTROL_RESULTS,
                content=self._format_run_summary(run_summary),
                content_hash=hashlib.sha256(json.dumps(run_summary, default=str).encode()).hexdigest(),
                title="Control Run Summary",
                source_id=run_id,
            ))
        
        # Get failed controls
        failed_controls = self._get_failed_controls(run_id)
        for control in failed_controls:
            structured.append(RetrievedDocument(
                document_id=str(uuid.uuid4()),
                source=RetrievalSource.STRUCTURED,
                scope=RetrievalScope.CONTROL_RESULTS,
                content=self._format_control_result(control),
                content_hash=hashlib.sha256(json.dumps(control, default=str).encode()).hexdigest(),
                title=f"Control Failure: {control['control_code']}",
                source_id=control['result_id'],
            ))
        
        # Get exceptions
        exceptions = self._get_exceptions_for_run(run_id)
        for exc in exceptions:
            structured.append(RetrievedDocument(
                document_id=str(uuid.uuid4()),
                source=RetrievalSource.STRUCTURED,
                scope=RetrievalScope.EXCEPTIONS,
                content=self._format_exception(exc),
                content_hash=hashlib.sha256(json.dumps(exc, default=str).encode()).hexdigest(),
                title=f"Exception: {exc['exception_code']}",
                source_id=exc['exception_id'],
            ))
        
        # 2. Lexical retrieval - policy sections for failed control types
        lexical = []
        
        if failed_controls:
            # Get relevant policy sections based on failed control categories
            categories = set(c['control_category'] for c in failed_controls)
            for category in categories:
                policy_sections = self._search_policies_by_category(category)
                for section in policy_sections:
                    lexical.append(RetrievedDocument(
                        document_id=str(uuid.uuid4()),
                        source=RetrievalSource.LEXICAL,
                        scope=RetrievalScope.POLICIES,
                        content=section['section_text'],
                        content_hash=hashlib.sha256(section['section_text'].encode()).hexdigest(),
                        title=section['policy_title'],
                        section=section['section_title'],
                        source_id=section['policy_code'],
                        effective_date=section.get('effective_date'),
                    ))
        
        # 3. Vector retrieval - only if needed for additional context
        vector = []
        
        if self.vector_store and failed_controls:
            # Search for semantically related content
            for control in failed_controls[:3]:  # Limit vector searches
                similar_docs = self._vector_search(
                    f"{control['control_name']} compliance requirements"
                )
                vector.extend(similar_docs)
        
        duration_ms = int((time.time() - start_time) * 1000)
        
        context = RetrievalContext(
            context_id=context_id,
            query=query,
            query_hash=hashlib.sha256(query.encode()).hexdigest(),
            structured_results=structured,
            lexical_results=lexical,
            vector_results=vector,
            total_documents=len(structured) + len(lexical) + len(vector),
            retrieval_duration_ms=duration_ms,
        )
        
        self._log_retrieval(context)
        
        return context
    
    def retrieve_for_exception_narrative(
        self,
        exception_id: str,
    ) -> RetrievalContext:
        """
        Retrieve context for generating an exception narrative.
        """
        import time
        start_time = time.time()
        
        context_id = str(uuid.uuid4())
        
        structured = []
        lexical = []
        vector = []
        
        # Get exception details
        exception = self._get_exception_details(exception_id)
        if not exception:
            raise ValueError(f"Exception {exception_id} not found")
        
        # Add exception as structured evidence
        structured.append(RetrievedDocument(
            document_id=str(uuid.uuid4()),
            source=RetrievalSource.STRUCTURED,
            scope=RetrievalScope.EXCEPTIONS,
            content=self._format_exception_detailed(exception),
            content_hash=hashlib.sha256(json.dumps(exception, default=str).encode()).hexdigest(),
            title=f"Exception Details: {exception['exception_code']}",
            source_id=exception_id,
        ))
        
        # Get the control result that triggered the exception
        control_result = self._get_control_result(exception['result_id'])
        if control_result:
            structured.append(RetrievedDocument(
                document_id=str(uuid.uuid4()),
                source=RetrievalSource.STRUCTURED,
                scope=RetrievalScope.CONTROL_RESULTS,
                content=self._format_control_result_detailed(control_result),
                content_hash=hashlib.sha256(json.dumps(control_result, default=str).encode()).hexdigest(),
                title=f"Control Result: {control_result['control_code']}",
                source_id=exception['result_id'],
            ))
        
        # Get relevant policy
        if exception.get('control_category'):
            policy_sections = self._search_policies_by_category(exception['control_category'])
            for section in policy_sections:
                lexical.append(RetrievedDocument(
                    document_id=str(uuid.uuid4()),
                    source=RetrievalSource.LEXICAL,
                    scope=RetrievalScope.POLICIES,
                    content=section['section_text'],
                    content_hash=hashlib.sha256(section['section_text'].encode()).hexdigest(),
                    title=section['policy_title'],
                    section=section['section_title'],
                    source_id=section['policy_code'],
                ))
        
        duration_ms = int((time.time() - start_time) * 1000)
        
        query = f"Exception narrative for {exception['exception_code']}"
        
        context = RetrievalContext(
            context_id=context_id,
            query=query,
            query_hash=hashlib.sha256(query.encode()).hexdigest(),
            structured_results=structured,
            lexical_results=lexical,
            vector_results=vector,
            total_documents=len(structured) + len(lexical) + len(vector),
            retrieval_duration_ms=duration_ms,
        )
        
        self._log_retrieval(context)
        
        return context
    
    def retrieve_for_filing_workpaper(
        self,
        filing_type: str,
        period_end: date,
        fund_ids: List[str],
    ) -> RetrievalContext:
        """
        Retrieve context for SEC filing workpaper generation.
        
        filing_type: 'form_pf', '13f', 'adv'
        """
        import time
        start_time = time.time()
        
        context_id = str(uuid.uuid4())
        query = f"{filing_type.upper()} workpaper for period ending {period_end}"
        
        structured = []
        lexical = []
        vector = []
        
        # Get prior filing for reference
        prior_filing = self._get_prior_filing(filing_type, period_end)
        if prior_filing:
            structured.append(RetrievedDocument(
                document_id=str(uuid.uuid4()),
                source=RetrievalSource.STRUCTURED,
                scope=RetrievalScope.PRIOR_FILINGS,
                content=prior_filing['content'],
                content_hash=hashlib.sha256(prior_filing['content'].encode()).hexdigest(),
                title=f"Prior {filing_type.upper()} Filing",
                source_id=prior_filing['filing_id'],
                effective_date=prior_filing.get('filing_date'),
            ))
        
        # Get filing-specific policy sections
        policy_sections = self._search_policies_by_filing_type(filing_type)
        for section in policy_sections:
            lexical.append(RetrievedDocument(
                document_id=str(uuid.uuid4()),
                source=RetrievalSource.LEXICAL,
                scope=RetrievalScope.POLICIES,
                content=section['section_text'],
                content_hash=hashlib.sha256(section['section_text'].encode()).hexdigest(),
                title=section['policy_title'],
                section=section['section_title'],
                source_id=section['policy_code'],
            ))
        
        # Get control results relevant to this filing
        if filing_type == 'form_pf':
            relevant_categories = ['liquidity', 'leverage', 'counterparty']
        elif filing_type == '13f':
            relevant_categories = ['regulatory']
        else:
            relevant_categories = ['regulatory']
        
        for category in relevant_categories:
            category_results = self._get_recent_results_by_category(category, period_end)
            for result in category_results:
                structured.append(RetrievedDocument(
                    document_id=str(uuid.uuid4()),
                    source=RetrievalSource.STRUCTURED,
                    scope=RetrievalScope.CONTROL_RESULTS,
                    content=self._format_control_result(result),
                    content_hash=hashlib.sha256(json.dumps(result, default=str).encode()).hexdigest(),
                    title=f"Control: {result['control_code']}",
                    source_id=result['result_id'],
                ))
        
        duration_ms = int((time.time() - start_time) * 1000)
        
        context = RetrievalContext(
            context_id=context_id,
            query=query,
            query_hash=hashlib.sha256(query.encode()).hexdigest(),
            structured_results=structured,
            lexical_results=lexical,
            vector_results=vector,
            total_documents=len(structured) + len(lexical) + len(vector),
            retrieval_duration_ms=duration_ms,
        )
        
        self._log_retrieval(context)
        
        return context
    
    def search_policies(
        self,
        query: str,
        scope: Optional[str] = None,
        max_results: int = 5,
    ) -> List[RetrievedDocument]:
        """
        Search policies using hybrid approach.
        
        1. First try lexical search for exact matches
        2. Then augment with vector search for semantic matches
        """
        results = []
        
        # Lexical search
        lexical_results = self._lexical_policy_search(query, scope, max_results)
        results.extend(lexical_results)
        
        # Vector search (if available and needed)
        if self.vector_store and len(results) < max_results:
            remaining = max_results - len(results)
            vector_results = self._vector_search(query, limit=remaining)
            
            # Deduplicate
            existing_ids = {r.source_id for r in results}
            for doc in vector_results:
                if doc.source_id not in existing_ids:
                    results.append(doc)
        
        return results[:max_results]
    
    # =========================================================================
    # STRUCTURED RETRIEVAL (SQL)
    # =========================================================================
    
    def _get_run_summary(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get control run summary from Postgres."""
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
                cr.run_timestamp_start,
                cr.run_timestamp_end,
                cr.config_hash
            FROM control_runs cr
            WHERE cr.run_id = %(run_id)s
        """
        rows = self._execute_query(query, {"run_id": run_id})
        return rows[0] if rows else None
    
    def _get_failed_controls(self, run_id: str) -> List[Dict[str, Any]]:
        """Get failed control results for a run."""
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
                cr.breach_percentage,
                cr.evidence_row_count,
                cr.created_at
            FROM control_results cr
            JOIN control_definitions cd ON cr.control_id = cd.control_id
            WHERE cr.run_id = %(run_id)s
              AND cr.result_status = 'fail'
            ORDER BY cd.control_category, cd.control_code
        """
        return self._execute_query(query, {"run_id": run_id})
    
    def _get_exceptions_for_run(self, run_id: str) -> List[Dict[str, Any]]:
        """Get exceptions created for a run."""
        query = """
            SELECT 
                e.exception_id,
                e.exception_code,
                cd.control_code,
                cd.control_name,
                e.severity,
                e.title,
                e.description,
                e.status,
                e.due_date,
                e.opened_at
            FROM exceptions e
            JOIN control_definitions cd ON e.control_id = cd.control_id
            WHERE e.run_id = %(run_id)s
            ORDER BY e.severity, e.opened_at
        """
        return self._execute_query(query, {"run_id": run_id})
    
    def _get_exception_details(self, exception_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed exception information."""
        query = """
            SELECT 
                e.exception_id,
                e.exception_code,
                e.result_id,
                cd.control_code,
                cd.control_name,
                cd.control_category,
                e.severity,
                e.title,
                e.description,
                e.status,
                e.breach_value,
                e.threshold_value,
                e.breach_amount,
                e.due_date,
                e.opened_at,
                e.resolved_at,
                e.resolution_type,
                e.resolution_notes,
                s.full_name as assigned_to
            FROM exceptions e
            JOIN control_definitions cd ON e.control_id = cd.control_id
            LEFT JOIN signatories s ON e.assigned_to = s.signatory_id
            WHERE e.exception_id = %(exception_id)s
        """
        rows = self._execute_query(query, {"exception_id": exception_id})
        return rows[0] if rows else None
    
    def _get_control_result(self, result_id: str) -> Optional[Dict[str, Any]]:
        """Get control result details."""
        query = """
            SELECT 
                cr.result_id,
                cd.control_code,
                cd.control_name,
                cd.control_category,
                cd.description as control_description,
                cd.regulatory_reference,
                cr.calculated_value,
                cr.threshold_value,
                cr.threshold_operator,
                cr.result_status,
                cr.breach_amount,
                cr.evidence_row_count,
                cr.evidence_sample_json,
                cr.computation_sql,
                r.run_code,
                r.run_date,
                r.snowflake_snapshot_id
            FROM control_results cr
            JOIN control_definitions cd ON cr.control_id = cd.control_id
            JOIN control_runs r ON cr.run_id = r.run_id
            WHERE cr.result_id = %(result_id)s
        """
        rows = self._execute_query(query, {"result_id": result_id})
        return rows[0] if rows else None
    
    def _get_recent_results_by_category(
        self,
        category: str,
        as_of_date: date,
    ) -> List[Dict[str, Any]]:
        """Get recent control results by category."""
        query = """
            SELECT 
                cr.result_id,
                cd.control_code,
                cd.control_name,
                cr.calculated_value,
                cr.threshold_value,
                cr.result_status,
                r.run_date
            FROM control_results cr
            JOIN control_definitions cd ON cr.control_id = cd.control_id
            JOIN control_runs r ON cr.run_id = r.run_id
            WHERE cd.control_category = %(category)s
              AND r.run_date <= %(as_of_date)s
              AND r.status = 'completed'
            ORDER BY r.run_date DESC
            LIMIT 10
        """
        return self._execute_query(query, {"category": category, "as_of_date": as_of_date})
    
    def _get_prior_filing(
        self,
        filing_type: str,
        period_end: date,
    ) -> Optional[Dict[str, Any]]:
        """Get prior filing for reference."""
        # This would query a filings table
        # Placeholder implementation
        return None
    
    # =========================================================================
    # LEXICAL RETRIEVAL (Full-Text Search)
    # =========================================================================
    
    def _search_policies_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Search policies by control category."""
        query = """
            SELECT 
                pd.policy_code,
                pd.title as policy_title,
                ps.section_number,
                ps.section_title,
                ps.section_text,
                pd.effective_date
            FROM policy_documents pd
            JOIN policy_sections ps ON pd.policy_id = ps.policy_id
            WHERE pd.status = 'active'
              AND (
                  pd.category ILIKE %(category)s
                  OR ps.section_text ILIKE %(pattern)s
              )
            ORDER BY pd.effective_date DESC
            LIMIT 5
        """
        pattern = f"%{category}%"
        return self._execute_query(query, {"category": category, "pattern": pattern})
    
    def _search_policies_by_filing_type(self, filing_type: str) -> List[Dict[str, Any]]:
        """Search policies relevant to a filing type."""
        filing_keywords = {
            'form_pf': ['Form PF', 'liquidity', 'leverage', 'systemic risk'],
            '13f': ['13F', '13f', 'institutional', 'holdings', 'securities'],
            'adv': ['ADV', 'Form ADV', 'disclosure', 'brochure'],
        }
        
        keywords = filing_keywords.get(filing_type, [filing_type])
        
        query = """
            SELECT 
                pd.policy_code,
                pd.title as policy_title,
                ps.section_number,
                ps.section_title,
                ps.section_text,
                pd.effective_date
            FROM policy_documents pd
            JOIN policy_sections ps ON pd.policy_id = ps.policy_id
            WHERE pd.status = 'active'
              AND (
                  pd.title ILIKE ANY(%(patterns)s)
                  OR ps.section_text ILIKE ANY(%(patterns)s)
              )
            ORDER BY pd.effective_date DESC
            LIMIT 10
        """
        patterns = [f"%{kw}%" for kw in keywords]
        return self._execute_query(query, {"patterns": patterns})
    
    def _lexical_policy_search(
        self,
        query_text: str,
        scope: Optional[str],
        max_results: int,
    ) -> List[RetrievedDocument]:
        """Perform lexical search on policies."""
        # Clean and tokenize query
        tokens = re.findall(r'\b\w+\b', query_text.lower())
        
        sql = """
            SELECT 
                pd.policy_id,
                pd.policy_code,
                pd.title as policy_title,
                ps.section_id,
                ps.section_number,
                ps.section_title,
                ps.section_text,
                pd.effective_date,
                pd.category
            FROM policy_documents pd
            JOIN policy_sections ps ON pd.policy_id = ps.policy_id
            WHERE pd.status = 'active'
              AND ps.section_text ILIKE ANY(%(patterns)s)
        """
        
        if scope:
            sql += " AND pd.category = %(scope)s"
        
        sql += f" LIMIT {max_results}"
        
        patterns = [f"%{token}%" for token in tokens[:5]]  # Use first 5 tokens
        params = {"patterns": patterns}
        if scope:
            params["scope"] = scope
        
        rows = self._execute_query(sql, params)
        
        results = []
        for row in rows:
            results.append(RetrievedDocument(
                document_id=str(uuid.uuid4()),
                source=RetrievalSource.LEXICAL,
                scope=RetrievalScope.POLICIES,
                content=row['section_text'],
                content_hash=hashlib.sha256(row['section_text'].encode()).hexdigest(),
                title=row['policy_title'],
                section=f"{row['section_number']} - {row['section_title']}",
                source_id=row['policy_code'],
                effective_date=row['effective_date'],
                relevance_score=1.0,
                match_type="lexical",
            ))
        
        return results
    
    # =========================================================================
    # VECTOR RETRIEVAL (Semantic Search)
    # =========================================================================
    
    def _vector_search(
        self,
        query: str,
        limit: int = 3,
    ) -> List[RetrievedDocument]:
        """Perform vector similarity search."""
        if not self.vector_store:
            return []
        
        try:
            # This would use the actual vector store API
            # Placeholder implementation showing the interface
            results = self.vector_store.search(
                query=query,
                limit=limit,
                filter={"status": "active"},
            )
            
            documents = []
            for result in results:
                documents.append(RetrievedDocument(
                    document_id=str(uuid.uuid4()),
                    source=RetrievalSource.VECTOR,
                    scope=RetrievalScope.POLICIES,
                    content=result.get('content', ''),
                    content_hash=hashlib.sha256(result.get('content', '').encode()).hexdigest(),
                    title=result.get('title'),
                    section=result.get('section'),
                    source_id=result.get('policy_code'),
                    relevance_score=result.get('score', 0.0),
                    match_type="semantic",
                ))
            
            return documents
        except Exception as e:
            self.logger.warning(f"Vector search failed: {e}")
            return []
    
    # =========================================================================
    # FORMATTING HELPERS
    # =========================================================================
    
    def _format_run_summary(self, run: Dict[str, Any]) -> str:
        """Format run summary as text."""
        pass_rate = run['controls_passed'] / max(run['total_controls'], 1) * 100
        
        return (
            f"Control Run: {run['run_code']}\n"
            f"Date: {run['run_date']}\n"
            f"Data Snapshot: {run['snowflake_snapshot_id']}\n"
            f"Total Controls: {run['total_controls']}\n"
            f"Passed: {run['controls_passed']}\n"
            f"Failed: {run['controls_failed']}\n"
            f"Warnings: {run['controls_warning']}\n"
            f"Pass Rate: {pass_rate:.1f}%\n"
            f"Config Hash: {run['config_hash'][:16]}...\n"
        )
    
    def _format_control_result(self, result: Dict[str, Any]) -> str:
        """Format control result as text."""
        return (
            f"Control: {result['control_code']} - {result['control_name']}\n"
            f"Category: {result['control_category']}\n"
            f"Calculated Value: {result['calculated_value']}\n"
            f"Threshold: {result['threshold_operator']} {result['threshold_value']}\n"
            f"Breach Amount: {result.get('breach_amount', 'N/A')}\n"
            f"Evidence Rows: {result.get('evidence_row_count', 0)}\n"
        )
    
    def _format_control_result_detailed(self, result: Dict[str, Any]) -> str:
        """Format detailed control result."""
        text = self._format_control_result(result)
        text += f"\nDescription: {result.get('control_description', 'N/A')}\n"
        text += f"Regulatory Reference: {result.get('regulatory_reference', 'N/A')}\n"
        text += f"Run: {result.get('run_code', 'N/A')}\n"
        text += f"Snapshot: {result.get('snowflake_snapshot_id', 'N/A')}\n"
        return text
    
    def _format_exception(self, exc: Dict[str, Any]) -> str:
        """Format exception as text."""
        return (
            f"Exception: {exc['exception_code']}\n"
            f"Control: {exc['control_code']} - {exc['control_name']}\n"
            f"Severity: {exc['severity']}\n"
            f"Status: {exc['status']}\n"
            f"Title: {exc['title']}\n"
            f"Due Date: {exc.get('due_date', 'N/A')}\n"
        )
    
    def _format_exception_detailed(self, exc: Dict[str, Any]) -> str:
        """Format detailed exception."""
        text = self._format_exception(exc)
        text += f"\nDescription:\n{exc.get('description', 'N/A')}\n"
        text += f"\nBreach Value: {exc.get('breach_value', 'N/A')}\n"
        text += f"Threshold: {exc.get('threshold_value', 'N/A')}\n"
        text += f"Breach Amount: {exc.get('breach_amount', 'N/A')}\n"
        text += f"Assigned To: {exc.get('assigned_to', 'Unassigned')}\n"
        
        if exc.get('resolution_type'):
            text += f"\nResolution: {exc['resolution_type']}\n"
            text += f"Resolution Notes: {exc.get('resolution_notes', 'N/A')}\n"
        
        return text
    
    # =========================================================================
    # INTERNAL HELPERS
    # =========================================================================
    
    def _execute_query(self, query: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute a SQL query."""
        cursor = self.postgres.cursor()
        try:
            cursor.execute(query, params)
            if cursor.description:
                columns = [desc[0] for desc in cursor.description]
                return [dict(zip(columns, row)) for row in cursor.fetchall()]
            return []
        finally:
            cursor.close()
    
    def _log_retrieval(self, context: RetrievalContext) -> None:
        """Log retrieval operation for audit."""
        self.logger.info(
            f"Retrieval completed: id={context.context_id}, "
            f"query_hash={context.query_hash[:16]}, "
            f"docs={context.total_documents}, "
            f"duration_ms={context.retrieval_duration_ms}"
        )
