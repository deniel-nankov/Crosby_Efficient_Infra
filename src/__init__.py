"""
Compliance RAG System for SEC-Registered Hedge Funds

A production-grade RAG system for automated compliance documentation with:
- Deterministic control execution (no LLM for calculations)
- Hybrid retrieval (SQL-first, lexical, then vector)
- LLM-assisted narrative generation with citations
- Immutable audit trail for SEC examination

Usage:
    from compliance_rag import ComplianceOrchestrator
    
    orchestrator = ComplianceOrchestrator(
        postgres_connection=pg_conn,
        snowflake_connection=sf_conn,
        llm_client=openai_client,
    )
    
    result = orchestrator.run_daily_compliance(
        run_date=date.today(),
        fund_ids=["FUND-001", "FUND-002"],
    )
"""

from .config import (
    Settings,
    get_settings,
    get_test_settings,
    Environment,
)

from .control_runner import (
    ControlRunner,
    ControlRunContext,
    ControlDefinition,
    ControlCategory,
    ControlResultStatus,
    get_all_controls,
    get_active_controls,
)

from .evidence_store import (
    EvidenceStore,
    ControlResultEvidence,
    ExceptionEvidence,
    DailyComplianceSummary,
)

from .retrieval import (
    HybridRetriever,
    RetrievalContext,
    RetrievedDocument,
    RetrievalSource,
)

from .narrative import (
    NarrativeGenerator,
    GeneratedNarrative,
    NarrativeType,
)

from .document_builder import (
    DocumentBuilder,
    GeneratedDocument,
    DocumentType,
    SectionType,
)

from .orchestrator import (
    ComplianceOrchestrator,
    ComplianceRunResult,
)

__version__ = "1.0.0"

__all__ = [
    # Config
    "Settings",
    "get_settings",
    "get_test_settings",
    "Environment",
    # Control Runner
    "ControlRunner",
    "ControlRunContext",
    "ControlDefinition",
    "ControlCategory",
    "ControlResultStatus",
    "get_all_controls",
    "get_active_controls",
    # Evidence Store
    "EvidenceStore",
    "ControlResultEvidence",
    "ExceptionEvidence",
    "DailyComplianceSummary",
    # Retrieval
    "HybridRetriever",
    "RetrievalContext",
    "RetrievedDocument",
    "RetrievalSource",
    # Narrative
    "NarrativeGenerator",
    "GeneratedNarrative",
    "NarrativeType",
    # Document Builder
    "DocumentBuilder",
    "GeneratedDocument",
    "DocumentType",
    "SectionType",
    # Orchestrator
    "ComplianceOrchestrator",
    "ComplianceRunResult",
]
