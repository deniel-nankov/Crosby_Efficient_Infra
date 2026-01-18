"""
Integration Package - Connect to Client Systems

Simplified adapters that trust client data and focus on AI value-add.
"""

from .client_adapter import (
    Position,
    ControlResult,
    DataSnapshot,
    ClientSystemAdapter,
    MockAdapter,
    CSVAdapter,
    DatabaseAdapter,
    get_adapter,
)

from .rag_pipeline import (
    PolicyContext,
    GeneratedNarrative,
    ComplianceReport,
    ComplianceRAGPipeline,
)

from .llm_config import (
    LLMProvider,
    LLMConfig,
    LLMClient,
    ComplianceLLM,
    DataAnonymizer,
    create_llm_client,
    get_compliance_llm,
)

__all__ = [
    # Data models
    'Position',
    'ControlResult',
    'DataSnapshot',
    # Adapters
    'ClientSystemAdapter',
    'MockAdapter',
    'CSVAdapter',
    'DatabaseAdapter',
    'get_adapter',
    # RAG Pipeline
    'PolicyContext',
    'GeneratedNarrative',
    'ComplianceReport',
    'ComplianceRAGPipeline',
    # LLM Configuration
    'LLMProvider',
    'LLMConfig',
    'LLMClient',
    'ComplianceLLM',
    'DataAnonymizer',
    'create_llm_client',
    'get_compliance_llm',
]
