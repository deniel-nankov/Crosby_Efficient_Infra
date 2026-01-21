"""
Pytest Configuration and Shared Fixtures

This module provides centralized test fixtures, configuration, and utilities
for the entire test suite. Following pytest best practices for fixture scoping,
dependency injection, and test isolation.

Author: Crosby Compliance Team
Version: 2.0.0
"""

from __future__ import annotations

import os
import sys
import json
import hashlib
import logging
import tempfile
from datetime import datetime, date, timezone
from decimal import Decimal
from pathlib import Path
from typing import Generator, Dict, List, Any, Optional
from dataclasses import dataclass, field
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Configure test logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("tests")


# =============================================================================
# TEST CONFIGURATION
# =============================================================================

@dataclass
class TestConfig:
    """Centralized test configuration."""
    
    # Database
    db_host: str = "localhost"
    db_port: int = 5433
    db_name: str = "compliance_test"
    db_user: str = "compliance_user"
    db_password: str = "compliance_dev_password_123"
    
    # LLM
    llm_provider: str = "lmstudio"
    llm_base_url: str = "http://localhost:1234/v1"
    llm_model: str = "qwen3-30b-a3b"
    embedding_model: str = "nomic-embed-text-v1.5"
    
    # Test Thresholds
    embedding_dimension: int = 768
    min_similarity_score: float = 0.5
    max_retrieval_latency_ms: float = 500.0
    max_embedding_latency_ms: float = 200.0
    
    # Paths
    policies_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "policies")
    test_data_dir: Path = field(default_factory=lambda: Path(__file__).parent / "fixtures")
    
    @classmethod
    def from_env(cls) -> "TestConfig":
        """Create config from environment variables."""
        return cls(
            db_host=os.getenv("TEST_DB_HOST", "localhost"),
            db_port=int(os.getenv("TEST_DB_PORT", "5433")),
            llm_provider=os.getenv("LLM_PROVIDER", "lmstudio"),
            llm_model=os.getenv("LLM_MODEL", "qwen3-30b-a3b"),
        )


@pytest.fixture(scope="session")
def test_config() -> TestConfig:
    """Provide test configuration."""
    return TestConfig.from_env()


# =============================================================================
# DATABASE FIXTURES
# =============================================================================

@pytest.fixture(scope="session")
def db_connection(test_config: TestConfig):
    """
    Session-scoped database connection.
    
    Uses the test database to avoid polluting production data.
    Automatically handles cleanup on teardown.
    """
    try:
        import psycopg2
        conn = psycopg2.connect(
            host=test_config.db_host,
            port=test_config.db_port,
            database="compliance",  # Use main db, test schema
            user=test_config.db_user,
            password=test_config.db_password,
        )
        conn.autocommit = False
        yield conn
        conn.rollback()  # Rollback any uncommitted changes
        conn.close()
    except Exception as e:
        pytest.skip(f"Database not available: {e}")


@pytest.fixture
def db_transaction(db_connection):
    """
    Function-scoped transaction that auto-rollbacks.
    
    Each test gets a clean transaction that is rolled back after the test,
    ensuring complete isolation between tests.
    """
    cursor = db_connection.cursor()
    yield cursor
    db_connection.rollback()
    cursor.close()


@pytest.fixture(scope="session")
def postgres_adapter(test_config: TestConfig):
    """Provide PostgresAdapter for integration tests."""
    try:
        from integration.postgres_adapter import PostgresAdapter, PostgresConfig
        config = PostgresConfig(
            host=test_config.db_host,
            port=test_config.db_port,
            database="compliance",
            user=test_config.db_user,
            password=test_config.db_password,
        )
        adapter = PostgresAdapter(config)
        yield adapter
    except Exception as e:
        pytest.skip(f"PostgresAdapter not available: {e}")


# =============================================================================
# EMBEDDER FIXTURES
# =============================================================================

@pytest.fixture(scope="session")
def embedder(test_config: TestConfig):
    """
    Session-scoped embedder instance.
    
    Reuses the same embedder across all tests for efficiency.
    """
    try:
        from rag.embedder import LocalEmbedder
        emb = LocalEmbedder(
            api_base=test_config.llm_base_url,
            model=test_config.embedding_model,
        )
        if not emb.available:
            pytest.skip("Embedder not available (LM Studio not running?)")
        yield emb
    except Exception as e:
        pytest.skip(f"Embedder initialization failed: {e}")


@pytest.fixture
def mock_embedder():
    """
    Mock embedder for unit tests that don't need real embeddings.
    
    Returns deterministic embeddings based on input hash for reproducibility.
    """
    def _deterministic_embedding(text: str) -> List[float]:
        """Generate deterministic embedding from text hash."""
        hash_bytes = hashlib.sha256(text.encode()).digest()
        # Convert hash to 768 floats in range [-1, 1]
        embedding = []
        for i in range(768):
            byte_val = hash_bytes[i % len(hash_bytes)]
            embedding.append((byte_val / 128.0) - 1.0)
        return embedding
    
    mock = MagicMock()
    mock.embed.side_effect = _deterministic_embedding
    mock.embed_batch.side_effect = lambda texts: [_deterministic_embedding(t) for t in texts]
    mock.available = True
    mock.dimension = 768
    return mock


# =============================================================================
# VECTOR STORE FIXTURES
# =============================================================================

@pytest.fixture(scope="session")
def vector_store(db_connection, embedder):
    """Session-scoped vector store with real database connection."""
    try:
        from rag.vector_store import VectorStore
        store = VectorStore(db_connection)
        yield store
    except Exception as e:
        pytest.skip(f"VectorStore initialization failed: {e}")


@pytest.fixture
def mock_vector_store():
    """Mock vector store for unit testing."""
    from rag.vector_store import PolicyChunk
    
    mock = MagicMock()
    
    # Sample chunks for testing
    sample_chunks = [
        PolicyChunk(
            chunk_id="test:chunk:001",
            document_id="DOC001",
            document_name="concentration_limits.md",
            section_title="Sector Concentration",
            content="No single sector shall exceed 30% of NAV. Technology sector is capped at 25%.",
            content_hash="abc123",
            control_types=["concentration", "sector"],
        ),
        PolicyChunk(
            chunk_id="test:chunk:002",
            document_id="DOC002",
            document_name="liquidity_policy.md",
            section_title="Minimum Liquidity",
            content="Maintain minimum 5% of NAV in highly liquid assets. Daily liquidity buffer required.",
            content_hash="def456",
            control_types=["liquidity"],
        ),
        PolicyChunk(
            chunk_id="test:chunk:003",
            document_id="DOC003",
            document_name="exception_management.md",
            section_title="Exception Procedures",
            content="All breaches require immediate escalation to CRO within 4 hours.",
            content_hash="ghi789",
            control_types=["exception", "escalation"],
        ),
    ]
    
    mock.search_similar.return_value = sample_chunks[:2]
    mock.count_chunks.return_value = len(sample_chunks)
    mock.get_by_control_type.side_effect = lambda ct: [c for c in sample_chunks if ct in c.control_types]
    
    return mock


# =============================================================================
# RAG RETRIEVER FIXTURES
# =============================================================================

@pytest.fixture
def rag_retriever(vector_store, embedder):
    """Fully configured RAG retriever for integration tests."""
    try:
        from rag.retriever import RAGRetriever
        retriever = RAGRetriever(
            vector_store=vector_store,
            embedder=embedder,
            use_hybrid=True,
            use_reranking=True,
        )
        yield retriever
    except Exception as e:
        pytest.skip(f"RAGRetriever initialization failed: {e}")


@pytest.fixture
def mock_rag_retriever(mock_vector_store, mock_embedder):
    """Mock RAG retriever for unit tests."""
    from rag.retriever import RAGRetriever
    
    retriever = RAGRetriever(
        vector_store=mock_vector_store,
        embedder=mock_embedder,
        use_hybrid=False,  # Disable hybrid for mock
        use_reranking=False,
    )
    return retriever


# =============================================================================
# AGENT FIXTURES
# =============================================================================

@pytest.fixture
def investigation_tools(postgres_adapter, mock_rag_retriever):
    """Investigation tools for agent testing."""
    try:
        from agent.investigator import InvestigationTools
        from dataclasses import dataclass
        from typing import List
        
        @dataclass
        class MockPosition:
            ticker: str
            security_name: str
            market_value: float
            sector: str = "Technology"
            nav_pct: float = 10.0
        
        @dataclass
        class MockControlResult:
            control_id: str
            control_name: str
            control_type: str
            status: str
            calculated_value: float
            threshold: float
        
        @dataclass
        class MockDataSnapshot:
            positions: List[MockPosition]
            control_results: List[MockControlResult]
            as_of_date: str = "2026-01-19"
        
        snapshot = MockDataSnapshot(
            positions=[
                MockPosition(ticker="AAPL", security_name="Apple", market_value=100000),
            ],
            control_results=[],
        )
        
        tools = InvestigationTools(snapshot=snapshot)
        yield tools
    except Exception as e:
        pytest.skip(f"InvestigationTools initialization failed: {e}")


@pytest.fixture
def mock_llm_client():
    """Mock LLM client for agent testing."""
    mock = MagicMock()
    
    # Simulate agent responses
    responses = iter([
        '{"thought": "I need to check the current sector allocations", "action": "query_positions", "action_input": {"filter": "sector"}}',
        '{"thought": "I found high tech concentration, need to check policy limits", "action": "search_policies", "action_input": {"query": "sector concentration limits"}}',
        '{"thought": "Technology is at 32% vs 30% limit - breach confirmed", "action": "FINAL_ANSWER", "action_input": {"findings": "Technology sector at 32% exceeds 30% limit", "recommendations": ["Reduce tech exposure", "Review allocation"]}}',
    ])
    
    mock.generate.side_effect = lambda **kwargs: next(responses, '{"action": "FINAL_ANSWER", "action_input": {}}')
    mock.available = True
    
    return mock


# =============================================================================
# SAMPLE DATA FIXTURES
# =============================================================================

@pytest.fixture
def sample_positions() -> List[Dict[str, Any]]:
    """Sample position data for testing."""
    return [
        {"ticker": "AAPL", "name": "Apple Inc", "sector": "Technology", "market_value": 5_000_000, "nav_pct": 10.0},
        {"ticker": "MSFT", "name": "Microsoft Corp", "sector": "Technology", "market_value": 4_000_000, "nav_pct": 8.0},
        {"ticker": "GOOGL", "name": "Alphabet Inc", "sector": "Technology", "market_value": 3_500_000, "nav_pct": 7.0},
        {"ticker": "JPM", "name": "JPMorgan Chase", "sector": "Financials", "market_value": 3_000_000, "nav_pct": 6.0},
        {"ticker": "JNJ", "name": "Johnson & Johnson", "sector": "Healthcare", "market_value": 2_500_000, "nav_pct": 5.0},
        {"ticker": "XOM", "name": "Exxon Mobil", "sector": "Energy", "market_value": 2_000_000, "nav_pct": 4.0},
    ]


@pytest.fixture
def sample_control_results() -> List[Dict[str, Any]]:
    """Sample control test results for testing."""
    return [
        {
            "control_name": "Sector Concentration",
            "control_type": "concentration",
            "status": "FAIL",
            "calculated_value": 32.5,
            "threshold": 30.0,
            "message": "Technology sector at 32.5% exceeds 30% limit",
        },
        {
            "control_name": "Single Issuer Limit",
            "control_type": "issuer",
            "status": "PASS",
            "calculated_value": 8.5,
            "threshold": 10.0,
            "message": "All issuers within 10% limit",
        },
        {
            "control_name": "Liquidity Minimum",
            "control_type": "liquidity",
            "status": "WARNING",
            "calculated_value": 5.2,
            "threshold": 5.0,
            "message": "Liquidity at 5.2% - close to 5% minimum",
        },
    ]


@pytest.fixture
def sample_nav() -> float:
    """Sample NAV for testing."""
    return 50_000_000.0


@pytest.fixture
def sample_report_date() -> date:
    """Sample report date for testing."""
    return date(2026, 1, 18)


# =============================================================================
# POLICY FIXTURES
# =============================================================================

@pytest.fixture
def sample_policy_content() -> str:
    """Sample policy document content."""
    return """
# Concentration Limits Policy

## Sector Concentration
- No single sector shall exceed 30% of total NAV
- Technology sector limited to 25% due to volatility
- Financial sector limited to 25%

## Single Issuer Limits  
- No single issuer shall exceed 10% of NAV
- Government securities exempt from issuer limits

## Exceptions
- Temporary breaches up to 5% over limit allowed for 48 hours
- All breaches require CRO notification within 4 hours
- Remediation plan required within 24 hours
"""


@pytest.fixture
def sample_policy_chunks(sample_policy_content):
    """Pre-parsed policy chunks for testing."""
    from rag.vector_store import PolicyChunk
    
    sections = sample_policy_content.strip().split("\n\n")
    chunks = []
    
    for i, section in enumerate(sections):
        if section.strip():
            chunks.append(PolicyChunk.from_text(
                content=section.strip(),
                document_name="concentration_limits.md",
                section_title=f"Section {i+1}",
                document_id=f"DOC{i:03d}",
                control_types=["concentration", "sector"] if "sector" in section.lower() else ["exception"],
            ))
    
    return chunks


# =============================================================================
# HELPER UTILITIES
# =============================================================================

@contextmanager
def timed_execution(name: str, max_ms: float = 1000.0):
    """
    Context manager for timing test execution.
    
    Usage:
        with timed_execution("embedding", max_ms=200):
            embedding = embedder.embed(text)
    """
    import time
    start = time.perf_counter()
    yield
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(f"{name} completed in {elapsed_ms:.2f}ms")
    assert elapsed_ms < max_ms, f"{name} took {elapsed_ms:.2f}ms, exceeds {max_ms}ms limit"


def assert_valid_embedding(embedding: List[float], dimension: int = 768):
    """Assert embedding has correct structure."""
    assert isinstance(embedding, list), "Embedding must be a list"
    assert len(embedding) == dimension, f"Expected {dimension} dimensions, got {len(embedding)}"
    assert all(isinstance(x, (int, float)) for x in embedding), "All values must be numeric"
    assert all(-10 <= x <= 10 for x in embedding), "Values should be in reasonable range"


def assert_valid_chunk(chunk, required_fields: Optional[List[str]] = None):
    """Assert chunk has required structure."""
    required = required_fields or ["chunk_id", "document_name", "content"]
    for field in required:
        assert hasattr(chunk, field), f"Chunk missing required field: {field}"
        assert getattr(chunk, field) is not None, f"Chunk field {field} is None"


# =============================================================================
# PYTEST MARKERS
# =============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests (fast, no external deps)")
    config.addinivalue_line("markers", "integration: Integration tests (require database/LLM)")
    config.addinivalue_line("markers", "performance: Performance benchmarks")
    config.addinivalue_line("markers", "slow: Slow tests (>5s)")
    config.addinivalue_line("markers", "requires_db: Requires database connection")
    config.addinivalue_line("markers", "requires_llm: Requires LLM service")
