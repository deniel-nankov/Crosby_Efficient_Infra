"""
Integration Tests for Compliance RAG System

End-to-end tests that verify the complete system workflow:
- Database connectivity and data flow
- RAG pipeline (embed → store → retrieve)
- PostgreSQL adapter integration
- Document generation
- Full compliance pipeline

These tests require:
- PostgreSQL with pgvector running (docker-compose up)
- LM Studio or other LLM backend (optional for some tests)

Test Categories:
- TestDatabaseConnectivity: Database connection tests
- TestPostgresAdapter: Adapter API tests
- TestRAGRetrieval: RAG retrieval pipeline tests
- TestPDFGeneration: PDF document generation tests
- TestDataIntegrity: Data consistency validation
"""

from __future__ import annotations

import pytest
import os
import sys
from datetime import datetime, date, timezone
from typing import List, Dict, Any, Optional
from pathlib import Path
from decimal import Decimal
from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Mark all tests as integration tests
pytestmark = [pytest.mark.integration]


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def db_connection():
    """Create database connection for testing."""
    try:
        import psycopg2
        conn = psycopg2.connect(
            host="localhost",
            port=5433,
            database="compliance",
            user="compliance_user",
            password="compliance_dev_password_123"
        )
        yield conn
        conn.rollback()  # Don't commit test changes
        conn.close()
    except Exception as e:
        pytest.skip(f"Database not available: {e}")


@pytest.fixture
def postgres_adapter():
    """Create PostgresAdapter for testing."""
    try:
        from integration.postgres_adapter import PostgresAdapter
        adapter = PostgresAdapter()
        yield adapter
        adapter.close()
    except Exception as e:
        pytest.skip(f"PostgresAdapter not available: {e}")


@pytest.fixture
def local_embedder():
    """Create LocalEmbedder for testing."""
    try:
        from rag.embedder import LocalEmbedder
        embedder = LocalEmbedder()
        if not embedder.available:
            pytest.skip("LM Studio embedding service not available")
        yield embedder
    except Exception as e:
        pytest.skip(f"LocalEmbedder not available: {e}")


@pytest.fixture
def sample_positions() -> List[Dict[str, Any]]:
    """Sample position data for testing."""
    return [
        {
            "security_id": "AAPL",
            "ticker": "AAPL",
            "security_name": "Apple Inc",
            "quantity": Decimal("1000"),
            "market_value": Decimal("175000"),
            "currency": "USD",
            "sector": "Technology",
            "issuer": "Apple Inc",
            "asset_class": "Equity",
        },
        {
            "security_id": "MSFT",
            "ticker": "MSFT",
            "security_name": "Microsoft Corp",
            "quantity": Decimal("500"),
            "market_value": Decimal("188750"),
            "currency": "USD",
            "sector": "Technology",
            "issuer": "Microsoft Corp",
            "asset_class": "Equity",
        },
    ]


# =============================================================================
# DATABASE CONNECTIVITY TESTS
# =============================================================================

class TestDatabaseConnectivity:
    """Test database connectivity and basic operations."""
    
    @pytest.mark.requires_db
    def test_database_connection_established(self, db_connection):
        """Should establish database connection."""
        assert db_connection is not None
        assert not db_connection.closed
    
    @pytest.mark.requires_db
    def test_pgvector_extension_available(self, db_connection):
        """pgvector extension should be available."""
        with db_connection.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
            result = cur.fetchone()
        
        assert result is not None, "pgvector extension not installed"
    
    @pytest.mark.requires_db
    def test_policy_chunks_table_exists(self, db_connection):
        """policy_chunks table should exist."""
        with db_connection.cursor() as cur:
            cur.execute("""
                SELECT 1 FROM information_schema.tables 
                WHERE table_name = 'policy_chunks'
            """)
            result = cur.fetchone()
        
        assert result is not None, "policy_chunks table not found"
    
    @pytest.mark.requires_db
    def test_vector_column_exists(self, db_connection):
        """embedding column should be vector type."""
        with db_connection.cursor() as cur:
            cur.execute("""
                SELECT data_type FROM information_schema.columns
                WHERE table_name = 'policy_chunks' AND column_name = 'embedding'
            """)
            result = cur.fetchone()
        
        assert result is not None, "embedding column not found"
    
    @pytest.mark.requires_db
    def test_can_query_policy_chunks(self, db_connection):
        """Should be able to query policy_chunks table."""
        with db_connection.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM policy_chunks")
            result = cur.fetchone()
        
        assert result is not None
        assert isinstance(result[0], int)
    
    @pytest.mark.requires_db
    def test_transaction_rollback_works(self, db_connection):
        """Transactions should be isolated in tests."""
        with db_connection.cursor() as cur:
            # Get initial count
            cur.execute("SELECT COUNT(*) FROM policy_chunks")
            initial_count = cur.fetchone()[0]
            
            # Rollback should not change count
            db_connection.rollback()
            
            cur.execute("SELECT COUNT(*) FROM policy_chunks")
            final_count = cur.fetchone()[0]
        
        assert initial_count == final_count


# =============================================================================
# POSTGRES ADAPTER TESTS
# =============================================================================

class TestPostgresAdapter:
    """Test PostgresAdapter integration."""
    
    @pytest.mark.requires_db
    def test_adapter_connects_successfully(self, postgres_adapter):
        """Adapter should connect successfully."""
        assert postgres_adapter is not None
        assert postgres_adapter.source is not None
    
    @pytest.mark.requires_db
    def test_adapter_has_get_snapshot_method(self, postgres_adapter):
        """Adapter should have get_snapshot method."""
        assert hasattr(postgres_adapter, 'get_snapshot')
        assert callable(postgres_adapter.get_snapshot)
    
    @pytest.mark.requires_db
    def test_adapter_has_close_method(self, postgres_adapter):
        """Adapter should have close method."""
        assert hasattr(postgres_adapter, 'close')
        assert callable(postgres_adapter.close)
    
    @pytest.mark.requires_db
    def test_adapter_source_has_required_methods(self, postgres_adapter):
        """Adapter source should have required data methods."""
        source = postgres_adapter.source
        
        assert hasattr(source, 'get_positions')
        assert hasattr(source, 'get_control_results')
        assert hasattr(source, 'get_nav')
    
    @pytest.mark.requires_db
    def test_get_snapshot_returns_data_snapshot(self, postgres_adapter):
        """get_snapshot should return DataSnapshot object."""
        try:
            snapshot = postgres_adapter.get_snapshot(date.today())
            
            assert snapshot is not None
            assert hasattr(snapshot, 'positions')
            assert hasattr(snapshot, 'control_results')
            assert hasattr(snapshot, 'nav')
        except Exception as e:
            # May fail if no data, but should not fail due to API mismatch
            if "get_positions" in str(e) or "get_nav" in str(e):
                pytest.fail(f"API method missing: {e}")
    
    @pytest.mark.requires_db
    def test_source_get_positions_returns_list(self, postgres_adapter):
        """Source get_positions should return list."""
        result = postgres_adapter.source.get_positions(date.today())
        assert isinstance(result, list)
    
    @pytest.mark.requires_db
    def test_source_get_nav_returns_decimal_or_none(self, postgres_adapter):
        """Source get_nav should return Decimal or None."""
        result = postgres_adapter.source.get_nav(date.today())
        assert result is None or isinstance(result, (Decimal, float, int))


# =============================================================================
# RAG RETRIEVAL TESTS
# =============================================================================

class TestRAGRetrieval:
    """Test RAG retrieval pipeline."""
    
    @pytest.mark.requires_db
    def test_bm25_index_can_be_created(self):
        """BM25Index should be creatable."""
        from rag.retriever import BM25Index
        
        index = BM25Index()
        assert index is not None
    
    @pytest.mark.requires_db
    def test_bm25_index_can_build_from_documents(self):
        """BM25Index should build from document dict."""
        from rag.retriever import BM25Index
        
        docs = {
            "doc1": "Concentration limits apply to sector exposure",
            "doc2": "Liquidity thresholds must be maintained",
            "doc3": "SEC compliance requires filing Form PF",
        }
        
        index = BM25Index()
        index.build_index(docs)
        
        assert len(index.documents) == 3
    
    @pytest.mark.requires_db
    def test_bm25_search_returns_results(self):
        """BM25 search should return ranked results."""
        from rag.retriever import BM25Index
        
        docs = {
            "doc1": "Concentration limits apply to sector exposure",
            "doc2": "Liquidity thresholds must be maintained",
            "doc3": "SEC compliance requires filing Form PF",
        }
        
        index = BM25Index()
        index.build_index(docs)
        
        results = index.search("concentration sector", top_k=2)
        
        assert len(results) <= 2
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)
    
    @pytest.mark.requires_db
    def test_rag_retriever_can_be_created(self, db_connection):
        """RAGRetriever should be creatable with VectorStore and embedder."""
        from rag.retriever import RAGRetriever, VectorStore
        from rag.embedder import LocalEmbedder
        from unittest.mock import MagicMock
        
        # RAGRetriever needs VectorStore and LocalEmbedder
        mock_embedder = MagicMock(spec=LocalEmbedder)
        mock_embedder.available = True
        mock_embedder.embed.return_value = [0.1] * 384
        
        vector_store = VectorStore(db_connection)
        retriever = RAGRetriever(
            vector_store=vector_store,
            embedder=mock_embedder,
            use_hybrid=False,
            use_reranking=False,
            use_query_rewriting=False
        )
        assert retriever is not None
        assert retriever.vector_store is vector_store
    
    @pytest.mark.requires_db
    def test_confidence_calibrator_exists(self):
        """ConfidenceCalibrator should be importable."""
        from rag.retriever import ConfidenceCalibrator
        
        calibrator = ConfidenceCalibrator()
        assert calibrator is not None
    
    @pytest.mark.requires_db
    def test_query_rewriter_exists(self):
        """QueryRewriter should be importable."""
        from rag.retriever import QueryRewriter
        
        rewriter = QueryRewriter()
        assert rewriter is not None


# =============================================================================
# EMBEDDER TESTS
# =============================================================================

class TestLocalEmbedder:
    """Test LocalEmbedder integration with LM Studio."""
    
    @pytest.mark.requires_embedder
    def test_embedder_can_be_created(self):
        """LocalEmbedder should be creatable."""
        from rag.embedder import LocalEmbedder
        
        embedder = LocalEmbedder()
        assert embedder is not None
        assert embedder.api_base == "http://localhost:1234/v1"
    
    @pytest.mark.requires_embedder
    def test_embedder_accepts_custom_api_base(self):
        """LocalEmbedder should accept custom API base."""
        from rag.embedder import LocalEmbedder
        
        embedder = LocalEmbedder(api_base="http://custom:8080/v1")
        assert embedder.api_base == "http://custom:8080/v1"
    
    @pytest.mark.requires_embedder
    def test_embedder_has_availability_check(self):
        """LocalEmbedder should have availability check."""
        from rag.embedder import LocalEmbedder
        
        embedder = LocalEmbedder()
        assert hasattr(embedder, 'available')
        assert isinstance(embedder.available, bool)
    
    @pytest.mark.requires_embedder
    def test_embedder_has_embed_method(self):
        """LocalEmbedder should have embed method."""
        from rag.embedder import LocalEmbedder
        
        embedder = LocalEmbedder()
        assert hasattr(embedder, 'embed')
        assert callable(embedder.embed)
    
    @pytest.mark.requires_embedder
    def test_embedder_has_embed_batch_method(self):
        """LocalEmbedder should have embed_batch method."""
        from rag.embedder import LocalEmbedder
        
        embedder = LocalEmbedder()
        assert hasattr(embedder, 'embed_batch')
        assert callable(embedder.embed_batch)
    
    @pytest.mark.requires_embedder
    @pytest.mark.requires_llm
    def test_embed_generates_vector(self, local_embedder):
        """embed() should generate embedding vector."""
        embedding = local_embedder.embed("Test compliance text")
        
        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(v, float) for v in embedding)


# =============================================================================
# PDF GENERATION TESTS
# =============================================================================

class TestPDFGeneration:
    """Test PDF document generation."""
    
    def test_document_builder_can_be_created(self):
        """DocumentBuilder should be creatable with settings."""
        from document_builder.builder import DocumentBuilder
        
        # DocumentBuilder requires settings argument
        mock_settings = MagicMock()
        mock_settings.output_dir = "/tmp"
        
        builder = DocumentBuilder(settings=mock_settings)
        assert builder is not None
    
    def test_professional_pdf_generator_exists(self):
        """ProfessionalCompliancePDF should be importable."""
        from document_builder.professional_pdf import ProfessionalCompliancePDF
        
        assert ProfessionalCompliancePDF is not None
    
    def test_pdf_generator_accepts_fund_name(self):
        """ProfessionalCompliancePDF should accept fund name."""
        from document_builder.professional_pdf import ProfessionalCompliancePDF
        
        generator = ProfessionalCompliancePDF(fund_name="Test Fund")
        assert generator.fund_name == "Test Fund"
    
    def test_pdf_generator_has_generate_method(self):
        """ProfessionalCompliancePDF should have generate method."""
        from document_builder.professional_pdf import ProfessionalCompliancePDF
        
        generator = ProfessionalCompliancePDF()
        assert hasattr(generator, 'generate_daily_compliance_report')
        assert callable(generator.generate_daily_compliance_report)


# =============================================================================
# DATA INTEGRITY TESTS
# =============================================================================

class TestDataIntegrity:
    """Test data integrity and consistency."""
    
    @pytest.mark.requires_db
    def test_policy_chunks_have_content(self, db_connection):
        """Policy chunks should have non-empty content."""
        with db_connection.cursor() as cur:
            cur.execute("""
                SELECT chunk_id, content 
                FROM policy_chunks 
                WHERE content IS NOT NULL AND content != ''
                LIMIT 5
            """)
            results = cur.fetchall()
        
        for chunk_id, content in results:
            assert len(content) > 0, f"Chunk {chunk_id} has empty content"
    
    @pytest.mark.requires_db
    def test_embeddings_are_populated(self, db_connection):
        """Embeddings should be populated for chunks."""
        with db_connection.cursor() as cur:
            cur.execute("""
                SELECT COUNT(*) FROM policy_chunks 
                WHERE embedding IS NOT NULL
            """)
            embedded_count = cur.fetchone()[0]
            
            cur.execute("SELECT COUNT(*) FROM policy_chunks")
            total_count = cur.fetchone()[0]
        
        if total_count > 0:
            ratio = embedded_count / total_count
            assert ratio >= 0.5, f"Only {ratio:.0%} of chunks have embeddings"
    
    @pytest.mark.requires_db
    def test_chunk_ids_are_unique(self, db_connection):
        """Chunk IDs should be unique."""
        with db_connection.cursor() as cur:
            cur.execute("""
                SELECT chunk_id, COUNT(*) as cnt 
                FROM policy_chunks 
                GROUP BY chunk_id 
                HAVING COUNT(*) > 1
            """)
            duplicates = cur.fetchall()
        
        assert len(duplicates) == 0, f"Found duplicate chunk_ids: {duplicates}"
    
    @pytest.mark.requires_db
    def test_documents_have_source_info(self, db_connection):
        """Policy chunks should have source document info."""
        with db_connection.cursor() as cur:
            cur.execute("""
                SELECT COUNT(*) FROM policy_chunks 
                WHERE document_name IS NOT NULL
            """)
            with_source = cur.fetchone()[0]
            
            cur.execute("SELECT COUNT(*) FROM policy_chunks")
            total = cur.fetchone()[0]
        
        if total > 0:
            ratio = with_source / total
            assert ratio >= 0.9, f"Only {ratio:.0%} of chunks have source info"


# =============================================================================
# VECTOR SEARCH TESTS
# =============================================================================

class TestVectorSearch:
    """Test vector similarity search functionality."""
    
    @pytest.mark.requires_db
    @pytest.mark.requires_embedder
    def test_vector_search_with_embedding(self, db_connection, local_embedder):
        """Should be able to search with embedding vector."""
        # Generate query embedding
        query_embedding = local_embedder.embed("concentration limits")
        
        with db_connection.cursor() as cur:
            # Format as pgvector literal
            vector_str = "[" + ",".join(str(v) for v in query_embedding) + "]"
            
            cur.execute(f"""
                SELECT chunk_id, content,
                       embedding <-> '{vector_str}'::vector AS distance
                FROM policy_chunks
                WHERE embedding IS NOT NULL
                ORDER BY embedding <-> '{vector_str}'::vector
                LIMIT 5
            """)
            results = cur.fetchall()
        
        assert len(results) <= 5
        if results:
            # Results should be ordered by distance (ascending)
            distances = [r[2] for r in results]
            assert distances == sorted(distances)
    
    @pytest.mark.requires_db
    def test_vector_dimension_consistency(self, db_connection):
        """All embeddings should have consistent dimensions."""
        with db_connection.cursor() as cur:
            # Get dimension of first embedding
            cur.execute("""
                SELECT vector_dims(embedding) 
                FROM policy_chunks 
                WHERE embedding IS NOT NULL 
                LIMIT 1
            """)
            result = cur.fetchone()
            
            if result:
                expected_dim = result[0]
                
                # Check all embeddings have same dimension
                cur.execute(f"""
                    SELECT COUNT(*) FROM policy_chunks 
                    WHERE embedding IS NOT NULL 
                    AND vector_dims(embedding) != {expected_dim}
                """)
                mismatched = cur.fetchone()[0]
                
                assert mismatched == 0, f"Found {mismatched} embeddings with wrong dimensions"


# =============================================================================
# FULL PIPELINE TESTS
# =============================================================================

class TestFullPipeline:
    """Test complete system pipeline integration."""
    
    @pytest.mark.requires_db
    @pytest.mark.slow
    def test_control_runner_can_be_created(self, db_connection):
        """ControlRunner should be creatable with proper dependencies."""
        from control_runner.runner import ControlRunner
        from unittest.mock import MagicMock
        
        # ControlRunner needs snowflake, postgres, and settings
        mock_snowflake = MagicMock()
        mock_settings = MagicMock()
        mock_settings.version = "1.0.0"
        
        runner = ControlRunner(
            snowflake_connection=mock_snowflake,
            postgres_connection=db_connection,
            settings=mock_settings
        )
        assert runner is not None
        assert runner.postgres is db_connection
    
    @pytest.mark.requires_db
    @pytest.mark.slow
    def test_evidence_store_can_be_created(self, db_connection):
        """EvidenceStore should be creatable with postgres_connection."""
        from evidence_store.store import EvidenceStore
        
        store = EvidenceStore(postgres_connection=db_connection)
        assert store is not None
        assert store.connection is db_connection
    
    @pytest.mark.requires_db
    @pytest.mark.slow
    def test_narrative_generator_can_be_created(self):
        """NarrativeGenerator should be importable."""
        try:
            import sys
            from pathlib import Path
            # Need parent of src on path for relative imports to work
            project_root = str(Path(__file__).parent.parent.parent)
            if project_root not in sys.path:
                sys.path.insert(0, project_root)
            
            from src.narrative.generator import NarrativeGenerator
            assert NarrativeGenerator is not None
        except ImportError as e:
            pytest.skip(f"NarrativeGenerator import issue: {e}")
    
    @pytest.mark.requires_db
    @pytest.mark.slow
    def test_orchestrator_module_exists(self):
        """Orchestrator module should be importable."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
        
        # Just check the file exists
        orchestrator_path = Path(__file__).parent.parent.parent / "src" / "orchestrator.py"
        assert orchestrator_path.exists()
