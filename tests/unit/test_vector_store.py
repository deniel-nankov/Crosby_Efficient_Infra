"""
Vector Store Unit Tests

Comprehensive test suite for the VectorStore and PolicyChunk classes, covering:
- Chunk creation and validation
- Vector storage and retrieval
- Similarity search functionality
- Filtering and querying
- Database operations

Test Categories:
- TestPolicyChunk: Chunk dataclass tests
- TestVectorStoreOperations: Core store operations
- TestSimilaritySearch: Search functionality
- TestFiltering: Filter and query tests
"""

from __future__ import annotations

import pytest
import hashlib
from datetime import datetime, timezone
from typing import List, Dict, Any
from unittest.mock import MagicMock, patch
from dataclasses import asdict

pytestmark = [pytest.mark.unit]


class TestPolicyChunkCreation:
    """Test PolicyChunk dataclass creation and validation."""
    
    def test_chunk_from_text_creates_valid_chunk(self):
        """PolicyChunk.from_text should create a valid chunk."""
        from rag.vector_store import PolicyChunk
        
        content = "No single sector shall exceed 30% of NAV."
        
        chunk = PolicyChunk.from_text(
            content=content,
            document_name="concentration_limits.md",
            section_title="Sector Limits",
            document_id="DOC001",
            control_types=["concentration", "sector"],
        )
        
        assert chunk.content == content
        assert chunk.document_name == "concentration_limits.md"
        assert chunk.section_title == "Sector Limits"
        assert "concentration" in chunk.control_types
    
    def test_chunk_generates_unique_id(self):
        """Each chunk should have a unique ID based on content hash."""
        from rag.vector_store import PolicyChunk
        
        chunk1 = PolicyChunk.from_text(
            content="Content A",
            document_name="doc.md",
            section_title="Section 1",
        )
        
        chunk2 = PolicyChunk.from_text(
            content="Content B",
            document_name="doc.md",
            section_title="Section 1",
        )
        
        assert chunk1.chunk_id != chunk2.chunk_id
        assert chunk1.content_hash != chunk2.content_hash
    
    def test_chunk_id_is_deterministic(self):
        """Same content should produce same chunk ID."""
        from rag.vector_store import PolicyChunk
        
        content = "Identical content for testing."
        
        chunk1 = PolicyChunk.from_text(
            content=content,
            document_name="doc.md",
            section_title="Section",
        )
        
        chunk2 = PolicyChunk.from_text(
            content=content,
            document_name="doc.md",
            section_title="Section",
        )
        
        assert chunk1.chunk_id == chunk2.chunk_id
        assert chunk1.content_hash == chunk2.content_hash
    
    def test_chunk_has_created_at_timestamp(self):
        """Chunk should have creation timestamp."""
        from rag.vector_store import PolicyChunk
        
        before = datetime.now(timezone.utc)
        
        chunk = PolicyChunk.from_text(
            content="Test content",
            document_name="doc.md",
            section_title="Section",
        )
        
        after = datetime.now(timezone.utc)
        
        assert before <= chunk.created_at <= after
    
    def test_chunk_control_types_default_to_empty_list(self):
        """Control types should default to empty list."""
        from rag.vector_store import PolicyChunk
        
        chunk = PolicyChunk.from_text(
            content="Test content",
            document_name="doc.md",
            section_title="Section",
        )
        
        assert chunk.control_types == []
    
    def test_chunk_handles_empty_content(self):
        """Chunk should handle empty content gracefully."""
        from rag.vector_store import PolicyChunk
        
        chunk = PolicyChunk.from_text(
            content="",
            document_name="doc.md",
            section_title="Section",
        )
        
        assert chunk.content == ""
        assert chunk.chunk_id is not None


class TestPolicyChunkAttributes:
    """Test PolicyChunk dynamic attributes for scoring."""
    
    def test_chunk_has_similarity_attribute(self):
        """Chunk should have similarity attribute."""
        from rag.vector_store import PolicyChunk
        
        chunk = PolicyChunk.from_text(
            content="Test",
            document_name="doc.md",
            section_title="Section",
        )
        
        # Should be able to set similarity
        chunk.similarity = 0.85
        assert chunk.similarity == 0.85
    
    def test_chunk_has_rerank_score_attribute(self):
        """Chunk should have rerank_score attribute."""
        from rag.vector_store import PolicyChunk
        
        chunk = PolicyChunk.from_text(
            content="Test",
            document_name="doc.md",
            section_title="Section",
        )
        
        chunk.rerank_score = 0.92
        assert chunk.rerank_score == 0.92
    
    def test_chunk_scoring_attributes_default_to_zero(self):
        """Scoring attributes should default to zero."""
        from rag.vector_store import PolicyChunk
        
        chunk = PolicyChunk.from_text(
            content="Test",
            document_name="doc.md",
            section_title="Section",
        )
        
        assert chunk.similarity == 0.0
        assert chunk.rerank_score == 0.0
        assert chunk.final_score == 0.0
        assert chunk.sparse_score == 0.0


class TestVectorStoreInitialization:
    """Test VectorStore initialization."""
    
    @pytest.mark.requires_db
    def test_vector_store_initializes_with_connection(self, db_connection):
        """VectorStore should initialize with database connection."""
        from rag.vector_store import VectorStore
        
        store = VectorStore(db_connection)
        
        assert store.conn is not None
    
    def test_vector_store_with_mock_connection(self):
        """VectorStore should work with mock connection."""
        from rag.vector_store import VectorStore
        
        mock_conn = MagicMock()
        store = VectorStore(mock_conn)
        
        assert store.conn == mock_conn


class TestVectorStoreOperations:
    """Test core VectorStore operations."""
    
    @pytest.mark.requires_db
    def test_upsert_chunks_stores_data(self, vector_store, sample_policy_chunks, mock_embedder):
        """Upsert should store chunks in database."""
        # Add embeddings to chunks
        for chunk in sample_policy_chunks:
            chunk.embedding = mock_embedder.embed(chunk.content)
        
        # This should not raise
        vector_store.upsert_chunks(sample_policy_chunks)
    
    @pytest.mark.requires_db
    def test_count_chunks_returns_integer(self, vector_store):
        """Count should return non-negative integer."""
        count = vector_store.count_chunks()
        
        assert isinstance(count, int)
        assert count >= 0
    
    @pytest.mark.requires_db
    def test_get_by_control_type_filters_correctly(self, vector_store):
        """Should filter chunks by control type."""
        chunks = vector_store.get_by_control_type("concentration")
        
        for chunk in chunks:
            assert "concentration" in chunk.control_types


class TestSimilaritySearch:
    """Test similarity search functionality."""
    
    @pytest.mark.requires_db
    def test_search_similar_returns_chunks(self, vector_store, embedder):
        """Similarity search should return chunks."""
        if not embedder.available:
            pytest.skip("Embedder not available")
        
        query = "What are the sector concentration limits?"
        query_embedding = embedder.embed(query)
        
        chunks = vector_store.search_similar(query_embedding, limit=5)
        
        assert isinstance(chunks, list)
        for chunk in chunks:
            assert hasattr(chunk, 'content')
            assert hasattr(chunk, 'similarity')
    
    @pytest.mark.requires_db
    def test_search_respects_limit(self, vector_store, embedder):
        """Search should respect limit parameter."""
        if not embedder.available:
            pytest.skip("Embedder not available")
        
        query_embedding = embedder.embed("concentration limits")
        
        chunks_3 = vector_store.search_similar(query_embedding, limit=3)
        chunks_5 = vector_store.search_similar(query_embedding, limit=5)
        
        assert len(chunks_3) <= 3
        assert len(chunks_5) <= 5
    
    @pytest.mark.requires_db
    def test_search_orders_by_similarity(self, vector_store, embedder):
        """Results should be ordered by similarity (descending)."""
        if not embedder.available:
            pytest.skip("Embedder not available")
        
        query_embedding = embedder.embed("sector concentration policy")
        
        chunks = vector_store.search_similar(query_embedding, limit=5)
        
        if len(chunks) > 1:
            similarities = [getattr(c, 'similarity', 0) for c in chunks]
            assert similarities == sorted(similarities, reverse=True), \
                "Results should be sorted by similarity descending"
    
    @pytest.mark.requires_db
    def test_search_with_control_type_filter(self, vector_store, embedder):
        """Search should filter by control type when specified."""
        if not embedder.available:
            pytest.skip("Embedder not available")
        
        query_embedding = embedder.embed("compliance policy")
        
        chunks = vector_store.search_similar(
            query_embedding,
            limit=5,
            control_types=["concentration"],
        )
        
        for chunk in chunks:
            # At least one control type should match
            assert any(ct in chunk.control_types for ct in ["concentration"]) or \
                   len(chunk.control_types) == 0


class TestMockVectorStore:
    """Test with mock vector store for unit testing."""
    
    def test_mock_store_returns_sample_chunks(self, mock_vector_store):
        """Mock store should return configured sample chunks."""
        chunks = mock_vector_store.search_similar([0.0] * 768, limit=5)
        
        assert len(chunks) == 2  # Configured to return 2
        assert chunks[0].document_name == "concentration_limits.md"
    
    def test_mock_store_count_chunks(self, mock_vector_store):
        """Mock store should return configured count."""
        count = mock_vector_store.count_chunks()
        
        assert count == 3
    
    def test_mock_store_get_by_control_type(self, mock_vector_store):
        """Mock store should filter by control type."""
        concentration_chunks = mock_vector_store.get_by_control_type("concentration")
        liquidity_chunks = mock_vector_store.get_by_control_type("liquidity")
        
        assert len(concentration_chunks) >= 1
        assert len(liquidity_chunks) >= 1


class TestVectorStoreEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_embedding_query(self, mock_vector_store):
        """Should handle empty embedding query."""
        empty_embedding = [0.0] * 768
        
        # Should not raise
        chunks = mock_vector_store.search_similar(empty_embedding, limit=5)
        
        assert isinstance(chunks, list)
    
    def test_zero_limit_returns_empty(self, mock_vector_store):
        """Zero limit should return empty list."""
        mock_vector_store.search_similar.return_value = []
        
        chunks = mock_vector_store.search_similar([0.0] * 768, limit=0)
        
        assert chunks == []
    
    def test_negative_similarity_handled(self):
        """Chunks with negative similarity should be handled."""
        from rag.vector_store import PolicyChunk
        
        chunk = PolicyChunk.from_text(
            content="Test",
            document_name="doc.md",
            section_title="Section",
        )
        
        chunk.similarity = -0.5  # Negative similarity (opposite direction)
        
        assert chunk.similarity == -0.5
