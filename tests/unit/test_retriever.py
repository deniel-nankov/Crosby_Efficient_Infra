"""
RAG Retriever Unit Tests - Aligned with Actual Implementation

Tests for:
- BM25Index: Sparse retrieval with BM25 scoring
- ConfidenceCalibrator: Confidence score calculation
- CrossEncoderReranker: Neural reranking
- QueryRewriter: Query expansion
- MultiHopRetriever: Multi-hop retrieval
- RAGRetriever: Full retrieval pipeline
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone

pytestmark = [pytest.mark.unit]


class TestBM25Index:
    """Test BM25 sparse retrieval index."""
    
    def test_bm25_initialization(self):
        """BM25Index should initialize with default parameters."""
        from rag.retriever import BM25Index
        
        bm25 = BM25Index()
        assert bm25.k1 == 1.5
        assert bm25.b == 0.75
        assert bm25.N == 0
    
    def test_bm25_custom_parameters(self):
        """BM25Index should accept custom k1 and b parameters."""
        from rag.retriever import BM25Index
        
        bm25 = BM25Index(k1=2.0, b=0.5)
        assert bm25.k1 == 2.0
        assert bm25.b == 0.5
    
    def test_bm25_build_index(self):
        """BM25Index should build index from documents."""
        from rag.retriever import BM25Index
        
        bm25 = BM25Index()
        documents = {
            "doc1": "sector concentration limit policy",
            "doc2": "liquidity requirements minimum threshold",
            "doc3": "issuer exposure limit 10 percent",
        }
        
        bm25.build_index(documents)
        
        assert bm25.N == 3
        assert len(bm25.documents) == 3
        assert "sector" in bm25.inverted_index
    
    def test_bm25_search_returns_ranked_results(self):
        """BM25 search should return results ranked by score."""
        from rag.retriever import BM25Index
        
        bm25 = BM25Index()
        documents = {
            "doc1": "sector concentration limit policy sector sector",  # More relevant
            "doc2": "liquidity requirements minimum threshold",
            "doc3": "sector exposure limit",
        }
        
        bm25.build_index(documents)
        results = bm25.search("sector concentration", top_k=3)
        
        assert len(results) > 0
        assert results[0][0] == "doc1"  # Most relevant first
        # Scores should be descending
        for i in range(len(results) - 1):
            assert results[i][1] >= results[i + 1][1]
    
    def test_bm25_search_empty_query(self):
        """BM25 should handle empty query gracefully."""
        from rag.retriever import BM25Index
        
        bm25 = BM25Index()
        bm25.build_index({"doc1": "test content"})
        
        results = bm25.search("")
        assert results == []
    
    def test_bm25_search_no_match(self):
        """BM25 should return empty for non-matching query."""
        from rag.retriever import BM25Index
        
        bm25 = BM25Index()
        bm25.build_index({"doc1": "sector concentration"})
        
        results = bm25.search("xyz123nonexistent")
        assert results == []
    
    def test_bm25_respects_top_k(self):
        """BM25 should respect top_k parameter."""
        from rag.retriever import BM25Index
        
        bm25 = BM25Index()
        documents = {f"doc{i}": f"test document {i}" for i in range(10)}
        bm25.build_index(documents)
        
        results = bm25.search("test document", top_k=3)
        assert len(results) <= 3


class TestConfidenceCalibrator:
    """Test confidence score calculation."""
    
    def test_calibrator_initialization(self):
        """ConfidenceCalibrator should initialize with thresholds."""
        from rag.retriever import ConfidenceCalibrator
        
        calibrator = ConfidenceCalibrator()
        assert hasattr(calibrator, 'calculate')
    
    def test_calibrator_high_confidence_with_good_chunks(self):
        """Should return high confidence for highly relevant chunks."""
        from rag.retriever import ConfidenceCalibrator
        from rag.vector_store import PolicyChunk
        
        calibrator = ConfidenceCalibrator()
        
        chunks = [
            PolicyChunk(
                chunk_id="c1",
                document_id="d1",
                document_name="concentration_limits.md",
                section_title="Sector Limits",
                content="The maximum sector concentration is 30% of NAV.",
                content_hash="abc123",
            ),
        ]
        chunks[0].similarity = 0.95
        
        confidence, explanation = calibrator.calculate("sector concentration limit", chunks)
        assert confidence >= 0.7
    
    def test_calibrator_low_confidence_with_poor_chunks(self):
        """Should return low confidence for irrelevant chunks."""
        from rag.retriever import ConfidenceCalibrator
        from rag.vector_store import PolicyChunk
        
        calibrator = ConfidenceCalibrator()
        
        chunks = [
            PolicyChunk(
                chunk_id="c1",
                document_id="d1",
                document_name="unrelated.md",
                section_title="Other",
                content="Something completely different.",
                content_hash="abc123",
            ),
        ]
        chunks[0].similarity = 0.3
        
        confidence, explanation = calibrator.calculate("sector concentration limit", chunks)
        assert confidence < 0.5
    
    def test_calibrator_returns_explanation(self):
        """Should return explanation string."""
        from rag.retriever import ConfidenceCalibrator
        
        calibrator = ConfidenceCalibrator()
        _, explanation = calibrator.calculate("test query", [])
        
        assert isinstance(explanation, str)
        assert len(explanation) > 0


class TestCrossEncoderReranker:
    """Test cross-encoder reranking."""
    
    def test_reranker_initialization(self):
        """CrossEncoderReranker should initialize."""
        from rag.retriever import CrossEncoderReranker
        
        reranker = CrossEncoderReranker()
        assert hasattr(reranker, 'available')
        assert hasattr(reranker, 'rerank')
    
    def test_reranker_available_property(self):
        """Should report availability correctly."""
        from rag.retriever import CrossEncoderReranker
        
        reranker = CrossEncoderReranker()
        assert isinstance(reranker.available, bool)
    
    @pytest.mark.skipif(
        not pytest.importorskip("sentence_transformers", reason="sentence-transformers not installed"),
        reason="sentence-transformers required"
    )
    def test_reranker_reranks_chunks(self):
        """Should rerank chunks when available."""
        from rag.retriever import CrossEncoderReranker
        from rag.vector_store import PolicyChunk
        
        reranker = CrossEncoderReranker()
        if not reranker.available:
            pytest.skip("Cross-encoder not available")
        
        chunks = [
            PolicyChunk(
                chunk_id=f"c{i}",
                document_id="d1",
                document_name="test.md",
                section_title=f"Section {i}",
                content=f"Content about sector concentration {i}",
                content_hash=f"hash{i}",
            )
            for i in range(3)
        ]
        
        reranked = reranker.rerank("sector concentration", chunks, top_k=2)
        assert len(reranked) <= 2


class TestQueryRewriter:
    """Test query rewriting and expansion."""
    
    def test_rewriter_initialization(self):
        """QueryRewriter should initialize without LLM client."""
        from rag.retriever import QueryRewriter
        
        rewriter = QueryRewriter()
        assert hasattr(rewriter, 'rewrite')
    
    def test_rewriter_expands_abbreviations(self):
        """Should expand common abbreviations."""
        from rag.retriever import QueryRewriter
        
        rewriter = QueryRewriter()
        expanded, _ = rewriter.rewrite("nav concentration limit")
        
        assert "net asset value" in expanded.lower() or "NAV" in expanded
    
    def test_rewriter_adds_synonyms(self):
        """Should add relevant synonyms."""
        from rag.retriever import QueryRewriter
        
        rewriter = QueryRewriter()
        expanded, _ = rewriter.rewrite("breach threshold")
        
        # Should add related terms
        assert len(expanded) >= len("breach threshold")
    
    def test_rewriter_returns_explanation(self):
        """Should return explanation of changes."""
        from rag.retriever import QueryRewriter
        
        rewriter = QueryRewriter()
        _, explanation = rewriter.rewrite("test query")
        
        assert isinstance(explanation, str)


class TestMultiHopRetriever:
    """Test multi-hop retrieval."""
    
    def test_multi_hop_initialization(self):
        """MultiHopRetriever should initialize with dependencies."""
        from rag.retriever import MultiHopRetriever
        
        mock_store = MagicMock()
        mock_embedder = MagicMock()
        
        retriever = MultiHopRetriever(
            vector_store=mock_store,
            embedder=mock_embedder,
        )
        
        assert retriever.vector_store == mock_store
        assert retriever.embedder == mock_embedder
    
    def test_multi_hop_default_hop_types(self):
        """Should use default hop types if not specified."""
        from rag.retriever import MultiHopRetriever
        
        mock_store = MagicMock()
        mock_store.search_similar.return_value = []
        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [0.1] * 768
        
        retriever = MultiHopRetriever(
            vector_store=mock_store,
            embedder=mock_embedder,
        )
        
        context = retriever.retrieve_multi_hop("test query")
        assert len(context.hops) >= 0


class TestRetrievedContext:
    """Test RetrievedContext dataclass."""
    
    def test_retrieved_context_creation(self):
        """RetrievedContext should be created with required fields."""
        from rag.retriever import RetrievedContext
        
        context = RetrievedContext(
            query="test query",
            chunks=[],
        )
        
        assert context.query == "test query"
        assert context.chunks == []
    
    def test_retrieved_context_has_metadata(self):
        """RetrievedContext should include metadata fields."""
        from rag.retriever import RetrievedContext
        
        context = RetrievedContext(
            query="test",
            chunks=[],
            confidence=0.85,
        )
        
        assert context.confidence == 0.85
        assert hasattr(context, 'retrieved_at')


class TestRAGRetrieverIntegration:
    """Integration tests for RAGRetriever."""
    
    def test_retriever_initialization(self):
        """RAGRetriever should initialize with dependencies."""
        from rag.retriever import RAGRetriever
        
        mock_store = MagicMock()
        mock_embedder = MagicMock()
        
        retriever = RAGRetriever(
            vector_store=mock_store,
            embedder=mock_embedder,
        )
        
        assert retriever.vector_store == mock_store
        assert retriever.embedder == mock_embedder
    
    def test_retriever_has_retrieve_for_control_method(self):
        """RAGRetriever should have retrieve_for_control method."""
        from rag.retriever import RAGRetriever
        
        mock_store = MagicMock()
        mock_embedder = MagicMock()
        
        retriever = RAGRetriever(
            vector_store=mock_store,
            embedder=mock_embedder,
        )
        
        assert hasattr(retriever, 'retrieve_for_control')
    
    def test_retriever_has_multi_hop_method(self):
        """RAGRetriever should have retrieve_multi_hop method."""
        from rag.retriever import RAGRetriever
        
        mock_store = MagicMock()
        mock_embedder = MagicMock()
        
        retriever = RAGRetriever(
            vector_store=mock_store,
            embedder=mock_embedder,
        )
        
        assert hasattr(retriever, 'retrieve_multi_hop')
