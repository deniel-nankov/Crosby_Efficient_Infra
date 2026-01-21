"""
RAG Retriever - Retrieves relevant policy context for compliance narratives

This is the core RAG component that:
1. Takes control results as input
2. Searches for relevant policy sections
3. Returns grounded context for LLM generation

SOTA Features:
- Hybrid Search: Dense embeddings + BM25 sparse retrieval
- Confidence Calibration: Multi-factor confidence scoring
- Query Decomposition: Complex queries → sub-queries
"""

from __future__ import annotations

import logging
import re
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone

from .vector_store import VectorStore, PolicyChunk
from .embedder import LocalEmbedder

logger = logging.getLogger(__name__)


# =============================================================================
# BM25 SPARSE RETRIEVAL (SOTA Enhancement)
# =============================================================================

class BM25Index:
    """
    BM25 (Best Matching 25) sparse retrieval.
    
    BM25 excels at exact keyword matching where semantic search might miss.
    For example: "30% limit" will match policy text containing "30%" exactly.
    """
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents: Dict[str, str] = {}
        self.doc_lengths: Dict[str, int] = {}
        self.avg_doc_length: float = 0
        self.term_doc_freq: Dict[str, int] = {}
        self.inverted_index: Dict[str, Dict[str, int]] = {}
        self.N: int = 0
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization - lowercase and split on non-alphanumeric."""
        text = text.lower()
        tokens = re.findall(r'\b[\w%]+\b', text)
        return tokens
    
    def build_index(self, documents: Dict[str, str]):
        """Build BM25 index from documents."""
        self.documents = documents
        self.N = len(documents)
        self.term_doc_freq = {}
        self.inverted_index = {}
        
        total_length = 0
        
        for doc_id, content in documents.items():
            tokens = self._tokenize(content)
            self.doc_lengths[doc_id] = len(tokens)
            total_length += len(tokens)
            
            term_counts = Counter(tokens)
            seen_terms = set()
            
            for term, count in term_counts.items():
                if term not in self.inverted_index:
                    self.inverted_index[term] = {}
                self.inverted_index[term][doc_id] = count
                
                if term not in seen_terms:
                    self.term_doc_freq[term] = self.term_doc_freq.get(term, 0) + 1
                    seen_terms.add(term)
        
        self.avg_doc_length = total_length / self.N if self.N > 0 else 0
        logger.debug(f"BM25 index built: {self.N} documents, {len(self.inverted_index)} terms")
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search using BM25 scoring."""
        query_tokens = self._tokenize(query)
        scores: Dict[str, float] = {}
        
        for term in query_tokens:
            if term not in self.inverted_index:
                continue
            
            doc_freq = self.term_doc_freq.get(term, 0)
            idf = math.log((self.N - doc_freq + 0.5) / (doc_freq + 0.5) + 1)
            
            for doc_id, term_freq in self.inverted_index[term].items():
                doc_length = self.doc_lengths[doc_id]
                numerator = term_freq * (self.k1 + 1)
                denominator = term_freq + self.k1 * (1 - self.b + self.b * doc_length / self.avg_doc_length)
                score = idf * (numerator / denominator)
                scores[doc_id] = scores.get(doc_id, 0) + score
        
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]


# =============================================================================
# CONFIDENCE CALIBRATION (SOTA Enhancement)
# =============================================================================

class ConfidenceCalibrator:
    """
    Calibrate confidence in RAG responses.
    
    Factors: retrieval quality, source agreement, query coverage, specificity
    """
    
    def calculate(
        self,
        query: str,
        chunks: List[PolicyChunk],
    ) -> Tuple[float, str]:
        """Calculate confidence score with explanation."""
        if not chunks:
            return 0.0, "No relevant documents found"
        
        factors = []
        
        # Factor 1: Top retrieval score
        top_score = max(getattr(c, 'similarity', 0) or getattr(c, 'final_score', 0) for c in chunks)
        score_factor = min(top_score / 0.8, 1.0)
        factors.append(("retrieval_quality", score_factor, 0.3))
        
        # Factor 2: Score distribution
        if len(chunks) >= 3:
            scores = [getattr(c, 'similarity', 0) or getattr(c, 'final_score', 0) for c in chunks[:3]]
            top_3_avg = sum(scores) / 3
            distribution_factor = min(top_3_avg / 0.7, 1.0)
        else:
            distribution_factor = 0.5
        factors.append(("source_agreement", distribution_factor, 0.2))
        
        # Factor 3: Query term coverage
        query_terms = set(query.lower().split())
        all_content = " ".join(c.content.lower() for c in chunks[:3])
        covered = sum(1 for term in query_terms if term in all_content)
        coverage_factor = covered / len(query_terms) if query_terms else 0
        factors.append(("query_coverage", coverage_factor, 0.3))
        
        # Factor 4: Specificity (numbers, percentages)
        specificity = 0
        for chunk in chunks[:3]:
            if re.search(r'\d+%|\$[\d,]+|\d+\s*(days|percent|limit)', chunk.content):
                specificity += 1
        specificity_factor = min(specificity / 2, 1.0)
        factors.append(("specificity", specificity_factor, 0.2))
        
        confidence = sum(score * weight for _, score, weight in factors)
        explanation = "; ".join(f"{n}: {'high' if s > 0.7 else 'medium' if s > 0.4 else 'low'}" 
                                for n, s, _ in factors)
        
        return round(confidence, 2), explanation


# =============================================================================
# CROSS-ENCODER RERANKING (SOTA Enhancement)
# =============================================================================

class CrossEncoderReranker:
    """
    Cross-encoder reranking for improved retrieval precision.
    
    Unlike bi-encoders (which embed query and document separately),
    cross-encoders process (query, document) pairs together, enabling
    deeper semantic understanding at the cost of speed.
    
    Usage:
        reranker = CrossEncoderReranker()
        reranked = reranker.rerank(query, chunks, top_k=5)
    """
    
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cpu",
        batch_size: int = 16,
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model = None
        self._available = None
    
    @property
    def available(self) -> bool:
        """Check if cross-encoder is available."""
        if self._available is None:
            try:
                from sentence_transformers import CrossEncoder
                self._available = True
            except ImportError:
                logger.warning(
                    "sentence-transformers not installed. "
                    "Install with: pip install sentence-transformers"
                )
                self._available = False
        return self._available
    
    def _load_model(self):
        """Lazy load the cross-encoder model."""
        if self._model is None and self.available:
            try:
                from sentence_transformers import CrossEncoder
                logger.info(f"Loading cross-encoder: {self.model_name}")
                self._model = CrossEncoder(self.model_name, device=self.device)
                logger.info("Cross-encoder loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load cross-encoder: {e}")
                self._available = False
    
    def rerank(
        self,
        query: str,
        chunks: List[PolicyChunk],
        top_k: int = 5,
    ) -> List[PolicyChunk]:
        """
        Rerank chunks using cross-encoder.
        
        Args:
            query: The search query
            chunks: List of candidate chunks from initial retrieval
            top_k: Number of top chunks to return after reranking
            
        Returns:
            Reranked list of chunks with updated scores
        """
        if not chunks:
            return chunks
        
        if not self.available:
            logger.debug("Cross-encoder not available, skipping reranking")
            return chunks[:top_k]
        
        self._load_model()
        
        if self._model is None:
            return chunks[:top_k]
        
        try:
            # Create (query, document) pairs
            pairs = [(query, chunk.content) for chunk in chunks]
            
            # Score all pairs
            scores = self._model.predict(pairs, batch_size=self.batch_size)
            
            # Attach scores and sort
            scored_chunks = []
            for chunk, score in zip(chunks, scores):
                chunk.rerank_score = float(score)
                scored_chunks.append((score, chunk))
            
            # Sort by rerank score (descending)
            scored_chunks.sort(key=lambda x: x[0], reverse=True)
            
            # Update final scores to incorporate reranking
            reranked = []
            for i, (score, chunk) in enumerate(scored_chunks[:top_k]):
                # Blend original score with rerank score
                original = getattr(chunk, 'final_score', chunk.similarity or 0)
                # Normalize rerank score (ms-marco outputs ~-10 to +10)
                normalized_rerank = (score + 10) / 20  # Map to 0-1
                chunk.final_score = 0.4 * original + 0.6 * normalized_rerank
                chunk.rerank_rank = i + 1
                reranked.append(chunk)
            
            logger.debug(
                f"Reranked {len(chunks)} chunks -> top {len(reranked)}, "
                f"best score: {reranked[0].rerank_score:.3f}" if reranked else ""
            )
            
            return reranked
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return chunks[:top_k]


# =============================================================================
# QUERY REWRITING (SOTA Enhancement)
# =============================================================================

class QueryRewriter:
    """
    LLM-powered query rewriting for improved retrieval.
    
    Before embedding, the LLM expands and clarifies the query:
    - "tech limit" → "technology sector concentration limit percentage NAV"
    - Adds synonyms, related terms, and domain context
    - Fixes typos and disambiguates terms
    
    This significantly improves retrieval for short or ambiguous queries.
    """
    
    REWRITE_PROMPT = """You are a query expansion expert for a compliance/investment policy search system.

Given the user's search query, rewrite it to improve retrieval from a policy document database.

Rules:
1. Expand abbreviations (e.g., "tech" → "technology", "NAV" → "Net Asset Value")
2. Add relevant synonyms and related terms
3. Include specific compliance/investment terminology
4. Keep the rewritten query concise (under 50 words)
5. Output ONLY the rewritten query, nothing else

Original query: {query}
Context: {context}

Rewritten query:"""

    def __init__(self, llm_client=None):
        """
        Initialize query rewriter.
        
        Args:
            llm_client: LLM client with generate() method. If None, uses simple expansion.
        """
        self.llm_client = llm_client
        self._expansion_map = {
            # Abbreviations
            "tech": "technology",
            "fin": "financials financial",
            "nav": "net asset value NAV",
            "aum": "assets under management AUM",
            "cio": "chief investment officer CIO",
            "pm": "portfolio manager",
            "otc": "over the counter OTC",
            # Common queries
            "limit": "limit threshold maximum",
            "breach": "breach violation exceed threshold",
            "concentration": "concentration exposure limit",
            "liquidity": "liquidity bucket redemption T+1 T+7",
            "sector": "sector GICS industry",
            "issuer": "issuer single name concentration",
        }
    
    def rewrite(
        self,
        query: str,
        context: str = "",
        use_llm: bool = True,
    ) -> Tuple[str, str]:
        """
        Rewrite query for better retrieval.
        
        Args:
            query: Original search query
            context: Optional context (e.g., control type)
            use_llm: Whether to use LLM for rewriting
            
        Returns:
            Tuple of (rewritten_query, explanation)
        """
        original = query
        
        # Try LLM rewriting first
        if use_llm and self.llm_client:
            try:
                rewritten = self._llm_rewrite(query, context)
                if rewritten and len(rewritten) > len(query):
                    return rewritten, f"LLM expanded: '{original}' → '{rewritten[:50]}...'"
            except Exception as e:
                logger.warning(f"LLM query rewriting failed: {e}")
        
        # Fall back to simple expansion
        rewritten = self._simple_expand(query)
        if rewritten != query:
            return rewritten, f"Expanded: '{original}' → '{rewritten[:50]}...'"
        
        return query, "No expansion needed"
    
    def _llm_rewrite(self, query: str, context: str) -> str:
        """Use LLM to rewrite query."""
        if self.llm_client is None:
            return query
            
        prompt = self.REWRITE_PROMPT.format(query=query, context=context or "compliance policy search")
        
        response = self.llm_client.generate(
            prompt=prompt,
            system_prompt="You are a search query optimization expert. Output only the rewritten query."
        )
        
        # Clean up response
        rewritten = response.strip()
        # Remove quotes if present
        rewritten = rewritten.strip('"\'')
        # Limit length
        if len(rewritten) > 200:
            rewritten = rewritten[:200]
        
        return rewritten
    
    def _simple_expand(self, query: str) -> str:
        """Simple rule-based query expansion."""
        words = query.lower().split()
        expanded = []
        
        for word in words:
            if word in self._expansion_map:
                expanded.append(self._expansion_map[word])
            else:
                expanded.append(word)
        
        return " ".join(expanded)


# =============================================================================
# MULTI-HOP REASONING (SOTA Enhancement)
# =============================================================================

@dataclass
class HopResult:
    """Result from a single hop in multi-hop retrieval."""
    hop_number: int
    hop_type: str  # e.g., "policy", "exception", "precedent"
    query: str
    chunks: List[PolicyChunk]
    reasoning: str


@dataclass
class MultiHopContext:
    """Context from multi-hop retrieval with full reasoning chain."""
    original_query: str
    hops: List[HopResult]
    final_chunks: List[PolicyChunk]
    reasoning_chain: List[str]
    total_tokens_estimate: int = 0
    confidence: float = 0.0
    
    def to_prompt_context(self) -> str:
        """Format multi-hop results for LLM prompt."""
        parts = ["MULTI-HOP RETRIEVAL RESULTS:"]
        parts.append("=" * 60)
        
        for hop in self.hops:
            parts.append(f"\n### HOP {hop.hop_number}: {hop.hop_type.upper()}")
            parts.append(f"Query: {hop.query[:80]}...")
            parts.append(f"Found: {len(hop.chunks)} relevant sections")
            
            for chunk in hop.chunks[:2]:  # Top 2 per hop
                parts.append(f"  - [{chunk.document_name}] {chunk.section_title}")
        
        parts.append("\n" + "=" * 60)
        parts.append("CONSOLIDATED CONTEXT:")
        
        for i, chunk in enumerate(self.final_chunks, 1):
            parts.append(f"\n[{i}] {chunk.document_name} | {chunk.section_title}")
            parts.append(chunk.content[:500] + "..." if len(chunk.content) > 500 else chunk.content)
        
        return "\n".join(parts)


class MultiHopRetriever:
    """
    Multi-hop reasoning for complex compliance queries.
    
    Chains multiple retrievals to build comprehensive context:
    1. Primary Policy: Find the main policy section
    2. Exceptions: Find any exceptions or special cases
    3. Precedents: Find historical interpretations or examples
    4. Related Controls: Find connected compliance requirements
    
    Example flow:
        "Technology sector at 28%" 
        → Hop 1: Find sector concentration policy (30% limit)
        → Hop 2: Find exceptions (40% with CIO approval)  
        → Hop 3: Find precedents (prior breaches, resolutions)
    """
    
    # Hop type configurations
    HOP_CONFIGS = {
        "policy": {
            "query_template": "{base_query}",
            "description": "Primary policy section",
        },
        "exception": {
            "query_template": "{base_query} exception waiver approval special case exemption",
            "description": "Exceptions and special cases",
        },
        "precedent": {
            "query_template": "{base_query} historical prior breach resolution remediation example",
            "description": "Historical precedents",
        },
        "escalation": {
            "query_template": "{base_query} escalation notification reporting CIO board committee",
            "description": "Escalation procedures",
        },
        "remediation": {
            "query_template": "{base_query} remediation cure action timeline reduction plan",
            "description": "Remediation requirements",
        },
    }
    
    def __init__(
        self,
        vector_store: VectorStore,
        embedder: LocalEmbedder,
        max_hops: int = 3,
        chunks_per_hop: int = 2,
    ):
        self.vector_store = vector_store
        self.embedder = embedder
        self.max_hops = max_hops
        self.chunks_per_hop = chunks_per_hop
    
    def retrieve_multi_hop(
        self,
        query: str,
        hop_types: Optional[List[str]] = None,
        control_type: Optional[str] = None,
    ) -> MultiHopContext:
        """
        Perform multi-hop retrieval.
        
        Args:
            query: Base query string
            hop_types: List of hop types to perform (default: policy, exception, precedent)
            control_type: Optional control type filter
            
        Returns:
            MultiHopContext with all hop results and consolidated chunks
        """
        if hop_types is None:
            hop_types = ["policy", "exception", "precedent"]
        
        # Limit to max hops
        hop_types = hop_types[:self.max_hops]
        
        hops: List[HopResult] = []
        all_chunks: List[PolicyChunk] = []
        seen_chunk_ids: set = set()
        reasoning_chain: List[str] = []
        
        reasoning_chain.append(f"Starting multi-hop retrieval for: '{query[:50]}...'")
        
        for hop_num, hop_type in enumerate(hop_types, 1):
            config = self.HOP_CONFIGS.get(hop_type)
            if not config:
                logger.warning(f"Unknown hop type: {hop_type}")
                continue
            
            # Build hop-specific query
            hop_query = config["query_template"].format(base_query=query)
            
            # Add context from previous hops
            if hops and hop_num > 1:
                # Extract key terms from previous results
                prev_content = " ".join(
                    chunk.section_title for hop in hops for chunk in hop.chunks[:1]
                )
                hop_query = f"{hop_query} {prev_content}"
            
            reasoning_chain.append(f"Hop {hop_num} ({hop_type}): {hop_query[:60]}...")
            
            try:
                # Generate embedding and search
                embedding = self.embedder.embed(hop_query)
                chunks = self.vector_store.search_similar(
                    query_embedding=embedding,
                    limit=self.chunks_per_hop * 2,  # Fetch extra for dedup
                    control_types=[control_type] if control_type else None,
                )
                
                # Deduplicate across hops
                unique_chunks = []
                for chunk in chunks:
                    if chunk.chunk_id not in seen_chunk_ids:
                        seen_chunk_ids.add(chunk.chunk_id)
                        unique_chunks.append(chunk)
                        if len(unique_chunks) >= self.chunks_per_hop:
                            break
                
                hop_result = HopResult(
                    hop_number=hop_num,
                    hop_type=hop_type,
                    query=hop_query,
                    chunks=unique_chunks,
                    reasoning=f"Found {len(unique_chunks)} unique {config['description']}",
                )
                hops.append(hop_result)
                all_chunks.extend(unique_chunks)
                
                reasoning_chain.append(
                    f"  → Found {len(unique_chunks)} chunks: " +
                    ", ".join(c.section_title[:30] for c in unique_chunks)
                )
                
            except Exception as e:
                logger.error(f"Hop {hop_num} failed: {e}")
                reasoning_chain.append(f"  → Hop {hop_num} failed: {e}")
        
        # Calculate confidence based on hop coverage
        hop_coverage = len([h for h in hops if h.chunks]) / len(hop_types) if hop_types else 0
        chunk_quality = sum(
            getattr(c, 'similarity', 0.5) for c in all_chunks
        ) / len(all_chunks) if all_chunks else 0
        confidence = 0.5 * hop_coverage + 0.5 * chunk_quality
        
        reasoning_chain.append(f"Multi-hop complete: {len(all_chunks)} total chunks, {confidence:.0%} confidence")
        
        # Estimate tokens
        total_chars = sum(len(c.content) for c in all_chunks)
        
        return MultiHopContext(
            original_query=query,
            hops=hops,
            final_chunks=all_chunks,
            reasoning_chain=reasoning_chain,
            total_tokens_estimate=total_chars // 4,
            confidence=confidence,
        )
    
    def retrieve_for_breach(
        self,
        control_name: str,
        current_value: float,
        threshold: float,
        control_type: Optional[str] = None,
    ) -> MultiHopContext:
        """
        Specialized multi-hop for breach investigation.
        
        Retrieves:
        1. The violated policy
        2. Any applicable exceptions
        3. Escalation requirements
        4. Remediation procedures
        """
        query = f"{control_name} {current_value}% threshold {threshold}% breach violation"
        
        return self.retrieve_multi_hop(
            query=query,
            hop_types=["policy", "exception", "escalation", "remediation"],
            control_type=control_type,
        )


@dataclass
class RetrievedContext:
    """Context retrieved for narrative generation."""
    query: str
    chunks: List[PolicyChunk]
    total_tokens_estimate: int = 0
    retrieved_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # SOTA additions
    confidence: float = 0.0
    confidence_explanation: str = ""
    reasoning_trace: List[str] = field(default_factory=list)
    
    def to_prompt_context(self) -> str:
        """Format retrieved chunks for LLM prompt."""
        if not self.chunks:
            return "No relevant policy context found."
        
        context_parts = ["RELEVANT POLICY CONTEXT:"]
        context_parts.append("=" * 60)
        
        for i, chunk in enumerate(self.chunks, 1):
            similarity = getattr(chunk, 'similarity', 0.0)
            context_parts.append(f"\n[Source {i}: {chunk.document_name} | {chunk.section_title}]")
            context_parts.append(f"[Document ID: {chunk.document_id}]")
            context_parts.append(f"[Relevance: {similarity:.2%}]")
            context_parts.append("")
            context_parts.append(chunk.content)
            context_parts.append("")
            context_parts.append("-" * 40)
        
        return "\n".join(context_parts)
    
    def get_citations(self) -> List[str]:
        """Get citation strings for all retrieved chunks."""
        return [
            f"[Policy: {chunk.document_name} | Section: {chunk.section_title}]"
            for chunk in self.chunks
        ]


# Mapping from control types to search queries
CONTROL_TYPE_QUERIES = {
    "concentration": "concentration limits single issuer sector exposure maximum threshold breach",
    "issuer": "single issuer concentration limit 10% NAV maximum exposure",
    "sector": "sector concentration GICS technology limit 30% exposure",
    "liquidity": "liquidity bucket T+1 T+7 T+30 minimum required redemption",
    "exposure": "gross exposure net exposure long short leverage limit",
    "cash": "cash minimum buffer operational margin settlement",
    "counterparty": "counterparty prime broker OTC exposure limit credit",
    "exception": "exception breach escalation remediation cure action",
}


class RAGRetriever:
    """
    Retrieves relevant policy context for compliance control results.
    
    Uses semantic search (vector similarity) combined with control type filtering.
    
    SOTA Features:
    - Hybrid search: Dense embeddings + BM25 sparse retrieval
    - Confidence calibration
    - Configurable weights
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        embedder: LocalEmbedder,
        use_hybrid: bool = True,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        use_reranking: bool = True,
        rerank_top_k: int = 20,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        use_query_rewriting: bool = True,
        llm_client = None,
    ):
        self.vector_store = vector_store
        self.embedder = embedder
        self.use_hybrid = use_hybrid
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.use_reranking = use_reranking
        self.rerank_top_k = rerank_top_k
        self.use_query_rewriting = use_query_rewriting
        
        # SOTA components
        self.bm25_index = BM25Index()
        self.calibrator = ConfidenceCalibrator()
        self.reranker = CrossEncoderReranker(model_name=reranker_model) if use_reranking else None
        self.query_rewriter = QueryRewriter(llm_client=llm_client) if use_query_rewriting else None
        self._bm25_initialized = False
        self._chunk_metadata: Dict[str, PolicyChunk] = {}
    
    def _ensure_bm25_index(self):
        """Lazy initialization of BM25 index."""
        if self._bm25_initialized:
            return
        
        try:
            # Load all chunks for BM25
            with self.vector_store.conn.cursor() as cur:
                cur.execute("""
                    SELECT chunk_id, document_id, document_name, section_title, 
                           content, content_hash, control_types, keywords
                    FROM policy_chunks
                """)
                documents = {}
                for row in cur.fetchall():
                    chunk_id = row[0]
                    documents[chunk_id] = row[4]  # content
                    self._chunk_metadata[chunk_id] = PolicyChunk(
                        chunk_id=row[0],
                        document_id=row[1],
                        document_name=row[2],
                        section_title=row[3],
                        content=row[4],
                        content_hash=row[5],
                        control_types=row[6] or [],
                        keywords=row[7] or [],
                    )
            
            self.bm25_index.build_index(documents)
            self._bm25_initialized = True
            logger.info(f"BM25 index initialized with {len(documents)} chunks")
        except Exception as e:
            logger.warning(f"Failed to initialize BM25 index: {e}")
    
    def _hybrid_search(
        self,
        query: str,
        query_embedding: List[float],
        limit: int,
        control_types: Optional[List[str]] = None,
    ) -> List[PolicyChunk]:
        """
        Perform hybrid search combining dense and sparse retrieval.
        """
        self._ensure_bm25_index()
        
        # Dense search
        dense_results = self.vector_store.search_similar(
            query_embedding=query_embedding,
            limit=limit * 2,
            control_types=control_types,
        )
        
        # Sparse search (BM25)
        sparse_results = self.bm25_index.search(query, top_k=limit * 2)
        max_sparse = max((s for _, s in sparse_results), default=1) or 1
        
        # Combine scores
        chunk_scores: Dict[str, Dict[str, Any]] = {}
        
        for chunk in dense_results:
            chunk_scores[chunk.chunk_id] = {
                'dense': getattr(chunk, 'similarity', 0),
                'sparse': 0,
                'chunk': chunk
            }
        
        for chunk_id, score in sparse_results:
            normalized_sparse = score / max_sparse
            if chunk_id in chunk_scores:
                chunk_scores[chunk_id]['sparse'] = normalized_sparse
            elif chunk_id in self._chunk_metadata:
                chunk_scores[chunk_id] = {
                    'dense': 0,
                    'sparse': normalized_sparse,
                    'chunk': self._chunk_metadata[chunk_id]
                }
        
        # Calculate final scores
        results = []
        for chunk_id, scores in chunk_scores.items():
            final_score = (
                self.dense_weight * scores['dense'] + 
                self.sparse_weight * scores['sparse']
            )
            chunk = scores['chunk']
            chunk.similarity = scores['dense']
            chunk.final_score = final_score
            chunk.sparse_score = scores['sparse']
            results.append((final_score, chunk))
        
        # Sort by final score
        results.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in results[:limit]]
    
    def retrieve_for_control(
        self,
        control_name: str,
        control_type: str,
        status: str,
        calculated_value: float,
        threshold: float,
        limit: int = 3,
    ) -> RetrievedContext:
        """
        Retrieve relevant policy context for a specific control result.
        
        Args:
            control_name: Name of the control (e.g., "Sector Concentration - Technology")
            control_type: Type of control (e.g., "concentration")
            status: Control status ("pass", "warning", "fail")
            calculated_value: Actual value
            threshold: Threshold value
            limit: Max chunks to retrieve
        
        Returns:
            RetrievedContext with relevant policy chunks
        """
        reasoning_trace = []
        
        # Build search query
        base_query = CONTROL_TYPE_QUERIES.get(control_type, control_type)
        query = f"{control_name} {base_query}"
        if status in ("warning", "fail"):
            query += " breach threshold exceeded remediation action"
        
        original_query = query
        
        # Query rewriting (expand/clarify before retrieval)
        if self.use_query_rewriting and self.query_rewriter:
            query, rewrite_explanation = self.query_rewriter.rewrite(
                query=query,
                context=f"Control type: {control_type}, Status: {status}",
                use_llm=self.query_rewriter.llm_client is not None,
            )
            reasoning_trace.append(f"Query rewriting: {rewrite_explanation}")
            logger.debug(f"Rewritten query: {query[:100]}")
        
        logger.debug(f"RAG query: {query}")
        reasoning_trace.append(f"Final query: {query[:50]}...")
        
        # Generate query embedding
        try:
            query_embedding = self.embedder.embed(query)
        except Exception as e:
            logger.warning(f"Failed to embed query: {e}")
            chunks = self.vector_store.get_by_control_type(control_type)[:limit]
            return RetrievedContext(query=query, chunks=chunks)
        
        # Use hybrid search if enabled - fetch more candidates for reranking
        fetch_limit = self.rerank_top_k if self.use_reranking else limit
        
        if self.use_hybrid:
            chunks = self._hybrid_search(
                query=query,
                query_embedding=query_embedding,
                limit=fetch_limit,
                control_types=[control_type] if control_type else None,
            )
            reasoning_trace.append(f"Hybrid search (dense + BM25): {len(chunks)} candidates")
        else:
            chunks = self.vector_store.search_similar(
                query_embedding=query_embedding,
                limit=fetch_limit,
                control_types=[control_type] if control_type else None,
            )
            reasoning_trace.append(f"Dense search: {len(chunks)} candidates")
        
        # Cross-encoder reranking
        if self.use_reranking and self.reranker and self.reranker.available:
            chunks = self.reranker.rerank(query, chunks, top_k=limit)
            reasoning_trace.append(f"Cross-encoder reranked to top {len(chunks)}")
        else:
            chunks = chunks[:limit]
        
        # Calculate confidence
        confidence, conf_explanation = self.calibrator.calculate(query, chunks)
        reasoning_trace.append(f"Confidence: {confidence:.0%} ({conf_explanation})")
        
        # Estimate tokens
        total_chars = sum(len(c.content) for c in chunks)
        token_estimate = total_chars // 4
        
        return RetrievedContext(
            query=query,
            chunks=chunks,
            total_tokens_estimate=token_estimate,
            confidence=confidence,
            confidence_explanation=conf_explanation,
            reasoning_trace=reasoning_trace,
        )
    
    def retrieve_for_controls(
        self,
        control_results: List[Any],
        focus_on_issues: bool = True,
        limit_per_control: int = 2,
        max_total: int = 10,
    ) -> RetrievedContext:
        """
        Retrieve context for multiple control results.
        
        Args:
            control_results: List of control result objects
            focus_on_issues: If True, prioritize warnings and failures
            limit_per_control: Max chunks per control
            max_total: Max total chunks to return
        
        Returns:
            Combined RetrievedContext
        """
        all_chunks = []
        seen_chunk_ids = set()
        
        # Sort controls: failures first, then warnings, then passes
        sorted_controls = sorted(
            control_results,
            key=lambda c: {"fail": 0, "warning": 1, "pass": 2}.get(c.status, 3)
        )
        
        # Filter if focusing on issues
        if focus_on_issues:
            sorted_controls = [c for c in sorted_controls if c.status in ("fail", "warning")]
        
        for control in sorted_controls:
            if len(all_chunks) >= max_total:
                break
            
            context = self.retrieve_for_control(
                control_name=control.control_name,
                control_type=control.control_type,
                status=control.status,
                calculated_value=float(control.calculated_value),
                threshold=float(control.threshold),
                limit=limit_per_control,
            )
            
            for chunk in context.chunks:
                if chunk.chunk_id not in seen_chunk_ids:
                    all_chunks.append(chunk)
                    seen_chunk_ids.add(chunk.chunk_id)
        
        # Estimate tokens
        total_chars = sum(len(c.content) for c in all_chunks)
        
        return RetrievedContext(
            query="Multiple controls",
            chunks=all_chunks[:max_total],
            total_tokens_estimate=total_chars // 4,
        )
    
    def is_available(self) -> bool:
        """Check if RAG retrieval is available."""
        try:
            count = self.vector_store.count_chunks()
            return count > 0 and self.embedder.available
        except Exception:
            return False
    
    def retrieve_multi_hop(
        self,
        query: str,
        hop_types: Optional[List[str]] = None,
        control_type: Optional[str] = None,
    ) -> MultiHopContext:
        """
        Perform multi-hop retrieval for complex queries.
        
        Chains multiple retrievals together:
        1. Find primary policy
        2. Find exceptions/special cases
        3. Find historical precedents
        
        Args:
            query: Base query string
            hop_types: List of hop types (default: policy, exception, precedent)
            control_type: Optional control type filter
            
        Returns:
            MultiHopContext with consolidated results
        """
        multi_hop = MultiHopRetriever(
            vector_store=self.vector_store,
            embedder=self.embedder,
        )
        return multi_hop.retrieve_multi_hop(
            query=query,
            hop_types=hop_types,
            control_type=control_type,
        )
    
    def investigate_breach(
        self,
        control_name: str,
        current_value: float,
        threshold: float,
        control_type: Optional[str] = None,
    ) -> MultiHopContext:
        """
        Comprehensive breach investigation using multi-hop retrieval.
        
        Retrieves policy, exceptions, escalation, and remediation in sequence.
        
        Args:
            control_name: Name of the breached control
            current_value: Current value (e.g., 32.5 for 32.5%)
            threshold: Threshold value (e.g., 30.0 for 30%)
            control_type: Control type for filtering
            
        Returns:
            MultiHopContext with full investigation context
        """
        multi_hop = MultiHopRetriever(
            vector_store=self.vector_store,
            embedder=self.embedder,
            max_hops=4,
            chunks_per_hop=2,
        )
        return multi_hop.retrieve_for_breach(
            control_name=control_name,
            current_value=current_value,
            threshold=threshold,
            control_type=control_type,
        )
