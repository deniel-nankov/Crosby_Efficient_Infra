"""
RAG Retriever - Retrieves relevant policy context for compliance narratives

This is the core RAG component that:
1. Takes control results as input
2. Searches for relevant policy sections
3. Returns grounded context for LLM generation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

from .vector_store import VectorStore, PolicyChunk
from .embedder import LocalEmbedder

logger = logging.getLogger(__name__)


@dataclass
class RetrievedContext:
    """Context retrieved for narrative generation."""
    query: str
    chunks: List[PolicyChunk]
    total_tokens_estimate: int = 0
    retrieved_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
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
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        embedder: LocalEmbedder,
    ):
        self.vector_store = vector_store
        self.embedder = embedder
    
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
        # Build search query
        base_query = CONTROL_TYPE_QUERIES.get(control_type, control_type)
        
        # Add control-specific context
        query = f"{control_name} {base_query}"
        if status in ("warning", "fail"):
            query += " breach threshold exceeded remediation action"
        
        logger.debug(f"RAG query: {query}")
        
        # Generate query embedding
        try:
            query_embedding = self.embedder.embed(query)
        except Exception as e:
            logger.warning(f"Failed to embed query: {e}")
            # Fallback to control type filtering only
            chunks = self.vector_store.get_by_control_type(control_type)[:limit]
            return RetrievedContext(query=query, chunks=chunks)
        
        # Search with control type filter
        chunks = self.vector_store.search_similar(
            query_embedding=query_embedding,
            limit=limit,
            control_types=[control_type] if control_type else None,
        )
        
        # Estimate tokens (rough: 4 chars per token)
        total_chars = sum(len(c.content) for c in chunks)
        token_estimate = total_chars // 4
        
        return RetrievedContext(
            query=query,
            chunks=chunks,
            total_tokens_estimate=token_estimate,
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
