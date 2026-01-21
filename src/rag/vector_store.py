"""
Vector Store for Policy Document Embeddings

Uses PostgreSQL with pgvector extension for efficient similarity search.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
import json

logger = logging.getLogger(__name__)


@dataclass
class PolicyChunk:
    # Dynamic attributes set at runtime (for type checking)
    similarity: float = field(default=0.0, init=False, repr=False)
    rerank_score: float = field(default=0.0, init=False, repr=False)
    final_score: float = field(default=0.0, init=False, repr=False)
    sparse_score: float = field(default=0.0, init=False, repr=False)
    """A chunk of policy document with its embedding."""
    chunk_id: str
    document_id: str           # e.g., "POL-CONC-001"
    document_name: str         # e.g., "concentration_limits.md"
    section_title: str         # e.g., "3.1 Limit Definition"
    content: str               # The actual text
    content_hash: str          # SHA-256 of content
    embedding: Optional[List[float]] = None
    
    # Metadata for retrieval
    control_types: List[str] = field(default_factory=list)  # ["concentration", "issuer"]
    keywords: List[str] = field(default_factory=list)
    
    # Audit
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @classmethod
    def from_text(
        cls,
        content: str,
        document_name: str,
        section_title: str,
        document_id: str = "",
        control_types: Optional[List[str]] = None,
    ) -> "PolicyChunk":
        """Create a chunk from text content."""
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        chunk_id = f"{document_name}:{section_title}:{content_hash[:8]}"
        
        return cls(
            chunk_id=chunk_id,
            document_id=document_id,
            document_name=document_name,
            section_title=section_title,
            content=content,
            content_hash=content_hash,
            control_types=control_types or [],
        )


class VectorStore:
    """
    PostgreSQL + pgvector store for policy embeddings.
    
    Uses the existing compliance-postgres container with pgvector extension.
    """
    
    # Schema for vector storage
    SCHEMA_SQL = """
    -- Enable pgvector extension
    CREATE EXTENSION IF NOT EXISTS vector;
    
    -- Policy chunks table with embeddings
    CREATE TABLE IF NOT EXISTS policy_chunks (
        chunk_id TEXT PRIMARY KEY,
        document_id TEXT NOT NULL,
        document_name TEXT NOT NULL,
        section_title TEXT NOT NULL,
        content TEXT NOT NULL,
        content_hash TEXT NOT NULL,
        embedding vector(768),  -- nomic-embed-text uses 768 dimensions
        control_types TEXT[],   -- array of control types this applies to
        keywords TEXT[],
        created_at TIMESTAMPTZ DEFAULT NOW(),
        
        -- Indexes for fast retrieval
        CONSTRAINT unique_content UNIQUE (content_hash)
    );
    
    -- Index for vector similarity search
    CREATE INDEX IF NOT EXISTS policy_chunks_embedding_idx 
    ON policy_chunks USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 10);
    
    -- Index for control type filtering
    CREATE INDEX IF NOT EXISTS policy_chunks_control_types_idx 
    ON policy_chunks USING GIN (control_types);
    """
    
    def __init__(self, connection_params: Dict[str, Any]):
        """
        Initialize vector store with database connection.
        
        Args:
            connection_params: Dict with host, port, database, user, password
        """
        self.connection_params = connection_params
        self._conn = None
    
    @property
    def conn(self):
        """Lazy database connection."""
        if self._conn is None:
            import psycopg2
            self._conn = psycopg2.connect(**self.connection_params)
        return self._conn
    
    def initialize(self) -> None:
        """Create schema if it doesn't exist."""
        with self.conn.cursor() as cur:
            cur.execute(self.SCHEMA_SQL)
        self.conn.commit()
        logger.info("Vector store schema initialized")
    
    def upsert_chunk(self, chunk: PolicyChunk) -> None:
        """Insert or update a policy chunk."""
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO policy_chunks 
                (chunk_id, document_id, document_name, section_title, 
                 content, content_hash, embedding, control_types, keywords)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (chunk_id) 
                DO UPDATE SET 
                    embedding = EXCLUDED.embedding,
                    control_types = EXCLUDED.control_types,
                    keywords = EXCLUDED.keywords
            """, (
                chunk.chunk_id,
                chunk.document_id,
                chunk.document_name,
                chunk.section_title,
                chunk.content,
                chunk.content_hash,
                chunk.embedding,
                chunk.control_types,
                chunk.keywords,
            ))
        self.conn.commit()
    
    def search_similar(
        self,
        query_embedding: List[float],
        limit: int = 5,
        control_types: Optional[List[str]] = None,
    ) -> List[PolicyChunk]:
        """
        Search for similar policy chunks.
        
        Args:
            query_embedding: Query vector
            limit: Max results to return
            control_types: Optional filter by control types
        
        Returns:
            List of PolicyChunk sorted by similarity
        """
        with self.conn.cursor() as cur:
            if control_types:
                # Filter by control types
                cur.execute("""
                    SELECT chunk_id, document_id, document_name, section_title,
                           content, content_hash, control_types, keywords,
                           1 - (embedding <=> %s::vector) as similarity
                    FROM policy_chunks
                    WHERE control_types && %s  -- array overlap
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                """, (query_embedding, control_types, query_embedding, limit))
            else:
                cur.execute("""
                    SELECT chunk_id, document_id, document_name, section_title,
                           content, content_hash, control_types, keywords,
                           1 - (embedding <=> %s::vector) as similarity
                    FROM policy_chunks
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                """, (query_embedding, query_embedding, limit))
            
            results = []
            for row in cur.fetchall():
                chunk = PolicyChunk(
                    chunk_id=row[0],
                    document_id=row[1],
                    document_name=row[2],
                    section_title=row[3],
                    content=row[4],
                    content_hash=row[5],
                    control_types=row[6] or [],
                    keywords=row[7] or [],
                )
                # Add similarity score as attribute
                chunk.similarity = row[8]
                results.append(chunk)
            
            return results
    
    def get_by_control_type(self, control_type: str) -> List[PolicyChunk]:
        """Get all chunks for a specific control type."""
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT chunk_id, document_id, document_name, section_title,
                       content, content_hash, control_types, keywords
                FROM policy_chunks
                WHERE %s = ANY(control_types)
                ORDER BY document_name, section_title
            """, (control_type,))
            
            return [
                PolicyChunk(
                    chunk_id=row[0],
                    document_id=row[1],
                    document_name=row[2],
                    section_title=row[3],
                    content=row[4],
                    content_hash=row[5],
                    control_types=row[6] or [],
                    keywords=row[7] or [],
                )
                for row in cur.fetchall()
            ]
    
    def count_chunks(self) -> int:
        """Count total chunks in store."""
        with self.conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM policy_chunks")
            return cur.fetchone()[0]
    
    def close(self):
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
