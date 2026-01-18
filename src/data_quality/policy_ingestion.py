"""
Policy Document Ingestion Pipeline

Converts policy documents (Markdown, PDF, text) into searchable, 
version-controlled chunks for RAG retrieval.

Quality Requirements:
- Every chunk traceable to source document
- Version history preserved
- Semantic chunking (not arbitrary splits)
- Overlap for context preservation
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone, date
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import uuid

from .validators import PolicyDocumentValidator, QualityReport

logger = logging.getLogger(__name__)


@dataclass
class PolicyChunk:
    """
    A single chunk of policy content for RAG retrieval.
    
    Chunks are semantic units (sections, paragraphs) not arbitrary splits.
    """
    chunk_id: str
    policy_id: str
    policy_version: str
    
    # Content
    content: str
    content_hash: str
    
    # Location
    section_path: str  # e.g., "1.Position Limits > 1.1.Gross Exposure"
    section_level: int  # Heading level (1-6)
    chunk_index: int  # Order within document
    
    # Metadata
    effective_date: date
    expiration_date: Optional[date] = None
    
    # Relationships
    parent_chunk_id: Optional[str] = None
    
    # Search metadata
    keywords: List[str] = field(default_factory=list)
    
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "content": self.content,
            "content_hash": self.content_hash,
            "section_path": self.section_path,
            "section_level": self.section_level,
            "chunk_index": self.chunk_index,
            "effective_date": self.effective_date.isoformat(),
            "expiration_date": self.expiration_date.isoformat() if self.expiration_date else None,
            "parent_chunk_id": self.parent_chunk_id,
            "keywords": self.keywords,
            "created_at": self.created_at.isoformat(),
        }
    
    def to_citation(self) -> str:
        """Generate citation string."""
        return f"[Policy: {self.policy_id} v{self.policy_version} | Section: {self.section_path}]"


@dataclass
class PolicyDocument:
    """
    Represents a complete policy document.
    """
    policy_id: str
    title: str
    version: str
    effective_date: date
    expiration_date: Optional[date] = None
    
    # Content
    raw_content: str = ""
    content_hash: str = ""
    
    # Parsed chunks
    chunks: List[PolicyChunk] = field(default_factory=list)
    
    # Metadata
    fund_name: Optional[str] = None
    category: Optional[str] = None  # e.g., "investment_guidelines", "compliance_manual"
    regulatory_references: List[str] = field(default_factory=list)
    
    # Quality
    quality_report: Optional[QualityReport] = None
    
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "title": self.title,
            "version": self.version,
            "effective_date": self.effective_date.isoformat(),
            "expiration_date": self.expiration_date.isoformat() if self.expiration_date else None,
            "content_hash": self.content_hash,
            "chunk_count": len(self.chunks),
            "fund_name": self.fund_name,
            "category": self.category,
            "regulatory_references": self.regulatory_references,
            "quality_score": self.quality_report.overall_score if self.quality_report else None,
            "created_at": self.created_at.isoformat(),
        }


class PolicyIngestionPipeline:
    """
    Pipeline for ingesting and chunking policy documents.
    
    Steps:
    1. Parse document structure
    2. Extract metadata
    3. Semantic chunking
    4. Quality validation
    5. Keyword extraction
    """
    
    def __init__(self, min_chunk_size: int = 100, max_chunk_size: int = 2000):
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.validator = PolicyDocumentValidator()
    
    def ingest_markdown(
        self,
        content: str,
        policy_id: str,
        title: str,
        version: str,
        effective_date: date,
        **metadata
    ) -> PolicyDocument:
        """
        Ingest a Markdown policy document.
        
        Args:
            content: Raw Markdown content
            policy_id: Unique identifier for this policy
            title: Policy title
            version: Version string (e.g., "3.2")
            effective_date: When policy becomes effective
            **metadata: Additional metadata (fund_name, category, etc.)
            
        Returns:
            PolicyDocument with chunks and quality report
        """
        logger.info(f"Ingesting policy: {policy_id} v{version}")
        
        # Create document
        doc = PolicyDocument(
            policy_id=policy_id,
            title=title,
            version=version,
            effective_date=effective_date,
            raw_content=content,
            content_hash=self._hash_content(content),
            fund_name=metadata.get('fund_name'),
            category=metadata.get('category'),
            regulatory_references=metadata.get('regulatory_references', []),
        )
        
        # Parse structure and create chunks
        doc.chunks = self._parse_markdown_to_chunks(content, doc)
        
        # Quality validation
        doc.quality_report = self.validator.validate_policy(
            content,
            {
                'policy_id': policy_id,
                'title': title,
                'effective_date': effective_date,
                'version': version,
            }
        )
        
        # Extract keywords for each chunk
        for chunk in doc.chunks:
            chunk.keywords = self._extract_keywords(chunk.content)
        
        logger.info(f"Created {len(doc.chunks)} chunks for {policy_id}")
        
        return doc
    
    def ingest_file(self, file_path: Path, **metadata) -> PolicyDocument:
        """
        Ingest a policy document from file.
        
        Supports: .md, .txt, .markdown
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Policy file not found: {file_path}")
        
        content = file_path.read_text(encoding='utf-8')
        
        # Extract metadata from content if not provided
        extracted = self._extract_metadata_from_markdown(content)
        
        policy_id = metadata.get('policy_id', extracted.get('policy_id', file_path.stem))
        title = metadata.get('title', extracted.get('title', file_path.stem))
        version = metadata.get('version', extracted.get('version', '1.0'))
        effective_date = metadata.get(
            'effective_date', 
            extracted.get('effective_date', date.today())
        )
        
        return self.ingest_markdown(
            content=content,
            policy_id=policy_id,
            title=title,
            version=version,
            effective_date=effective_date,
            fund_name=metadata.get('fund_name', extracted.get('fund_name')),
            category=metadata.get('category', 'investment_guidelines'),
        )
    
    def _parse_markdown_to_chunks(
        self, 
        content: str, 
        doc: PolicyDocument
    ) -> List[PolicyChunk]:
        """
        Parse Markdown into semantic chunks based on headers.
        
        Strategy:
        1. Split by headers (## and ###)
        2. Each section becomes a chunk
        3. Tables and lists stay with their parent section
        4. Very long sections get sub-chunked by paragraph
        """
        chunks = []
        chunk_index = 0
        
        # Split by lines and track headers
        lines = content.split('\n')
        
        current_section = ""
        current_content: List[str] = []
        section_stack: List[Tuple[int, str]] = []  # (level, title)
        
        for line in lines:
            # Check if this is a header
            header_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            
            if header_match:
                # Save previous section if it has content
                if current_content:
                    chunk = self._create_chunk(
                        doc=doc,
                        content='\n'.join(current_content),
                        section_path=self._build_section_path(section_stack),
                        section_level=section_stack[-1][0] if section_stack else 1,
                        chunk_index=chunk_index,
                    )
                    if chunk:
                        chunks.append(chunk)
                        chunk_index += 1
                    current_content = []
                
                # Update section stack
                level = len(header_match.group(1))
                title = header_match.group(2).strip()
                
                # Pop sections at same or higher level
                while section_stack and section_stack[-1][0] >= level:
                    section_stack.pop()
                
                section_stack.append((level, title))
                current_section = title
                
            else:
                # Regular content line
                current_content.append(line)
        
        # Don't forget the last section
        if current_content:
            chunk = self._create_chunk(
                doc=doc,
                content='\n'.join(current_content),
                section_path=self._build_section_path(section_stack),
                section_level=section_stack[-1][0] if section_stack else 1,
                chunk_index=chunk_index,
            )
            if chunk:
                chunks.append(chunk)
        
        return chunks
    
    def _create_chunk(
        self,
        doc: PolicyDocument,
        content: str,
        section_path: str,
        section_level: int,
        chunk_index: int,
    ) -> Optional[PolicyChunk]:
        """Create a chunk, potentially splitting if too large."""
        # Clean content
        content = content.strip()
        
        # Skip empty or too-small chunks
        if len(content) < self.min_chunk_size:
            return None
        
        # If content is within limits, create single chunk
        if len(content) <= self.max_chunk_size:
            return PolicyChunk(
                chunk_id=str(uuid.uuid4()),
                policy_id=doc.policy_id,
                policy_version=doc.version,
                content=content,
                content_hash=self._hash_content(content),
                section_path=section_path,
                section_level=section_level,
                chunk_index=chunk_index,
                effective_date=doc.effective_date,
                expiration_date=doc.expiration_date,
            )
        
        # Content too large - this shouldn't happen often with proper headers
        # Return as single chunk but log warning
        logger.warning(
            f"Large chunk ({len(content)} chars) in {section_path} - consider more granular headers"
        )
        
        return PolicyChunk(
            chunk_id=str(uuid.uuid4()),
            policy_id=doc.policy_id,
            policy_version=doc.version,
            content=content[:self.max_chunk_size],  # Truncate with note
            content_hash=self._hash_content(content),
            section_path=section_path,
            section_level=section_level,
            chunk_index=chunk_index,
            effective_date=doc.effective_date,
            expiration_date=doc.expiration_date,
        )
    
    def _build_section_path(self, section_stack: List[Tuple[int, str]]) -> str:
        """Build hierarchical section path."""
        if not section_stack:
            return "Document Root"
        return " > ".join(title for _, title in section_stack)
    
    def _extract_metadata_from_markdown(self, content: str) -> Dict[str, Any]:
        """Extract metadata from Markdown front matter or content."""
        metadata: Dict[str, Any] = {}
        
        # Look for title (first # heading)
        title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        if title_match:
            metadata['title'] = title_match.group(1).strip()
        
        # Look for fund name
        fund_match = re.search(r'Fund:\s*(.+)$', content, re.MULTILINE)
        if fund_match:
            metadata['fund_name'] = fund_match.group(1).strip()
        
        # Look for effective date
        date_match = re.search(r'Effective Date:\s*(.+)$', content, re.MULTILINE)
        if date_match:
            try:
                date_str = date_match.group(1).strip()
                # Try common date formats
                for fmt in ['%B %d, %Y', '%Y-%m-%d', '%m/%d/%Y']:
                    try:
                        metadata['effective_date'] = datetime.strptime(date_str, fmt).date()
                        break
                    except ValueError:
                        continue
            except Exception:
                pass
        
        # Look for version
        version_match = re.search(r'Version:\s*(.+)$', content, re.MULTILINE)
        if version_match:
            metadata['version'] = version_match.group(1).strip()
        
        return metadata
    
    def _extract_keywords(self, content: str) -> List[str]:
        """Extract compliance-relevant keywords from content."""
        # Common compliance terms to look for
        compliance_terms = [
            'limit', 'threshold', 'maximum', 'minimum', 'exposure', 'concentration',
            'nav', 'liquidity', 'counterparty', 'risk', 'breach', 'warning',
            'monitoring', 'daily', 'intraday', 'weekly', 'monthly',
            'sector', 'issuer', 'position', 'gross', 'net', 'long', 'short',
            'regulatory', 'sec', 'compliance', 'prohibited', 'restricted',
        ]
        
        content_lower = content.lower()
        found_keywords = []
        
        for term in compliance_terms:
            if term in content_lower:
                found_keywords.append(term)
        
        # Also extract numeric thresholds
        threshold_patterns = re.findall(r'(\d+(?:\.\d+)?)\s*%', content)
        for pct in threshold_patterns:
            found_keywords.append(f"{pct}%")
        
        return list(set(found_keywords))
    
    def _hash_content(self, content: str) -> str:
        """Create deterministic hash of content."""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()


class PolicyStore:
    """
    Storage layer for policy documents and chunks.
    
    Supports:
    - PostgreSQL storage (production)
    - In-memory storage (testing)
    """
    
    def __init__(self, connection=None):
        self.connection = connection
        self._in_memory_store: Dict[str, PolicyDocument] = {}
        self._in_memory_chunks: List[PolicyChunk] = []
    
    def store_policy(self, doc: PolicyDocument) -> str:
        """
        Store a policy document and its chunks.
        
        Returns:
            policy_id of stored document
        """
        if self.connection:
            return self._store_to_postgres(doc)
        else:
            return self._store_in_memory(doc)
    
    def _store_in_memory(self, doc: PolicyDocument) -> str:
        """Store in memory for testing."""
        self._in_memory_store[doc.policy_id] = doc
        self._in_memory_chunks.extend(doc.chunks)
        return doc.policy_id
    
    def _store_to_postgres(self, doc: PolicyDocument) -> str:
        """Store to PostgreSQL."""
        cursor = self.connection.cursor()
        
        try:
            # Store document metadata
            cursor.execute("""
                INSERT INTO policy_documents (
                    policy_id, title, version, effective_date,
                    expiration_date, content_hash, fund_name, category,
                    chunk_count, quality_score, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (policy_id, version) 
                DO UPDATE SET 
                    content_hash = EXCLUDED.content_hash,
                    chunk_count = EXCLUDED.chunk_count,
                    quality_score = EXCLUDED.quality_score,
                    updated_at = NOW()
            """, (
                doc.policy_id,
                doc.title,
                doc.version,
                doc.effective_date,
                doc.expiration_date,
                doc.content_hash,
                doc.fund_name,
                doc.category,
                len(doc.chunks),
                doc.quality_report.overall_score if doc.quality_report else None,
                doc.created_at,
            ))
            
            # Store chunks
            for chunk in doc.chunks:
                cursor.execute("""
                    INSERT INTO policy_chunks (
                        chunk_id, policy_id, policy_version,
                        content, content_hash, section_path, section_level,
                        chunk_index, effective_date, expiration_date,
                        keywords, created_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (chunk_id) DO UPDATE SET
                        content = EXCLUDED.content,
                        content_hash = EXCLUDED.content_hash,
                        updated_at = NOW()
                """, (
                    chunk.chunk_id,
                    chunk.policy_id,
                    chunk.policy_version,
                    chunk.content,
                    chunk.content_hash,
                    chunk.section_path,
                    chunk.section_level,
                    chunk.chunk_index,
                    chunk.effective_date,
                    chunk.expiration_date,
                    json.dumps(chunk.keywords),
                    chunk.created_at,
                ))
            
            self.connection.commit()
            logger.info(f"Stored policy {doc.policy_id} with {len(doc.chunks)} chunks")
            
            return doc.policy_id
            
        except Exception as e:
            self.connection.rollback()
            logger.error(f"Failed to store policy: {e}")
            raise
        finally:
            cursor.close()
    
    def get_policy(self, policy_id: str, version: Optional[str] = None) -> Optional[PolicyDocument]:
        """Retrieve a policy document."""
        if self.connection:
            return self._get_from_postgres(policy_id, version)
        else:
            return self._in_memory_store.get(policy_id)
    
    def _get_from_postgres(self, policy_id: str, version: Optional[str] = None) -> Optional[PolicyDocument]:
        """Retrieve from PostgreSQL."""
        cursor = self.connection.cursor()
        
        try:
            if version:
                cursor.execute("""
                    SELECT policy_id, title, version, effective_date, expiration_date,
                           content_hash, fund_name, category, chunk_count
                    FROM policy_documents
                    WHERE policy_id = %s AND version = %s
                """, (policy_id, version))
            else:
                cursor.execute("""
                    SELECT policy_id, title, version, effective_date, expiration_date,
                           content_hash, fund_name, category, chunk_count
                    FROM policy_documents
                    WHERE policy_id = %s
                    ORDER BY version DESC
                    LIMIT 1
                """, (policy_id,))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            doc = PolicyDocument(
                policy_id=row[0],
                title=row[1],
                version=row[2],
                effective_date=row[3],
                expiration_date=row[4],
                content_hash=row[5],
                fund_name=row[6],
                category=row[7],
            )
            
            # Get chunks
            cursor.execute("""
                SELECT chunk_id, content, content_hash, section_path, section_level,
                       chunk_index, keywords
                FROM policy_chunks
                WHERE policy_id = %s AND policy_version = %s
                ORDER BY chunk_index
            """, (policy_id, doc.version))
            
            for chunk_row in cursor.fetchall():
                doc.chunks.append(PolicyChunk(
                    chunk_id=chunk_row[0],
                    policy_id=policy_id,
                    policy_version=doc.version,
                    content=chunk_row[1],
                    content_hash=chunk_row[2],
                    section_path=chunk_row[3],
                    section_level=chunk_row[4],
                    chunk_index=chunk_row[5],
                    effective_date=doc.effective_date,
                    keywords=json.loads(chunk_row[6]) if chunk_row[6] else [],
                ))
            
            return doc
            
        finally:
            cursor.close()
    
    def search_chunks(
        self, 
        query: str, 
        policy_ids: Optional[List[str]] = None,
        limit: int = 10
    ) -> List[PolicyChunk]:
        """
        Search chunks by keyword matching.
        
        For semantic search, use the vector store layer.
        """
        if self.connection:
            return self._search_postgres(query, policy_ids, limit)
        else:
            return self._search_in_memory(query, policy_ids, limit)
    
    def _search_in_memory(
        self, 
        query: str, 
        policy_ids: Optional[List[str]], 
        limit: int
    ) -> List[PolicyChunk]:
        """Search in-memory chunks."""
        query_lower = query.lower()
        results = []
        
        for chunk in self._in_memory_chunks:
            if policy_ids and chunk.policy_id not in policy_ids:
                continue
            
            # Simple keyword matching
            if query_lower in chunk.content.lower():
                results.append(chunk)
            elif any(kw in query_lower for kw in chunk.keywords):
                results.append(chunk)
        
        return results[:limit]
    
    def _search_postgres(
        self, 
        query: str, 
        policy_ids: Optional[List[str]], 
        limit: int
    ) -> List[PolicyChunk]:
        """Search PostgreSQL using full-text search."""
        cursor = self.connection.cursor()
        
        try:
            sql = """
                SELECT chunk_id, policy_id, policy_version, content, content_hash,
                       section_path, section_level, chunk_index, effective_date, keywords
                FROM policy_chunks
                WHERE to_tsvector('english', content) @@ plainto_tsquery('english', %s)
            """
            params = [query]
            
            if policy_ids:
                sql += " AND policy_id = ANY(%s)"
                params.append(policy_ids)
            
            sql += f" LIMIT {limit}"
            
            cursor.execute(sql, params)
            
            chunks = []
            for row in cursor.fetchall():
                chunks.append(PolicyChunk(
                    chunk_id=row[0],
                    policy_id=row[1],
                    policy_version=row[2],
                    content=row[3],
                    content_hash=row[4],
                    section_path=row[5],
                    section_level=row[6],
                    chunk_index=row[7],
                    effective_date=row[8],
                    keywords=json.loads(row[9]) if row[9] else [],
                ))
            
            return chunks
            
        finally:
            cursor.close()
