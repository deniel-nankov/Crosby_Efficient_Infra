"""
Document Embedder for Policy Documents

Chunks and embeds policy documents using local embeddings via LM Studio
or falls back to simple keyword-based retrieval if embeddings unavailable.
"""

from __future__ import annotations

import os
import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from .vector_store import VectorStore, PolicyChunk

logger = logging.getLogger(__name__)


# Mapping from document names to control types
DOCUMENT_CONTROL_MAPPING = {
    "concentration_limits.md": ["concentration", "issuer", "sector", "geographic"],
    "liquidity_policy.md": ["liquidity", "cash", "redemption"],
    "investment_guidelines.md": ["exposure", "concentration", "liquidity", "counterparty"],
    "exposure_limits.md": ["exposure", "gross", "net", "leverage"],
    "exception_management.md": ["exception", "breach", "escalation"],
    "sec_compliance.md": ["regulatory", "filing", "disclosure"],
    "commodity_trading.md": ["commodity", "derivatives", "trading"],
}


class LocalEmbedder:
    """
    Generate embeddings using LM Studio's local API.
    
    Uses the nomic-embed-text model which produces 768-dimensional vectors.
    """
    
    def __init__(
        self,
        api_base: str = "http://localhost:1234/v1",
        model: str = "text-embedding-nomic-embed-text-v1.5",
    ):
        self.api_base = api_base
        self.model = model
        self._available = None
    
    @property
    def available(self) -> bool:
        """Check if embedding service is available."""
        if self._available is None:
            try:
                import requests
                response = requests.get(f"{self.api_base}/models", timeout=2)
                self._available = response.status_code == 200
            except Exception:
                self._available = False
        return self._available
    
    def embed(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        import requests
        
        response = requests.post(
            f"{self.api_base}/embeddings",
            json={"model": self.model, "input": text},
            timeout=30,
        )
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        import requests
        
        response = requests.post(
            f"{self.api_base}/embeddings",
            json={"model": self.model, "input": texts},
            timeout=60,
        )
        response.raise_for_status()
        
        data = response.json()["data"]
        # Sort by index to maintain order
        sorted_data = sorted(data, key=lambda x: x["index"])
        return [item["embedding"] for item in sorted_data]


def chunk_markdown(content: str, document_name: str) -> List[Dict[str, Any]]:
    """
    Split markdown content into chunks by section.
    
    Returns list of {section_title, content, document_id}
    """
    chunks = []
    
    # Extract document ID if present
    doc_id_match = re.search(r"Document ID:\s*(\S+)", content)
    document_id = doc_id_match.group(1) if doc_id_match else document_name
    
    # Split by headers (##)
    sections = re.split(r'\n(?=##\s)', content)
    
    current_section = "Introduction"
    
    for section in sections:
        if not section.strip():
            continue
        
        # Extract section title
        title_match = re.match(r'##\s*(.+?)(?:\n|$)', section)
        if title_match:
            current_section = title_match.group(1).strip()
        
        # Clean content
        section_content = section.strip()
        
        # Skip very short sections
        if len(section_content) < 50:
            continue
        
        # If section is too long, split by subsections (###)
        if len(section_content) > 1500:
            subsections = re.split(r'\n(?=###\s)', section_content)
            for subsection in subsections:
                if len(subsection.strip()) > 50:
                    sub_title_match = re.match(r'###\s*(.+?)(?:\n|$)', subsection)
                    sub_title = sub_title_match.group(1).strip() if sub_title_match else current_section
                    chunks.append({
                        "section_title": f"{current_section} - {sub_title}" if sub_title != current_section else current_section,
                        "content": subsection.strip(),
                        "document_id": document_id,
                    })
        else:
            chunks.append({
                "section_title": current_section,
                "content": section_content,
                "document_id": document_id,
            })
    
    return chunks


def embed_policies(
    policies_dir: Path,
    vector_store: VectorStore,
    embedder: LocalEmbedder,
) -> int:
    """
    Embed all policy documents and store in vector database.
    
    Args:
        policies_dir: Path to policies directory
        vector_store: VectorStore instance
        embedder: LocalEmbedder instance
    
    Returns:
        Number of chunks embedded
    """
    # Initialize schema
    vector_store.initialize()
    
    total_chunks = 0
    
    for policy_file in policies_dir.glob("*.md"):
        document_name = policy_file.name
        control_types = DOCUMENT_CONTROL_MAPPING.get(document_name, [])
        
        logger.info(f"Processing {document_name}...")
        
        # Read and chunk
        content = policy_file.read_text(encoding="utf-8")
        chunks = chunk_markdown(content, document_name)
        
        # Embed and store
        for chunk_data in chunks:
            # Create chunk
            chunk = PolicyChunk.from_text(
                content=chunk_data["content"],
                document_name=document_name,
                section_title=chunk_data["section_title"],
                document_id=chunk_data["document_id"],
                control_types=control_types,
            )
            
            # Generate embedding
            try:
                chunk.embedding = embedder.embed(chunk.content)
            except Exception as e:
                logger.warning(f"Failed to embed chunk {chunk.chunk_id}: {e}")
                continue
            
            # Store
            vector_store.upsert_chunk(chunk)
            total_chunks += 1
        
        logger.info(f"  Embedded {len(chunks)} chunks from {document_name}")
    
    logger.info(f"Total chunks embedded: {total_chunks}")
    return total_chunks


def embed_policies_cli():
    """CLI entry point for embedding policies."""
    import argparse
    
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    
    parser = argparse.ArgumentParser(description="Embed policy documents for RAG")
    parser.add_argument(
        "--policies-dir",
        type=Path,
        default=Path(__file__).parent.parent.parent / "policies",
        help="Path to policies directory",
    )
    parser.add_argument(
        "--db-host", default=os.environ.get("DB_HOST", "localhost"),
    )
    parser.add_argument(
        "--db-port", type=int, default=int(os.environ.get("DB_PORT", "5433")),
    )
    parser.add_argument(
        "--embedding-api", default="http://localhost:1234/v1",
        help="LM Studio embedding API base URL",
    )
    
    args = parser.parse_args()
    
    # Initialize
    connection_params = {
        "host": args.db_host,
        "port": args.db_port,
        "database": "compliance",
        "user": "compliance_user",
        "password": "compliance_dev_password_123",
    }
    
    vector_store = VectorStore(connection_params)
    embedder = LocalEmbedder(api_base=args.embedding_api)
    
    if not embedder.available:
        print("ERROR: Embedding service not available at", args.embedding_api)
        print("Please start LM Studio server with an embedding model loaded.")
        return 1
    
    print(f"Embedding policies from {args.policies_dir}")
    print(f"Using embedding API at {args.embedding_api}")
    print()
    
    count = embed_policies(args.policies_dir, vector_store, embedder)
    
    print()
    print(f"Successfully embedded {count} policy chunks")
    print("RAG pipeline is now ready!")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(embed_policies_cli())
