"""
RAG (Retrieval-Augmented Generation) Pipeline

This module provides true RAG functionality:
1. Embed and store policy documents in pgvector
2. Retrieve relevant context based on control results
3. Ground LLM generation in actual policy text
"""

from .vector_store import VectorStore, PolicyChunk
from .embedder import LocalEmbedder, embed_policies
from .retriever import RAGRetriever

__all__ = [
    "VectorStore",
    "PolicyChunk", 
    "LocalEmbedder",
    "embed_policies",
    "RAGRetriever",
]
